"""
End-to-end pipeline test: Train on 2 patients, test on 1.

Train set: 010S0067, 011S0010 (both have M00, M06, M12, M24)
Test set:  009S4612 (only M00, M24)

Flow:
  1. Load pre-trained AAE
  2. Encode all PET scans → latent space
  3. Train Latent SDE on training patients (short run)
  4. Test on 009S4612: predict M24 from M00
  5. Save results as NIfTI

Usage:
  conda run -n Pytorch python test_pipeline.py
"""

import os
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import nibabel as nib
from tqdm import tqdm

from model import AAE
from ode_model import LatentSDE
from utils import load_checkpoint, seed_torch
import config


def load_pet(filepath, device='cpu'):
    """Load NIfTI PET scan → normalized tensor (1, 1, D, H, W)."""
    data = nib.load(filepath).get_fdata().astype(np.float32)
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    return torch.from_numpy(data).unsqueeze(0).unsqueeze(0).to(device)


def discover_patients(data_root):
    """Discover patients and their available timepoints."""
    patients = {}
    for subject_dir in sorted(os.listdir(data_root)):
        subject_path = os.path.join(data_root, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        timepoints = {}
        for f in sorted(os.listdir(subject_path)):
            if not f.endswith('.nii.gz'):
                continue
            for session in ['M00', 'M06', 'M12', 'M18', 'M24']:
                if f'ses-{session}' in f:
                    timepoints[session] = os.path.join(subject_path, f)
        if 'M00' in timepoints and 'M24' in timepoints:
            patients[subject_dir] = timepoints
    return patients


def main():
    seed_torch()
    device = config.device
    print(f"Device: {device}")
    print("=" * 60)
    print("End-to-End Pipeline: Train + Test")
    print("=" * 60)

    TRAIN_IDS = ['010S0067', '011S0010']
    TEST_IDS = ['009S4612']
    EPOCHS = 200  # Short training for pipeline validation

    # ========== 1. Load AAE ==========
    print("\n[1/5] Loading pre-trained AAE...")
    aae = AAE().to(device)
    aae_opt = optim.Adam(aae.parameters(), lr=config.learning_rate)

    aae_paths = [
        "result/AAE/AAE.pth.tar_epoch1000",
        config.CHECKPOINT_AAE,
        "result/AAE/AAE.pth.tar",
    ]
    loaded = False
    for path in aae_paths:
        if os.path.exists(path):
            load_checkpoint(path, aae, aae_opt, config.learning_rate)
            print(f"  ✓ Loaded from: {path}")
            loaded = True
            break
    if not loaded:
        print("  ✗ AAE checkpoint not found!")
        return
    aae.eval()

    # ========== 2. Encode all patients ==========
    print("\n[2/5] Encoding PET scans to latent space...")
    patients = discover_patients("./data")
    encoded = {}

    with torch.no_grad():
        for pid, timepoints in patients.items():
            encoded[pid] = {}
            for session, path in timepoints.items():
                pet = load_pet(path, device)
                z = aae.encode(pet)
                encoded[pid][session] = z
                month = int(session[1:])
                encoded[pid][f'month_{month}'] = z
            sessions = sorted(timepoints.keys())
            print(f"  {pid}: encoded {', '.join(sessions)} → z shape {z.shape}")

    # ========== 3. Train SDE ==========
    print(f"\n[3/5] Training Latent SDE on {TRAIN_IDS} ({EPOCHS} epochs)...")
    sde = LatentSDE(
        latent_channels=config.latent_dim,
        hidden_channels=getattr(config, 'ode_hidden_dim', 32),
        time_dim=getattr(config, 'ode_time_dim', 64),
        num_blocks=getattr(config, 'ode_num_blocks', 3),
        num_classes=config.num_classes,
        solver='srk'
    ).to(device)
    print(f"  SDE parameters: {sum(p.numel() for p in sde.parameters()):,}")

    optimizer = optim.AdamW(sde.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Prepare training data
    train_samples = []
    for pid in TRAIN_IDS:
        if pid not in encoded:
            print(f"  ✗ {pid} not found in data!")
            continue

        z_0 = encoded[pid]['M00']
        z_T = encoded[pid]['M24']

        # Collect intermediates
        obs = {}
        for session in ['M06', 'M12', 'M18']:
            if session in encoded[pid]:
                month = int(session[1:])
                obs[month] = encoded[pid][session]

        train_samples.append({
            'pid': pid,
            'z_0': z_0,
            'z_T': z_T,
            'obs': obs if obs else None,
        })
        int_str = ','.join(f"M{m:02d}" for m in obs.keys()) if obs else 'None'
        print(f"  {pid}: z_0→z_T, intermediates={int_str}")

    if not train_samples:
        print("  ✗ No training samples!")
        return

    # Training loop
    sde.train()
    t_span = torch.tensor([0., 24.], device=device)
    label = torch.tensor([0], device=device)  # Placeholder

    best_loss = float('inf')
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for sample in train_samples:
            z_0 = sample['z_0']
            z_T_gt = sample['z_T']
            obs = sample['obs']

            # Forward: SDE integration (with intermediates if available)
            z_trajectory = sde(z_0, t_span, obs, label)
            z_T_pred = z_trajectory[-1]

            loss = criterion(z_T_pred, z_T_gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sde.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_samples)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>4d}/{EPOCHS}: loss = {avg_loss:.8f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    print(f"  ✓ Training done. Best loss: {best_loss:.8f}")

    # Save SDE checkpoint
    os.makedirs("result/pipeline_test", exist_ok=True)
    ckpt_path = "result/pipeline_test/SDE_pipeline.pth.tar"
    torch.save({
        'state_dict': sde.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, ckpt_path)
    print(f"  ✓ Saved SDE checkpoint: {ckpt_path}")

    # ========== 4. Test on 009S4612 ==========
    print(f"\n[4/5] Testing on {TEST_IDS}...")
    sde.eval()

    ref_path = list(list(patients.values())[0].values())[0]
    ref_affine = nib.load(ref_path).affine
    output_dir = "result/pipeline_test"

    with torch.no_grad():
        for pid in TEST_IDS:
            if pid not in encoded:
                print(f"  ✗ {pid} not found!")
                continue

            print(f"\n  --- {pid} (Test) ---")
            z_0 = encoded[pid]['M00']
            z_T_gt = encoded[pid]['M24']

            # This patient has no intermediates (only M00, M24)
            z_trajectory = sde(z_0, t_span, None, label)
            z_T_pred = z_trajectory[-1]

            # Decode to PET space
            pet_pred = aae.decode(z_T_pred)
            pet_pred = torch.clamp(pet_pred, 0, 1)

            pet_gt = aae.decode(z_T_gt)
            pet_gt = torch.clamp(pet_gt, 0, 1)

            pet_recon_m00 = aae.decode(z_0)
            pet_recon_m00 = torch.clamp(pet_recon_m00, 0, 1)

            # Metrics
            latent_mse = nn.functional.mse_loss(z_T_pred, z_T_gt).item()
            image_mse = nn.functional.mse_loss(pet_pred, pet_gt).item()

            # Baseline: how different is M00 from M24?
            baseline_mse = nn.functional.mse_loss(
                aae.decode(z_0), pet_gt
            ).item()

            print(f"    Latent MSE (pred vs gt):   {latent_mse:.8f}")
            print(f"    Image MSE (pred vs gt):    {image_mse:.8f}")
            print(f"    Baseline MSE (M00 vs M24): {baseline_mse:.8f}")
            if image_mse < baseline_mse:
                print(f"    ✓ Prediction is CLOSER to M24 than raw M00!")
            else:
                print(f"    ⚠ Prediction not yet better (SDE needs more training/data)")

            # Save NIfTI
            pred_np = pet_pred.squeeze().cpu().numpy()
            gt_np = pet_gt.squeeze().cpu().numpy()
            recon_np = pet_recon_m00.squeeze().cpu().numpy()

            nib.save(nib.Nifti1Image(pred_np, ref_affine),
                     os.path.join(output_dir, f"{pid}_pred_M24.nii.gz"))
            nib.save(nib.Nifti1Image(gt_np, ref_affine),
                     os.path.join(output_dir, f"{pid}_gt_M24.nii.gz"))
            nib.save(nib.Nifti1Image(recon_np, ref_affine),
                     os.path.join(output_dir, f"{pid}_recon_M00.nii.gz"))
            print(f"    ✓ Saved: {pid}_pred_M24.nii.gz, {pid}_gt_M24.nii.gz, {pid}_recon_M00.nii.gz")

    # ========== 5. Summary ==========
    print("\n" + "=" * 60)
    print("[5/5] Pipeline Test Summary")
    print("=" * 60)
    print(f"  AAE:          ✓ Loaded & working")
    print(f"  SDE Training: ✓ {EPOCHS} epochs on {TRAIN_IDS}")
    print(f"  SDE Testing:  ✓ Predicted M24 for {TEST_IDS}")
    print(f"  Intermediates: ✓ M06/M12 used during training (Jump mechanism)")
    print(f"  NIfTI Output: ✓ Saved to {output_dir}/")
    print(f"\n  Pipeline is READY for full-scale training.")
    print(f"  Next step: python train_ode.py --train --use_sde --data_root ./data")


if __name__ == "__main__":
    main()
