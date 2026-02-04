"""
Training script for Neural ODE longitudinal PET prediction.

This script trains the Neural ODE model to predict T24 PET from T0 PET
(and optionally intermediate timepoints) in the latent space.
"""

import os
import csv
import copy
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import nibabel as nib

from ode_model import LatentODE, LatentSDE
from dataset_longitudinal import SimpleLongitudinalDataset, collate_longitudinal
from model import AAE, EMA
from utils import seed_torch, save_checkpoint, load_checkpoint
import config

# Metrics
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure


def compute_metrics(pred, gt, data_range=1.0):
    """Compute PSNR and SSIM between predicted and ground truth."""
    if pred.ndim == 4:
        pred = pred.unsqueeze(1)
    if gt.ndim == 4:
        gt = gt.unsqueeze(1)
    
    pred = pred.float()
    gt = gt.float()
    
    dims = tuple(range(1, pred.ndim))
    
    psnr_vals = peak_signal_noise_ratio(
        pred, gt, data_range=data_range, reduction='none', dim=dims
    )
    
    ssim_vals = structural_similarity_index_measure(
        pred, gt, data_range=data_range, reduction='none',
        gaussian_kernel=False, kernel_size=7
    )
    
    return psnr_vals.sum().item(), ssim_vals.sum().item()


def train_ode(args):
    """Main training function for Neural ODE model."""
    
    print("=" * 60)
    print("Neural ODE Longitudinal PET Prediction Training")
    print("=" * 60)
    
    device = config.device
    print(f"Using device: {device}")
    
    # Create result directory
    exp_dir = f"result/{config.exp}"
    os.makedirs(exp_dir, exist_ok=True)
    
    # ========== Load Pre-trained AAE ==========
    print("\nLoading pre-trained AAE...")
    aae = AAE().to(device)
    aae_opt = optim.Adam(aae.parameters(), lr=config.learning_rate)
    
    if os.path.exists(config.CHECKPOINT_AAE):
        load_checkpoint(config.CHECKPOINT_AAE, aae, aae_opt, config.learning_rate)
        print(f"Loaded AAE from {config.CHECKPOINT_AAE}")
    else:
        print(f"WARNING: AAE checkpoint not found at {config.CHECKPOINT_AAE}")
        print("Make sure to train Stage 1 (AAE) first!")
        if not args.skip_aae_check:
            return
    
    aae.eval()  # Freeze AAE
    
    # ========== Create Neural ODE Model ==========
    print("\nCreating model...")
    
    if args.use_sde:
        # Latent SDE: Drift (ODE) + Diffusion
        ode_model = LatentSDE(
            latent_channels=config.latent_dim,
            hidden_channels=getattr(config, 'ode_hidden_dim', 32),
            time_dim=getattr(config, 'ode_time_dim', 64),
            num_blocks=getattr(config, 'ode_num_blocks', 3),
            num_classes=config.num_classes,
            solver=getattr(config, 'ode_solver', 'dopri5'),
            diffusion_steps=getattr(config, 'diffusion_steps', 100)
        ).to(device)
    else:
        # Neural ODE (Drift only)
        # Note: Both models automatically handle intermediate observations if provided
        ode_model = LatentODE(
            latent_channels=config.latent_dim,
            hidden_channels=getattr(config, 'ode_hidden_dim', 32),
            time_dim=getattr(config, 'ode_time_dim', 64),
            num_blocks=getattr(config, 'ode_num_blocks', 3),
            num_classes=config.num_classes,
            solver=getattr(config, 'ode_solver', 'dopri5')
        ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in ode_model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(ode_model.parameters(), lr=config.learning_rate)
    
    # EMA for stable predictions
    ema = EMA(0.9999)
    ema_model = copy.deepcopy(ode_model).eval().requires_grad_(False)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # ========== Create Datasets ==========
    print("\nLoading datasets...")
    
    train_dataset = SimpleLongitudinalDataset(
        data_root=args.data_root,
        label_file=args.label_file,
        stage='train'
    )
    
    val_dataset = SimpleLongitudinalDataset(
        data_root=args.data_root,
        label_file=args.label_file,
        stage='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.numworker,
        collate_fn=collate_longitudinal,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.numworker,
        collate_fn=collate_longitudinal,
        pin_memory=True
    )
    
    if len(train_dataset) == 0:
        print("ERROR: No training samples found!")
        return
    
    # ========== Training Loop ==========
    print("\nStarting training...")
    
    best_val_loss = float('inf')
    
    # CSV logging
    log_file = os.path.join(exp_dir, "ode_training_log.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Loss', 'Val_PSNR', 'Val_SSIM'])
    
    epochs = getattr(config, 'ode_epochs', config.epochs)
    
    for epoch in range(epochs):
        # ===== Training =====
        ode_model.train()
        train_loss_sum = 0.0
        train_count = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in loop:
            z_0 = batch['z_0'].to(device)  # (B, 1, D, H, W)
            z_T_gt = batch['z_T'].to(device)
            labels = batch['labels'].to(device)
            
            # For now, work directly with input data (assumed to be latent)
            # In full pipeline, would encode with AAE first
            
            # Predict z_T from z_0
            t_span = torch.tensor([0., 24.], device=device)
            
            # Get intermediates if available
            obs_dict = None
            if batch['intermediates'] and len(batch['intermediates']) > 0:
                # batch['intermediates'] is a list of dicts, one per sample
                # For simplicity, we pass None here and let the model handle it
                # In full implementation, would pass observations properly
                pass
            
            # Forward pass (handles with or without intermediates)
            # Both LatentODE and LatentSDE accept obs_dict
            z_trajectory = ode_model(z_0, t_span, obs_dict, labels)
            
            z_T_pred = z_trajectory[-1]  # Prediction at T=24
            
            # Loss
            loss = criterion(z_T_pred, z_T_gt)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ode_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # EMA update
            ema.step_ema(ema_model, ode_model)
            
            train_loss_sum += loss.item() * z_0.shape[0]
            train_count += z_0.shape[0]
            
            loop.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss_sum / train_count
        
        # ===== Validation =====
        ode_model.eval()
        val_loss_sum = 0.0
        val_psnr_sum = 0.0
        val_ssim_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                z_0 = batch['z_0'].to(device)
                z_T_gt = batch['z_T'].to(device)
                labels = batch['labels'].to(device)
                
                t_span = torch.tensor([0., 24.], device=device)
                
                t_span = torch.tensor([0., 24.], device=device)
                
                # Use EMA model for validation (both accept intermediates)
                z_trajectory = ema_model(z_0, t_span, None, labels)
                
                z_T_pred = z_trajectory[-1]
                z_T_pred = torch.clamp(z_T_pred, 0, 1)
                
                loss = criterion(z_T_pred, z_T_gt)
                val_loss_sum += loss.item() * z_0.shape[0]
                
                # Compute metrics
                psnr, ssim = compute_metrics(z_T_pred, z_T_gt)
                val_psnr_sum += psnr
                val_ssim_sum += ssim
                val_count += z_0.shape[0]
        
        avg_val_loss = val_loss_sum / val_count if val_count > 0 else 0
        avg_val_psnr = val_psnr_sum / val_count if val_count > 0 else 0
        avg_val_ssim = val_ssim_sum / val_count if val_count > 0 else 0
        
        # Logging
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, "
              f"Val Loss={avg_val_loss:.6f}, PSNR={avg_val_psnr:.2f}, SSIM={avg_val_ssim:.4f}")
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, avg_val_psnr, avg_val_ssim])
        
        # Save best model
        if avg_val_loss < best_val_loss or (epoch + 1) % 100 == 0:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            
            checkpoint_path = os.path.join(exp_dir, f"ODE_epoch{epoch+1}.pth.tar")
            save_checkpoint(ema_model, optimizer, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


def test_ode(args):
    """Test the trained Neural ODE model."""
    
    print("=" * 60)
    print("Neural ODE Model Testing")
    print("=" * 60)
    
    device = config.device
    
    # Load models
    aae = AAE().to(device)
    aae_opt = optim.Adam(aae.parameters(), lr=config.learning_rate)
    if os.path.exists(config.CHECKPOINT_AAE):
        load_checkpoint(config.CHECKPOINT_AAE, aae, aae_opt, config.learning_rate)
    aae.eval()
    
    ode_model = LatentODE(
        latent_channels=config.latent_dim,
        hidden_channels=getattr(config, 'ode_hidden_dim', 32),
        num_blocks=getattr(config, 'ode_num_blocks', 3),
        num_classes=config.num_classes
    ).to(device)
    
    ode_opt = optim.AdamW(ode_model.parameters(), lr=config.learning_rate)
    
    checkpoint_path = args.checkpoint
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, ode_model, ode_opt, config.learning_rate)
        print(f"Loaded ODE model from {checkpoint_path}")
    else:
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    ode_model.eval()
    
    # Load test dataset
    test_dataset = SimpleLongitudinalDataset(
        data_root=args.data_root,
        label_file=args.label_file,
        stage='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_longitudinal
    )
    
    # Test
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            z_0 = batch['z_0'].to(device)
            z_T_gt = batch['z_T'].to(device)
            labels = batch['labels'].to(device)
            
            t_span = torch.tensor([0., 24.], device=device)
            z_trajectory = ode_model(z_0, t_span, labels)
            z_T_pred = torch.clamp(z_trajectory[-1], 0, 1)
            
            # Decode to image space for evaluation
            pet_pred = aae.decoder(z_T_pred)
            pet_gt = aae.decoder(z_T_gt)
            
            pet_pred = torch.clamp(pet_pred, 0, 1)
            pet_gt = torch.clamp(pet_gt, 0, 1)
            
            psnr, ssim = compute_metrics(pet_pred, pet_gt)
            total_psnr += psnr
            total_ssim += ssim
            count += 1
    
    print(f"\nTest Results:")
    print(f"  PSNR: {total_psnr / count:.2f}")
    print(f"  SSIM: {total_ssim / count:.4f}")


def generate_predictions(args):
    """Generate predictions and save as NIfTI files."""
    
    print("=" * 60)
    print("Generating Predictions")
    print("=" * 60)
    
    device = config.device
    
    # Load models
    aae = AAE().to(device)
    aae_opt = optim.Adam(aae.parameters(), lr=config.learning_rate)
    if os.path.exists(config.CHECKPOINT_AAE):
        load_checkpoint(config.CHECKPOINT_AAE, aae, aae_opt, config.learning_rate)
    aae.eval()
    
    ode_model = LatentODE(
        latent_channels=config.latent_dim,
        hidden_channels=getattr(config, 'ode_hidden_dim', 32),
        num_blocks=getattr(config, 'ode_num_blocks', 3),
        num_classes=config.num_classes
    ).to(device)
    
    ode_opt = optim.AdamW(ode_model.parameters(), lr=config.learning_rate)
    
    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, ode_model, ode_opt, config.learning_rate)
    
    ode_model.eval()
    
    # Output directory
    output_dir = args.output_dir or config.syn_Abeta
    os.makedirs(output_dir, exist_ok=True)
    
    # Reference image for affine
    ref_image = nib.load(config.path)
    
    # Dataset
    dataset = SimpleLongitudinalDataset(
        data_root=args.data_root,
        stage='test'
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_longitudinal)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating"):
            z_0 = batch['z_0'].to(device)
            labels = batch['labels'].to(device)
            subject_id = batch['subject_ids'][0]
            
            t_span = torch.tensor([0., 24.], device=device)
            z_trajectory = ode_model(z_0, t_span, labels)
            z_T_pred = torch.clamp(z_trajectory[-1], 0, 1)
            
            # Decode
            pet_pred = aae.decoder(z_T_pred)
            pet_pred = torch.clamp(pet_pred, 0, 1)
            
            # Save
            pet_np = pet_pred.squeeze().cpu().numpy().astype(np.float32)
            nifti = nib.Nifti1Image(pet_np, ref_image.affine)
            
            output_path = os.path.join(output_dir, f"{subject_id}_pred_T24.nii.gz")
            nib.save(nifti, output_path)
    
    print(f"\nPredictions saved to {output_dir}")


if __name__ == "__main__":
    seed_torch()
    
    parser = argparse.ArgumentParser(description="Neural ODE Longitudinal PET Prediction")
    
    # Mode
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--generate', action='store_true', help='Generate predictions')
    
    # Data
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory of PET data')
    parser.add_argument('--label_file', type=str, default='data_info/data_info.csv',
                        help='Path to label CSV file')
    
    # Model
    parser.add_argument('--use_sde', action='store_true',
                        help='Use Latent SDE (Drift + Diffusion) for uncertainty estimation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for testing/generation')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for generated predictions')
    
    # Debug
    parser.add_argument('--skip_aae_check', action='store_true',
                        help='Skip AAE checkpoint check')
    
    args = parser.parse_args()
    
    if args.train:
        train_ode(args)
    elif args.test:
        if args.checkpoint is None:
            print("ERROR: --checkpoint required for testing")
        else:
            test_ode(args)
    elif args.generate:
        if args.checkpoint is None:
            print("ERROR: --checkpoint required for generation")
        else:
            generate_predictions(args)
    else:
        print("Please specify --train, --test, or --generate")
        parser.print_help()
