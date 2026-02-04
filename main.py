"""
Main training script for longitudinal PET prediction.

Stage 1: Train AAE (Adversarial Autoencoder) for PET image compression.
Stage 2: Use train_ode.py for Neural ODE training.
"""

import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
import os
import csv
import nibabel as nib

from utils import *
from model import AAE, Discriminator
from dataset import OneDataset
import config

import warnings
warnings.filterwarnings("ignore")

# Metrics
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure


def batch_psnr_ssim_torchmetrics(gt: torch.Tensor, pred: torch.Tensor, data_range: float = 1.0):
    """Compute PSNR and SSIM using torchmetrics (GPU accelerated)."""
    if gt.ndim == 4:
        gt = gt.unsqueeze(1)
    if pred.ndim == 4:
        pred = pred.unsqueeze(1)

    gt = gt.float()
    pred = pred.float()

    dims = tuple(range(1, pred.ndim))

    psnr_vals = peak_signal_noise_ratio(
        pred, gt, data_range=data_range, reduction='none', dim=dims
    )

    ssim_vals = structural_similarity_index_measure(
        pred, gt, data_range=data_range, reduction='none',
        gaussian_kernel=False, kernel_size=7,
    )

    psnr_sum = psnr_vals.sum().item()
    ssim_sum = ssim_vals.sum().item()
    return psnr_sum, ssim_sum


# ============================================================
# Stage 1: AAE Training
# ============================================================

def train_AAE():
    """Train the Adversarial Autoencoder for PET image compression."""
    
    print("=" * 60)
    print("Stage 1: Training AAE (Adversarial Autoencoder)")
    print("=" * 60)
    
    # Create result directory
    os.makedirs(f"result/{config.exp}", exist_ok=True)
    
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    
    disc = Discriminator().to(config.device)
    opt_disc = optim.Adam(disc.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))

    best_score = 0

    for epoch in range(config.epochs):
        print(f"\nEpoch: {epoch + 1}/{config.epochs}")
        
        # Training
        model.train()
        disc.train()
        
        dataset = OneDataset(root_Abeta=config.whole_Abeta, task=config.train, stage="train")
        loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True,
            num_workers=config.numworker, pin_memory=True, drop_last=True
        )
        
        loop = tqdm(loader, leave=True, desc="Training")
        length = dataset.length_dataset
        recon_loss_epoch = 0
        disc_loss_epoch = 0

        for idx, (Abeta, name) in enumerate(loop):
            Abeta = np.expand_dims(Abeta, axis=1)
            Abeta = torch.tensor(Abeta)
            Abeta = Abeta.to(config.device)
            
            decoded_Abeta = model(Abeta)
            
            disc_real = disc(Abeta)
            disc_fake = disc(decoded_Abeta)

            recon_loss = torch.abs(Abeta - decoded_Abeta).mean()
            g_loss = -torch.mean(disc_fake)
            loss = recon_loss * config.Lambda + g_loss

            d_loss_real = torch.mean(F.relu(1. - disc_real))
            d_loss_fake = torch.mean(F.relu(1. + disc_fake))
            disc_loss = (d_loss_real + d_loss_fake) / 2

            opt_model.zero_grad()
            loss.backward(retain_graph=True)

            opt_disc.zero_grad()
            disc_loss.backward()

            opt_model.step()
            opt_disc.step()

            recon_loss_epoch += recon_loss.item()
            disc_loss_epoch += disc_loss.item()
            
            loop.set_postfix(recon_loss=recon_loss.item(), disc_loss=disc_loss.item())

        # Log training loss
        with open(f"result/{config.exp}loss_curve.csv", 'a+', newline='') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(["Epoch", "recon_loss", "disc_loss"])
            writer.writerow([epoch + 1, recon_loss_epoch / len(loader), disc_loss_epoch / len(loader)])

        # Validation
        model.eval()
        dataset = OneDataset(root_Abeta=config.whole_Abeta, task=config.validation, stage="validation")
        loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=False,
            num_workers=config.numworker, pin_memory=True, drop_last=True
        )
        
        length = dataset.length_dataset
        psnr_sum = 0
        ssim_sum = 0

        with torch.no_grad():
            for idx, (Abeta, name) in enumerate(tqdm(loader, desc="Validation", leave=False)):
                Abeta = np.expand_dims(Abeta, axis=1)
                Abeta = torch.tensor(Abeta).to(config.device)
                
                decoded_Abeta = model(Abeta)
                decoded_Abeta = torch.clamp(decoded_Abeta, 0, 1)

                psnr_val, ssim_val = batch_psnr_ssim_torchmetrics(Abeta, decoded_Abeta, data_range=1.0)
                psnr_sum += psnr_val
                ssim_sum += ssim_val

        avg_psnr = psnr_sum / length
        avg_ssim = ssim_sum / length
        score = avg_psnr + avg_ssim * 10

        # Log validation metrics
        with open(f"result/{config.exp}validation.csv", 'a+', newline='') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(['Epoch', 'PSNR', 'SSIM'])
            writer.writerow([epoch + 1, avg_psnr, avg_ssim])

        print(f"Validation - PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")

        # Save checkpoint
        if score > best_score or (epoch + 1) % 100 == 0:
            if score > best_score:
                best_score = score
            save_checkpoint(model, opt_model, filename=f"{config.CHECKPOINT_AAE}_epoch{epoch + 1}")
            print(f"Saved checkpoint at epoch {epoch + 1}")

    print("\nAAE training complete!")


def test_AAE():
    """Test the trained AAE model."""
    
    print("=" * 60)
    print("Testing AAE Model")
    print("=" * 60)
    
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.9))
    load_checkpoint(config.CHECKPOINT_AAE, model, opt_model, config.learning_rate)
    model.eval()
    
    dataset = OneDataset(root_Abeta=config.whole_Abeta, task=config.test, stage="test")
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.numworker, pin_memory=True, drop_last=True
    )
    
    length = dataset.length_dataset
    psnr_sum = 0
    ssim_sum = 0

    with torch.no_grad():
        for idx, (Abeta, name) in enumerate(tqdm(loader, desc="Testing")):
            Abeta = np.expand_dims(Abeta, axis=1)
            Abeta = torch.tensor(Abeta).to(config.device)
            
            decoded_Abeta = model(Abeta)
            decoded_Abeta = torch.clamp(decoded_Abeta, 0, 1)

            psnr_val, ssim_val = batch_psnr_ssim_torchmetrics(Abeta, decoded_Abeta, data_range=1.0)
            psnr_sum += psnr_val
            ssim_sum += ssim_val

    print(f"\nTest Results:")
    print(f"  PSNR: {psnr_sum / length:.2f}")
    print(f"  SSIM: {ssim_sum / length:.4f}")


def encode_data(task_file, stage_name="train"):
    """Encode PET images to latent space using trained AAE."""
    
    print(f"\nEncoding {stage_name} data...")
    
    model = AAE().to(config.device)
    opt_model = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.9))
    load_checkpoint(config.CHECKPOINT_AAE, model, opt_model, config.learning_rate)
    model.eval()
    
    # Reference image for affine
    ref_image = nib.load(config.path)
    
    # Create output directory
    os.makedirs(config.latent_Abeta, exist_ok=True)

    dataset = OneDataset(root_Abeta=config.whole_Abeta, task=task_file, stage="Non")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=config.numworker, pin_memory=True)

    with torch.no_grad():
        for idx, (Abeta, name) in enumerate(tqdm(loader, desc=f"Encoding {stage_name}")):
            Abeta = np.expand_dims(Abeta, axis=1)
            Abeta = torch.tensor(Abeta).to(config.device)
            
            latent_Abeta = model.encoder(Abeta)
            latent_Abeta = latent_Abeta.detach().cpu().numpy()
            latent_Abeta = np.squeeze(latent_Abeta)
            latent_Abeta = latent_Abeta.astype(np.float32)

            latent_nifti = nib.Nifti1Image(latent_Abeta, ref_image.affine)
            nib.save(latent_nifti, config.latent_Abeta + str(name[0]))
    
    print(f"Encoded {len(dataset)} samples to {config.latent_Abeta}")


def encoding():
    """Encode training data to latent space."""
    encode_data(config.train, "train")


def encoding_test():
    """Encode test data to latent space."""
    encode_data(config.test, "test")


def encoding_all():
    """Encode all data (train, val, test) to latent space."""
    encode_data(config.train, "train")
    encode_data(config.validation, "validation")
    encode_data(config.test, "test")


# ============================================================
# Main Entry Point
# ============================================================

if __name__ == '__main__':
    seed_torch()

    import argparse

    parser = argparse.ArgumentParser(description="Longitudinal PET Prediction - AAE Training")

    # Stage 1: AAE
    parser.add_argument('--train_aae', action='store_true', help='Train the AAE model')
    parser.add_argument('--test_aae', action='store_true', help='Test the AAE model')
    parser.add_argument('--enc', action='store_true', help='Encode training data to latent space')
    parser.add_argument('--enc_test', action='store_true', help='Encode test data to latent space')
    parser.add_argument('--enc_all', action='store_true', help='Encode all data to latent space')

    args = parser.parse_args()

    if args.train_aae:
        train_AAE()

    if args.test_aae:
        test_AAE()

    if args.enc:
        encoding()

    if args.enc_test:
        encoding_test()

    if args.enc_all:
        encoding_all()

    if not any(vars(args).values()):
        print("Usage:")
        print("  Stage 1 (AAE):")
        print("    python main.py --train_aae    # Train AAE")
        print("    python main.py --test_aae     # Test AAE")
        print("    python main.py --enc          # Encode training data")
        print("    python main.py --enc_all      # Encode all data")
        print("")
        print("  Stage 2 (Neural ODE):")
        print("    python train_ode.py --train   # Train Neural ODE")
        print("    python train_ode.py --test    # Test Neural ODE")
        parser.print_help()
