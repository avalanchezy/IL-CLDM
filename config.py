"""
Configuration for Longitudinal PET Prediction.

This file contains all hyperparameters and paths for training.
"""

import torch

# ============================================================
# Device Configuration
# ============================================================

# Automatically select GPU: prefer cuda:1, fallback to cuda:0, then cpu
device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0" if torch.cuda.is_available() else "cpu"
gpus = [0]  # GPU IDs for DataParallel (if needed)

# ============================================================
# Model Hyperparameters
# ============================================================

# AAE (Adversarial Autoencoder)
latent_dim = 1              # Latent channels (1 for single-channel latent)
image_size = 28             # Latent spatial size (112/4=28)
Lambda = 100                # Weight for reconstruction loss vs adversarial loss

# Training
learning_rate = 1e-4
batch_size = 2
numworker = 2               # DataLoader workers
epochs = 1000               # AAE training epochs

# Classification
num_classes = 4             # Disease stages (e.g., NC, EMCI, LMCI, AD)

# ============================================================
# Data Paths
# ============================================================

# PET data directories
whole_Abeta = "./data/process/whole_Abeta_112"     # Preprocessed PET images (112×128×112)
latent_Abeta = "./data/latent_Abeta/latent_Abeta_112/"  # Encoded latent representations
syn_Abeta = "./data/syn_Abeta/"                    # Generated/synthetic PET outputs

# Reference image for NIfTI affine matrix
path = "./data/process/whole_Abeta_112/sample.nii.gz"

# Data split files
train = "data_info/train.txt"
validation = "data_info/val.txt"
test = "data_info/test.txt"

# ============================================================
# Experiment Settings
# ============================================================

exp = "exp_longitudinal/"   # Experiment output folder

# Checkpoints
CHECKPOINT_AAE = "result/" + exp + "AAE.pth.tar"
CHECKPOINT_ODE = "result/" + exp + "ODE.pth.tar"

# ============================================================
# Neural ODE Settings
# ============================================================

ode_hidden_dim = 32         # Hidden channels in ODE function
ode_time_dim = 64           # Time embedding dimension
ode_num_blocks = 3          # Number of conv blocks in ODE function
ode_solver = "dopri5"       # ODE solver: 'dopri5', 'euler', 'rk4', 'midpoint'
ode_rtol = 1e-5             # Relative tolerance for adaptive solvers
ode_atol = 1e-7             # Absolute tolerance for adaptive solvers
ode_epochs = 500            # Epochs for ODE training

# Diffusion refinement (for ODE+Diffusion hybrid)
diffusion_steps = 100       # Number of diffusion steps for refinement

# Timepoints (in months)
timepoints = [0, 6, 12, 18, 24]
baseline_time = 0           # Starting timepoint
target_time = 24            # Target timepoint to predict