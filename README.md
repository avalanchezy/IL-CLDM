# Longitudinal PET Prediction with Neural ODE

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Predict future PET scans from baseline using Neural ODEs in latent space, with robust handling of **missing intermediate timepoints**.

## Overview

This project predicts longitudinal PET scans (T24) from baseline (T0), handling missing intermediate scans (T6/T12/T18) that may be unavailable due to data collection issues.

### Key Features

- **Neural ODE**: Continuous-time dynamics modeling in latent space
- **ODE + Diffusion Hybrid**: Combines deterministic trajectory with stochastic refinement
- **Missing Data Handling**: Robust to variable available timepoints per subject
- **Disease Conditioning**: Optional conditioning on disease stage labels

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Pipeline: T0 PET → T24 PET Prediction                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Stage 1: AAE (Adversarial Autoencoder)                                 │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  PET Image (112×128×112) → Encoder → Latent (28×32×28)        │    │
│  │  Latent (28×32×28) → Decoder → PET Image (112×128×112)        │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Stage 2: Neural ODE (choose one)                                       │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Option A: Pure Neural ODE                                     │    │
│  │    z_0 → dz/dt = f(z,t) → ODE Solve → z_24                    │    │
│  │                                                                │    │
│  │  Option B: ODE + Diffusion (recommended)                       │    │
│  │    z_0 → Neural ODE → z_mean → Diffusion Refine → z_24        │    │
│  │              (deterministic)    (stochastic)                   │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/avalanchezy/IL-CLDM.git
cd IL-CLDM

# Create environment
conda create -n pet-ode python=3.11
conda activate pet-ode

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### Directory Structure

```
IL-CLDM/
├── data/
│   └── {SubjectID}/
│       ├── *_ses-M00_*_pet.nii.gz    # Baseline (required)
│       ├── *_ses-M06_*_pet.nii.gz    # Month 6 (optional)
│       ├── *_ses-M12_*_pet.nii.gz    # Month 12 (optional)
│       └── *_ses-M24_*_pet.nii.gz    # Target (required)
├── data_info/
│   ├── data_info.csv                 # Labels: filename,label_id
│   ├── train.txt                     # Training subject IDs
│   ├── val.txt                       # Validation subject IDs
│   └── test.txt                      # Test subject IDs
└── result/                           # Checkpoints & logs
```

## Training

### Stage 1: Train AAE

```bash
# Train autoencoder
python main.py --train_aae

# Encode data to latent space
python main.py --enc_all

# Test reconstruction quality
python main.py --test_aae
```

### Stage 2: Train Prediction Model

```bash
# Option A: Pure Neural ODE (handles missing intermediates automatically)
python train_ode.py --train --data_root ./data

# Option B: Neural ODE + Diffusion (recommended for uncertainty estimation)
python train_ode.py --train --use_diffusion --data_root ./data
```

> **Note**: The model automatically handles missing intermediate timepoints.
> When available, T6/T12/T18 scans are used to improve trajectory accuracy.

### Inference

```bash
# Test model
python train_ode.py --test --checkpoint result/exp/ODE_best.pth.tar

# Generate predictions
python train_ode.py --generate --checkpoint result/exp/ODE_best.pth.tar
```

## Model Variants

| Model | Description | Use Case |
|-------|-------------|----------|
| `LatentODE` | Pure Neural ODE | Fast, deterministic predictions |
| `LatentODEWithIntermediates` | ODE with intermediate observations | When T6/T12 scans are available |
| `LatentODEDiffusion` | ODE + Diffusion hybrid | Best quality, uncertainty estimation |

## Configuration

Edit `config.py` to customize hyperparameters:

```python
# Device
device = "cuda:0"

# AAE Training
epochs = 1000
batch_size = 2

# Neural ODE
ode_hidden_dim = 32
ode_num_blocks = 3
ode_solver = "dopri5"
ode_epochs = 500

# Disease classification
num_classes = 4  # e.g., NC, EMCI, LMCI, AD
```

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Stage 1: AAE training |
| `train_ode.py` | Stage 2: Neural ODE training |
| `model.py` | AAE architecture |
| `ode_model.py` | Neural ODE + Diffusion models |
| `dataset.py` | AAE dataset |
| `dataset_longitudinal.py` | Longitudinal dataset with missing data handling |
| `config.py` | Configuration |
| `utils.py` | Utilities |

## Citation

If you use this code, please cite:

```bibtex
@misc{longitudinal-pet-ode,
  title={Longitudinal PET Prediction with Neural ODE},
  author={Your Name},
  year={2024},
  url={https://github.com/avalanchezy/IL-CLDM}
}
```

## License

MIT License
