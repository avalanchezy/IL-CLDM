"""
Model architectures for longitudinal PET prediction.

Contains:
- AAE (Adversarial Autoencoder) for image compression
- EMA (Exponential Moving Average) for stable training

For Neural ODE models, see ode_model.py
"""

import torch
from torch import nn
import config


# ============================================================
# Basic Components
# ============================================================

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm(nn.Module):
    """Group Normalization wrapper."""
    def __init__(self, channels, num_groups=16):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(
            num_groups=min(num_groups, channels), 
            num_channels=channels, 
            eps=1e-6, 
            affine=True
        )

    def forward(self, x):
        return self.gn(x)


class Upsample(nn.Module):
    """3D Upsampling with transposed convolution."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose3d(dim, dim, 4, 2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    """3D Downsampling with strided convolution."""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 4, 2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ============================================================
# Adversarial Autoencoder (AAE)
# ============================================================

class ResidualBlock(nn.Module):
    """Residual block for 3D data."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class Encoder(nn.Module):
    """
    3D Encoder for PET images.
    
    Input: (B, 1, 112, 128, 112)
    Output: (B, latent_dim, 28, 32, 28)
    
    Downsamples by factor of 4 in each spatial dimension.
    """
    def __init__(self, image_channels=1, latent_dim=config.latent_dim):
        super(Encoder, self).__init__()
        
        channels = [16, 32, 64]
        num_res_blocks = 1
        
        layers = [nn.Conv3d(image_channels, channels[0], 3, 1, 1)]
        
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if i != len(channels) - 1:
                layers.append(Downsample(channels[i + 1]))
        
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(channels[-1], latent_dim, 3, 1, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    """
    3D Decoder for PET images.
    
    Input: (B, latent_dim, 28, 32, 28)
    Output: (B, 1, 112, 128, 112)
    
    Upsamples by factor of 4 in each spatial dimension.
    """
    def __init__(self, image_channels=1, latent_dim=config.latent_dim):
        super(Decoder, self).__init__()
        
        channels = [64, 32, 16]
        num_res_blocks = 1

        in_channels = channels[0]
        layers = [
            nn.Conv3d(latent_dim, in_channels, 3, 1, 1),
            ResidualBlock(in_channels, in_channels)
        ]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if i != 0:
                layers.append(Upsample(in_channels))

        layers.append(GroupNorm(in_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Conv3d(in_channels, image_channels, 3, 1, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AAE(nn.Module):
    """
    Adversarial Autoencoder for 3D PET images.
    
    Compresses PET images to a lower-dimensional latent space
    for efficient temporal modeling with Neural ODE.
    
    Input: (B, 1, 112, 128, 112)
    Latent: (B, 1, 28, 32, 28)
    Output: (B, 1, 112, 128, 112)
    """
    def __init__(self):
        super(AAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, data):
        encoded_data = self.encoder(data)
        decoded_data = self.decoder(encoded_data)
        return decoded_data
    
    def encode(self, data):
        """Encode data to latent space."""
        return self.encoder(data)
    
    def decode(self, latent):
        """Decode latent representation to image."""
        return self.decoder(latent)


class Discriminator(nn.Module):
    """
    3D Discriminator for adversarial training.
    
    Distinguishes between real and reconstructed PET images.
    """
    def __init__(self, image_channels=1, channels=[16, 32, 64, 128]):
        super(Discriminator, self).__init__()

        layers = [
            nn.Conv3d(image_channels, channels[0], 4, 2, 1), 
            nn.LeakyReLU(0.2)
        ]
        
        layers += [
            nn.Conv3d(channels[0], channels[1], 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels[1]),
            nn.LeakyReLU(0.2, True)
        ]
        
        layers += [
            nn.Conv3d(channels[1], channels[2], 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels[2]),
            nn.LeakyReLU(0.2, True)
        ]
        
        layers += [
            nn.Conv3d(channels[2], channels[3], 4, 2, 1, bias=False),
            nn.BatchNorm3d(channels[3]),
            nn.LeakyReLU(0.2, True)
        ]

        layers.append(nn.Conv3d(channels[3], image_channels, 4, 2, 1))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ============================================================
# Training Utilities
# ============================================================

class EMA:
    """
    Exponential Moving Average for model parameters.
    
    Maintains a smoothed version of model weights for more stable predictions.
    """
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    device = config.device
    
    # Test AAE
    print("Testing AAE...")
    model = AAE().to(device)
    x = torch.randn(2, 1, 112, 128, 112).to(device)
    
    # Encode
    z = model.encode(x)
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {z.shape}")
    
    # Decode
    x_recon = model.decode(z)
    print(f"Reconstructed shape: {x_recon.shape}")
    
    # Full forward
    x_out = model(x)
    print(f"Full forward output shape: {x_out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\nAll tests passed!")
