"""
Neural ODE model for longitudinal PET prediction in latent space.

This module implements continuous-time dynamics for predicting future PET scans
from baseline, learning dz/dt = f(z, t; θ) in the latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
import config


class Swish(nn.Module):
    """Swish activation function."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for conditioning on time."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: (B,) or scalar tensor of time values
        Returns:
            (B, dim) time embeddings
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ConvBlock3D(nn.Module):
    """3D convolutional block with GroupNorm and Swish activation."""
    def __init__(self, in_channels, out_channels, time_dim=None, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.act = Swish()
        
        # Time embedding projection
        if time_dim is not None:
            self.time_mlp = nn.Sequential(
                Swish(),
                nn.Linear(time_dim, out_channels)
            )
        else:
            self.time_mlp = None
            
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x, t_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Add time embedding
        if t_emb is not None and self.time_mlp is not None:
            t_emb = self.time_mlp(t_emb)
            h = h + t_emb[:, :, None, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.residual(x)


class LatentODEFunc(nn.Module):
    """
    ODE function that defines dz/dt = f(z, t; θ).
    
    This network operates on 3D latent representations and is conditioned on time.
    The architecture is a lightweight 3D U-Net style network.
    """
    def __init__(
        self,
        latent_channels=1,
        hidden_channels=32,
        time_dim=64,
        num_blocks=3,
        label_dim=None,  # Optional disease label conditioning
        num_classes=4
    ):
        super().__init__()
        
        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            Swish(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # Optional label embedding
        if label_dim is not None or num_classes > 0:
            self.label_embed = nn.Embedding(num_classes, time_dim)
        else:
            self.label_embed = None
        
        # Encoder path
        self.input_conv = nn.Conv3d(latent_channels, hidden_channels, 3, padding=1)
        
        self.encoder_blocks = nn.ModuleList([
            ConvBlock3D(hidden_channels, hidden_channels, time_dim)
            for _ in range(num_blocks)
        ])
        
        # Middle
        self.middle_block = ConvBlock3D(hidden_channels, hidden_channels, time_dim)
        
        # Decoder path (same resolution, no downsampling)
        self.decoder_blocks = nn.ModuleList([
            ConvBlock3D(hidden_channels * 2, hidden_channels, time_dim)  # *2 for skip connection
            for _ in range(num_blocks)
        ])
        
        # Output: predict dz/dt
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(8, hidden_channels), hidden_channels),
            Swish(),
            nn.Conv3d(hidden_channels, latent_channels, 3, padding=1)
        )
        
        # Store label for forward pass
        self._label = None
    
    def set_label(self, label):
        """Set the disease label for conditioning."""
        self._label = label
    
    def forward(self, t, z):
        """
        Forward pass computing dz/dt.
        
        Args:
            t: scalar time (normalized to [0, 1] for ODE solver)
            z: (B, C, D, H, W) latent tensor
            
        Returns:
            dz_dt: (B, C, D, H, W) time derivative of latent
        """
        # Time embedding
        batch_size = z.shape[0]
        t_tensor = t.expand(batch_size) if t.dim() == 0 else t
        t_emb = self.time_embed(t_tensor)
        t_emb = self.time_mlp(t_emb)
        
        # Add label embedding if available
        if self.label_embed is not None and self._label is not None:
            label_emb = self.label_embed(self._label.long())
            t_emb = t_emb + label_emb
        
        # Forward through network
        h = self.input_conv(z)
        
        # Encoder with skip connections
        skips = []
        for block in self.encoder_blocks:
            h = block(h, t_emb)
            skips.append(h)
        
        # Middle
        h = self.middle_block(h, t_emb)
        
        # Decoder with skip connections
        for block, skip in zip(self.decoder_blocks, reversed(skips)):
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
        
        # Output dz/dt
        dz_dt = self.output_conv(h)
        
        return dz_dt


class LatentODE(nn.Module):
    """
    Neural ODE for longitudinal PET prediction.
    
    Given z_0 (latent at T0), predicts z_T (latent at target time T).
    Can optionally incorporate intermediate observations.
    """
    def __init__(
        self,
        latent_channels=1,
        hidden_channels=32,
        time_dim=64,
        num_blocks=3,
        num_classes=4,
        solver='dopri5',
        rtol=1e-5,
        atol=1e-7,
        use_adjoint=True
    ):
        super().__init__()
        
        self.odefunc = LatentODEFunc(
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            time_dim=time_dim,
            num_blocks=num_blocks,
            num_classes=num_classes
        )
        
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint
        
        # Choose ODE integrator
        self.odeint = odeint_adjoint if use_adjoint else odeint
    
    def forward(self, z_0, t_span, label=None):
        """
        Integrate from z_0 at t=0 to target times.
        
        Args:
            z_0: (B, C, D, H, W) initial latent state
            t_span: (T,) tensor of times to evaluate [0, t1, t2, ..., T]
                    e.g., [0, 6, 12, 24] for month timepoints
            label: (B,) optional disease labels
            
        Returns:
            z_trajectory: (T, B, C, D, H, W) latent states at each time
        """
        # Normalize times to [0, 1] for stable integration
        t_normalized = t_span / t_span[-1]
        
        # Set label for conditioning
        self.odefunc.set_label(label)
        
        # Integrate ODE
        z_trajectory = self.odeint(
            self.odefunc,
            z_0,
            t_normalized,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )
        
        return z_trajectory
    
    def predict(self, z_0, target_time=24, label=None):
        """
        Convenience method to predict latent at a single target time.
        
        Args:
            z_0: (B, C, D, H, W) initial latent state at T0
            target_time: target time in months (default 24)
            label: (B,) optional disease labels
            
        Returns:
            z_T: (B, C, D, H, W) predicted latent at target time
        """
        t_span = torch.tensor([0.0, float(target_time)], device=z_0.device)
        z_trajectory = self.forward(z_0, t_span, label)
        return z_trajectory[-1]  # Return prediction at final time


class LatentODEWithIntermediates(nn.Module):
    """
    Neural ODE that can incorporate intermediate observations.
    
    When intermediate timepoints are available (e.g., T6, T12), 
    they are used to refine the trajectory via observation matching.
    """
    def __init__(
        self,
        latent_channels=1,
        hidden_channels=32,
        time_dim=64,
        num_blocks=3,
        num_classes=4,
        solver='dopri5'
    ):
        super().__init__()
        
        self.ode = LatentODE(
            latent_channels=latent_channels,
            hidden_channels=hidden_channels,
            time_dim=time_dim,
            num_blocks=num_blocks,
            num_classes=num_classes,
            solver=solver
        )
        
        # Observation encoder: refines latent using actual observations
        self.obs_encoder = nn.Sequential(
            nn.Conv3d(latent_channels * 2, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            Swish(),
            nn.Conv3d(hidden_channels, latent_channels, 3, padding=1)
        )
    
    def forward(self, z_0, t_span, observations=None, label=None):
        """
        Integrate with optional intermediate observations.
        
        Args:
            z_0: (B, C, D, H, W) initial latent
            t_span: (T,) times to evaluate
            observations: dict {time: z_obs} of available observations
            label: (B,) disease labels
            
        Returns:
            z_trajectory: (T, B, C, D, H, W) refined trajectory
        """
        if observations is None or len(observations) == 0:
            # No intermediates, just integrate
            return self.ode(z_0, t_span, label)
        
        # Integrate segment by segment, updating at observations
        z_current = z_0
        z_trajectory = [z_0]
        
        times = t_span.tolist()
        for i in range(1, len(times)):
            t_segment = torch.tensor([times[i-1], times[i]], device=z_0.device)
            
            # Integrate this segment
            z_segment = self.ode(z_current, t_segment, label)
            z_pred = z_segment[-1]
            
            # Check if we have observation at this time
            t_current = times[i]
            if t_current in observations:
                z_obs = observations[t_current]
                # Combine prediction with observation
                combined = torch.cat([z_pred, z_obs], dim=1)
                z_refined = z_pred + self.obs_encoder(combined)
                z_trajectory.append(z_refined)
                z_current = z_refined
            else:
                z_trajectory.append(z_pred)
                z_current = z_pred
        
        return torch.stack(z_trajectory, dim=0)


# For testing
if __name__ == "__main__":
    # Test the model
    device = config.device
    
    # Create model
    model = LatentODE(
        latent_channels=1,
        hidden_channels=32,
        num_blocks=2,
        num_classes=4
    ).to(device)
    
    # Test input
    batch_size = 2
    z_0 = torch.randn(batch_size, 1, 28, 32, 28).to(device)
    t_span = torch.tensor([0., 6., 12., 24.]).to(device)
    labels = torch.randint(0, 4, (batch_size,)).to(device)
    
    # Forward pass
    print("Testing LatentODE...")
    z_trajectory = model(z_0, t_span, labels)
    print(f"Input shape: {z_0.shape}")
    print(f"Output trajectory shape: {z_trajectory.shape}")
    print(f"Prediction at T24 shape: {z_trajectory[-1].shape}")
    
    # Test prediction method
    z_pred = model.predict(z_0, target_time=24, label=labels)
    print(f"Predict method output shape: {z_pred.shape}")
    
    print("\nAll tests passed!")
