"""
Neural ODE and SDE models for longitudinal PET prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
import torchsde
import config


# ============================================================
# Basic Blocks
# ============================================================

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
        
        if time_dim is not None:
            self.time_mlp = nn.Sequential(
                Swish(),
                nn.Linear(time_dim, out_channels)
            )
        else:
            self.time_mlp = None
            
        if in_channels != out_channels:
            self.residual = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x, t_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        if t_emb is not None and self.time_mlp is not None:
            t_emb = self.time_mlp(t_emb)
            h = h + t_emb[:, :, None, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.residual(x)


# ============================================================
# Core Functions (Drift f & Diffusion g)
# ============================================================

class LatentODEFunc(nn.Module):
    """
    Drift function f(t, y) for both ODE and SDE.
    Architecture: Lightweight 3D U-Net.
    """
    def __init__(
        self,
        latent_channels=1,
        hidden_channels=32,
        time_dim=64,
        num_blocks=3,
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
        if num_classes > 0:
            self.label_embed = nn.Embedding(num_classes, time_dim)
        else:
            self.label_embed = None
        
        # Encoder
        self.input_conv = nn.Conv3d(latent_channels, hidden_channels, 3, padding=1)
        self.encoder_blocks = nn.ModuleList([
            ConvBlock3D(hidden_channels, hidden_channels, time_dim)
            for _ in range(num_blocks)
        ])
        
        # Middle
        self.middle_block = ConvBlock3D(hidden_channels, hidden_channels, time_dim)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList([
            ConvBlock3D(hidden_channels * 2, hidden_channels, time_dim)
            for _ in range(num_blocks)
        ])
        
        # Output dz/dt
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(8, hidden_channels), hidden_channels),
            Swish(),
            nn.Conv3d(hidden_channels, latent_channels, 3, padding=1)
        )
        
        self._label = None
    
    def set_label(self, label):
        self._label = label
    
    def forward(self, t, z):
        # Time embedding
        batch_size = z.shape[0]
        t_tensor = t.expand(batch_size) if t.dim() == 0 else t
        t_emb = self.time_embed(t_tensor)
        t_emb = self.time_mlp(t_emb)
        
        if self.label_embed is not None and self._label is not None:
            t_emb = t_emb + self.label_embed(self._label.long())
        
        # U-Net structure
        h = self.input_conv(z)
        skips = []
        for block in self.encoder_blocks:
            h = block(h, t_emb)
            skips.append(h)
        
        h = self.middle_block(h, t_emb)
        
        for block, skip in zip(self.decoder_blocks, reversed(skips)):
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            
        return self.output_conv(h)


class DiffusionFunc(nn.Module):
    """
    Diffusion function g(t, y) for SDE.
    Architecture: Similar to Drift but simpler, predicting log-sigma for diagonal noise.
    """
    def __init__(
        self,
        latent_channels=1,
        hidden_channels=32,
        time_dim=64,
        num_classes=4
    ):
        super().__init__()
        
        self.time_embed = TimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            Swish(),
            nn.Linear(time_dim, time_dim)
        )
        
        if num_classes > 0:
            self.label_embed = nn.Embedding(num_classes, time_dim)
        else:
            self.label_embed = None

        # Simple 3D CNN for diffusion coefficient
        self.net = nn.Sequential(
            nn.Conv3d(latent_channels, hidden_channels, 3, padding=1),
            Swish(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            Swish(),
            nn.Conv3d(hidden_channels, latent_channels, 3, padding=1)
        )
        
        self._label = None

    def set_label(self, label):
        self._label = label

    def forward(self, t, y):
        batch_size = y.shape[0]
        t_tensor = t.expand(batch_size) if t.dim() == 0 else t
        t_emb = self.time_embed(t_tensor)
        t_emb = self.time_mlp(t_emb)
        
        if self.label_embed is not None and self._label is not None:
            t_emb = t_emb + self.label_embed(self._label.long())
        
        # Combine y and t (simplified injection for diffusion)
        # Note: Ideally inject t into convs, here we add it to input scaling for simplicity
        # or just use the net. For robustness, let's keep it simple.
        # Diffusion needs to be positive. We output log_sigma and exp() it.
        # Standard SDE: g(t, y).
        
        out = self.net(y)
        # Add time effect global modulation
        t_scale = torch.sigmoid(t_emb).mean(dim=1).view(batch_size, 1, 1, 1, 1)
        
        # Return sigma. Softplus for positivity + small epsilon
        return F.softplus(out) * t_scale + 1e-3


# ============================================================
# SDE System Wrapper
# ============================================================

class SDEWrapper(nn.Module):
    """Wraps f and g for torchsde, handling flattening for 2D requirement."""
    def __init__(self, f_func, g_func, shape):
        super().__init__()
        self.f_func = f_func
        self.g_func = g_func
        self.shape = shape # (C, D, H, W) without batch
        self.noise_type = "diagonal"
        self.sde_type = "ito"
    
    def f(self, t, y):
        # y is (B, D*H*W*C)
        B = y.shape[0]
        y_reshaped = y.view(B, *self.shape)
        out = self.f_func(t, y_reshaped)
        return out.view(B, -1)
    
    def g(self, t, y):
        B = y.shape[0]
        y_reshaped = y.view(B, *self.shape)
        out = self.g_func(t, y_reshaped)
        return out.view(B, -1)


# ============================================================
# Models
# ============================================================

class LatentODE(nn.Module):
    """Standard Neural ODE (Drift only, with optional Jump for intermediates)."""
    def __init__(self, latent_channels=1, hidden_channels=32, time_dim=64, num_blocks=3, num_classes=4, solver='dopri5'):
        super().__init__()
        self.odefunc = LatentODEFunc(latent_channels, hidden_channels, time_dim, num_blocks, num_classes)
        self.solver = solver
        
        # Observation encoder for jump updates
        self.obs_encoder = nn.Sequential(
            nn.Conv3d(latent_channels * 2, hidden_channels, 3, padding=1),
            nn.GroupNorm(min(8, hidden_channels), hidden_channels),
            Swish(),
            nn.Conv3d(hidden_channels, latent_channels, 3, padding=1)
        )
        
    def forward(self, z_0, t_span, observations=None, label=None):
        self.odefunc.set_label(label)
        
        # Case 1: No intermediates - single-span integration
        if observations is None or len(observations) == 0:
            t_normalized = t_span / t_span[-1]
            z = odeint(self.odefunc, z_0, t_normalized, method=self.solver, rtol=1e-4, atol=1e-5)
            return z
        
        # Case 2: With intermediates - segment-by-segment with jumps
        obs_times = sorted(observations.keys())
        all_times = sorted(set([0] + obs_times + [t_span[-1].item()]))
        max_t = float(t_span[-1])
        
        z_current = z_0
        z_trajectory = [z_0]
        
        for i in range(len(all_times) - 1):
            t_start = all_times[i]
            t_end = all_times[i + 1]
            
            t_segment = torch.tensor(
                [t_start / max_t, t_end / max_t],
                device=z_0.device
            )
            
            seg_result = odeint(
                self.odefunc, z_current, t_segment,
                method=self.solver, rtol=1e-4, atol=1e-5
            )
            z_pred = seg_result[-1]
            
            # Jump update if observation available at t_end
            t_end_int = int(t_end)
            if t_end_int in observations:
                z_obs = observations[t_end_int]
                combined = torch.cat([z_pred, z_obs], dim=1)
                z_refined = z_pred + self.obs_encoder(combined)
                z_trajectory.append(z_refined)
                z_current = z_refined
            else:
                z_trajectory.append(z_pred)
                z_current = z_pred
        
        return torch.stack(z_trajectory, dim=0)
        
    def predict(self, z_0, target_time=24, label=None):
        t = torch.tensor([0., float(target_time)], device=z_0.device)
        return self.forward(z_0, t, None, label)[-1]


class LatentSDE(nn.Module):
    """
    True Latent SDE using torchsde.
    Wrapper around torchsde.sdeint.
    """
    def __init__(self, latent_channels=1, hidden_channels=32, time_dim=64, num_blocks=3, num_classes=4, solver='srk', diffusion_steps=None):
        super().__init__()
        
        self.f_func = LatentODEFunc(latent_channels, hidden_channels, time_dim, num_blocks, num_classes)
        self.g_func = DiffusionFunc(latent_channels, hidden_channels // 2, time_dim, num_classes)
        
        # SDEWrapper initialized with shape later or passed?
        # We need shape for reshaping. Since shape is fixed (C, D, H, W), we can infer or pass.
        # But D, H, W might vary? In this project seems fixed (28x32x28).
        # Let's handle it dynamically if possible, or store shape from first forward?
        # torchsde requires the object to be passed.
        # We will create SDEWrapper in forward pass or update its shape.
        self.sde_system = None # Created on fly
        self.latent_channels = latent_channels
        
        self.solver = solver 
        self.dt = 0.05
        
        # Observation encoder for jump updates (same as LatentODE)
        hidden_ch = hidden_channels
        self.obs_encoder = nn.Sequential(
            nn.Conv3d(latent_channels * 2, hidden_ch, 3, padding=1),
            nn.GroupNorm(min(8, hidden_ch), hidden_ch),
            Swish(),
            nn.Conv3d(hidden_ch, latent_channels, 3, padding=1)
        )
        
    def _get_sde_system(self, shape):
        # shape: (C, D, H, W)
        return SDEWrapper(self.f_func, self.g_func, shape)
        
    def forward(self, z_0, t_span, observations=None, label=None):
        """
        Forward pass with optional segment-by-segment integration.
        
        Args:
            z_0: (B, C, D, H, W) initial latent state
            t_span: tensor of timepoints, e.g. [0, 24] or [0, 6, 12, 18, 24]
            observations: dict {month: (B, C, D, H, W)} of intermediate observations, or None
            label: (B,) disease labels
        """
        self.f_func.set_label(label)
        self.g_func.set_label(label)
        
        batch_size = z_0.shape[0]
        shape_without_batch = z_0.shape[1:]  # (C, D, H, W)
        
        # Case 1: No intermediates - single-span integration
        if observations is None or len(observations) == 0:
            z_0_flat = z_0.view(batch_size, -1)
            sde_system = self._get_sde_system(shape_without_batch)
            
            z_traj_flat = torchsde.sdeint(
                sde_system, z_0_flat, t_span,
                method='srk', dt=self.dt,
                names={'drift': 'f', 'diffusion': 'g'}
            )
            T = z_traj_flat.shape[0]
            return z_traj_flat.view(T, batch_size, *shape_without_batch)
        
        # Case 2: With intermediates - segment-by-segment with jumps
        # Build sorted list of all timepoints to integrate through
        obs_times = sorted(observations.keys())
        all_times = sorted(set([0] + obs_times + [t_span[-1].item()]))
        
        sde_system = self._get_sde_system(shape_without_batch)
        z_current = z_0
        z_trajectory = [z_0]
        max_t = float(t_span[-1])
        
        for i in range(len(all_times) - 1):
            t_start = all_times[i]
            t_end = all_times[i + 1]
            
            # Normalize to [0, 1] for numerical stability
            t_segment = torch.tensor(
                [t_start / max_t, t_end / max_t],
                device=z_0.device
            )
            
            z_flat = z_current.view(batch_size, -1)
            seg_traj = torchsde.sdeint(
                sde_system, z_flat, t_segment,
                method='srk', dt=self.dt,
                names={'drift': 'f', 'diffusion': 'g'}
            )
            z_pred = seg_traj[-1].view(batch_size, *shape_without_batch)
            
            # Jump update if observation available at t_end
            t_end_int = int(t_end)
            if t_end_int in observations:
                z_obs = observations[t_end_int]
                combined = torch.cat([z_pred, z_obs], dim=1)
                z_refined = z_pred + self.obs_encoder(combined)
                z_trajectory.append(z_refined)
                z_current = z_refined
            else:
                z_trajectory.append(z_pred)
                z_current = z_pred
        
        return torch.stack(z_trajectory, dim=0)

    def predict(self, z_0, target_time=24, label=None, num_samples=1):
        t_span = torch.tensor([0., float(target_time)], device=z_0.device)
        self.f_func.set_label(label)
        self.g_func.set_label(label)
        
        shape_without_batch = z_0.shape[1:]
        sde_system = self._get_sde_system(shape_without_batch)
        
        is_expanded = False 
        
        if num_samples > 1:
            z_0_start = z_0.repeat(num_samples, 1, 1, 1, 1)
            if label is not None:
                expanded_label = label.repeat(num_samples)
                self.f_func.set_label(expanded_label)
                self.g_func.set_label(expanded_label)
            is_expanded = True
            current_batch_size = z_0.shape[0] * num_samples
        else:
            z_0_start = z_0
            current_batch_size = z_0.shape[0]
            
        z_0_flat = z_0_start.view(current_batch_size, -1)
        
        traj_flat = torchsde.sdeint(sde_system, z_0_flat, t_span, method='srk', dt=self.dt)
        z_T_flat = traj_flat[-1]
        
        # Reshape
        z_T = z_T_flat.view(current_batch_size, *shape_without_batch)
        
        if is_expanded:
            # Reshape to (samples, batch, ...)
            z_T = z_T.view(num_samples, z_0.shape[0], *shape_without_batch)
            # Restore labels
            self.f_func.set_label(label)
            self.g_func.set_label(label)
            return z_T.mean(dim=0)
            
        return z_T

    def compute_loss(self, z_0, z_T_gt, label=None):
        """
        SDE Training Loss.
        Ideally: Variational loss.
        Simplification: MSE of trajectory samples.
        """
        # Generate prediction (single sample path)
        t_span = torch.tensor([0., 24.], device=z_0.device)
        z_pred = self.forward(z_0, t_span, label=label)[-1]
        
        loss = F.mse_loss(z_pred, z_T_gt)
        # Add regularization to encourage non-zero diffusion?
        # For now, basic MSE.
        
        return loss, {'loss': loss.item()}


if __name__ == "__main__":
    # Test
    print("Testing SDE...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    z = torch.randn(2, 1, 28, 32, 28).to(device)
    t = torch.tensor([0., 24.]).to(device)
    model = LatentSDE().to(device)
    out = model(z, t)
    print(out.shape)
