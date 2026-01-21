
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from torchvision.transforms import ToPILImage
from torchvision import models
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL


class NoiseScheduler:
    """Standard DDPM noise scheduler – linear or cosine beta schedule."""
    def __init__(self, timesteps=1000, beta_schedule="linear", device='cpu'):
        self.timesteps = timesteps
        self.device = torch.device(device)

        if beta_schedule == "linear":
            betas = self._linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas.to(self.device)
        self.alphas = (1.0 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(self.device)

        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod).to(self.device)

    def _linear_beta_schedule(self, timesteps, start=0.0001, end=0.02):
        return torch.linspace(start, end, timesteps, dtype=torch.float32)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        t = torch.arange(timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_bar = f_t / f_t[0]
        betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
        return torch.clip(betas, 0.0001, 0.999)

    def get_noisy_image(self, x_0, t, noise=None):
        """
        Adds noise to the original image x_0 at time t.
        """
        if noise is None:
            noise = torch.randn_like(x_0, device=self.device)
        if t.ndim == 1:
            alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        else:
            alpha_bar_t = self.alphas_cumprod[t]

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    def denoise_image(self, x_t, t, noise_unet):
        """
        Denoises an image x_t given the predicted noise (noise_unet) at time t.
        This is the reverse step.
        """
        if t.ndim == 1:
            alpha_bar_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        else:
            alpha_bar_t = self.alphas_cumprod[t]

        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        x_0 = (x_t - sqrt_one_minus_alpha_bar_t * noise_unet) / sqrt_alpha_bar_t

        return x_0

class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal Positional Embedding for time steps.
    Transforms a scalar time step 't' into a high-dimensional vector.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.embeddings = math.log(10000) / (self.half_dim - 1)
        self.embeddings = torch.exp(torch.arange(self.half_dim) * -self.embeddings)

    def forward(self, time):
        time_embedding = time.unsqueeze(1) * self.embeddings.to(time.device)

        time_embedding = torch.cat((time_embedding.sin(), time_embedding.cos()), dim=-1)
        return time_embedding

class AdaptiveGroupNorm(nn.Module):
    """
    Adaptive Group Normalization (AdaGN) layer.
    Applies GroupNorm, then scales and shifts features based on conditioning (time embedding).
    """
    def __init__(self, num_groups, num_channels, time_emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.time_proj = nn.Linear(time_emb_dim, 2 * num_channels)

    def forward(self, x, time_emb):

        normed_x = self.norm(x)

        scale_shift = self.time_proj(time_emb)
        scale, shift = scale_shift.chunk(2, dim=1)

        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)

        output = normed_x * (1 + scale) + shift
        return output

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Fully Connected Layers (bottleneck)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: Learn weights
        y = self.fc(y).view(b, c, 1, 1)
        # Scale the original input
        return x * y.expand_as(x)


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        # Reshape for Attention: (Batch, Channels, H, W) -> (Batch, H*W, Channels)
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)

        # Norm and Attention
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        x = x + attention_value

        # Feed Forward
        x = x + self.ff_self(x)

        # Reshape back: (Batch, H*W, Channels) -> (Batch, Channels, H, W)
        return x.swapaxes(1, 2).view(-1, self.channels, size, size)

class ContractingBlock(nn.Module):
    """
    Encoder blocks of the Unet, using two convolutions, SE Block and residual connections.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, use_dropout=False, use_attention=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        self.norm1 = AdaptiveGroupNorm(num_groups=8, num_channels=out_channels, time_emb_dim=time_emb_dim)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = AdaptiveGroupNorm(num_groups=8, num_channels=out_channels, time_emb_dim=time_emb_dim)

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        if use_dropout:
            self.dropout = nn.Dropout(0.3)
        self.use_dropout = use_dropout

        self.use_attention = use_attention

        self.se = SEBlock(out_channels)

        self.selfatt = SelfAttention(out_channels)

    def forward(self, x, time_emb):
        identity = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x, time_emb)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x, time_emb)
        if self.use_dropout:
            x = self.dropout(x)
        skip_features = self.activation(x)

        if self.use_attention:
            skip_features = self.selfatt(skip_features)

        x = self.downsample(x + identity)
        return x

class ExpandingBlock(nn.Module):
    """
    Decoder blocks of the Unet, using two convolutions, SE Block, residual connections and skip connections.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, use_dropout=False, use_attention=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.norm1 = AdaptiveGroupNorm(num_groups=8, num_channels=out_channels, time_emb_dim=time_emb_dim)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = AdaptiveGroupNorm(num_groups=8, num_channels=out_channels, time_emb_dim=time_emb_dim)

        self.activation = nn.SiLU()
        if use_dropout:
            self.dropout = nn.Dropout(0.3)
        self.use_dropout = use_dropout

        self.use_attention = use_attention

        self.selfatt = SelfAttention(out_channels)

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.se = SEBlock(out_channels)

    def forward(self, x, skip_con_x, time_emb):
        x = self.upsample(x)

        if x.shape != skip_con_x.shape:
             x = torch.nn.functional.interpolate(x, size=skip_con_x.shape[2:], mode='nearest')
        # Skip connection
        x = torch.cat([x, skip_con_x], dim=1)

        identity = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x, time_emb)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x, time_emb)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)

        if self.use_attention:
            x = self.selfatt(x)

        return x + identity

class BottleNeck(nn.Module):
    """
    Decoder blocks of the Unet, using two convolutions, SE Block, residual connections and skip connections.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, use_dropout=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = AdaptiveGroupNorm(num_groups=8, num_channels=out_channels, time_emb_dim=time_emb_dim)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = AdaptiveGroupNorm(num_groups=8, num_channels=out_channels, time_emb_dim=time_emb_dim)

        self.activation = nn.SiLU()
        if use_dropout:
            self.dropout = nn.Dropout(0.3)
        self.use_dropout = use_dropout

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.se = SEBlock(out_channels)

    def forward(self, x, time_emb):

        identity = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x, time_emb)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x, time_emb)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)

        x = self.se(x)

        return x + identity

class FeatureMapBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    Unet network using six encoder layers and six decoder layers.
    """
    def __init__(self, input_channels, output_channels, hidden_channels=256, time_emb_dim=256): # Added time_emb_dim
        super().__init__()

        # Applying an MLP to the Sinusoidal positional embedding to create a richer vector of the time (t).

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        self.time_emb_dim = time_emb_dim

        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)

        self.contract1 = ContractingBlock(hidden_channels, hidden_channels*2, time_emb_dim)
        self.contract2 = ContractingBlock(hidden_channels*2, hidden_channels*4, time_emb_dim, use_attention=True)
        self.contract3 = ContractingBlock(hidden_channels*4, hidden_channels*8, time_emb_dim, use_attention=True)

        self.bottleneck = BottleNeck(hidden_channels*8, hidden_channels*8, time_emb_dim)

        self.expand0 = ExpandingBlock(hidden_channels*8, hidden_channels*4, time_emb_dim, use_attention=True)
        self.expand1 = ExpandingBlock(hidden_channels*4, hidden_channels*2, time_emb_dim, use_attention=True)
        self.expand2 = ExpandingBlock(hidden_channels*2, hidden_channels, time_emb_dim)

        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)

    def forward(self, x, time):
        time_emb = self.time_mlp(time)

        x0 = self.upfeature(x)

        x1 = self.contract1(x0, time_emb) # 32x32
        x2 = self.contract2(x1, time_emb) # 16x16
        x3 = self.contract3(x2, time_emb) # 8x8

        x4 = self.bottleneck(x3, time_emb) # 8x8

        x5 = self.expand0(x4, x2, time_emb) # 16x16
        x6 = self.expand1(x5, x1, time_emb) # 32x32
        x7 = self.expand2(x6, x0, time_emb) # 64x64

        x8 = self.downfeature(x7)

        return x8

