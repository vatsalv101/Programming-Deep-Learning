import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):

    """
    Sinoidal time embedding used by the U-Net.

    Args:
        dim (int): Output embedding dimension. Must be even.
    """

    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        h = dim // 2
        self.register_buffer("freqs", torch.exp(-torch.arange(h, dtype=torch.float32) / (h - 1) * math.log(10000)), persistent=False)

    def forward(self, t):

        """
        Embeds integer or continuous time values.

        Args:
            t (Tensor): Time tensor of shape (B,). For DDPM this contains
                integer diffusion steps. For score-based diffusion this
                contains continuous values in [eps, 1].

        Returns:
            emb (Tensor): Sinusoidal embedding of shape (B, dim).
        """

        emb = None
        t = t.float()
        emb = t[:, None] * self.freqs[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=emb.device)], dim=-1)
        return emb
    
    
class ResBlock(nn.Module):

    """
    Small convolutional residual block with a projected time embedding.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        time_dim (int): Dimension of the time embedding.
        groups (int): Number of groups for group normalization.
    """

    def __init__(self, in_channels, out_channels, time_dim, groups=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(groups if in_channels % groups == 0 else 1, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(groups if out_channels % groups == 0 else 1, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, t_emb):

        """
        Applies the block to an image feature map.

        Args:
            x (Tensor): Feature map of shape (B, C, H, W).
            t_emb (Tensor): Time embedding of shape (B, time_dim).

        Returns:
            h (Tensor): Output feature map of shape (B, out_channels, H, W).
        """

        h = None
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(h)
        h = h + self.skip(x)
        return h
    

class Downsample(nn.Module):

    """
    Downsampling layer that halves the spatial resolution of a feature map.

    Uses a strided convolution to reduce height and width by a factor of 2
    while keeping the number of channels unchanged.

    Args:
        channels (int): Number of input and output feature channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.down = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):

        """
        Applies spatial downsampling to a feature map.

        Args:
            x (Tensor): Feature map of shape (B, C, H, W).

        Returns:
            x (Tensor): Downsampled feature map of shape (B, C, H/2, W/2).
        """

        x = self.down(x)

        return x


class Upsample(nn.Module):

    """
    Upsampling layer that doubles the spatial resolution of a feature map.

    Uses nearest-neighbor interpolation followed by a convolution to increase
    height and width by a factor of 2 while keeping the number of channels
    unchanged.

    Args:
        channels (int): Number of input and output feature channels.
    """

    def __init__(self, channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):

        """
        Applies spatial upsampling to a feature map.

        Args:
            x (Tensor): Feature map of shape (B, C, H, W).

        Returns:
            x (Tensor): Upsampled feature map of shape (B, C, 2H, 2W).
        """

        x = self.up(x)

        return x


class UNet(nn.Module):

    """
    U-Net for noise or score prediction.

    Args:
        in_channels (int): Number of image channels.
        base_channels (int): Number of feature channels in the first level.
        time_dim (int): Time embedding dimension.
    """

    def __init__(self, in_channels, base_channels, time_dim, groups):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.groups = groups

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
            )
        
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.downblock1 = ResBlock(base_channels, base_channels, time_dim, groups)
        self.down1 = Downsample(base_channels)
        self.downblock2 = ResBlock(base_channels, base_channels * 2, time_dim, groups)
        self.down2 = Downsample(base_channels * 2)
        self.downblock3 = ResBlock(base_channels * 2, base_channels * 4, time_dim, groups)
        self.down3 = Downsample(base_channels * 4)

        
        self.bottleneck = ResBlock(base_channels * 4, base_channels * 4, time_dim, groups)
        
        self.up1 = Upsample(base_channels * 4)
        self.upblock1 = ResBlock(base_channels * 8, base_channels * 4, time_dim, groups)
        self.up2 = Upsample(base_channels * 4)
        self.upblock2 = ResBlock(base_channels * 6, base_channels * 2, time_dim, groups)
        self.up3 = Upsample(base_channels * 2)
        self.upblock3 = ResBlock(base_channels * 3, base_channels, time_dim, groups)

        self.out = nn.Sequential(
            nn.GroupNorm(groups if base_channels % groups == 0 else 1, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, t):

        """
        Predicts noise or score for a noised image batch.

        Args:
            x (Tensor): Noised image batch of shape (B, 1, 32, 32).
            t (Tensor): Time tensor of shape (B,).

        Returns:
            out (Tensor): Prediction of shape (B, 1, 32, 32).
        """

        out = None
        t_emb = self.time_embedding(t)
        
        h = self.init_conv(x)

        skip1 = self.downblock1(h, t_emb)
        h = self.down1(skip1)
        skip2 = self.downblock2(h, t_emb)
        h = self.down2(skip2)
        skip3 = self.downblock3(h, t_emb)
        h = self.down3(skip3)
        
        h = self.bottleneck(h, t_emb)
        
        h = self.up1(h)
        h = torch.cat([h, skip3], dim=1)
        h = self.upblock1(h, t_emb)
        h = self.up2(h)
        h = torch.cat([h, skip2], dim=1)
        h = self.upblock2(h, t_emb)
        h = self.up3(h)
        h = torch.cat([h, skip1], dim=1)
        h = self.upblock3(h, t_emb)

        out = self.out(h)

        return out
