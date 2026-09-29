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
        ################################################################
        # TODO
        self.h = dim // 2
        self.w = torch.exp(-torch.arange(self.h) / (self.h - 1) * math.log(10000))
        
        ################################################################

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
        ################################################################
        # TODO
        embs = torch.sin(t[:, None] * self.w[None, :])
        embc = torch.cos(t[:, None] * self.w[None, :])

        emb = torch.cat([embs, embc], dim = 1)

        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1)], dim = 1)

        ################################################################
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

        ################################################################
        # TODO
        self.conv1 = nn.Sequential(
            nn.GroupNorm(groups if in_channels % groups == 0 else 1, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        )

        self.time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(groups if out_channels % groups == 0 else 1, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        ################################################################

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
        ################################################################
        # TODO
        h = self.conv1(x)
        t = self.time(t_emb)
        h = h + t[:, :, None, None]
        h = self.conv2(h)
        h = h + self.skip(x)

        ################################################################
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

        ################################################################
        # TODO
        self.down = nn.Conv2d(channels, channels, 4, 2, 1)

        ################################################################

    def forward(self, x):

        """
        Applies spatial downsampling to a feature map.

        Args:
            x (Tensor): Feature map of shape (B, C, H, W).

        Returns:
            x (Tensor): Downsampled feature map of shape (B, C, H/2, W/2).
        """

        ################################################################
        # TODO
        x = self.down(x)

        ################################################################

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

        ################################################################
        # TODO
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

        ################################################################

    def forward(self, x):

        """
        Applies spatial upsampling to a feature map.

        Args:
            x (Tensor): Feature map of shape (B, C, H, W).

        Returns:
            x (Tensor): Upsampled feature map of shape (B, C, 2H, 2W).
        """

        ################################################################
        # TODO
        x = self.up(x)

        ################################################################

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

        ################################################################
        # TODO
        self.te = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, 4 * time_dim),
            nn.SiLU(),
            nn.Linear(4 * time_dim, time_dim),
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, 1, 1)

        self.db1 = ResBlock(base_channels, base_channels, time_dim, groups)
        self.d1 = Downsample(base_channels)
        self.db2 = ResBlock(base_channels, 2 * base_channels, time_dim, groups)
        self.d2 = Downsample(2 * base_channels)
        self.db3 = ResBlock(2 * base_channels, 4 * base_channels, time_dim, groups)
        self.d3 = Downsample(4 * base_channels)

        self.bn = ResBlock(4 * base_channels, 4 * base_channels, time_dim, groups)

        self.u1 = Upsample(4 * base_channels)
        self.up1 = ResBlock(8 * base_channels, 4 * base_channels, time_dim, groups)
        self.u2 = Upsample(4 * base_channels)
        self.up2 = ResBlock(6 * base_channels, 2 * base_channels, time_dim, groups)
        self.u3 = Upsample(2 * base_channels)
        self.up3 = ResBlock(3 * base_channels, base_channels, time_dim, groups)

        self.out = nn.Sequential(
            nn.GroupNorm(groups if base_channels % groups == 0 else 1, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, 1, 1)
        )

        
        ################################################################

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
        ################################################################
        # TODO
        t = self.te(t)

        h = self.init_conv(x)

        skip1 = self.db1(h, t)
        h = self.d1(skip1)
        skip2 = self.db2(h, t)
        h = self.d2(skip2)
        skip3 = self.db3(h, t)
        h = self.d3(skip3)

        h = self.bn(h, t)

        h = self.u1(h)
        h = torch.cat([h, skip3], dim = 1)
        h = self.up1(h, t)
        h = self.u2(h)
        h = torch.cat([h, skip2], dim = 1)
        h = self.up2(h, t)
        h = self.u3(h)
        h = torch.cat([h, skip1], dim = 1)
        h = self.up3(h, t)

        out = self.out(h)
        
        ################################################################

        return out