"""
1D Convolutional Block Attention Module for spectral models.

The module preserves input shape throughout:
    (B, C, T) -> (B, C, T)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention1D(nn.Module):
    """
    Channel attention using shared MLP gates over avg- and max-pooled spectra.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if reduction < 1:
            raise ValueError("reduction must be >= 1")

        hidden_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _ = x.shape
        avg = self.avg_pool(x).view(batch_size, channels)
        max_values = self.max_pool(x).view(batch_size, channels)
        attention = self.sigmoid(self.mlp(avg) + self.mlp(max_values))
        return x * attention.view(batch_size, channels, 1)


class SpatialAttention1D(nn.Module):
    """
    True 1D spectral attention over temporal/spectral positions.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve spectral length")

        self.conv = nn.Conv1d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_values, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg, max_values], dim=1)))
        return x * attention


class CBAM1D(nn.Module):
    """
    Sequential channel and spatial attention for 1D spectral feature maps.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention1D(
            channels=channels,
            reduction=reduction,
        )
        self.spatial_attention = SpatialAttention1D(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        return self.spatial_attention(x)
