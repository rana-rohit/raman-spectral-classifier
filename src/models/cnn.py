"""
src/models/cnn.py

1D CNN baseline for spectral signal classification.

Design decisions:
- Four kernel sizes (7, 15, 15, 31) capturing multi-scale spectral features
- BatchNorm after every conv for training stability
- Global Average Pooling instead of Flatten gives spatial invariance,
  reduces parameter count, and is required for Grad-CAM to work correctly
- Dropout before classifier head regularises the 30-class output
- No normalization after feature extraction to preserve discriminative amplitude information
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1d + BatchNorm + ReLU. The fundamental building unit."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = "same",
    ) -> None:
        del padding
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=0.05),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1D(nn.Module):
    def __init__(
        self,
        signal_length: int = 1000,
        n_classes: int = 30,
        channels: List[int] | None = None,
        kernel_sizes: List[int] | None = None,
        dropout: float = 0.3,
        in_channels: int = 1,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128, 256]
        kernel_sizes = kernel_sizes or [7, 15, 15, 31]

        assert len(channels) == 4, "channels must have exactly 4 values"
        assert len(kernel_sizes) == 4, "kernel_sizes must have exactly 4 values"

        c1, c2, c3, c4 = channels
        k1, k2, k3, k4 = kernel_sizes

        self.features = nn.Sequential(
            ConvBlock(in_channels, c1, k1),
            nn.MaxPool1d(2),
            ConvBlock(c1, c2, k2),
            nn.MaxPool1d(2),
            ConvBlock(c2, c3, k3),
            ConvBlock(c3, c4, k4),
            nn.MaxPool1d(2),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embedding_dim = c4

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c4, n_classes),
        )

        self._init_weights()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.gap(feat)
        return feat.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
