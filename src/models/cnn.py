"""
src/models/cnn.py

1D CNN baseline for spectral signal classification.

Design decisions:
- Three kernel sizes (7, 15, 31) to capture narrow peaks, medium features,
  and broad spectral envelopes simultaneously
- BatchNorm after every conv for training stability
- Global Average Pooling instead of Flatten — gives spatial invariance,
  reduces parameter count, and is required for Grad-CAM to work correctly
- Dropout before classifier head to regularise the 30-class output

Receptive field grows with depth:
  Block 1 (k=7):  RF =  7
  Block 2 (k=15): RF = 29   (7 + 15 - 1 + pooling)
  Block 3 (k=31): RF = 75+  (grows quickly with pooling)
"""

import torch
import torch.nn as nn
from typing import List


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
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=0.1),   # spatial dropout — zeros entire channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for spectral classification.

    Architecture:
        Input  (B, 1, L)
        Block1: Conv(1→C1, k=7)  + BN + ReLU               → (B, C1, L)
        Pool1:  MaxPool(2)                                   → (B, C1, L/2)
        Block2: Conv(C1→C2, k=15) + BN + ReLU               → (B, C2, L/2)
        Pool2:  MaxPool(2)                                   → (B, C2, L/4)
        Block3: Conv(C2→C3, k=15) + BN + ReLU               → (B, C3, L/4)
        Block4: Conv(C3→C4, k=31) + BN + ReLU               → (B, C4, L/4)
        Pool3:  MaxPool(2)                                   → (B, C4, L/8)
        GAP:    GlobalAveragePool                            → (B, C4)
        Head:   Dropout → Linear(C4, n_classes)             → (B, n_classes)
    """

    def __init__(
        self,
        signal_length: int = 1000,
        n_classes: int = 30,
        channels: List[int] = None,
        kernel_sizes: List[int] = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        channels     = channels     or [32, 64, 128, 256]
        kernel_sizes = kernel_sizes or [7, 15, 15, 31]

        assert len(channels) == 4, "channels must have exactly 4 values"
        assert len(kernel_sizes) == 4, "kernel_sizes must have exactly 4 values"

        C1, C2, C3, C4 = channels
        K1, K2, K3, K4 = kernel_sizes

        self.features = nn.Sequential(
            ConvBlock(1,  C1, K1),
            nn.MaxPool1d(2),
            ConvBlock(C1, C2, K2),
            nn.MaxPool1d(2),
            ConvBlock(C2, C3, K3),
            ConvBlock(C3, C4, K4),
            nn.MaxPool1d(2),
        )
        # Global Average Pooling — collapses temporal dimension
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(C4, n_classes),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L)
        feat = self.features(x)      # (B, C4, L')
        feat = self.gap(feat)        # (B, C4, 1)
        feat = feat.squeeze(-1)      # (B, C4)
        return self.classifier(feat) # (B, n_classes)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the final conv feature maps before GAP.
        Required by Grad-CAM — shape: (B, C4, L')
        """
        return self.features(x)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)