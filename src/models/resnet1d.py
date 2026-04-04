"""
src/models/resnet1d.py

1D ResNet for spectral classification — strong baseline that allows
much deeper networks than plain CNN without degradation.

Residual connections let gradients flow directly to early layers,
enabling 8+ convolutional blocks without vanishing gradients.

Architecture follows ResNet-18 spirit adapted for 1D signals:
  - Initial stem conv (wide kernel to capture broad structure)
  - 4 stages of residual blocks with progressive channel doubling
  - Global Average Pooling + linear head

Shortcut connections use 1x1 conv when channel dimensions change.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ResidualBlock1D(nn.Module):
    """
    Basic 1D residual block:
        x → Conv(k=3) → BN → ReLU → Conv(k=3) → BN → + x → ReLU

    If in_channels != out_channels, a 1×1 conv shortcut projects x.
    Stride=2 in the first conv halves the temporal dimension (like MaxPool).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=pad, bias=False
        )
        self.bn1   = nn.BatchNorm1d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=pad, bias=False
        )
        self.bn2   = nn.BatchNorm1d(out_channels)

        # Shortcut: project when dimensions change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class ResNet1D(nn.Module):
    """
    1D ResNet for spectral classification.

    Architecture:
        Stem:    Conv(1→64, k=15) + BN + ReLU + MaxPool(2)   → (B,  64, L/2)
        Stage1:  2× ResBlock(64→64,   stride=1)               → (B,  64, L/2)
        Stage2:  2× ResBlock(64→128,  stride=2)               → (B, 128, L/4)
        Stage3:  2× ResBlock(128→256, stride=2)               → (B, 256, L/8)
        Stage4:  2× ResBlock(256→512, stride=2)               → (B, 512, L/16)
        GAP + Head                                             → (B, n_classes)

    Configurable n_blocks per stage and channel widths.
    """

    def __init__(
        self,
        signal_length: int = 1000,
        n_classes: int = 30,
        channels: List[int] = None,
        n_blocks: List[int] = None,
        stem_kernel: int = 15,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        channels = channels or [64, 128, 256, 512]
        n_blocks = n_blocks or [2, 2, 2, 2]     # ResNet-18 config
        assert len(channels) == 4 and len(n_blocks) == 4

        C1, C2, C3, C4 = channels

        # Stem: wide kernel captures broad spectral structure first
        self.stem = nn.Sequential(
            nn.Conv1d(1, C1, stem_kernel, stride=1, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(C1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        # Residual stages
        self.stage1 = self._make_stage(C1, C1, n_blocks[0], stride=1)
        self.stage2 = self._make_stage(C1, C2, n_blocks[1], stride=2)
        self.stage3 = self._make_stage(C2, C3, n_blocks[2], stride=2)
        self.stage4 = self._make_stage(C3, C4, n_blocks[3], stride=2)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(C4, n_classes),
        )

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_ch: int,
        out_ch: int,
        n_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [ResidualBlock1D(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            layers.append(ResidualBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).squeeze(-1)
        return self.classifier(x)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Returns final stage feature maps for Grad-CAM."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.stage4(x)

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