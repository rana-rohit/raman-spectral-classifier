"""
src/models/resnet1d.py

1D ResNet for spectral classification.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

class DepthwiseSeparableConv1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),

            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),

            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.conv1 = DepthwiseSeparableConv1D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConv1D(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
        )
        self.bn2 = nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.Identity(),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)

class ResNet1D(nn.Module):
    def __init__(
        self,
        signal_length: int = 1000,
        n_classes: int = 30,
        channels: List[int] | None = None,
        n_blocks: List[int] | None = None,
        stem_kernel: int = 7,
        dropout: float = 0.3,
        in_channels: int = 2,
    ) -> None:
        del signal_length
        super().__init__()
        channels = channels or [32, 64, 128, 256]
        n_blocks = n_blocks or [2, 2, 2, 2]
        assert len(channels) == 4 and len(n_blocks) == 4

        c1, c2, c3, c4 = channels

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, c1, stem_kernel, stride=1, padding=stem_kernel // 2, bias=False),
            nn.Identity(),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(c1, c1, n_blocks[0], stride=1, kernel_size=7)
        self.stage2 = self._make_stage(c1, c2, n_blocks[1], stride=2, kernel_size=9)
        self.stage3 = self._make_stage(c2, c3, n_blocks[2], stride=2, kernel_size=13)
        self.stage4 = self._make_stage(c3, c4, n_blocks[3], stride=2, kernel_size=17)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embedding_dim = c4

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c4, n_classes),
        )
        self._init_weights()

    @staticmethod
    def _make_stage(
        in_ch: int,
        out_ch: int,
        n_blocks: int,
        stride: int,
        kernel_size: int,
    ) -> nn.Sequential:
        layers = [ResidualBlock1D(in_ch, out_ch, stride=stride, kernel_size=kernel_size)]
        for _ in range(n_blocks - 1):
            layers.append(ResidualBlock1D(out_ch, out_ch, stride=1, kernel_size=kernel_size))
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        return x.squeeze(-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        return {
            "main_logits": self.classifier(features),
            "features": features,
        }

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.stage4(x)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
