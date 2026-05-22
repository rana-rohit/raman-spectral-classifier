"""
src/models/inception1d.py

Inception1D backbone for spectral classification.
Ported from the reference implementation (inception1d_gradcam_fixed.py)
with optional bottleneck and residual support for experimentation.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] | None = None,
        bottleneck_channels: int = 0,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        kernel_sizes = kernel_sizes or [3, 5, 7]
        n_branches = len(kernel_sizes) + 1
        if out_channels % n_branches != 0:
            raise ValueError(
                "out_channels must be divisible by number of branches "
                f"({n_branches}), got {out_channels}"
            )
        branch_channels = out_channels // n_branches

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU(inplace=True),
        )

        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            if bottleneck_channels and bottleneck_channels > 0:
                branch = nn.Sequential(
                    nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(bottleneck_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(
                        bottleneck_channels,
                        branch_channels,
                        kernel_size=k,
                        padding=k // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                branch = nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        branch_channels,
                        kernel_size=k,
                        padding=k // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(branch_channels),
                    nn.ReLU(inplace=True),
                )
            self.branches.append(branch)

        self.use_residual = use_residual
        if use_residual:
            if in_channels != out_channels:
                self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            else:
                self.shortcut = nn.Identity()
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [self.branch1(x)]
        for branch in self.branches:
            outputs.append(branch(x))
        out = torch.cat(outputs, dim=1)
        if self.use_residual:
            out = out + self.shortcut(x)
        return out


class Inception1D(nn.Module):
    def __init__(
        self,
        signal_length: int = 1000,
        n_classes: int = 30,
        in_channels: int = 1,
        stem_channels: int = 32,
        inception_channels: List[int] | None = None,
        kernel_sizes: List[int] | None = None,
        bottleneck_channels: int = 0,
        use_residual: bool = False,
        dropout: float = 0.5,
        fc_dim: int = 256,
    ) -> None:
        del signal_length
        super().__init__()

        inception_channels = inception_channels or [64, 128, 256]
        kernel_sizes = kernel_sizes or [3, 5, 7]
        if len(inception_channels) != 3:
            raise ValueError("inception_channels must have exactly 3 values")

        self.kernel_sizes = [1] + list(kernel_sizes)
        self.use_residual = use_residual
        self.bottleneck_channels = bottleneck_channels

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

        c1, c2, c3 = inception_channels
        self.block1 = InceptionBlock(
            stem_channels,
            c1,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
        )
        self.pool1 = nn.MaxPool1d(2)

        self.block2 = InceptionBlock(
            c1,
            c2,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
        )
        self.pool2 = nn.MaxPool1d(2)

        self.block3 = InceptionBlock(
            c2,
            c3,
            kernel_sizes=kernel_sizes,
            bottleneck_channels=bottleneck_channels,
            use_residual=use_residual,
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embedding_dim = c3

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, n_classes),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(c3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        self._init_weights()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.gap(x)
        return x.squeeze(-1)

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        logits = self.forward_logits(features)
        return {
            "main_logits": logits,
            "aux_logits": None,
            "features": features,
        }

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        return x

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
