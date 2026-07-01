"""
src/models/inception1d.py

Inception1D backbone for spectral classification.
Designed for Raman spectra with multi-scale receptive fields,
default bottlenecks, and residual connections.
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
        kernel_sizes: List[int],
        bottleneck_channels: int,
        use_residual: bool,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        n_branches = len(kernel_sizes) + 1
        if out_channels % n_branches != 0:
            raise ValueError(
                "out_channels must be divisible by number of branches "
                f"({n_branches}), got {out_channels}"
            )
        branch_channels = out_channels // n_branches

        act = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(branch_channels),
            act,
        )

        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(bottleneck_channels),
                act,
                nn.Conv1d(
                    bottleneck_channels,
                    branch_channels,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(branch_channels),
                act,
            )
            self.branches.append(branch)

        self.use_residual = use_residual
        if use_residual:
            if in_channels != out_channels:
                self.shortcut = nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, bias=False
                )
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
        in_channels: int = 2,
        base_filters: int = 64,
        depth: int = 6,
        kernel_sizes: List[int] | None = None,
        bottleneck_channels: int = 32,
        use_residual: bool = True,
        dropout: float = 0.3,
        fc_dim: int = 256,
        activation: str = "relu",
    ) -> None:
        del signal_length
        super().__init__()
        kernel_sizes = kernel_sizes or [9, 19, 39]
        if bottleneck_channels < 1:
            raise ValueError("bottleneck_channels must be >= 1")
        if depth < 3:
            raise ValueError("depth must be >= 3")

        self.kernel_sizes = [1] + list(kernel_sizes)
        self.use_residual = use_residual
        self.bottleneck_channels = bottleneck_channels
        self.in_channels = in_channels

        act = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            act,
            nn.MaxPool1d(2),
        )

        channels = [base_filters, base_filters * 2, base_filters * 3, base_filters * 4]
        blocks = []
        in_ch = base_filters
        for i in range(depth):
            out_ch = channels[min(i // 2, len(channels) - 1)]
            blocks.append(
                InceptionBlock(
                    in_ch,
                    out_ch,
                    kernel_sizes=kernel_sizes,
                    bottleneck_channels=bottleneck_channels,
                    use_residual=use_residual,
                    activation=activation,
                )
            )
            in_ch = out_ch
            if i % 2 == 1:
                blocks.append(nn.MaxPool1d(2))

        self.blocks = nn.Sequential(*blocks)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embedding_dim = in_ch

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, fc_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(fc_dim, n_classes),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(in_ch, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

        self._init_weights()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
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
        x = self.blocks(x)
        return x

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
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
