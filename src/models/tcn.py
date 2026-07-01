"""
src/models/tcn.py

Temporal Convolutional Network (TCN) for 1D spectral classification.

Design rationale:
- Dilated causal=False convolutions expand the receptive field
  exponentially while preserving local spectral peak structure.
- Raman spectra are NOT autoregressive: peaks at any wavenumber
  can depend on peaks at any other wavenumber, so we use
  symmetric (non-causal) padding.
- Residual temporal blocks with pre-activation normalization
  stabilise gradient flow through deep dilation stacks.
- The architecture intentionally mirrors the ResNet1D API surface
  so that all downstream tools (embedding analysis, checkpoint
  loading, DANN hooks, evaluators, XAI) work unchanged.

Scientific hypothesis:
TCN may preserve local Raman peak inductive bias while improving
broader biochemical context modelling through dilated receptive
fields, potentially yielding stronger semantic transfer and
improved clinical robustness compared to strided ResNet.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# TEMPORAL BLOCK
class TemporalBlock(nn.Module):
    """
    Single residual temporal block with dilated convolutions.

    Structure:
        input
        → dilated Conv1D → BatchNorm → ReLU → Dropout
        → dilated Conv1D → BatchNorm
        → residual add
        → ReLU

    Padding is symmetric (causal=False) because Raman spectra
    are spatial signals, not autoregressive sequences.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Symmetric padding to preserve sequence length
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # 1x1 projection for channel mismatch
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        # Trim to input length if padding is asymmetric
        out = out[:, :, : x.size(2)]
        out = self.relu(self.bn1(out))
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, : x.size(2)]
        out = self.bn2(out)

        out = self.relu(out + residual)
        return out


# TEMPORAL STAGE
class TemporalStage(nn.Module):
    """
    A stage consisting of multiple TemporalBlocks at a fixed
    channel width but with increasing dilation rates.

    Each stage processes the signal at one resolution level.
    Dilation growth within a stage expands the receptive field
    without reducing spatial resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilations: List[int],
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers = []
        for i, d in enumerate(dilations):
            ch_in = in_channels if i == 0 else out_channels
            layers.append(
                TemporalBlock(
                    ch_in,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


# TCN BACKBONE
class TCN1D(nn.Module):
    """
    Temporal Convolutional Network for 1D spectral classification.

    Architecture:
        Stem (Conv1D) → Stage1 → Stage2 → Stage3 → Stage4 → GAP → Classifier

    Each stage applies a stack of dilated residual temporal blocks.
    Downsampling between stages is achieved via strided convolution
    to control sequence length and computational cost.

    The API matches ResNet1D exactly so that all downstream tools
    (embedding analysis, checkpoint loading, DANN, evaluators, XAI)
    work without modification.
    """

    def __init__(
        self,
        signal_length: int = 1000,
        n_classes: int = 30,
        channels: List[int] | None = None,
        kernel_size: int = 5,
        dilations: List[int] | None = None,
        dropout: float = 0.2,
        in_channels: int = 2,
    ) -> None:
        del signal_length  # Unused; kept for registry compatibility
        super().__init__()

        channels = channels or [32, 64, 128, 256]
        dilations = dilations or [1, 2, 4, 8]
        assert len(channels) == 4, "channels must have exactly 4 values"

        c1, c2, c3, c4 = channels

        # --------------------------------------------------------
        # Stem: initial projection to c1 channels
        # --------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                c1,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
        )

        # --------------------------------------------------------
        # Four temporal stages with inter-stage downsampling
        #
        # Downsampling via strided conv preserves the dilated
        # receptive field structure while reducing sequence length.
        # --------------------------------------------------------
        self.stage1 = TemporalStage(
            c1,
            c1,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )

        self.down1 = nn.Conv1d(c1, c2, kernel_size=1, stride=2, bias=False)

        self.stage2 = TemporalStage(
            c2,
            c2,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )

        self.down2 = nn.Conv1d(c2, c3, kernel_size=1, stride=2, bias=False)

        self.stage3 = TemporalStage(
            c3,
            c3,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )

        self.down3 = nn.Conv1d(c3, c4, kernel_size=1, stride=2, bias=False)

        self.stage4 = TemporalStage(
            c4,
            c4,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )

        # --------------------------------------------------------
        # Global average pooling and classifier
        # --------------------------------------------------------
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.embedding_dim = c4

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c4, n_classes),
        )

        # --------------------------------------------------------
        # Domain classifier for DANN / domain adaptation
        # Topology matches ResNet1D domain classifier exactly
        # --------------------------------------------------------
        self.domain_classifier = nn.Sequential(
            nn.Linear(c4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # 2 domains: reference vs clinical
        )

        self._init_weights()

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent embedding vector (B, embedding_dim)."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.gap(x)
        return x.squeeze(-1)

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        return {
            "main_logits": self.classifier(features),
            "aux_logits": None,
            "features": features,
        }

    # --------------------------------------------------------
    # Classifier head (for embedding_analysis / XAI)
    # --------------------------------------------------------

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply classifier head to latent embeddings.

        Args:
            features:
                Tensor of shape (B, embedding_dim)

        Returns:
            logits tensor of shape (B, n_classes)
        """
        return self.classifier(features)

    # --------------------------------------------------------
    # Feature maps (for Grad-CAM / XAI)
    # --------------------------------------------------------

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return pre-GAP feature maps for Grad-CAM visualisation."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        return self.stage4(x)

    # --------------------------------------------------------
    # Weight initialisation
    # --------------------------------------------------------

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
