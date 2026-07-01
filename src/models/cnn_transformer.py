"""
src/models/cnn_transformer.py

CNN+Transformer Hybrid Architecture for 1D signal classification.
Combining local feature extraction via CNN with global context modeling via Transformer.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.models.transformer import PositionalEncoding, TransformerEncoderLayer


class ConvBlock(nn.Module):
    """
    Conv1d + BatchNorm1d + ReLU block for local feature extraction.
    No dropout is applied here to avoid interfering with local representation learning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNTransformer(nn.Module):
    """
    CNN-Transformer hybrid model.
    Extracts local features using a CNN backbone, projects them,
    and then models global dependencies via a Transformer Encoder.
    """

    def __init__(
        self,
        signal_length: int = 1000,
        n_classes: int = 30,
        in_channels: int = 1,
        channels: List[int] | None = None,
        cnn_kernel_size: int = 5,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128, 256]

        self.d_model = d_model
        self.embedding_dim = d_model
        self.in_channels = in_channels

        # 1. CNN Feature Extractor Stem
        modules = []
        curr_in = in_channels
        for c in channels:
            modules.append(ConvBlock(curr_in, c, cnn_kernel_size))
            modules.append(nn.MaxPool1d(2))
            curr_in = c
        self.cnn_stem = nn.Sequential(*modules)

        # 2. Feature Projection (downsampled output channels -> d_model)
        final_cnn_channels = channels[-1]
        if final_cnn_channels != d_model:
            self.proj = nn.Conv1d(final_cnn_channels, d_model, kernel_size=1)
        else:
            self.proj = nn.Identity()

        # 3. Transformer CLS token and Positional Encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Pre-allocate large enough max_len for positional encoding to handle various lengths safely
        self.pos_enc = PositionalEncoding(d_model, max_len=1024, dropout=dropout)

        # 4. Transformer Encoder Layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # 5. Normalization and Classifiers
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        # Optional Domain adaptation classifier for DANN integration
        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
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
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # 1. Local feature extraction via CNN stem
        feat = self.cnn_stem(x)  # (B, final_cnn_channels, L')

        # 2. Projection to d_model dimension
        feat = self.proj(feat)  # (B, d_model, L')
        feat = feat.transpose(1, 2)  # (B, L', d_model)

        # 3. Prepend CLS token and apply positional encoding
        batch_size = feat.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, feat], dim=1)  # (B, L'+1, d_model)
        tokens = self.pos_enc(tokens)

        # 4. Process global context via Transformer layers
        attn_maps = []
        for layer in self.layers:
            tokens, attn = layer(tokens, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn)
        return tokens, attn_maps

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the CLS token representation from the input spectrum.
        """
        tokens, _ = self._encode(x, return_attn=False)
        return self.norm(tokens[:, 0, :])

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply the classifier head to latent features.
        """
        return self.classifier(features)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass for training and evaluation.
        """
        tokens, attn_maps = self._encode(x, return_attn=return_attn)
        cls_out = self.norm(tokens[:, 0, :])
        logits = self.forward_logits(cls_out)

        if return_attn:
            return logits, attn_maps

        return {
            "main_logits": logits,
            "aux_logits": None,
            "features": cls_out,
        }

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns CNN feature maps for Grad-CAM visualization.
        """
        return self.cnn_stem(x)

    def get_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns CNN features. Alias for get_feature_maps for compatibility.
        """
        return self.cnn_stem(x)

    def get_attention_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Returns attention maps from each transformer layer.
        """
        _, attn_maps = self.forward(x, return_attn=True)
        return attn_maps

    def n_parameters(self) -> int:
        """
        Returns the number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
