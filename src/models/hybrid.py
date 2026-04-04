"""
src/models/hybrid.py

Hybrid CNN-Transformer for spectral classification.

Architecture:
  CNN stem  →  linear projection  →  Transformer encoder  →  classifier

The CNN stem acts as a learned tokeniser:
  - Extracts local spectral features (peaks, shoulders, doublets)
  - Reduces sequence length (making attention tractable)
  - Learns spatially-aware representations before global attention

The Transformer back-end then models long-range dependencies between
the CNN-extracted local features — e.g. the ratio between a peak at
position 100 and one at position 800.

Research contribution:
  The handoff_blocks parameter controls how many CNN blocks run before
  the Transformer takes over. Ablating this (1, 2, 3 blocks) is a
  direct experimental contribution that quantifies the optimal
  local-to-global feature handoff point for spectral data.

Key design differences from plain Transformer:
  - No patch splitting — CNN produces the tokens naturally
  - Token sequence length depends on CNN pooling, not a fixed patch size
  - CNN features are already spatially structured, so positional encoding
    is less critical (but still applied)
"""

import torch
import torch.nn as nn
from typing import List, Optional

from src.models.cnn import ConvBlock
from src.models.transformer import TransformerEncoderLayer, PositionalEncoding


class CNNStem(nn.Module):
    """
    Configurable CNN front-end. Produces a sequence of local feature vectors
    that the Transformer treats as tokens.

    handoff_blocks controls depth:
      1 block: (B, 64,  500)  — 500 tokens of dim 64  → project to d_model
      2 blocks: (B, 128, 250)  — 250 tokens
      3 blocks: (B, 256, 125)  — 125 tokens  ← sweet spot for attention efficiency
    """

    BLOCK_CONFIGS = [
        # (in_ch, out_ch, kernel, pool)
        (1,   64,  7,  True),
        (64,  128, 15, True),
        (128, 256, 15, True),
    ]

    def __init__(self, n_blocks: int = 2) -> None:
        super().__init__()
        assert 1 <= n_blocks <= 3, "n_blocks must be 1, 2, or 3"
        self.n_blocks = n_blocks

        layers = []
        for i in range(n_blocks):
            in_ch, out_ch, k, pool = self.BLOCK_CONFIGS[i]
            layers.append(ConvBlock(in_ch, out_ch, k))
            if pool:
                layers.append(nn.MaxPool1d(2))

        self.stem = nn.Sequential(*layers)
        self.out_channels = self.BLOCK_CONFIGS[n_blocks - 1][1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)   # (B, out_channels, T)


class HybridCNNTransformer(nn.Module):
    """
    CNN-Transformer hybrid model.

    Full architecture (with n_blocks=2, d_model=256, n_layers=4):
        Input           (B, 1, 1000)
        CNN stem        (B, 128, 250)     — 2 conv blocks + pooling
        Projection      (B, 250, 256)     — Linear(128, 256), transposed
        + CLS token     (B, 251, 256)
        + Pos encoding  (B, 251, 256)
        Transformer ×4  (B, 251, 256)
        CLS → head      (B, n_classes)
    """

    def __init__(
        self,
        signal_length: int = 1000,
        n_classes: int = 30,
        handoff_blocks: int = 2,   # CNN blocks before Transformer (1, 2, or 3)
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.handoff_blocks = handoff_blocks
        self.d_model = d_model

        # CNN stem
        self.cnn_stem = CNNStem(n_blocks=handoff_blocks)
        cnn_out_ch = self.cnn_stem.out_channels

        # Project CNN channels to Transformer d_model
        self.proj = nn.Sequential(
            nn.Linear(cnn_out_ch, d_model),
            nn.LayerNorm(d_model),
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional encoding (seq_len determined at runtime)
        self.pos_enc = PositionalEncoding(d_model, max_len=1024, dropout=dropout)

        # Transformer encoder
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, attn_dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        self._init_weights()

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor:
        B = x.size(0)

        # CNN stem: (B, 1, L) → (B, C, T)
        cnn_out = self.cnn_stem(x)           # (B, C, T)

        # Reformat for Transformer: (B, C, T) → (B, T, d_model)
        tokens = cnn_out.transpose(1, 2)     # (B, T, C)
        tokens = self.proj(tokens)           # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)   # (B, T+1, d_model)

        # Positional encoding
        tokens = self.pos_enc(tokens)

        # Transformer encoder
        attn_maps = []
        for layer in self.transformer_layers:
            tokens, attn = layer(tokens, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn)

        # CLS output → classifier
        cls_out = self.norm(tokens[:, 0, :])
        logits  = self.classifier(cls_out)

        if return_attn:
            return logits, attn_maps
        return logits

    def get_attention_maps(self, x: torch.Tensor) -> list:
        _, attn_maps = self.forward(x, return_attn=True)
        return attn_maps

    def get_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns CNN stem output for Grad-CAM on the local features."""
        return self.cnn_stem(x)

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)