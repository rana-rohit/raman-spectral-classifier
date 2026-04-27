"""
src/models/hybrid.py

Hybrid CNN-Transformer for spectral classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.cnn import ConvBlock
from src.models.transformer import PositionalEncoding, TransformerEncoderLayer


class CNNStem(nn.Module):
    BLOCK_CONFIGS = [
        (1, 64, 7, True),
        (64, 128, 15, True),
        (128, 256, 15, True),
    ]

    def __init__(self, n_blocks: int = 2, in_channels: int = 1) -> None:
        super().__init__()
        assert 1 <= n_blocks <= 3, "n_blocks must be 1, 2, or 3"
        self.n_blocks = n_blocks

        layers = []
        for i in range(n_blocks):
            in_ch, out_ch, kernel, pool = self.BLOCK_CONFIGS[i]
            if i == 0:
                in_ch = in_channels  # Override first block input channels
            layers.append(ConvBlock(in_ch, out_ch, kernel))
            if pool:
                layers.append(nn.MaxPool1d(2))

        self.stem = nn.Sequential(*layers)
        self.out_channels = self.BLOCK_CONFIGS[n_blocks - 1][1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class HybridCNNTransformer(nn.Module):
    def __init__(
        self,
        signal_length: int = 1000,
        n_classes: int = 30,
        handoff_blocks: int = 2,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        in_channels: int = 1,
    ) -> None:
        del signal_length
        super().__init__()
        self.handoff_blocks = handoff_blocks
        self.d_model = d_model
        self.embedding_dim = d_model

        self.cnn_stem = CNNStem(n_blocks=handoff_blocks, in_channels=in_channels)
        cnn_out_ch = self.cnn_stem.out_channels

        self.proj = nn.Sequential(
            nn.Linear(cnn_out_ch, d_model),
            nn.LayerNorm(d_model),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, max_len=1024, dropout=dropout)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, attn_dropout)
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        self._init_weights()

    def _encode(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size = x.size(0)
        cnn_out = self.cnn_stem(x)
        tokens = cnn_out.transpose(1, 2)
        tokens = self.proj(tokens)

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.pos_enc(tokens)

        attn_maps = []
        for layer in self.transformer_layers:
            tokens, attn = layer(tokens, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn)
        return tokens, attn_maps

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens, _ = self._encode(x, return_attn=False)
        return self.norm(tokens[:, 0, :])

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        tokens, attn_maps = self._encode(x, return_attn=return_attn)
        cls_out = self.norm(tokens[:, 0, :])
        logits = self.classifier(cls_out)
        if return_attn:
            return logits, attn_maps
        return {
            "main_logits": logits,
            "features": cls_out,
        }

    def get_attention_maps(self, x: torch.Tensor) -> list:
        _, attn_maps = self.forward(x, return_attn=True)
        return attn_maps

    def get_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn_stem(x)

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
