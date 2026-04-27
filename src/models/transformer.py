"""
src/models/transformer.py

Spectral Transformer for 1D signal classification.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, d_model: int, in_channels: int = 1) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.transpose(1, 2)
        return self.norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self._attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed)
        self._attn_weights = attn_weights.detach()
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, attn_weights if return_attn else None


class SpectralTransformer(nn.Module):
    def __init__(
        self,
        signal_length: int = 1000,
        n_classes: int = 30,
        patch_size: int = 20,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        pos_encoding: str = "sinusoidal",
        in_channels: int = 1,
    ) -> None:
        super().__init__()

        assert signal_length % patch_size == 0, (
            f"signal_length ({signal_length}) must be divisible by patch_size ({patch_size})"
        )
        self.n_patches = signal_length // patch_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.embedding_dim = d_model
        self._use_learned_pos = pos_encoding != "sinusoidal"

        self.patch_embed = PatchEmbedding(patch_size, d_model, in_channels=in_channels)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        seq_len = self.n_patches + 1
        if self._use_learned_pos:
            self.pos_embedding = nn.Embedding(seq_len, d_model)
            self.pos_dropout = nn.Dropout(dropout)
        else:
            self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        self.layers = nn.ModuleList(
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

    def _apply_positional_encoding(self, tokens: torch.Tensor) -> torch.Tensor:
        if self._use_learned_pos:
            positions = torch.arange(tokens.size(1), device=tokens.device)
            pos = self.pos_embedding(positions).unsqueeze(0)
            return self.pos_dropout(tokens + pos)
        return self.pos_enc(tokens)

    def _encode(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size = x.size(0)
        tokens = self.patch_embed(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self._apply_positional_encoding(tokens)

        attn_maps = []
        for layer in self.layers:
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

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
