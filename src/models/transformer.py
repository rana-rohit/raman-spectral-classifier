"""
src/models/transformer.py

Spectral Transformer for 1D signal classification.

Adapts the Vision Transformer (ViT) paradigm to 1D spectral signals:
  - Signal is divided into non-overlapping patches (like image patches in ViT)
  - Each patch is linearly projected to d_model dimensions
  - A learnable CLS token is prepended — its output is the classification vector
  - Sinusoidal or learned positional encoding preserves spectral ordering
  - Standard Transformer encoder layers (Multi-Head Attention + FFN + LayerNorm)

Design choices specific to spectral data:
  - Patch size 20 → 50 tokens from a 1000-point signal (manageable for attention)
  - Pre-norm architecture (LayerNorm before attention) — more stable than post-norm
  - Attention dropout in addition to FFN dropout
  - Label smoothing is applied at the loss level, not here

Attention maps are extractable for interpretability (see interpretability/attention_viz.py).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PatchEmbedding(nn.Module):
    """
    Splits a 1D signal into non-overlapping patches and projects each
    patch to d_model dimensions via a learned linear projection.

    Input:  (B, 1, L)
    Output: (B, n_patches, d_model)

    With L=1000 and patch_size=20: n_patches = 50
    """

    def __init__(self, patch_size: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        # Conv1d with stride=patch_size acts as a learned patch projection
        self.proj = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L)
        x = self.proj(x)        # (B, d_model, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, d_model)
        return self.norm(x)

    @property
    def n_patches(self) -> int:
        return -1  # Computed at runtime


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).
    Preserves the spectral ordering of patches without learned parameters.
    Learned encoding (nn.Embedding) is an alternative — config-controlled.
    """

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
        # Register as buffer (saved in state_dict, not a parameter)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Pre-norm Transformer encoder layer:
        x → LN → MHA → + x → LN → FFN → + x

    Pre-norm (vs post-norm) is more stable for deep models and
    doesn't require warmup to the same degree.
    """

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

        self._attn_weights: Optional[torch.Tensor] = None  # For interpretability

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, normed, normed)
        self._attn_weights = attn_weights.detach()
        x = x + attn_out

        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))

        return x, attn_weights if return_attn else None


class SpectralTransformer(nn.Module):
    """
    Transformer encoder for spectral signal classification.

    Architecture:
        Input   (B, 1, 1000)
        Patch embed                  → (B, 50, d_model)    [patch_size=20]
        Prepend CLS token            → (B, 51, d_model)
        + Positional encoding
        N × TransformerEncoderLayer
        CLS token output             → (B, d_model)
        LayerNorm + Classifier       → (B, n_classes)
    """

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
        pos_encoding: str = "sinusoidal",  # "sinusoidal" or "learned"
    ) -> None:
        super().__init__()

        assert signal_length % patch_size == 0, (
            f"signal_length ({signal_length}) must be divisible by patch_size ({patch_size})"
        )
        self.n_patches = signal_length // patch_size
        self.d_model   = d_model
        self.n_layers  = n_layers

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, d_model)

        # CLS token — learnable classification vector
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Positional encoding
        seq_len = self.n_patches + 1  # +1 for CLS
        if pos_encoding == "sinusoidal":
            self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)
        else:
            self.pos_enc = nn.Embedding(seq_len, d_model)
            self._use_learned_pos = True

        # Transformer encoder layers
        self.layers = nn.ModuleList([
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

        # Patch embedding: (B, 1, L) → (B, n_patches, d_model)
        tokens = self.patch_embed(x)

        # Prepend CLS token: (B, n_patches+1, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        # Positional encoding
        tokens = self.pos_enc(tokens)

        # Transformer encoder
        attn_maps = []
        for layer in self.layers:
            tokens, attn = layer(tokens, return_attn=return_attn)
            if return_attn:
                attn_maps.append(attn)

        # CLS token → classification
        cls_out = self.norm(tokens[:, 0, :])   # (B, d_model)
        logits  = self.classifier(cls_out)

        if return_attn:
            return logits, attn_maps
        return logits

    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Extract attention weights from all layers.
        Returns: list of (B, n_heads, seq_len, seq_len) tensors.
        Used by attention_viz.py for interpretability.
        """
        _, attn_maps = self.forward(x, return_attn=True)
        return attn_maps

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)