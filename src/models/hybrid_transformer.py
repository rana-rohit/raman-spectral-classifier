"""
src/models/hybrid_transformer.py

Hybrid CNN-Transformer for Raman spectral classification.
Extracts local motifs via CNN, then models global relationships via Transformer.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class CNNFrontend(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_sizes: List[int],
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes)
        
        layers = []
        current_in = in_channels
        
        for out_ch, k in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(current_in, out_ch, kernel_size=k, stride=1, padding=k//2, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ])
            current_in = out_ch
            
        self.net = nn.Sequential(*layers)
        self.out_channels = current_in

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HybridCNNTransformer(nn.Module):
    def __init__(
        self,
        signal_length: int = 1000,
        n_classes: int = 5,
        in_channels: int = 2,
        cnn_channels: List[int] | None = None,
        cnn_kernel_sizes: List[int] | None = None,
        transformer_dim: int = 128,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_dropout: float = 0.1,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        del signal_length
        super().__init__()
        
        cnn_channels = cnn_channels or [32, 64, 128]
        cnn_kernel_sizes = cnn_kernel_sizes or [7, 5, 3]
        
        self.embedding_dim = embedding_dim
        
        # 1. CNN Frontend
        self.cnn = CNNFrontend(
            in_channels=in_channels,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
        )
        
        # Projection to transformer dimension if needed
        self.proj = nn.Identity()
        if self.cnn.out_channels != transformer_dim:
            self.proj = nn.Conv1d(self.cnn.out_channels, transformer_dim, kernel_size=1)
            
        # 2. Transformer Stage
        self.pos_encoder = PositionalEncoding(transformer_dim, transformer_dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # Optional projection from transformer dim to embedding dim
        self.embed_proj = nn.Identity()
        if transformer_dim != embedding_dim:
            self.embed_proj = nn.Linear(transformer_dim, embedding_dim)
            
        # 3. Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, n_classes)
        )
        
        # 4. Domain Classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )
        
        self._init_weights()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        features = self.cnn(x)
        features = self.proj(features)
        
        # Prepare for transformer: (B, C, L') -> (B, L', C)
        features = features.transpose(1, 2)
        
        # Transformer encoding
        features = self.pos_encoder(features)
        features = self.transformer(features)
        
        # Global mean pooling across sequence dimension
        features = features.mean(dim=1)
        
        # Map to final embedding dim
        features = self.embed_proj(features)
        
        return features

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        logits = self.classifier(features)
        
        return {
            "main_logits": logits,
            "aux_logits": None,
            "features": features,
        }
        
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
