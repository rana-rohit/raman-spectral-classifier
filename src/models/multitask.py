from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MultiHeadSpectralModel(nn.Module):
    """
    Wrap a backbone with an auxiliary shared-class classifier.

    Optional auxiliary head over the same feature representation. The primary
    production recipe already uses the shared 5-class classifier.
    """

    def __init__(
        self,
        backbone: nn.Module,
        shared_class_ids: Sequence[int],
        aux_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not hasattr(backbone, "forward_features"):
            raise AttributeError("Backbone must implement forward_features().")
        if not hasattr(backbone, "embedding_dim"):
            raise AttributeError("Backbone must expose embedding_dim.")
        if not hasattr(backbone, "classifier"):
            raise AttributeError("Backbone must expose classifier.")

        self.backbone = backbone
        self.embedding_dim = int(backbone.embedding_dim)
        self.shared_class_ids = [int(cls) for cls in shared_class_ids]
        self.shared_classifier = nn.Sequential(
            nn.Dropout(aux_dropout),
            nn.Linear(self.embedding_dim, len(self.shared_class_ids)),
        )
        if hasattr(backbone, "domain_classifier"):
            self.domain_classifier = backbone.domain_classifier

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        main_logits = self.backbone.classifier(features)
        aux_logits = self.shared_classifier(features)
        return {
            "main_logits": main_logits,
            "aux_logits": aux_logits,
            "features": features,
        }

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.backbone, "get_feature_maps"):
            raise AttributeError("Backbone does not implement get_feature_maps().")
        return self.backbone.get_feature_maps(x)

    def get_attention_maps(self, x: torch.Tensor) -> list:
        if not hasattr(self.backbone, "get_attention_maps"):
            raise AttributeError("Backbone does not implement get_attention_maps().")
        return self.backbone.get_attention_maps(x)

    def get_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.backbone, "get_cnn_features"):
            raise AttributeError("Backbone does not implement get_cnn_features().")
        return self.backbone.get_cnn_features(x)
