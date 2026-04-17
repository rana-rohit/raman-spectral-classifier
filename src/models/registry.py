"""
src/models/registry.py

Model factory for all supported spectral architectures.
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from src.models.cnn import CNN1D
from src.models.hybrid import HybridCNNTransformer
from src.models.multitask import MultiHeadSpectralModel
from src.models.resnet1d import ResNet1D
from src.models.transformer import SpectralTransformer


MODEL_REGISTRY = {
    "cnn": CNN1D,
    "resnet1d": ResNet1D,
    "transformer": SpectralTransformer,
    "hybrid": HybridCNNTransformer,
}


def get_model(name: str, cfg: Dict[str, Any]) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[name]
    model_cfg = cfg.get("model", {})

    common = {
        "signal_length": model_cfg.get("signal_length", 1000),
        "n_classes": model_cfg.get("n_classes", 30),
    }
    specific = {
        k: v
        for k, v in model_cfg.items()
        if k not in ("name", "signal_length", "n_classes")
    }

    model = model_cls(**common, **specific)

    aux_cfg = cfg.get("multitask", {}).get("auxiliary_shared_head", {})
    if aux_cfg.get("enabled", False):
        shared_class_ids = aux_cfg.get("classes", cfg.get("dataset", {}).get("shared_classes", []))
        model = MultiHeadSpectralModel(
            backbone=model,
            shared_class_ids=shared_class_ids,
            aux_dropout=aux_cfg.get("dropout", 0.0),
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {name} | Parameters: {n_params:,}")
    return model


def model_summary(model: nn.Module, input_shape=(1, 1, 1000)) -> None:
    del input_shape
    total = 0
    print(f"\n{'Layer':<45} {'Params':>10}")
    print("-" * 57)
    for name, module in model.named_modules():
        if not list(module.children()):
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name:<43} {params:>10,}")
                total += params
    print("-" * 57)
    print(f"  {'Total trainable parameters':<43} {total:>10,}\n")
