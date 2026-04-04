"""
src/models/registry.py

Model factory — instantiate any model by name from config.
The training script never imports model classes directly;
it calls get_model(name, cfg) and receives the right architecture.

Adding a new model: register it in MODEL_REGISTRY, done.
"""

import torch.nn as nn
from typing import Dict, Any

from src.models.cnn import CNN1D
from src.models.resnet1d import ResNet1D
from src.models.transformer import SpectralTransformer
from src.models.hybrid import HybridCNNTransformer


MODEL_REGISTRY = {
    "cnn":         CNN1D,
    "resnet1d":    ResNet1D,
    "transformer": SpectralTransformer,
    "hybrid":      HybridCNNTransformer,
}


def get_model(name: str, cfg: Dict[str, Any]) -> nn.Module:
    """
    Instantiate a model from config.

    Args:
        name: One of "cnn", "resnet1d", "transformer", "hybrid"
        cfg:  Full config dict — model-specific params read from cfg["model"]

    Returns: Initialised (untrained) model.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[name]
    model_cfg = cfg.get("model", {})

    # Common params present in all models
    common = {
        "signal_length": model_cfg.get("signal_length", 1000),
        "n_classes":     model_cfg.get("n_classes", 30),
    }

    # Model-specific params (pass only what the constructor accepts)
    specific = {k: v for k, v in model_cfg.items()
                if k not in ("name", "signal_length", "n_classes")}

    model = model_cls(**common, **specific)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {name} | Parameters: {n_params:,}")

    return model


def model_summary(model: nn.Module, input_shape=(1, 1, 1000)) -> None:
    """Print a concise layer-by-layer parameter summary."""
    import torch
    total = 0
    print(f"\n{'Layer':<45} {'Params':>10}")
    print("-" * 57)
    for name, module in model.named_modules():
        if not list(module.children()):   # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name:<43} {params:>10,}")
                total += params
    print("-" * 57)
    print(f"  {'Total trainable parameters':<43} {total:>10,}\n")