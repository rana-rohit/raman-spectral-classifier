"""
src/models/registry.py

Model factory for all supported spectral architectures.

Supports:
- isolate-space pretraining
- compact transfer-space finetuning
- ontology-aware multitask heads
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn
import torch

from src.models.cnn import CNN1D
from src.models.multitask import MultiHeadSpectralModel
from src.models.resnet1d import ResNet1D
from src.models.inception1d import Inception1D
from src.models.tcn import TCN1D
from src.models.transformer import SpectralTransformer


MODEL_REGISTRY = {
    "cnn": CNN1D,
    "resnet1d": ResNet1D,
    "seresnet1d": ResNet1D,
    "tcn": TCN1D,
    "transformer": SpectralTransformer,
    "inception1d": Inception1D,
}


def get_model(name: str, cfg: Dict[str, Any]) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[name]
    model_cfg = cfg.get("model", {})
    clinical_sparse_ids = (
        cfg.get("task", {}).get("clinical_sparse_global_ids", [])
    )
    stage = cfg.get("task", {}).get("stage", "transfer_5class")

    n_classes = int(
        model_cfg.get(
            "n_classes",
            len(clinical_sparse_ids) or 5,
        )
    )
    
    # --------------------------------------------------------
    # Transfer-learning semantic integrity
    #
    # transfer_5class operates in:
    # compact transfer-space
    #
    # Compact labels:
    # [0,1,2,3,4]
    #
    # derived from sparse clinical IDs:
    # [0,2,3,5,6]
    # --------------------------------------------------------
    if stage == "transfer_5class" and len(clinical_sparse_ids) > 0:
        assert n_classes == len(clinical_sparse_ids), (
            "transfer_5class requires "
            "n_classes == number of "
            "clinical sparse IDs, "
            f"got {n_classes} vs "
            f"{len(clinical_sparse_ids)}"
        )

    # Determine input channels (1 = raw only, 2 = raw + derivative)
    deriv_cfg = cfg.get("preprocessing", cfg.get("derivative", {}))
    if isinstance(deriv_cfg, dict) and "derivative" in deriv_cfg:
        deriv_cfg = deriv_cfg["derivative"]
    elif isinstance(deriv_cfg, dict) and "derivative" not in deriv_cfg:
        deriv_cfg = cfg.get("derivative", {})
    use_derivative = deriv_cfg.get("enabled", False) if isinstance(deriv_cfg, dict) else False
    default_in_channels = 2 if use_derivative else 1

    common = {
        "signal_length": model_cfg.get("signal_length", 1000),
        "n_classes": n_classes,
        "in_channels": model_cfg.get("in_channels", default_in_channels),
    }
    specific = {
        k: v
        for k, v in model_cfg.items()
        if k not in ("name", "signal_length", "n_classes", "in_channels")
    }

    model_kwargs = {
        **common,
        **specific,
    }

    non_constructor_keys = {
        "semantic_space",
        "contrastive",
        "projection_dim",
    }

    for key in non_constructor_keys:
        model_kwargs.pop(key, None)

    model = model_cls(**model_kwargs)
    model.semantic_space = model_cfg.get(
        "semantic_space",
        None,
    )
   
    # --------------------------------------------------------
    # Optional ontology-aware auxiliary head
    #
    # Uses sparse clinical ontology IDs to
    # supervise auxiliary transfer objectives.
    # --------------------------------------------------------
    aux_cfg = cfg.get("multitask", {}).get("auxiliary_clinical_head", {})
    if not aux_cfg or not aux_cfg.get("enabled", False):
        aux_cfg = cfg.get("multitask", {}).get("auxiliary_shared_head", {})
    if aux_cfg.get("enabled", False):
        clinical_sparse_ids_aux = aux_cfg.get(
            "classes",
            cfg.get("task", {}).get(
                "clinical_sparse_global_ids",
                []
            )
        )
        model = MultiHeadSpectralModel(
            backbone=model,
            shared_class_ids=clinical_sparse_ids_aux,
            aux_dropout=aux_cfg.get("dropout", 0.0),
        )

    # --------------------------------------------------------
    # Optional SupCon projection head.
    #
    # Canonical activation lives in training.supcon.enabled.
    # model.contrastive is accepted only as a legacy compatibility
    # alias for older experiment configs.
    # --------------------------------------------------------
    supcon_cfg = cfg.get("training", {}).get("supcon", {})
    legacy_contrastive = bool(model_cfg.get("contrastive", False))
    supcon_enabled = bool(supcon_cfg.get("enabled", False) or legacy_contrastive)
    if legacy_contrastive and not supcon_cfg.get("enabled", False):
        print(
            "  [Config] Legacy model.contrastive=True detected; "
            "using canonical training.supcon.enabled behavior."
        )
    if supcon_enabled:
        projection_dim = int(
            supcon_cfg.get(
                "projection_dim",
                model_cfg.get("projection_dim", 128),
            )
        )
        model.projection_head = nn.Sequential(
            nn.Linear(model.embedding_dim, model.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model.embedding_dim, projection_dim),
        )
        model.contrastive = True
        
        orig_forward = model.forward
        def contrastive_forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            out = orig_forward(x)
            if getattr(self, "bypass_projection", False):
                return out
            features = out["features"]
            proj = self.projection_head(features)
            out["projection_features"] = nn.functional.normalize(proj, p=2, dim=-1)
            return out
            
        import types
        model.forward = types.MethodType(contrastive_forward, model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {name} | Parameters: {n_params:,}")
    if name == "inception1d":
        kernel_sizes = getattr(model, "kernel_sizes", [])
        use_residual = getattr(model, "use_residual", False)
        bottleneck_channels = getattr(model, "bottleneck_channels", 0)
        input_channels = getattr(model, "in_channels", None)
        print("\n============================================================")
        print("MODEL SUMMARY")
        print("=============")
        print("Model: Inception1D")
        print(f"Kernel Sizes: {kernel_sizes}")
        print(f"Residual Connections: {'Enabled' if use_residual else 'Disabled'}")
        print(f"Bottleneck Channels: {bottleneck_channels}")
        if input_channels is not None:
            print(f"Input Channels: {input_channels}")
        print(f"Total Parameters: {n_params:,}")
        print("============================================================\n")
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
