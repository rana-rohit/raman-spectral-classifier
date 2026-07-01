"""
src/training/finetuner.py

Fine-tuning protocol for clinical domain adaptation.

Supports:
- compact transfer-space finetuning
- clinical OOD adaptation
- ontology-aware transfer evaluation

Clinical sparse labels:
[0,2,3,5,6]

are remapped into:
compact transfer-space labels:
[0,1,2,3,4]
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.evaluation.evaluator import ModelEvaluator
from src.training.trainer import build_trainer
from src.utils.checkpoint import load_best_model


def finetune(
    model: nn.Module,
    pretrained_exp_dir: str,
    loaders: Dict,
    cfg: dict,
    exp_dir: str,
    n_shots_per_class: Optional[int] = None,
    freeze_epochs: int = 0,
    n_classes: int = None,
) -> Dict:
    stage = cfg.get(
        "task",
        {},
    ).get(
        "stage",
        None,
    )

    assert stage == "transfer_5class", (
        "finetuner.py currently supports only " "transfer_5class stage"
    )

    # --------------------------------------------------------
    # Stage 1: Load isolate-space pretrained backbone
    # Pretraining learned: 30-class spectral representations.
    # Fine-tuning adapts these features into:
    # compact transfer-space.
    # --------------------------------------------------------
    checkpoint = load_best_model(pretrained_exp_dir, model)
    monitor_metric = checkpoint["metrics"].get("monitor_metric", "metric")
    monitor_value = checkpoint["metrics"].get("monitor_value", float("nan"))

    from src.utils.logging import print_checkpoint_info

    print_checkpoint_info(
        pretrained_exp_dir,
        loaded=True,
        details={
            "monitor_metric": monitor_metric,
            "monitor_value": f"{monitor_value:.4f}",
        },
    )

    reference_state = {
        name: param.detach().cpu().clone() for name, param in model.named_parameters()
    }

    from src.utils.logging import print_split_provenance

    print_split_provenance(loaders, cfg, context="finetuning")

    # Only reset the classifier head if the number of classes has changed;
    # preserves the pretrained class-feature mapping when possible.
    if n_classes is None:
        n_classes = cfg.get(
            "model",
            {},
        ).get(
            "n_classes",
            5,
        )
        print(f"  * Detected {n_classes} clinical classes")
        if n_classes != 5:
            raise AssertionError(
                "Clinical transfer finetuning must "
                "operate in compact transfer-space "
                f"with exactly 5 classes, got {n_classes}"
            )

    if not hasattr(model, "classifier"):
        raise AttributeError(f"{type(model).__name__} missing classifier")
    if not isinstance(model.classifier, nn.Sequential):
        raise TypeError("Expected classifier to be nn.Sequential")
    in_features = model.classifier[-1].in_features
    current_out = model.classifier[-1].out_features
    # --------------------------------------------------------
    # Transfer-space classifier adaptation
    # Reset classifier ONLY if output semantic
    # space changes.
    # Preserves pretrained decision geometry
    # whenever dimensions already match.
    # --------------------------------------------------------
    if n_classes != current_out:
        model.classifier[-1] = nn.Linear(in_features, n_classes)
        print(f"  * Classifier head reset to {n_classes} classes (clinical)")
    else:
        print(f"  * Keeping pretrained classifier head ({n_classes} classes)")

    # --------------------------------------------------------
    # Fine-tuning validation protocol
    # Validation switches from:
    # reference-domain
    # to:
    # clinical-domain validation
    # while retaining reference-domain
    # test evaluation separately.
    # --------------------------------------------------------
    local_loaders = {
        **loaders,
        "val": loaders["clinical_val"],  # only validation changes
    }
    if n_shots_per_class is not None:
        local_loaders["clinical_train"] = _subsample_loader(
            loaders["clinical_train"],
            n_shots=n_shots_per_class,
            n_classes=n_classes,
        )
        print(f"  * Few-shot clinical adaptation: {n_shots_per_class} shots/class")

    ft_cfg = _make_finetune_cfg(cfg, freeze_epochs)
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    phase_summaries = []
    total_epochs = ft_cfg["training"]["max_epochs"]

    if freeze_epochs > 0:
        print(f"  * Stem frozen for first {freeze_epochs} epochs")
        _freeze_stem(model)
        frozen_cfg = _phase_cfg(ft_cfg, max_epochs=freeze_epochs)
        frozen_dir = os.path.join(exp_dir, "phase1_frozen")
        frozen_trainer = build_trainer(
            model,
            local_loaders,
            frozen_cfg,
            frozen_dir,
            n_classes,
            reference_state=reference_state,
        )
        phase_summaries.append(
            {
                "phase": "phase1_frozen",
                "best": frozen_trainer.fit(),
            }
        )
        load_best_model(frozen_dir, model)

    remaining_epochs = max(total_epochs - freeze_epochs, 0)
    best_metrics = {}
    if remaining_epochs > 0:
        _unfreeze_all(model)
        print("  * Stem unfrozen - continuing full fine-tuning")
        full_lr = ft_cfg["training"]["lr"] * (0.1 if freeze_epochs > 0 else 1.0)
        full_cfg = _phase_cfg(ft_cfg, max_epochs=remaining_epochs, lr=full_lr)
        full_dir = os.path.join(exp_dir, "phase2_full")
        full_trainer = build_trainer(
            model,
            local_loaders,
            full_cfg,
            full_dir,
            n_classes,
            reference_state=reference_state,
        )
        best_metrics = full_trainer.fit()
        phase_summaries.append(
            {
                "phase": "phase2_full",
                "best": best_metrics,
            }
        )
        load_best_model(full_dir, model)

    evaluator = ModelEvaluator(
        model=model,
        model_name=cfg.get("model", {}).get("name", "model"),
        n_classes=n_classes,
        device=_infer_model_device(model),
        cfg=cfg,
    )
    val_metrics = evaluator.evaluate_split(loaders["clinical_val"], "clinical_val")
    test_metrics = evaluator.evaluate_split(loaders["test"], "test")
    ood_results = {}

    # --------------------------------------------------------
    # Clinical OOD evaluation
    #
    # Measures domain generalization across:
    # hospitals
    # collection years
    # patient populations
    # --------------------------------------------------------
    for ood_name, ood_loader in loaders.get("ood", {}).items():
        ood_results[ood_name] = evaluator.evaluate_split(ood_loader, ood_name)

    evaluator.results["summary"] = evaluator._build_summary("test")
    evaluator.save(os.path.join(exp_dir, "results.json"))

    results = {
        "val": val_metrics,
        "test": test_metrics,
        "ood": ood_results,
        "best_metrics": best_metrics,
        "n_shots": n_shots_per_class,
        "phases": phase_summaries,
        "semantic_space": "compact_transfer_space",
    }
    with open(os.path.join(exp_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# --------------------------------------------------------
# Few-shot transfer simulation
#
# Randomly subsamples compact transfer-space
# examples per class.
#
# Used for:
# - low-data adaptation studies
# - robustness experiments
# - transfer efficiency analysis
# --------------------------------------------------------
def _subsample_loader(
    loader: DataLoader,
    n_shots: int,
    n_classes: int,
) -> DataLoader:
    dataset = loader.dataset
    if hasattr(dataset, "y"):
        targets = np.asarray(dataset.y)
    else:
        targets = np.array(
            [
                dataset[i]["y"] if isinstance(dataset[i], dict) else dataset[i][1]
                for i in range(len(dataset))
            ]
        )

    selected = []
    rng = np.random.default_rng(42)
    for cls in range(n_classes):
        idx = np.where(targets == cls)[0]
        if len(idx) == 0:

            raise ValueError(f"No samples found for " f"compact transfer label {cls}")
        chosen = rng.choice(idx, n_shots, replace=(len(idx) < n_shots))
        selected.extend(chosen.tolist())

    subset = Subset(dataset, selected)
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        worker_init_fn=loader.worker_init_fn,
        generator=loader.generator,
    )


def _freeze_stem(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if (
            "classifier" not in name
            and "domain_classifier" not in name
            and "fc" not in name
        ):
            param.requires_grad = False


def _unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


# --------------------------------------------------------
# Domain adaptation finetuning policy
#
# Preserves explicitly configured optional features.
# DANN/CORAL are never activated implicitly during finetuning.
# --------------------------------------------------------
def _make_finetune_cfg(base_cfg: dict, freeze_epochs: int) -> dict:
    del freeze_epochs
    cfg = copy.deepcopy(base_cfg)
    ft = cfg.setdefault("training", {})

    # Preserve explicit domain-adaptation settings. Finetuning no longer
    # enables DANN or CORAL implicitly.
    ft["dann"] = {
        "enabled": ft.get("dann", {}).get("enabled", False),
        "weight": ft.get("dann", {}).get("weight", 0.0),
    }
    ft["coral"] = {
        "enabled": ft.get("coral", {}).get("enabled", False),
        "weight": ft.get("coral", {}).get("weight", 0.0),
    }
    ft["use_dann"] = bool(ft.get("use_dann", False) or ft["dann"]["enabled"])
    ft["use_coral"] = bool(ft.get("use_coral", False) or ft["coral"]["enabled"])
    ft["freeze_bn"] = ft.get("freeze_bn", False)

    ft["lr"] = ft.get("finetune_lr", 1e-4)
    ft["max_epochs"] = ft.get("finetune_epochs", 50)
    ft["early_stopping_patience"] = ft.get("finetune_patience", 8)
    ft["scheduler"] = "warmup_cosine"
    ft["scheduler_cfg"] = {
        "warmup_epochs": 2,
        "total_epochs": ft["max_epochs"],
        "eta_min": 1e-6,
    }
    return cfg


def _phase_cfg(base_cfg: dict, max_epochs: int, lr: Optional[float] = None) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["training"]["max_epochs"] = max_epochs
    if lr is not None:
        cfg["training"]["lr"] = lr
    cfg["training"]["scheduler_cfg"]["total_epochs"] = max_epochs
    cfg["training"]["scheduler_cfg"]["warmup_epochs"] = min(
        cfg["training"]["scheduler_cfg"].get("warmup_epochs", 2),
        max_epochs,
    )
    return cfg


def _infer_model_device(model: nn.Module) -> str:
    return str(next(model.parameters()).device)
