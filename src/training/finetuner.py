"""
src/training/finetuner.py

Fine-tuning protocol for domain adaptation (Experiment 3).

Starting from a pretrained checkpoint, adapts the model to a new domain
using the finetune split (100 samples/class, 30 classes).

Supports:
- Full fine-tuning (all layers updated)
- Frozen stem (only classifier head updated initially)
- Few-shot subsampling for learning curve experiments (Experiment 4)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.training.trainer import Trainer, build_trainer
from src.utils.checkpoint import load_best_model
from src.utils.logging import ExperimentLogger


def finetune(
    model: nn.Module,
    pretrained_exp_dir: str,
    loaders: Dict,
    cfg: dict,
    exp_dir: str,
    n_shots_per_class: Optional[int] = None,
    freeze_epochs: int = 0,
    n_classes: int = 30,
) -> Dict:
    """
    Fine-tune a pretrained model on the finetune split.

    Args:
        model:              Model architecture (weights will be overwritten)
        pretrained_exp_dir: Directory of the pretrained experiment (contains best.pt)
        loaders:            Standard loader dict from build_all_loaders()
        cfg:                Training config for fine-tuning
        exp_dir:            Output directory for this fine-tuning run
        n_shots_per_class:  If set, subsample finetune split to this many per class
        freeze_epochs:      Number of epochs to freeze all layers except head
        n_classes:          Number of output classes
    """
    # Load pretrained weights
    print(f"\n  Loading pretrained weights from {pretrained_exp_dir}")
    checkpoint = load_best_model(pretrained_exp_dir, model)
    print(f"  Pretrained val_acc: {checkpoint['metrics'].get('val_accuracy', '?'):.4f}")

    # Optionally subsample the finetune loader
    ft_loader = loaders["finetune"]
    if n_shots_per_class is not None:
        ft_loader = _subsample_loader(ft_loader, n_shots_per_class, n_classes)
        n_samples = n_shots_per_class * n_classes
        print(f"  Few-shot: using {n_shots_per_class} samples/class ({n_samples} total)")

    # Replace finetune loader in dict
    local_loaders = {**loaders, "train": ft_loader}

    # Optionally freeze stem for initial epochs
    if freeze_epochs > 0:
        _freeze_stem(model)
        print(f"  Stem frozen for first {freeze_epochs} epochs")

    # Override config for fine-tuning
    ft_cfg = _make_finetune_cfg(cfg, freeze_epochs)
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    trainer = build_trainer(model, local_loaders, ft_cfg, exp_dir, n_classes)

    # Phase 1: frozen stem (if requested)
    if freeze_epochs > 0:
        orig_max = ft_cfg["training"]["max_epochs"]
        ft_cfg["training"]["max_epochs"] = freeze_epochs
        trainer.cfg["max_epochs"] = freeze_epochs
        _unfreeze_all(model)
        print("  Stem unfrozen — continuing full fine-tuning")

        ft_cfg["training"]["lr"] *= 0.1

        trainer = build_trainer(model, local_loaders, ft_cfg, exp_dir, n_classes)

        trainer.fit()
        ft_cfg["training"]["max_epochs"] = orig_max - freeze_epochs
        trainer.cfg["max_epochs"] = orig_max - freeze_epochs

    # Phase 2: full fine-tuning
    best_metrics = trainer.fit()

    # Evaluate on all relevant splits
    results = {
        "val":          trainer.evaluate(loaders["val"],  "val"),
        "test":         trainer.evaluate(loaders["test"], "test"),
        "ood":          trainer.evaluate_ood(),
        "best_metrics": best_metrics,
        "n_shots":      n_shots_per_class,
    }
    return results


def _subsample_loader(
    loader: DataLoader,
    n_shots: int,
    n_classes: int,
) -> DataLoader:
    """Return a DataLoader with exactly n_shots samples per class."""
    dataset = loader.dataset
    # Get all indices per class
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    selected = []
    rng = np.random.default_rng(42)
    for cls in range(n_classes):
        idx = np.where(targets == cls)[0]
        if len(idx) >= n_shots:
            chosen = rng.choice(idx, n_shots, replace=False)
        else:
            chosen = idx
        selected.extend(chosen.tolist())

    subset = Subset(dataset, selected)
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
    )


def _freeze_stem(model: nn.Module) -> None:
    """Freeze all parameters except the final classification head."""
    for name, param in model.named_parameters():
        if "head" not in name and "classifier" not in name and "fc" not in name:
            param.requires_grad = False


def _unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def _make_finetune_cfg(base_cfg: dict, freeze_epochs: int) -> dict:
    """Override training config for fine-tuning (lower LR, fewer epochs)."""
    import copy
    cfg = copy.deepcopy(base_cfg)
    ft = cfg.setdefault("training", {})
    ft["lr"]          = ft.get("finetune_lr", 1e-4)    # 10x lower than initial
    ft["max_epochs"]  = ft.get("finetune_epochs", 50)
    ft["early_stopping_patience"] = ft.get("finetune_patience", 8)
    ft["scheduler"]   = "warmup_cosine"
    ft["scheduler_cfg"] = {
        "warmup_epochs": 2,
        "total_epochs":  ft["max_epochs"],
    }
    return cfg