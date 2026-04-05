"""
src/data/dataloader.py

Builds all DataLoaders from the DataRegistry in a single call.
Returns a typed dict so the trainer can access loaders by name.

Usage:
    loaders = build_all_loaders(registry, preprocessor, aug_pipeline, cfg)
    for batch_x, batch_y in loaders["train"]:
        ...
    for name, loader in loaders["ood"].items():
        evaluate(model, loader)
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from src.data.dataset import SpectralDataset, make_train_val_split
from src.data.registry import DataRegistry
from src.data.preprocessing import SpectralPreprocessor


def build_all_loaders(
    registry: DataRegistry,
    preprocessor: SpectralPreprocessor,
    augmentation,
    cfg: dict,
) -> Dict:
    """
    Constructs and returns all DataLoaders.

    Returns a dict with keys:
        "train"   -> DataLoader  (augmented, shuffled)
        "val"     -> DataLoader  (no augmentation)
        "test"    -> DataLoader  (no augmentation)
        "finetune"-> DataLoader  (augmented, shuffled)
        "ood"     -> Dict[str, DataLoader]  (one per OOD split, class-filtered)
    """
    batch_size   = cfg.get("batch_size", 256)
    num_workers  = cfg.get("num_workers", 4)
    val_fraction = cfg["validation"]["val_fraction"]
    val_seed     = cfg["validation"]["random_seed"]

    loaders = {}

    # ---- Reference split -> train + val ----
    X_ref, y_ref = registry.get_arrays("reference")
    X_ref = preprocessor.transform(X_ref)

    (X_tr, y_tr), (X_val, y_val) = make_train_val_split(
        X_ref, y_ref, val_fraction=val_fraction, random_seed=val_seed
    )

    loaders["train"] = _make_loader(
        X_tr, y_tr,
        augmentation=augmentation,
        training=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    loaders["val"] = _make_loader(
        X_val, y_val,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ---- Holdout test split ----
    X_test, y_test = registry.get_arrays("test")
    X_test = preprocessor.transform(X_test)
    loaders["test"] = _make_loader(
        X_test, y_test,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ---- Fine-tune split ----
    X_ft, y_ft = registry.get_arrays("finetune")
    X_ft = preprocessor.transform(X_ft)
    loaders["finetune"] = _make_loader(
        X_ft, y_ft,
        augmentation=augmentation,
        training=True,
        batch_size=min(batch_size, 64),   # Smaller batch for small dataset
        num_workers=num_workers,
        shuffle=True,
    )

    # ---- OOD / clinical splits (class-filtered) ----
    loaders["ood"] = {}
    for split_name in registry.ood_split_names():
        X_ood, y_ood = registry.get_arrays(split_name)
        X_ood = preprocessor.transform(X_ood)
        eval_classes = registry.get_eval_classes(split_name)
        loaders["ood"][split_name] = _make_loader(
            X_ood, y_ood,
            class_filter=eval_classes,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    return loaders


def _make_loader(
    X,
    y,
    augmentation=None,
    training: bool = False,
    class_filter=None,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = False,
) -> DataLoader:
    dataset = SpectralDataset(
        X, y,
        augmentation=augmentation,
        training=training,
        class_filter=class_filter,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,   # Drop incomplete last batch during training only
    )