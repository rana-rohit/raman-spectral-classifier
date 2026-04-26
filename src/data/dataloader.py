"""
src/data/dataloader.py

Build all DataLoaders used by the pipeline.
Supports optional 2-channel input (raw + first derivative).
"""

from __future__ import annotations

import copy
import random
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.preprocessing import SpectralPreprocessor
from src.data.dataset import SpectralDataset, make_train_val_split
from src.data.registry import DataRegistry
from src.utils.class_subset import filter_and_remap_classes
from sklearn.model_selection import train_test_split


def build_all_loaders(
    registry: DataRegistry,
    preprocessor: SpectralPreprocessor,
    augmentation,
    cfg: dict,
    shared_classes
) -> Dict:
    batch_size = cfg.get("batch_size", 256)
    num_workers = cfg.get("num_workers", 4)
    seed = cfg.get("seed", 42)
    val_fraction = cfg["validation"]["val_fraction"]
    val_seed = cfg["validation"]["random_seed"]
    consistency_cfg = cfg.get("consistency") or {}
    
    train_views = 2 if consistency_cfg.get("enabled", False) else 1
    # If consistency training enabled, return multiple augmented views per sample
    if train_views not in (1, 2):
        raise ValueError("train_views must be 1 or 2")

    loaders = {}

    X_ref, y_ref = registry.get_arrays("reference")

    # THEN filter
    X_ref, y_ref = filter_and_remap_classes(X_ref, y_ref, shared_classes)

    (X_tr, y_tr), (X_val, y_val) = make_train_val_split(
        X_ref, y_ref,
        val_fraction=val_fraction,
        random_seed=val_seed,
    )

    loaders["train"] = _make_loader(
        X_tr, y_tr,
        augmentation=_clone_augmentation(augmentation),
        training=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        n_views=train_views,
        preprocessor=preprocessor, 
    )
    loaders["val"] = _make_loader(
        X_val, y_val,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed + 1,
        preprocessor=preprocessor, 
        shuffle=False,
    )

    X_test, y_test = registry.get_arrays("test", allow_holdout=True)

    X_test, y_test = filter_and_remap_classes(X_test, y_test, shared_classes)
    loaders["test"] = _make_loader(
        X_test, y_test,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed + 2,
        shuffle=False,
        preprocessor=preprocessor, 
    )
    
    # -----------------------------
    # CLINICAL LOADER (for domain adaptation / DANN)
    # -----------------------------
    try:
        X_clin1, y_clin1 = registry.get_arrays("2018clinical")
        X_clin2, y_clin2 = registry.get_arrays("2019clinical")

        X_clin = np.concatenate([X_clin1, X_clin2], axis=0)
        y_clin = np.concatenate([y_clin1, y_clin2], axis=0)
        X_clin, y_clin = filter_and_remap_classes(X_clin, y_clin, shared_classes)

        
        # Stratified split
        X_clin_tr, X_clin_val, y_clin_tr, y_clin_val = train_test_split(
            X_clin,
            y_clin,
            test_size=0.2,
            stratify=y_clin if len(np.unique(y_clin)) > 1 else None,
            random_state=seed,
        )

        # Train loader
        loaders["clinical_train"] = _make_loader(
            X_clin_tr,
            y_clin_tr,
            augmentation=_clone_augmentation(augmentation),
            training=True,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            seed=seed + 5,
            n_views=train_views,   
            preprocessor=preprocessor,  
        )

        # Validation loader
        loaders["clinical_val"] = _make_loader(
            X_clin_val,
            y_clin_val,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            seed=seed + 6,
            preprocessor=preprocessor, 
        )

    except Exception as e:
        print("WARNING: Clinical data not found:", e)

    X_ft, y_ft = registry.get_arrays("finetune")

    X_ft, y_ft = filter_and_remap_classes(X_ft, y_ft, shared_classes)
    loaders["finetune"] = _make_loader(
        X_ft, y_ft,
        augmentation=_clone_augmentation(augmentation),
        training=True,
        batch_size=min(batch_size, 64),
        num_workers=num_workers,
        shuffle=True,
        seed=seed + 3,
        n_views=train_views,
        preprocessor=preprocessor, 
    )

    loaders["ood"] = {}
    for idx, split_name in enumerate(registry.ood_split_names()):
        X_ood, y_ood = registry.get_arrays(split_name)

        X_ood, y_ood = filter_and_remap_classes(X_ood, y_ood, shared_classes)
        loaders["ood"][split_name] = _make_loader(
            X_ood, y_ood,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            seed=seed + 10 + idx,
            preprocessor=preprocessor, 
        )

    return loaders

def _make_loader(
    X,
    y,
    augmentation=None,
    training: bool = False,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = False,
    seed: int = 42,
    n_views: int = 1,
    preprocessor=None, 
) -> DataLoader:
    dataset = SpectralDataset(
        X,
        y,
        augmentation=augmentation,
        training=training,
        n_views=n_views,
        preprocessor=preprocessor,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        drop_last=training,
    )


def _clone_augmentation(augmentation):
    if augmentation is None:
        return None
    return copy.deepcopy(augmentation)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return

    dataset = worker_info.dataset
    if hasattr(dataset, "dataset"):
        dataset = dataset.dataset

    augmentation = getattr(dataset, "augmentation", None)
    if augmentation is not None:
        augmentation._rng = np.random.default_rng(worker_seed)
