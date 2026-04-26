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

from src.data.dataset import SpectralDataset, make_train_val_split
from src.data.preprocessing import SpectralPreprocessor, FirstDerivative
from src.data.registry import DataRegistry
from src.utils.class_subset import filter_and_remap_classes
from sklearn.model_selection import train_test_split

SHARED_CLASSES = [0, 2, 3, 5, 6]

def build_all_loaders(
    registry: DataRegistry,
    preprocessor: SpectralPreprocessor,
    augmentation,
    cfg: dict,
    derivative_cfg: Optional[dict] = None,
) -> Dict:
    derivative_cfg = derivative_cfg or {}
    batch_size = cfg.get("batch_size", 256)
    num_workers = cfg.get("num_workers", 4)
    seed = cfg.get("seed", 42)
    val_fraction = cfg["validation"]["val_fraction"]
    val_seed = cfg["validation"]["random_seed"]
    consistency_cfg = cfg.get("consistency", {})

    # If consistency training enabled, return multiple augmented views per sample
    train_views = 2 if consistency_cfg.get("enabled", False) else 1

    # Build derivative transform if enabled
    deriv_transform = None
    if derivative_cfg is not None and derivative_cfg.get("enabled", False):
        deriv_transform = FirstDerivative(
            window_length=derivative_cfg.get("window_length", 11),
            polyorder=derivative_cfg.get("polyorder", 3),
        )

    loaders = {}

    X_ref, y_ref = registry.get_arrays("reference")

    # Apply preprocessing FIRST
    X_ref = preprocessor.transform(X_ref)

    # THEN filter
    X_ref, y_ref = filter_and_remap_classes(X_ref, y_ref, SHARED_CLASSES)

    if deriv_transform is not None:
        dX_ref = deriv_transform.transform(X_ref)
        assert dX_ref.shape == X_ref.shape, "Derivative shape mismatch"
    else:
        dX_ref = None

    train_split, val_split = make_train_val_split(
        X_ref, y_ref,
        val_fraction=val_fraction,
        random_seed=val_seed,
        derivative_X=dX_ref,
    )
    X_tr, y_tr, dX_tr = train_split
    X_val, y_val, dX_val = val_split

    loaders["train"] = _make_loader(
        X_tr, y_tr,
        augmentation=_clone_augmentation(augmentation),
        training=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        n_views=train_views,
        derivative_X=dX_tr,
    )
    loaders["val"] = _make_loader(
        X_val, y_val,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed + 1,
        derivative_X=dX_val,
    )

    X_test, y_test = registry.get_arrays("test", allow_holdout=True)

    X_test = preprocessor.transform(X_test)
    X_test, y_test = filter_and_remap_classes(X_test, y_test, SHARED_CLASSES)
    dX_test = _apply_deriv(deriv_transform, X_test)
    loaders["test"] = _make_loader(
        X_test, y_test,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed + 2,
        derivative_X=dX_test,
    )
    
    # -----------------------------
    # CLINICAL LOADER (for domain adaptation / DANN)
    # -----------------------------
    try:
        X_clin1, y_clin1 = registry.get_arrays("2018clinical")
        X_clin2, y_clin2 = registry.get_arrays("2019clinical")

        X_clin = np.concatenate([X_clin1, X_clin2], axis=0)
        y_clin = np.concatenate([y_clin1, y_clin2], axis=0)
        X_clin = preprocessor.transform(X_clin)
        X_clin, y_clin = filter_and_remap_classes(X_clin, y_clin, SHARED_CLASSES)

        
        # Stratified split
        X_clin_tr, X_clin_val, y_clin_tr, y_clin_val = train_test_split(
            X_clin,
            y_clin,
            test_size=0.2,
            stratify=y_clin if len(np.unique(y_clin)) > 1 else None,
            random_state=seed,
        )

        # Apply derivative separately
        dX_clin_tr = _apply_deriv(deriv_transform, X_clin_tr)
        dX_clin_val = _apply_deriv(deriv_transform, X_clin_val)

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
            derivative_X=dX_clin_tr,
        )

        # Validation loader
        loaders["clinical_val"] = _make_loader(
            X_clin_val,
            y_clin_val,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            seed=seed + 6,
            derivative_X=dX_clin_val,
        )

    except Exception as e:
        print("WARNING: Clinical data not found:", e)

    X_ft, y_ft = registry.get_arrays("finetune")

    X_ft = preprocessor.transform(X_ft)
    X_ft, y_ft = filter_and_remap_classes(X_ft, y_ft, SHARED_CLASSES)
    dX_ft = _apply_deriv(deriv_transform, X_ft)
    loaders["finetune"] = _make_loader(
        X_ft, y_ft,
        augmentation=_clone_augmentation(augmentation),
        training=True,
        batch_size=min(batch_size, 64),
        num_workers=num_workers,
        shuffle=True,
        seed=seed + 3,
        n_views=train_views,
        derivative_X=dX_ft,
    )

    loaders["ood"] = {}
    for idx, split_name in enumerate(registry.ood_split_names()):
        X_ood, y_ood = registry.get_arrays(split_name)

        X_ood = preprocessor.transform(X_ood)
        X_ood, y_ood = filter_and_remap_classes(X_ood, y_ood, SHARED_CLASSES)
        dX_ood = _apply_deriv(deriv_transform, X_ood)
        loaders["ood"][split_name] = _make_loader(
            X_ood, y_ood,
            class_filter=None,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed + 10 + idx,
            derivative_X=dX_ood,
        )

    return loaders
    
def _apply_deriv(deriv_transform, X):
    if deriv_transform is None:
        return None
    return deriv_transform.transform(X)


def _make_loader(
    X,
    y,
    augmentation=None,
    training: bool = False,
    class_filter=None,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = False,
    seed: int = 42,
    n_views: int = 1,
    derivative_X=None,
) -> DataLoader:
    dataset = SpectralDataset(
        X,
        y,
        augmentation=augmentation,
        training=training,
        class_filter=class_filter,
        n_views=n_views,
        derivative_X=derivative_X,
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
