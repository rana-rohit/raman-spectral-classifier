"""
src/data/dataloader.py

Build all DataLoaders used by the pipeline.
"""

from __future__ import annotations

import copy
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import SpectralDataset, make_train_val_split
from src.data.preprocessing import SpectralPreprocessor
from src.data.registry import DataRegistry


def build_all_loaders(
    registry: DataRegistry,
    preprocessor: SpectralPreprocessor,
    augmentation,
    cfg: dict,
) -> Dict:
    batch_size = cfg.get("batch_size", 256)
    num_workers = cfg.get("num_workers", 4)
    seed = cfg.get("seed", 42)
    val_fraction = cfg["validation"]["val_fraction"]
    val_seed = cfg["validation"]["random_seed"]
    consistency_cfg = cfg.get("consistency", {})
    train_views = 2 if consistency_cfg.get("enabled", False) else 1

    loaders = {}

    X_ref, y_ref = registry.get_arrays("reference")
    X_ref = preprocessor.transform(X_ref)
    (X_tr, y_tr), (X_val, y_val) = make_train_val_split(
        X_ref,
        y_ref,
        val_fraction=val_fraction,
        random_seed=val_seed,
    )

    loaders["train"] = _make_loader(
        X_tr,
        y_tr,
        augmentation=_clone_augmentation(augmentation),
        training=True,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        seed=seed,
        n_views=train_views,
    )
    loaders["val"] = _make_loader(
        X_val,
        y_val,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed + 1,
    )

    X_test, y_test = registry.get_arrays("test")
    X_test = preprocessor.transform(X_test)
    loaders["test"] = _make_loader(
        X_test,
        y_test,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed + 2,
    )

    X_ft, y_ft = registry.get_arrays("finetune")
    X_ft = preprocessor.transform(X_ft)
    loaders["finetune"] = _make_loader(
        X_ft,
        y_ft,
        augmentation=_clone_augmentation(augmentation),
        training=True,
        batch_size=min(batch_size, 64),
        num_workers=num_workers,
        shuffle=True,
        seed=seed + 3,
        n_views=train_views,
    )

    loaders["ood"] = {}
    for idx, split_name in enumerate(registry.ood_split_names()):
        X_ood, y_ood = registry.get_arrays(split_name)
        X_ood = preprocessor.transform(X_ood)
        eval_classes = registry.get_eval_classes(split_name)
        loaders["ood"][split_name] = _make_loader(
            X_ood,
            y_ood,
            class_filter=eval_classes,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed + 10 + idx,
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
    seed: int = 42,
    n_views: int = 1,
) -> DataLoader:
    dataset = SpectralDataset(
        X,
        y,
        augmentation=augmentation,
        training=training,
        class_filter=class_filter,
        n_views=n_views,
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
