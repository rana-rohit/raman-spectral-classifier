"""
src/data/dataloader.py

Build all DataLoaders used by the pipeline.
Supports optional 2-channel input (raw + first derivative).
"""

from __future__ import annotations

import copy
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.preprocessing import SpectralPreprocessor
from src.data.dataset import SpectralDataset
from src.data.registry import DataRegistry
from src.utils.class_subset import (
    class_maps,
    filter_and_remap_classes,
)



def build_all_loaders(
    registry: DataRegistry,
    preprocessor: SpectralPreprocessor,
    augmentation,
    cfg: dict,
    clinical_sparse_ids,
    n_classes,
) -> Dict:
    train_cfg = cfg.get("training", {})
    batch_size = cfg.get("batch_size", train_cfg.get("batch_size", 256))
    num_workers = cfg.get("num_workers", train_cfg.get("num_workers", 4))
    seed = cfg.get("seed", 42)
    val_fraction = cfg["validation"]["val_fraction"]
    val_seed = cfg["validation"]["random_seed"]
    clinical_val_fraction = cfg["validation"].get("clinical_val_fraction", val_fraction)
    clinical_eval_fraction = cfg["validation"].get("clinical_eval_fraction", val_fraction)
    consistency_cfg = cfg.get("consistency") or train_cfg.get("consistency", {})
    stage = cfg.get("task", {}).get(
        "stage",
        None,
    )

    if stage is None:
        raise ValueError(
            "Missing task.stage in dataloader config"
        )
    if clinical_sparse_ids is not None:
        clinical_sparse_ids = [int(cls) for cls in sorted(clinical_sparse_ids)]

        class_map, inverse_class_map = class_maps(clinical_sparse_ids)

    else:
        class_map = None
        inverse_class_map = None

    train_views = 2 if consistency_cfg.get("enabled", False) else 1
    # If consistency training enabled, return multiple augmented views per sample
    if train_views not in (1, 2):
        raise ValueError("train_views must be 1 or 2")

    loaders = {}
    
    # --------------------------------------------------------
    # Stage-dependent reference-domain behavior:
    #
    #- pretrain_30class:
    #   isolate-space labels
    #
    #- pretrain_treatment_8class:
    #   global treatment-space labels
    #2
    #- transfer_5class:
    #   compact transfer-space labels
    # --------------------------------------------------------

    X_ref, y_ref, ref_ids = _load_shared_split(registry, "reference", clinical_sparse_ids, stage=stage)

    X_tr, X_val, y_tr, y_val, tr_ids, val_ids = _contiguous_split(
        X_ref, y_ref, ref_ids, val_fraction
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
        expected_n_classes=n_classes,
        class_filter=clinical_sparse_ids,
        class_map=class_map,
        inverse_class_map=inverse_class_map,
        sample_ids=tr_ids,
    )
    loaders["source_val"] = _make_loader(
        X_val, y_val,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed + 1,
        preprocessor=preprocessor,
        shuffle=False,
        expected_n_classes=n_classes,
        class_filter=clinical_sparse_ids,
        class_map=class_map,
        inverse_class_map=inverse_class_map,
        sample_ids=val_ids,
    )
    loaders["val"] = loaders["source_val"]
    
    # --------------------------------------------------------
    # REFERENCE-DOMAIN HOLDOUT EVALUATION
    #
    # IMPORTANT:
    # In transfer mode, this evaluates compact
    # transfer-space performance on reference-domain spectra.
    #
    # This is NOT 30-class isolate evaluation.
    # --------------------------------------------------------

    X_test, y_test, test_ids = _load_shared_split(
        registry,
        "test",
        clinical_sparse_ids,
        allow_holdout=True,
        stage=stage,
    )
    loaders["test"] = _make_loader(
        X_test, y_test,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed + 2,
        shuffle=False,
        preprocessor=preprocessor,
        expected_n_classes=n_classes,
        class_filter=clinical_sparse_ids,
        class_map=class_map,
        inverse_class_map=inverse_class_map,
        sample_ids=test_ids,
    )
    
    # --------------------------------------------------------
    # CLINICAL OOD DOMAIN
    #
    # Clinical datasets already exist in sparse
    # clinical treatment-space:
    #
    # [0,2,3,5,6]
    #
    # and are remapped internally to compact labels.
    # --------------------------------------------------------

    # -----------------------------
    # CLINICAL LOADER (for domain adaptation / DANN)
    # -----------------------------
    loaders["ood"] = {}
    try:
        clinical_train_parts = []
        clinical_val_parts = []
        clinical_train_ids = []
        clinical_val_ids = []

        for idx, split_name in enumerate(registry.ood_split_names()):
            X_clin, y_clin, clin_ids = _load_shared_split(
                registry,
                split_name,
                clinical_sparse_ids,
                stage=stage,
            )
            (
                X_pool,
                X_eval,
                y_pool,
                y_eval,
                ids_pool,
                ids_eval,
            ) = _contiguous_split(
                X_clin,
                y_clin,
                clin_ids,
                test_fraction=clinical_eval_fraction,
            )

            val_relative = clinical_val_fraction / max(1e-8, (1.0 - clinical_eval_fraction))
            val_relative = min(max(val_relative, 0.0), 0.5)
            (
                X_clin_tr,
                X_clin_val,
                y_clin_tr,
                y_clin_val,
                ids_clin_tr,
                ids_clin_val,
            ) = _contiguous_split(
                X_pool,
                y_pool,
                ids_pool,
                test_fraction=val_relative,
            )

            clinical_train_parts.append((X_clin_tr, y_clin_tr))
            clinical_val_parts.append((X_clin_val, y_clin_val))
            clinical_train_ids.extend(ids_clin_tr.tolist())
            clinical_val_ids.extend(ids_clin_val.tolist())

            loaders["ood"][split_name] = _make_loader(
                X_eval,
                y_eval,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                seed=seed + 10 + idx,
                preprocessor=preprocessor,
                expected_n_classes=n_classes,
                class_filter=clinical_sparse_ids,
                class_map=class_map,
                inverse_class_map=inverse_class_map,
                sample_ids=ids_eval,
            )

        X_clin_tr = np.concatenate([part[0] for part in clinical_train_parts], axis=0)
        y_clin_tr = np.concatenate([part[1] for part in clinical_train_parts], axis=0)
        X_clin_val = np.concatenate([part[0] for part in clinical_val_parts], axis=0)
        y_clin_val = np.concatenate([part[1] for part in clinical_val_parts], axis=0)
        clinical_eval_ids = [
            sample_id
            for loader in loaders["ood"].values()
            for sample_id in loader.dataset.sample_ids.tolist()
        ]
        _assert_disjoint("clinical_train", clinical_train_ids, "clinical_eval", clinical_eval_ids)
        _assert_disjoint("clinical_val", clinical_val_ids, "clinical_eval", clinical_eval_ids)

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
            expected_n_classes=n_classes,
            class_filter=clinical_sparse_ids,
            class_map=class_map,
            inverse_class_map=inverse_class_map,
            sample_ids=np.asarray(clinical_train_ids),
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
            expected_n_classes=n_classes,
            class_filter=clinical_sparse_ids,
            class_map=class_map,
            inverse_class_map=inverse_class_map,
            sample_ids=np.asarray(clinical_val_ids),
        )
        loaders["val"] = loaders["clinical_val"]

    except FileNotFoundError as e:
        if stage == "transfer_5class":
            raise
        print("WARNING: Clinical data not found; skipping optional OOD loaders:", e)

    X_ft, y_ft, ft_ids = _load_shared_split(registry, "finetune", clinical_sparse_ids, stage=stage)
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
        expected_n_classes=n_classes,
        class_filter=clinical_sparse_ids,
        class_map=class_map,
        inverse_class_map=inverse_class_map,
        sample_ids=ft_ids,
    )

    return loaders


def _contiguous_split(X, y, sample_ids, test_fraction: float):
    """
    Splits data contiguously within each class to prevent biological
    replicate leakage. This relies on spectra being ordered sequentially
    by replicate/acquisition during dataset creation.
    """
    tr_idx_list = []
    val_idx_list = []
    
    for c in np.unique(y):
        c_indices = np.where(y == c)[0]
        n_val = max(1, int(len(c_indices) * test_fraction))
        
        if n_val > 0:
            val_idx_list.extend(c_indices[-n_val:])
            tr_idx_list.extend(c_indices[:-n_val])
        else:
            tr_idx_list.extend(c_indices)
            
    tr_idx = np.array(tr_idx_list)
    val_idx = np.array(val_idx_list)
    
    return X[tr_idx], X[val_idx], y[tr_idx], y[val_idx], sample_ids[tr_idx], sample_ids[val_idx]

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
    expected_n_classes: int | None = None,
    class_filter: list[int] | None = None,
    class_map: dict[int, int] | None = None,
    inverse_class_map: dict[int, int] | None = None,
    sample_ids=None,
) -> DataLoader:
    dataset = SpectralDataset(
        X,
        y,
        augmentation=augmentation,
        training=training,
        n_views=n_views,
        preprocessor=preprocessor,
        expected_n_classes=expected_n_classes,
        class_filter=class_filter,
        class_map=class_map,
        inverse_class_map=inverse_class_map,
        sample_ids=sample_ids,
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


def _load_shared_split(
    registry: DataRegistry,
    split_name: str,
    clinical_sparse_ids: list[int] | None,
    allow_holdout: bool = False,
    stage: str = "transfer_5class",
):
    from metadata.ontology import ISOLATE_TO_TREATMENT

    X, y = registry.get_arrays(
        split_name,
        allow_holdout=allow_holdout,
    )

    # --------------------------------------------------------
    # PRETRAINING MODE 1: ISOLATE SPACE
    # --------------------------------------------------------
    if stage == "pretrain_30class":
        sample_ids = np.asarray(
            [f"{split_name}:{idx}" for idx in range(len(y))]
        )
        return X, y, sample_ids

    split_cfg = registry.cfg["splits"][split_name]

    label_space = split_cfg.get("label_space")

    if label_space is None:
        raise ValueError(
            f"{split_name} missing label_space metadata"
        )

    is_isolate_space = (
        label_space == "isolate_space"
    )

    # --------------------------------------------------------
    # PRETRAINING MODE 2: GLOBAL TREATMENT SPACE (Stage 2)
    # --------------------------------------------------------
    if stage == "pretrain_treatment_8class":
        if is_isolate_space:
            y_treatment = np.array(
                [ISOLATE_TO_TREATMENT[int(label)] for label in y],
                dtype=np.int64,
            )
        else:
            y_treatment = y

        sample_ids = np.asarray(
            [f"{split_name}:{idx}" for idx in range(len(y))]
        )

        _assert_label_range(split_name, y_treatment, 8)
        return X, y_treatment, sample_ids

    # --------------------------------------------------------
    # TRANSFER MODE (Stage 3): COMPACT TRANSFER SPACE
    # --------------------------------------------------------
    assert clinical_sparse_ids is not None
    assert len(clinical_sparse_ids) > 0, (
        "transfer_5class requires "
        "clinical_sparse_ids"
    )
    if is_isolate_space:
        # Step 1: Map isolate IDs → global treatment IDs
        y_treatment = np.array(
            [ISOLATE_TO_TREATMENT[int(label)] for label in y],
            dtype=np.int64,
        )

        # Step 2: Filter to clinical treatment subset
        mask = np.isin(y_treatment, clinical_sparse_ids)

        sample_ids = np.asarray(
            [f"{split_name}:{idx}" for idx in np.flatnonzero(mask)]
        )

        X_filtered = X[mask]
        y_filtered = y_treatment[mask]

        # Step 3: Remap global treatment → compact
        X_filtered, y_compact = filter_and_remap_classes(
            X_filtered,
            y_filtered,
            clinical_sparse_ids,
        )
    else:
        # Clinical data: already in global treatment-space
        mask = np.isin(y, clinical_sparse_ids)

        sample_ids = np.asarray(
            [f"{split_name}:{idx}" for idx in np.flatnonzero(mask)]
        )

        X_filtered, y_compact = filter_and_remap_classes(
            X,
            y,
            clinical_sparse_ids,
        )

    _assert_label_range(
        split_name,
        y_compact,
        len(clinical_sparse_ids),
    )

    return X_filtered, y_compact, sample_ids


def _assert_label_range(split_name: str, labels: np.ndarray, n_classes: int) -> None:
    if labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError(
            f"{split_name} labels must be in [0, {n_classes - 1}], "
            f"got [{labels.min()}, {labels.max()}]"
        )


def _assert_disjoint(left_name: str, left_ids, right_name: str, right_ids) -> None:
    overlap = set(left_ids).intersection(set(right_ids))
    if overlap:
        example = sorted(overlap)[0]
        raise RuntimeError(
            f"{left_name} and {right_name} overlap at sample {example}; "
            "clinical train/eval leakage is not allowed"
        )
