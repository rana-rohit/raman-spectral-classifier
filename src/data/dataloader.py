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
from src.utils.split_modes import (
    HOLDOUT,
    IID_REFERENCE,
    PATIENT_CV,
    resolve_iid_reference_split_config,
    resolve_split_mode,
)
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
    fold_index: int | None = None,
) -> Dict:
    train_cfg = cfg.get("training", {})
    batch_size = cfg.get("batch_size", train_cfg.get("batch_size", 256))
    num_workers = cfg.get("num_workers", train_cfg.get("num_workers", 4))
    seed = cfg.get("seed", 42)
    val_fraction = cfg["validation"]["val_fraction"]
    split_mode = resolve_split_mode(cfg)
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

    source_batch_size = batch_size
    # Clinical train loader uses the same batch size as source by default.
    clinical_train_batch_size = batch_size

    loaders = {}
    clinical_enabled = _clinical_splits_enabled(stage, cfg)
    
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

    if split_mode in {HOLDOUT, PATIENT_CV}:
        X_tr, X_val, y_tr, y_val, tr_ids, val_ids = _contiguous_split(
            X_ref, y_ref, ref_ids, val_fraction
        )
        X_test, y_test, test_ids = _load_shared_split(
            registry,
            "test",
            clinical_sparse_ids,
            allow_holdout=True,
            stage=stage,
        )
    elif split_mode == IID_REFERENCE:
        (
            X_tr,
            X_val,
            X_test,
            y_tr,
            y_val,
            y_test,
            tr_ids,
            val_ids,
            test_ids,
        ) = _iid_reference_split(X_ref, y_ref, ref_ids, cfg)
    else:
        raise AssertionError(f"Unhandled split_mode: {split_mode}")

    if split_mode == PATIENT_CV and stage == "transfer_5class":
        include_finetune_in_source = (
            cfg.get("validation", {})
            .get("patient_cv", {})
            .get("include_finetune_in_train", True)
        )
        if include_finetune_in_source:
            X_ft_source, y_ft_source, ft_source_ids = _load_shared_split(
                registry,
                "finetune",
                clinical_sparse_ids,
                stage=stage,
            )
            X_tr = np.concatenate([X_tr, X_ft_source], axis=0)
            y_tr = np.concatenate([y_tr, y_ft_source], axis=0)
            tr_ids = np.concatenate([tr_ids, ft_source_ids], axis=0)

    _assert_disjoint("train", tr_ids, "source_val", val_ids)
    _assert_disjoint("train", tr_ids, "test", test_ids)
    _assert_disjoint("source_val", val_ids, "test", test_ids)

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
        label_validation="contiguous",
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
        label_validation="contiguous",
    )
    loaders["val"] = loaders["source_val"]
    
    # --------------------------------------------------------
    # REFERENCE-DOMAIN EVALUATION
    #
    # holdout:
    #   dedicated X_test / y_test split.
    #
    # iid_reference:
    #   held-out stratified partition of X_reference / y_reference.
    #
    # In transfer mode, both evaluate compact transfer-space
    # performance on reference-domain spectra.
    # --------------------------------------------------------
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
        label_validation="contiguous",
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
    if clinical_enabled:
        if split_mode == PATIENT_CV:
            from src.data.patient_cv import build_patient_folds, get_fold_indices

            patient_cv_cfg = cfg.get("validation", {}).get("patient_cv", {})
            n_patient_folds = int(patient_cv_cfg.get("n_folds", 5))
            clinical_ys = {}
            for split_name in registry.ood_split_names():
                _, y_raw = registry.get_arrays(split_name)
                clinical_ys[split_name] = y_raw

            folds = build_patient_folds(clinical_ys, n_folds=n_patient_folds, seed=seed)
            if fold_index is None:
                fold_index = cfg.get("fold_index") or cfg.get("validation", {}).get("fold_index", 0)
            fold_index = int(fold_index)
            if fold_index < 0 or fold_index >= len(folds):
                raise ValueError(
                    f"fold_index must be in [0, {len(folds) - 1}], got {fold_index}"
                )
            fold = folds[fold_index]

            clinical_train_parts = []
            clinical_train_ids = []

            for idx, split_name in enumerate(registry.ood_split_names()):
                X_clin, y_clin, clin_ids = _load_shared_split(
                    registry,
                    split_name,
                    clinical_sparse_ids,
                    stage=stage,
                )

                train_idx = get_fold_indices(clin_ids, fold, "train")
                test_idx = get_fold_indices(clin_ids, fold, "test")

                if len(train_idx) > 0:
                    clinical_train_parts.append((X_clin[train_idx], y_clin[train_idx]))
                    clinical_train_ids.extend(clin_ids[train_idx].tolist())

                loaders["ood"][split_name] = _make_loader(
                    X_clin[test_idx],
                    y_clin[test_idx],
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    seed=seed + 10 + idx,
                    preprocessor=preprocessor,
                    expected_n_classes=n_classes,
                    class_filter=clinical_sparse_ids,
                    class_map=class_map,
                    inverse_class_map=inverse_class_map,
                    sample_ids=clin_ids[test_idx],
                    label_validation=_clinical_label_validation(stage),
                    valid_label_ids=_valid_label_ids(stage, clinical_sparse_ids),
                )

            if not clinical_train_parts:
                raise RuntimeError(
                    f"Patient CV fold {fold_index} produced no clinical training spectra"
                )

            X_clin_tr = np.concatenate([part[0] for part in clinical_train_parts], axis=0)
            y_clin_tr = np.concatenate([part[1] for part in clinical_train_parts], axis=0)

            clinical_val_parts = []
            clinical_val_ids = []
            for loader in loaders["ood"].values():
                clinical_val_parts.append((loader.dataset.X, loader.dataset.y))
                clinical_val_ids.extend(loader.dataset.sample_ids.tolist())

            if not clinical_val_parts:
                raise RuntimeError(
                    f"Patient CV fold {fold_index} produced no clinical evaluation spectra"
                )

            X_clin_val = np.concatenate([part[0] for part in clinical_val_parts], axis=0)
            y_clin_val = np.concatenate([part[1] for part in clinical_val_parts], axis=0)

        else:
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
                    label_validation=_clinical_label_validation(stage),
                    valid_label_ids=_valid_label_ids(stage, clinical_sparse_ids),
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
        if split_mode != PATIENT_CV:
            _assert_disjoint("clinical_val", clinical_val_ids, "clinical_eval", clinical_eval_ids)

        # Train loader
        loaders["clinical_train"] = _make_loader(
            X_clin_tr,
            y_clin_tr,
            augmentation=_clone_augmentation(augmentation),
            training=True,
            batch_size=clinical_train_batch_size,
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
            label_validation=_clinical_label_validation(stage),
            valid_label_ids=_valid_label_ids(stage, clinical_sparse_ids),
        )

        if split_mode != PATIENT_CV:
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
                label_validation=_clinical_label_validation(stage),
                valid_label_ids=_valid_label_ids(stage, clinical_sparse_ids),
            )
            loaders["val"] = loaders["clinical_val"]

    if _finetune_split_enabled(stage, cfg):
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
            label_validation="contiguous",
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


def _iid_reference_split(X, y, sample_ids, cfg: dict):
    """
    Build train/val/test partitions from reference groups only.

    Raman grouped evaluation assumes contiguous spectra_per_group chunks.
    IID reference mode therefore partitions those chunks, not individual
    spectra, so no isolate/replicate group can span train/val/test.
    """
    iid_cfg = resolve_iid_reference_split_config(cfg)
    groups_by_label = _reference_groups_by_label(
        y,
        spectra_per_group=iid_cfg.spectra_per_group,
    )
    group_counts = {label: len(groups) for label, groups in groups_by_label.items()}
    total_groups = sum(group_counts.values())

    if iid_cfg.test_groups >= total_groups:
        raise ValueError(
            "validation.iid_reference.test_groups must be smaller than the "
            f"number of reference groups; got {iid_cfg.test_groups} for "
            f"{total_groups} groups"
        )

    rng = np.random.default_rng(iid_cfg.random_seed)
    shuffled_groups = {
        label: list(rng.permutation(groups))
        for label, groups in groups_by_label.items()
    }

    test_counts = _allocate_group_counts(group_counts, iid_cfg.test_groups)
    test_groups_by_label = {}
    remaining_groups_by_label = {}
    for label, groups in shuffled_groups.items():
        n_test = test_counts[label]
        test_groups_by_label[label] = groups[:n_test]
        remaining_groups_by_label[label] = groups[n_test:]

    remaining_counts = {
        label: len(groups)
        for label, groups in remaining_groups_by_label.items()
    }
    val_group_target = int(round(total_groups * iid_cfg.val_fraction))
    val_group_target = max(len(remaining_counts), val_group_target)
    val_group_target = min(sum(remaining_counts.values()) - len(remaining_counts), val_group_target)
    val_counts = _allocate_group_counts(remaining_counts, val_group_target)

    val_groups_by_label = {}
    train_groups_by_label = {}
    for label, groups in remaining_groups_by_label.items():
        n_val = val_counts[label]
        val_groups_by_label[label] = groups[:n_val]
        train_groups_by_label[label] = groups[n_val:]

    train_idx = _flatten_group_indices(train_groups_by_label)
    val_idx = _flatten_group_indices(val_groups_by_label)
    test_idx = _flatten_group_indices(test_groups_by_label)

    _assert_group_keys_disjoint(
        "train",
        sample_ids[train_idx],
        "source_val",
        sample_ids[val_idx],
        iid_cfg.spectra_per_group,
    )
    _assert_group_keys_disjoint(
        "train",
        sample_ids[train_idx],
        "test",
        sample_ids[test_idx],
        iid_cfg.spectra_per_group,
    )
    _assert_group_keys_disjoint(
        "source_val",
        sample_ids[val_idx],
        "test",
        sample_ids[test_idx],
        iid_cfg.spectra_per_group,
    )

    return (
        X[train_idx],
        X[val_idx],
        X[test_idx],
        y[train_idx],
        y[val_idx],
        y[test_idx],
        sample_ids[train_idx],
        sample_ids[val_idx],
        sample_ids[test_idx],
    )


def _reference_groups_by_label(y, spectra_per_group: int):
    y = np.asarray(y)
    groups_by_label = {}

    for label in sorted(np.unique(y).astype(int).tolist()):
        label_indices = np.where(y == label)[0]
        if len(label_indices) % spectra_per_group != 0:
            raise ValueError(
                f"Reference label {label} has {len(label_indices)} spectra, "
                f"which is not divisible by spectra_per_group={spectra_per_group}"
            )

        groups = []
        for start in range(0, len(label_indices), spectra_per_group):
            group = label_indices[start:start + spectra_per_group]
            if len(np.unique(y[group])) != 1:
                raise RuntimeError(
                    f"Reference group for label {label} contains mixed labels"
                )
            groups.append(group)

        groups_by_label[label] = groups

    return groups_by_label


def _allocate_group_counts(group_counts: dict[int, int], target_total: int) -> dict[int, int]:
    labels = sorted(group_counts)
    available = sum(group_counts.values())
    if target_total <= 0 or target_total > available:
        raise ValueError(
            f"Cannot allocate {target_total} groups from {available} available groups"
        )
    if target_total < len(labels):
        raise ValueError(
            f"Need at least one group per class; got target_total={target_total} "
            f"for {len(labels)} classes"
        )

    exact = {
        label: target_total * group_counts[label] / available
        for label in labels
    }
    allocation = {
        label: min(group_counts[label], max(1, int(np.floor(exact[label]))))
        for label in labels
    }

    while sum(allocation.values()) > target_total:
        candidates = [
            label for label in labels
            if allocation[label] > 1
        ]
        label = min(
            candidates,
            key=lambda item: (exact[item] - np.floor(exact[item]), allocation[item]),
        )
        allocation[label] -= 1

    while sum(allocation.values()) < target_total:
        candidates = [
            label for label in labels
            if allocation[label] < group_counts[label]
        ]
        label = max(
            candidates,
            key=lambda item: (exact[item] - np.floor(exact[item]), group_counts[item]),
        )
        allocation[label] += 1

    return allocation


def _flatten_group_indices(groups_by_label: dict[int, list[np.ndarray]]) -> np.ndarray:
    groups = [
        group
        for label in sorted(groups_by_label)
        for group in groups_by_label[label]
    ]
    if not groups:
        return np.asarray([], dtype=np.int64)
    groups = sorted(groups, key=lambda group: int(group[0]))
    return np.concatenate(groups).astype(np.int64)

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
    label_validation: str = "contiguous",
    valid_label_ids: list[int] | None = None,
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
        label_validation=label_validation,
        valid_label_ids=valid_label_ids,
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


def _clinical_splits_enabled(stage: str, cfg: dict) -> bool:
    """Clinical OOD loaders are stage-aware and opt-in outside transfer."""
    if stage == "transfer_5class":
        return True
    clinical_cfg = cfg.get("evaluation", {}).get("clinical_ood", {})
    return bool(clinical_cfg.get("enabled", False)) and stage == "pretrain_treatment_8class"


def _finetune_split_enabled(stage: str, cfg: dict) -> bool:
    """The adaptation split is only built when explicitly requested."""
    if cfg.get("training", {}).get("finetune", {}).get("enabled", False):
        return True
    return bool(cfg.get("data", {}).get("include_finetune_split", False))


def _clinical_label_validation(stage: str) -> str:
    if stage == "pretrain_treatment_8class":
        return "range"
    if stage == "transfer_5class":
        return "range"
    return "membership"


def _valid_label_ids(stage: str, clinical_sparse_ids: list[int] | None) -> list[int] | None:
    if stage == "pretrain_treatment_8class":
        return list(range(8))
    if stage == "transfer_5class" and clinical_sparse_ids is not None:
        return list(range(len(clinical_sparse_ids)))
    return clinical_sparse_ids


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

        if split_name in ("2018clinical", "2019clinical"):
            from metadata.patient_ids import generate_patient_ids
            patient_ids = generate_patient_ids(y, split_name)
            sample_ids = patient_ids[mask]
        else:
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


def _assert_group_keys_disjoint(
    left_name: str,
    left_ids,
    right_name: str,
    right_ids,
    spectra_per_group: int,
) -> None:
    left_keys = _sample_group_keys(left_ids, spectra_per_group)
    right_keys = _sample_group_keys(right_ids, spectra_per_group)
    overlap = left_keys.intersection(right_keys)
    if overlap:
        example = sorted(overlap)[0]
        raise RuntimeError(
            f"{left_name} and {right_name} share reference group {example}; "
            "group-aware IID mode requires isolate/replicate groups to be disjoint"
        )


def _sample_group_keys(sample_ids, spectra_per_group: int) -> set[str]:
    keys = set()
    for raw_id in sample_ids:
        sample_id = str(raw_id)
        split_name, index_text = sample_id.rsplit(":", 1)
        keys.add(f"{split_name}:{int(index_text) // spectra_per_group}")
    return keys
