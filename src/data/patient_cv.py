"""
src/data/patient_cv.py

Patient-aware 5-fold cross-validation for clinical transfer learning.

Guarantees:
- No patient appears in both train and test within any fold.
- Every patient appears in test exactly once across all folds.
- Class balance is maintained as much as possible within folds.
- Reference and finetune data always appear in training.

This module provides the splitting logic only. It does not
build DataLoaders — that remains in dataloader.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from metadata.patient_ids import (generate_patient_ids, get_unique_patients,
                                  patient_to_label)


@dataclass
class PatientFold:
    """A single fold of patient-level cross-validation."""
    fold_index: int
    train_patients: list[str]
    test_patients: list[str]
    n_folds: int

    def __post_init__(self):
        overlap = set(self.train_patients) & set(self.test_patients)
        if overlap:
            raise ValueError(
                f"Fold {self.fold_index}: patient leakage detected. "
                f"Overlap: {sorted(overlap)}"
            )


def build_patient_folds(
    y_arrays: Dict[str, np.ndarray],
    n_folds: int = 5,
    seed: int = 42,
) -> list[PatientFold]:
    """
    Build patient-level cross-validation folds from clinical datasets.

    Parameters
    ----------
    y_arrays : dict
        Mapping of dataset_name -> label array.
        Example: {"2018clinical": y_2018, "2019clinical": y_2019}
    n_folds : int
        Number of cross-validation folds.
    seed : int
        Random seed for reproducible fold assignment.

    Returns
    -------
    folds : list[PatientFold]
        One PatientFold per fold with train/test patient lists.
    """
    # Generate patient IDs for all clinical datasets
    all_patient_ids = []
    for dataset_name, y in sorted(y_arrays.items()):
        pids = generate_patient_ids(y, dataset_name)
        unique_pids = get_unique_patients(pids)
        all_patient_ids.extend(unique_pids)

    all_patient_ids = sorted(set(all_patient_ids))
    n_patients = len(all_patient_ids)

    if n_patients < n_folds:
        raise ValueError(
            f"Cannot create {n_folds} folds from {n_patients} patients"
        )

    # --------------------------------------------------------
    # Stratified fold assignment
    # Group patients by treatment label, then distribute
    # evenly across folds to preserve class balance.
    # --------------------------------------------------------
    patients_by_label: Dict[int, list[str]] = {}
    for pid in all_patient_ids:
        label = patient_to_label(pid)
        patients_by_label.setdefault(label, []).append(pid)

    rng = np.random.default_rng(seed)

    # Shuffle patients within each label group
    for label in patients_by_label:
        patients_by_label[label] = list(
            rng.permutation(patients_by_label[label])
        )

    # Assign patients to folds in round-robin fashion per label
    fold_assignments: Dict[int, list[str]] = {
        i: [] for i in range(n_folds)
    }

    for label in sorted(patients_by_label.keys()):
        patients = patients_by_label[label]
        for idx, pid in enumerate(patients):
            fold_idx = idx % n_folds
            fold_assignments[fold_idx].append(pid)

    # Build fold objects
    folds = []
    for fold_idx in range(n_folds):
        test_patients = sorted(fold_assignments[fold_idx])
        train_patients = sorted(
            pid
            for other_fold, pids in fold_assignments.items()
            if other_fold != fold_idx
            for pid in pids
        )

        fold = PatientFold(
            fold_index=fold_idx,
            train_patients=train_patients,
            test_patients=test_patients,
            n_folds=n_folds,
        )
        folds.append(fold)

    # --------------------------------------------------------
    # Integrity checks
    # --------------------------------------------------------
    # Every patient appears in test exactly once
    all_test = []
    for fold in folds:
        all_test.extend(fold.test_patients)

    if sorted(all_test) != sorted(all_patient_ids):
        raise RuntimeError(
            "Patient coverage check failed: not every patient "
            "appears in test exactly once"
        )

    if len(all_test) != len(set(all_test)):
        raise RuntimeError(
            "Patient uniqueness check failed: some patients "
            "appear in test more than once"
        )

    return folds


def get_fold_indices(
    patient_ids: np.ndarray,
    fold: PatientFold,
    split: str = "test",
) -> np.ndarray:
    """
    Get spectrum indices belonging to a fold's train or test set.

    Parameters
    ----------
    patient_ids : np.ndarray
        Per-spectrum patient IDs for the full clinical dataset.
    fold : PatientFold
        The fold specification.
    split : str
        "train" or "test".

    Returns
    -------
    indices : np.ndarray of int
        Indices into the original array for this fold split.
    """
    if split == "test":
        target_patients = set(fold.test_patients)
    elif split == "train":
        target_patients = set(fold.train_patients)
    else:
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    patient_ids = np.asarray(patient_ids)
    mask = np.array([pid in target_patients for pid in patient_ids])
    return np.where(mask)[0]


def print_fold_summary(folds: list[PatientFold]) -> None:
    """Print a structured summary of patient fold assignments."""
    border = "=" * 60
    print(f"\n{border}")
    print("PATIENT CROSS-VALIDATION FOLDS")
    print("=" * 30)
    print(f"Number of Folds: {len(folds)}")

    for fold in folds:
        test_labels = sorted(set(
            patient_to_label(pid) for pid in fold.test_patients
        ))
        train_labels = sorted(set(
            patient_to_label(pid) for pid in fold.train_patients
        ))
        print(
            f"\n  Fold {fold.fold_index}: "
            f"train={len(fold.train_patients)} patients "
            f"(labels: {train_labels}), "
            f"test={len(fold.test_patients)} patients "
            f"(labels: {test_labels})"
        )
        for pid in fold.test_patients:
            print(f"    test: {pid}")

    print(f"\n{border}\n")