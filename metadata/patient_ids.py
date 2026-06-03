"""
metadata/patient_ids.py

Deterministic patient identity generation for clinical datasets.

Clinical datasets are structured as contiguous blocks of spectra
per patient within each treatment class. This module reconstructs
patient IDs from the known dataset structure:

  2018clinical:
    5 classes × 2000 spectra/class
    5 patients per class × 400 spectra per patient
    25 patients total

  2019clinical:
    5 classes × 500 spectra/class
    5 patients per class × 100 spectra per patient
    25 patients total

Patient IDs are formatted as:
  {dataset}_{class_label}_p{patient_index}

Example: "2018clinical_0_p0" identifies the first patient
with treatment label 0 in the 2018 clinical dataset.

These IDs are deterministic and reproducible from dataset
ordering alone — no external metadata files required.
"""

from __future__ import annotations

import numpy as np

# ============================================================
# CLINICAL DATASET STRUCTURE
# ============================================================

CLINICAL_STRUCTURE = {
    "2018clinical": {
        "samples_per_class": 2000,
        "patients_per_class": 5,
        "spectra_per_patient": 400,
        "n_classes": 5,
        "total_patients": 25,
    },
    "2019clinical": {
        "samples_per_class": 500,
        "patients_per_class": 5,
        "spectra_per_patient": 100,
        "n_classes": 5,
        "total_patients": 25,
    },
}


def generate_patient_ids(
    y: np.ndarray,
    dataset_name: str,
) -> np.ndarray:
    """
    Generate deterministic patient IDs for a clinical dataset.

    Parameters
    ----------
    y : np.ndarray
        Label array (sparse global treatment IDs: [0,2,3,5,6]).
    dataset_name : str
        One of "2018clinical" or "2019clinical".

    Returns
    -------
    patient_ids : np.ndarray of str
        One patient ID per spectrum, same length as y.
    """
    if dataset_name not in CLINICAL_STRUCTURE:
        raise ValueError(
            f"Unknown clinical dataset: {dataset_name}. "
            f"Expected one of {list(CLINICAL_STRUCTURE.keys())}"
        )

    structure = CLINICAL_STRUCTURE[dataset_name]
    spectra_per_patient = structure["spectra_per_patient"]
    patients_per_class = structure["patients_per_class"]

    y = np.asarray(y)
    patient_ids = np.empty(len(y), dtype=object)

    for label in sorted(np.unique(y)):
        label_int = int(label)
        label_mask = y == label
        label_indices = np.where(label_mask)[0]

        n_spectra = len(label_indices)
        expected = patients_per_class * spectra_per_patient
        if n_spectra != expected:
            raise ValueError(
                f"{dataset_name} label {label_int}: expected "
                f"{expected} spectra, got {n_spectra}"
            )

        for p in range(patients_per_class):
            start = p * spectra_per_patient
            end = start + spectra_per_patient
            patient_indices = label_indices[start:end]
            pid = f"{dataset_name}_{label_int}_p{p}"
            patient_ids[patient_indices] = pid

    # Verify no missing assignments
    if any(pid is None for pid in patient_ids):
        raise RuntimeError(
            "Some spectra were not assigned a patient ID"
        )

    return patient_ids


def get_unique_patients(patient_ids: np.ndarray) -> list[str]:
    """Return sorted list of unique patient IDs."""
    return sorted(set(patient_ids.tolist()))


def patient_to_label(patient_id: str) -> int:
    """Extract the sparse treatment label from a patient ID string."""
    # Format: {dataset}_{label}_p{index}
    parts = patient_id.rsplit("_p", 1)
    label_part = parts[0].rsplit("_", 1)[1]
    return int(label_part)


def patient_to_dataset(patient_id: str) -> str:
    """Extract the dataset name from a patient ID string."""
    # Format: {dataset}_{label}_p{index}
    parts = patient_id.rsplit("_p", 1)
    dataset_and_label = parts[0]
    return dataset_and_label.rsplit("_", 1)[0]
