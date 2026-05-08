"""
metadata/helpers.py

Shared helper utilities for ontology-aware
label and semantic-space operations.

All canonical semantics originate from:
metadata/ontology.py
"""

from metadata.ontology import (
    ISOLATES,
    GLOBAL_TREATMENTS,
    ISOLATE_TO_TREATMENT,
    CLINICAL_LABELS,
    CLINICAL_SPARSE_IDS,
    COMPACT_LABEL_MAP,
    INVERSE_COMPACT_LABEL_MAP,
)

# ============================================================
# ISOLATE HELPERS
# ============================================================

def get_isolate_name(isolate_id: int) -> str:
    """
    Return human-readable isolate strain name.
    """

    if isolate_id not in ISOLATES:
        raise KeyError(f"Unknown isolate ID: {isolate_id}")

    return ISOLATES[isolate_id]["strain"]


def get_species_name(isolate_id: int) -> str:
    """
    Return species name for isolate ID.
    """

    if isolate_id not in ISOLATES:
        raise KeyError(f"Unknown isolate ID: {isolate_id}")

    return ISOLATES[isolate_id]["species"]


# ============================================================
# TREATMENT HELPERS
# ============================================================

def isolate_to_treatment(isolate_id: int) -> int:
    """
    Map isolate ID -> global treatment ID.
    """

    if isolate_id not in ISOLATE_TO_TREATMENT:
        raise KeyError(
            f"No treatment mapping found for isolate ID {isolate_id}"
        )

    return ISOLATE_TO_TREATMENT[isolate_id]


def get_treatment_name(treatment_id: int) -> str:
    """
    Return treatment name from global treatment ID.
    """

    if treatment_id not in GLOBAL_TREATMENTS:
        raise KeyError(f"Unknown treatment ID: {treatment_id}")

    return GLOBAL_TREATMENTS[treatment_id]


# ============================================================
# CLINICAL HELPERS
# ============================================================

def is_clinical_sparse_label(label: int) -> bool:
    """
    Check whether label belongs to the
    clinical sparse-ID subset.
    """

    return label in CLINICAL_SPARSE_IDS


def sparse_to_compact(label: int) -> int:
    """
    Convert sparse clinical label ->
    compact contiguous transfer label.
    """

    if label not in COMPACT_LABEL_MAP:
        raise KeyError(
            f"Unknown sparse clinical label: {label}"
        )

    return COMPACT_LABEL_MAP[label]


def compact_to_sparse(label: int) -> int:
    """
    Convert compact transfer label ->
    sparse clinical label.
    """

    if label not in INVERSE_COMPACT_LABEL_MAP:
        raise KeyError(
            f"Unknown compact label: {label}"
        )

    return INVERSE_COMPACT_LABEL_MAP[label]


def sparse_label_to_clinical_info(label: int) -> dict:
    """
    Return clinical metadata associated
    with sparse clinical label.
    """

    if label not in CLINICAL_LABELS:
        raise KeyError(
            f"Unknown clinical sparse label: {label}"
        )

    return CLINICAL_LABELS[label]