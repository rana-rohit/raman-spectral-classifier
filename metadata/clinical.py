"""
metadata/clinical.py

Backward-compatible clinical metadata accessors.

Canonical semantics now live in:
metadata/ontology.py
"""

from metadata.ontology import (
    CLINICAL_LABELS,
    CLINICAL_SPARSE_IDS,
    COMPACT_LABEL_MAP,
    INVERSE_COMPACT_LABEL_MAP,
)

# Backward compatibility aliases

CLINICAL_LABEL_INVERSE_REMAP = INVERSE_COMPACT_LABEL_MAP