"""
metadata/isolates.py

Backward-compatible isolate metadata accessors.

Canonical isolate ontology now lives in:
metadata/ontology.py

This module exists only to support legacy imports.
"""

from metadata.ontology import (
    ISOLATES,
    ISOLATE_TO_TREATMENT,
)

# ------------------------------------------------------------
# Legacy compatibility exports
# ------------------------------------------------------------

# Older code may still expect:
#
#   STRAINS
#   ATCC_GROUPINGS
#
# Preserve compatibility while routing all semantics
# through the canonical ontology layer.

STRAINS = {
    isolate_id: info["strain"]
    for isolate_id, info in ISOLATES.items()
}

ATCC_GROUPINGS = ISOLATE_TO_TREATMENT

# Original isolate ordering used in the Raman paper
# confusion matrices and visualization layouts.

ORDER = [
    16, 17, 14, 18, 15,
    20, 21,
    24, 23,
    26, 27, 28, 29, 25,
    6, 7, 5,
    3, 4,
    9, 10, 2, 8, 11, 22,
    19,
    12, 13,
    0, 1,
]