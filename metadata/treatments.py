"""
metadata/treatments.py

Backward-compatible treatment metadata accessors.

Canonical ontology definitions now live in:
metadata/ontology.py

This module exists only to support legacy imports.
"""

from metadata.ontology import (
    GLOBAL_TREATMENTS,
    ISOLATE_TO_TREATMENT,
)

# ------------------------------------------------------------
# Legacy aliases
# ------------------------------------------------------------

# Old code may still import:
#
#   antibiotics
#   ATCC_GROUPINGS
#
# Keep aliases for backward compatibility.

antibiotics = GLOBAL_TREATMENTS

ATCC_GROUPINGS = ISOLATE_TO_TREATMENT