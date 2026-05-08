"""
metadata/tasks.py

Canonical task definitions for the Raman
spectral classification project.

Task semantics are derived from:
metadata/ontology.py
"""

from metadata.ontology import (
    CLINICAL_SPARSE_IDS,
    COMPACT_LABEL_MAP,
    INVERSE_COMPACT_LABEL_MAP,
    SEMANTIC_SPACES,
    TRANSFER_TASK,
)

# ============================================================
# TASK REGISTRY
# ============================================================

TASKS = {

    # --------------------------------------------------------
    # 30-class isolate identification
    # --------------------------------------------------------

    "isolate_30": {

        "semantic_space": "isolate_space",

        "n_classes": 30,

        "description": (
            "30-class isolate identification task used "
            "for spectral representation learning."
        ),
    },

    # --------------------------------------------------------
    # 8-class treatment grouping
    # --------------------------------------------------------

    "treatment_8": {

        "semantic_space": "global_treatment_space",

        "n_classes": 8,

        "description": (
            "8-class empiric antibiotic treatment grouping task."
        ),
    },

    # --------------------------------------------------------
    # 5-class clinical transfer task
    # --------------------------------------------------------

    "clinical_5_transfer": {

        "semantic_space": "compact_transfer_space",

        "n_classes": 5,

        "clinical_sparse_ids": CLINICAL_SPARSE_IDS,

        "compact_label_map": COMPACT_LABEL_MAP,

        "inverse_compact_map": INVERSE_COMPACT_LABEL_MAP,

        "description": (
            "Clinical transfer-learning task using "
            "compact remapped labels derived from "
            "clinical sparse treatment IDs."
        ),
    },
}

# ============================================================
# DEFAULT TASK
# ============================================================

DEFAULT_TASK = "clinical_5_transfer"

# ============================================================
# PUBLIC EXPORTS
# ============================================================

__all__ = [
    "TASKS",
    "DEFAULT_TASK",
    "SEMANTIC_SPACES",
    "TRANSFER_TASK",
]