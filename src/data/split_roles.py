"""
src/data/split_roles.py
Defines the semantic role of each dataset split.
Roles control what operations are permitted on each split.
"""

from enum import Enum, auto


class SplitRole(Enum):
    SOURCE     = auto()   # Primary supervised learning (reference)
    ADAPTATION = auto()   # Fine-tuning / domain bridging (finetune)
    HOLDOUT    = auto()   # Final evaluation only — touch once per model (test)
    OOD_EVAL   = auto()   # Out-of-distribution generalisation (clinical splits)
    VALIDATION = auto()   # Carved from SOURCE at runtime — never saved to disk


ROLE_PERMISSIONS = {
    SplitRole.SOURCE:     {"train", "validate", "inspect"},
    SplitRole.ADAPTATION: {"train", "inspect"},
    SplitRole.HOLDOUT:    {"evaluate"},
    SplitRole.OOD_EVAL:   {"evaluate", "inspect"},
    SplitRole.VALIDATION: {"validate", "inspect"},
}


def role_from_str(s: str) -> SplitRole:
    mapping = {
        "source":     SplitRole.SOURCE,
        "adaptation": SplitRole.ADAPTATION,
        "holdout":    SplitRole.HOLDOUT,
        "ood_eval":   SplitRole.OOD_EVAL,
        "validation": SplitRole.VALIDATION,
    }
    if s not in mapping:
        raise ValueError(f"Unknown split role '{s}'. Valid: {list(mapping)}")
    return mapping[s]