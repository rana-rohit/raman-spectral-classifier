"""
Task definitions for the project.
"""

from .isolates import STRAINS
from .treatments import TREATMENTS
from .clinical import CLINICAL_LABELS


TASKS = {

    "isolate_30": {

        "num_classes": 30,
        "label_space": STRAINS
    },

    "treatment_8": {

        "num_classes": 8,
        "label_space": TREATMENTS
    },

    "clinical_5": {

        "num_classes": 5,
        "label_space": CLINICAL_LABELS
    }
}