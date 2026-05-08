"""
Treatment-group metadata.

Defines:
- isolate -> treatment mapping
- treatment names
- semantic treatment groups
"""

from .isolates import STRAINS


# ISOLATE -> TREATMENT GROUP

ATCC_GROUPINGS = {

    3: 0,
    4: 0,
    9: 0,
    10: 0,
    2: 0,
    8: 0,
    11: 0,
    22: 0,

    19: 1,

    12: 2,
    13: 2,

    14: 3,
    15: 3,
    16: 3,
    17: 3,
    18: 3,
    20: 3,
    21: 3,

    23: 4,
    24: 4,

    6: 5,
    7: 5,
    25: 5,
    26: 5,
    27: 5,
    28: 5,
    29: 5,

    5: 6,

    0: 7,
    1: 7
}


# TREATMENT LABELS

TREATMENTS = {

    0: "Meropenem",
    1: "Ciprofloxacin",
    2: "TZP",
    3: "Vancomycin",
    4: "Ceftriaxone",
    5: "Penicillin",
    6: "Daptomycin",
    7: "Caspofungin"
}


# TREATMENT GROUP SEMANTICS

SEMANTIC_GROUPS = {

    treatment_id: {
        "treatment": treatment_name,
        "isolates": [

            isolate_id
            for isolate_id, group_id
            in ATCC_GROUPINGS.items()

            if group_id == treatment_id
        ],

        "strain_names": [

            STRAINS[isolate_id]

            for isolate_id, group_id
            in ATCC_GROUPINGS.items()

            if group_id == treatment_id
        ]
    }

    for treatment_id, treatment_name
    in TREATMENTS.items()
}