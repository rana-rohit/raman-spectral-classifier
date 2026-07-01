"""
metadata/ontology.py

Canonical ontology definitions for the Raman bacterial
classification and empiric antibiotic treatment project.

This file is the SINGLE SOURCE OF TRUTH for:

1. 30-class isolate-space
2. 8-class global treatment-space
3. Clinical sparse treatment-space
4. Compact transfer-training label-space

All loaders, evaluators, trainers, metrics,
and visualization utilities must import from here.

This ontology is reconstructed and VERIFIED from:
- original dataset metadata
- ATCC_GROUPINGS
- Raman paper confusion matrices
- clinical evaluation figures
- treatment-group mappings
"""

# ============================================================
# 30-CLASS ISOLATE SPACE
# ============================================================

# Canonical isolate IDs used during:
# - reference pretraining
# - 30-class isolate classification
# - spectral representation learning

ISOLATES = {

    0: {
        "strain": "C. albicans",
        "species": "Candida albicans",
        "group": "Yeast",
    },

    1: {
        "strain": "C. glabrata",
        "species": "Candida glabrata",
        "group": "Yeast",
    },

    2: {
        "strain": "K. aerogenes",
        "species": "Klebsiella aerogenes",
        "group": "Gram-negative",
    },

    3: {
        "strain": "E. coli 1",
        "species": "Escherichia coli",
        "group": "Gram-negative",
    },

    4: {
        "strain": "E. coli 2",
        "species": "Escherichia coli",
        "group": "Gram-negative",
    },

    5: {
        "strain": "E. faecium",
        "species": "Enterococcus faecium",
        "group": "Gram-positive",
    },

    6: {
        "strain": "E. faecalis 1",
        "species": "Enterococcus faecalis",
        "group": "Gram-positive",
    },

    7: {
        "strain": "E. faecalis 2",
        "species": "Enterococcus faecalis",
        "group": "Gram-positive",
    },

    8: {
        "strain": "E. cloacae",
        "species": "Enterobacter cloacae",
        "group": "Gram-negative",
    },

    9: {
        "strain": "K. pneumoniae 1",
        "species": "Klebsiella pneumoniae",
        "group": "Gram-negative",
    },

    10: {
        "strain": "K. pneumoniae 2",
        "species": "Klebsiella pneumoniae",
        "group": "Gram-negative",
    },

    11: {
        "strain": "P. mirabilis",
        "species": "Proteus mirabilis",
        "group": "Gram-negative",
    },

    12: {
        "strain": "P. aeruginosa 1",
        "species": "Pseudomonas aeruginosa",
        "group": "Gram-negative",
    },

    13: {
        "strain": "P. aeruginosa 2",
        "species": "Pseudomonas aeruginosa",
        "group": "Gram-negative",
    },

    14: {
        "strain": "MSSA 1",
        "species": "Methicillin-sensitive Staphylococcus aureus",
        "group": "Gram-positive",
    },

    15: {
        "strain": "MSSA 3",
        "species": "Methicillin-sensitive Staphylococcus aureus",
        "group": "Gram-positive",
    },

    16: {
        "strain": "MRSA 1 (isogenic)",
        "species": "Methicillin-resistant Staphylococcus aureus",
        "group": "Gram-positive",
    },

    17: {
        "strain": "MRSA 2",
        "species": "Methicillin-resistant Staphylococcus aureus",
        "group": "Gram-positive",
    },

    18: {
        "strain": "MSSA 2",
        "species": "Methicillin-sensitive Staphylococcus aureus",
        "group": "Gram-positive",
    },

    19: {
        "strain": "S. enterica",
        "species": "Salmonella enterica",
        "group": "Gram-negative",
    },

    20: {
        "strain": "S. epidermidis",
        "species": "Staphylococcus epidermidis",
        "group": "Gram-positive",
    },

    21: {
        "strain": "S. lugdunensis",
        "species": "Staphylococcus lugdunensis",
        "group": "Gram-positive",
    },

    22: {
        "strain": "S. marcescens",
        "species": "Serratia marcescens",
        "group": "Gram-negative",
    },

    23: {
        "strain": "S. pneumoniae 2",
        "species": "Streptococcus pneumoniae",
        "group": "Gram-positive",
    },

    24: {
        "strain": "S. pneumoniae 1",
        "species": "Streptococcus pneumoniae",
        "group": "Gram-positive",
    },

    25: {
        "strain": "S. sanguinis",
        "species": "Streptococcus sanguinis",
        "group": "Gram-positive",
    },

    26: {
        "strain": "Group A Strep.",
        "species": "Group A Streptococcus",
        "group": "Gram-positive",
    },

    27: {
        "strain": "Group B Strep.",
        "species": "Group B Streptococcus",
        "group": "Gram-positive",
    },

    28: {
        "strain": "Group C Strep.",
        "species": "Group C Streptococcus",
        "group": "Gram-positive",
    },

    29: {
        "strain": "Group G Strep.",
        "species": "Group G Streptococcus",
        "group": "Gram-positive",
    },
}

# ============================================================
# GLOBAL 8-CLASS TREATMENT SPACE
# ============================================================

# VERIFIED from:
# - original ATCC_GROUPINGS
# - papers confusion matrix 

GLOBAL_TREATMENTS = {

    0: "Meropenem",
    1: "Ciprofloxacin",
    2: "TZP",
    3: "Vancomycin",
    4: "Ceftriaxone",
    5: "Penicillin",
    6: "Daptomycin",
    7: "Caspofungin",
}

# ============================================================
# 30 ISOLATES -> 8 TREATMENT GROUPS
# ============================================================

# VERIFIED from:
# - original ATCC_GROUPINGS
# - paper confusion matrices
# - paper isolate grouping figure

ISOLATE_TO_TREATMENT = {

    # --------------------------------------------------------
    # 0 = Meropenem
    # Enterobacteriaceae
    # --------------------------------------------------------

    2: 0,   # K. aerogenes
    3: 0,   # E. coli 1
    4: 0,   # E. coli 2
    8: 0,   # E. cloacae
    9: 0,   # K. pneumoniae 1
    10: 0,  # K. pneumoniae 2
    11: 0,  # P. mirabilis
    22: 0,  # S. marcescens

    # --------------------------------------------------------
    # 1 = Ciprofloxacin
    # --------------------------------------------------------

    19: 1,  # S. enterica

    # --------------------------------------------------------
    # 2 = TZP
    # --------------------------------------------------------

    12: 2,  # P. aeruginosa 1
    13: 2,  # P. aeruginosa 2

    # --------------------------------------------------------
    # 3 = Vancomycin
    # Staphylococcus group
    # CORRECTED: includes 20, 21 (were wrongly in Ceftriaxone)
    # --------------------------------------------------------

    14: 3,  # MSSA 1
    15: 3,  # MSSA 3
    16: 3,  # MRSA 1 (isogenic)
    17: 3,  # MRSA 2
    18: 3,  # MSSA 2
    20: 3,  # S. epidermidis
    21: 3,  # S. lugdunensis

    # --------------------------------------------------------
    # 4 = Ceftriaxone
    # CORRECTED: 23, 24 restored here (were wrongly in Penicillin)
    # --------------------------------------------------------

    23: 4,  # S. pneumoniae 2
    24: 4,  # S. pneumoniae 1

    # --------------------------------------------------------
    # 5 = Penicillin
    # CORRECTED: 6,7,25,26,27,28,29 restored here
    # (were wrongly in Daptomycin)
    # --------------------------------------------------------

    6: 5,   # E. faecalis 1
    7: 5,   # E. faecalis 2
    25: 5,  # S. sanguinis
    26: 5,  # Group A Strep.
    27: 5,  # Group B Strep.
    28: 5,  # Group C Strep.
    29: 5,  # Group G Strep.

    # --------------------------------------------------------
    # 6 = Daptomycin
    # CORRECTED: only E. faecium (was wrongly grouped
    # with E. faecalis and Strep species)
    # --------------------------------------------------------

    5: 6,   # E. faecium

    # --------------------------------------------------------
    # 7 = Caspofungin
    # --------------------------------------------------------

    0: 7,   # C. albicans
    1: 7,   # C. glabrata
}

# ============================================================
# CLINICAL SPARSE TREATMENT SPACE
# ============================================================

# Clinical datasets use a sparse subset of GLOBAL treatment IDs:
#
# [0, 2, 3, 5, 6]
#
# These ARE global treatment IDs (same numbering as
# GLOBAL_TREATMENTS above). The clinical cohort simply
# does not include treatments 1, 4, or 7.
#
# Meaning:
# 0 = Meropenem
# 2 = TZP
# 3 = Vancomycin
# 5 = Penicillin
# 6 = Daptomycin
#
# CLINICAL_LABELS maps each global treatment ID to the
# primary clinical species encountered in the hospital
# datasets. These species are a SUBSET of the reference
# isolates within each treatment group.
#
# IMPORTANT: Verify against y_2018clinical.npy and
# y_2019clinical.npy if any doubt about label semantics.

CLINICAL_LABELS = {

    0: {
        "global_treatment": "Meropenem",
        "clinical_species": "E. coli",
        "note": "Ref group includes all Enterobacteriaceae",
    },

    2: {
        "global_treatment": "TZP",
        "clinical_species": "P. aeruginosa",
        "note": "Exact genus match with reference",
    },

    3: {
        "global_treatment": "Vancomycin",
        "clinical_species": "S. aureus",
        "note": "Ref group includes MSSA/MRSA + CoNS",
    },

    5: {
        "global_treatment": "Penicillin",
        "clinical_species": "E. faecalis",
        "note": "Ref group includes Strep spp + Enterococci",
    },

    6: {
        "global_treatment": "Daptomycin",
        "clinical_species": "E. faecium",
        "note": "Exact species match with reference",
    },
}

# Sparse global treatment IDs present in clinical datasets.

CLINICAL_SPARSE_IDS = [
    0,
    2,
    3,
    5,
    6,
]

# ============================================================
# COMPACT TRANSFER LABEL SPACE
# ============================================================

# During transfer learning,
# sparse global treatment IDs are remapped to contiguous IDs:
#
# Global [0, 2, 3, 5, 6]
# ->  Compact [0, 1, 2, 3, 4]
#
# Semantic alignment:
# Compact 0 = Global 0 = Meropenem
# Compact 1 = Global 2 = TZP
# Compact 2 = Global 3 = Vancomycin
# Compact 3 = Global 5 = Penicillin
# Compact 4 = Global 6 = Daptomycin

COMPACT_LABEL_MAP = {
    0: 0,  # Meropenem
    2: 1,  # TZP
    3: 2,  # Vancomycin
    5: 3,  # Penicillin
    6: 4,  # Daptomycin
}

INVERSE_COMPACT_LABEL_MAP = {
    0: 0,  # -> Meropenem
    1: 2,  # -> TZP
    2: 3,  # -> Vancomycin
    3: 5,  # -> Penicillin
    4: 6,  # -> Daptomycin
}

# ============================================================
# GLOBAL TREATMENT <-> COMPACT BRIDGE MAPPINGS
# ============================================================

# Maps global treatment IDs to compact transfer labels.
# Used when converting reference data (after isolate->treatment)
# into the compact space shared with clinical data.

GLOBAL_TO_COMPACT = dict(COMPACT_LABEL_MAP) # Same mapping

COMPACT_TO_GLOBAL = dict(INVERSE_COMPACT_LABEL_MAP) # Same inverse

# Set of global treatment IDs that have clinical representation.
CLINICAL_GLOBAL_TREATMENT_IDS = frozenset(CLINICAL_SPARSE_IDS)

from collections import defaultdict

TREATMENT_TO_ISOLATES = defaultdict(list)

for isolate_id, treatment_id in ISOLATE_TO_TREATMENT.items():
    TREATMENT_TO_ISOLATES[treatment_id].append(isolate_id)

TREATMENT_TO_ISOLATES = {
    k: tuple(sorted(v))
    for k, v in TREATMENT_TO_ISOLATES.items()
}
# ============================================================
# SEMANTIC SPACE DEFINITIONS
# ============================================================

SEMANTIC_SPACES = {

    "isolate_space": {
        "n_classes": 30,
        "description": "30-class isolate identification space",
    },

    "global_treatment_space": {
        "n_classes": 8,
        "description": "8-class empiric antibiotic treatment grouping",
    },

    "sparse_global_treatment_space": {
        "n_classes": 5,
        "description": (
            "Sparse subset of global treatment IDs [0,2,3,5,6] "
            "present in clinical hospital datasets"
        ),
    },

    "compact_transfer_space": {
        "n_classes": 5,
        "description": (
            "Contiguous [0..4] internal label space used during "
            "transfer training and evaluation. Maps 1:1 with "
            "sparse global treatment space."
        ),
    },
}

# ============================================================
# TRANSFER TASK CONFIGURATION
# ============================================================

TRANSFER_TASK = {

    "reference_input_space": "isolate_space",

    "pretraining_output_space": "global_treatment_space",

    "transfer_output_space": "compact_transfer_space",

    "clinical_sparse_ids": CLINICAL_SPARSE_IDS,

    "compact_map": COMPACT_LABEL_MAP,

    "inverse_compact_map": INVERSE_COMPACT_LABEL_MAP,
}

TREATMENT_PRETRAINING_TASK = {

    "input_space": "isolate_space",

    "output_space": "global_treatment_space",

    "n_classes": 8,

    "mapping": ISOLATE_TO_TREATMENT,
}

# ============================================================
# ONTOLOGY INTEGRITY CHECKS
# ============================================================

assert len(ISOLATES) == 30

assert set(GLOBAL_TREATMENTS.keys()) == set(range(8))

assert set(CLINICAL_SPARSE_IDS) == {0, 2, 3, 5, 6}

assert set(COMPACT_LABEL_MAP.values()) == set(range(5))

assert set(INVERSE_COMPACT_LABEL_MAP.keys()) == set(range(5))

assert all(
    isolate_id in ISOLATES
    for isolate_id in ISOLATE_TO_TREATMENT
)

assert all(
    treatment_id in GLOBAL_TREATMENTS
    for treatment_id in ISOLATE_TO_TREATMENT.values()
)

assert set(GLOBAL_TO_COMPACT.keys()) == set(CLINICAL_SPARSE_IDS)

assert set(COMPACT_TO_GLOBAL.values()) == set(CLINICAL_SPARSE_IDS)

assert set(TREATMENT_TO_ISOLATES.keys()) == set(range(8))

ONTOLOGY_VERSION = "v2_verified"