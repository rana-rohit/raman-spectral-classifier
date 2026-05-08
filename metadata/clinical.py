"""
Clinical validation metadata.

Clinical datasets use treatment-group indices,
NOT original isolate IDs.
"""

# CLINICAL TASK LABELS

CLINICAL_LABELS = {

    0: {
        "species": "S. aureus",
        "treatment": "Vancomycin"
    },

    2: {
        "species": "E. faecalis",
        "treatment": "Penicillin"
    },

    3: {
        "species": "E. faecium",
        "treatment": "Daptomycin"
    },

    5: {
        "species": "E. coli",
        "treatment": "Ciprofloxacin"
    },

    6: {
        "species": "P. aeruginosa",
        "treatment": "TZP"
    }
}

# Compact training labels -> original clinical labels
# Used after dataloader remapping for semantic restoration

CLINICAL_LABEL_INVERSE_REMAP = {
    0: 0,   # S. aureus
    1: 2,   # E. faecalis
    2: 3,   # E. faecium
    3: 5,   # E. coli
    4: 6,   # P. aeruginosa
}