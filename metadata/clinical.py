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