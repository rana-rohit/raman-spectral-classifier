"""
Helper functions for metadata access.
"""

from .isolates import STRAINS
from .treatments import TREATMENTS
from .clinical import CLINICAL_LABELS


# ISOLATE HELPERS

def get_strain_name(label_id):

    return STRAINS[label_id]


# TREATMENT HELPERS

def get_treatment_name(label_id):

    return TREATMENTS[label_id]


# CLINICAL HELPERS

def get_clinical_species(label_id):

    return CLINICAL_LABELS[label_id]["species"]


def get_clinical_treatment(label_id):

    return CLINICAL_LABELS[label_id]["treatment"]