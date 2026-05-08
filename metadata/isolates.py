"""
Isolate-level metadata.

This file defines:
- 30 isolate IDs
- strain names
- biological categories
- display ordering
"""

# DISPLAY ORDER

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
    0, 1
]


# STRAIN NAMES

STRAINS = {

    0: "C. albicans",
    1: "C. glabrata",

    2: "K. aerogenes",

    3: "E. coli 1",
    4: "E. coli 2",

    5: "E. faecium",

    6: "E. faecalis 1",
    7: "E. faecalis 2",

    8: "E. cloacae",

    9: "K. pneumoniae 1",
    10: "K. pneumoniae 2",

    11: "P. mirabilis",

    12: "P. aeruginosa 1",
    13: "P. aeruginosa 2",

    14: "MSSA 1",
    15: "MSSA 3",

    16: "MRSA 1 (isogenic)",
    17: "MRSA 2",

    18: "MSSA 2",

    19: "S. enterica",

    20: "S. epidermidis",
    21: "S. lugdunensis",

    22: "S. marcescens",

    23: "S. pneumoniae 2",
    24: "S. pneumoniae 1",

    25: "S. sanguinis",

    26: "Group A Strep.",
    27: "Group B Strep.",
    28: "Group C Strep.",
    29: "Group G Strep."
}


# BIOLOGICAL CATEGORIES

BIOLOGICAL_CATEGORIES = {

    "Gram-negative": [
        2, 3, 4, 8, 9, 10,
        11, 12, 13, 19, 22
    ],

    "Gram-positive": [
        5, 6, 7,
        14, 15, 16, 17, 18,
        20, 21,
        23, 24, 25,
        26, 27, 28, 29
    ],

    "Yeast": [
        0, 1
    ]
}