"""
Split-mode configuration helpers.

The pipeline supports the original holdout evaluation path plus an IID
reference-only path used for standard Raman classification experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping

HOLDOUT = "holdout"
IID_REFERENCE = "iid_reference"
PATIENT_CV = "patient_cv"
VALID_SPLIT_MODES = {HOLDOUT, IID_REFERENCE, PATIENT_CV}


@dataclass(frozen=True)
class IIDReferenceSplitConfig:
    val_fraction: float
    test_groups: int
    spectra_per_group: int
    random_seed: int


def resolve_split_mode(cfg: Mapping[str, Any]) -> str:
    """Return the active split mode, defaulting to the legacy holdout path."""
    training_cfg = cfg.get("training", {}) or {}
    validation_cfg = cfg.get("validation", {}) or {}
    mode = (
        cfg.get("split_mode")
        or validation_cfg.get("split_mode")
        or training_cfg.get("split_mode")
        or HOLDOUT
    )
    mode = str(mode).strip().lower()
    if mode not in VALID_SPLIT_MODES:
        raise ValueError(
            f"Unknown split_mode={mode!r}. Expected one of: "
            f"{sorted(VALID_SPLIT_MODES)}"
        )
    return mode


def canonicalize_split_mode_config(
    cfg: MutableMapping[str, Any],
    split_mode: str | None = None,
) -> str:
    """
    Resolve and persist the active split mode in every runtime location.

    The top-level key is the source of truth after canonicalization, while
    training.split_mode is kept in sync for existing consumers and saved
    experiment configs.
    """
    if split_mode is not None:
        cfg["split_mode"] = split_mode

    mode = resolve_split_mode(cfg)
    cfg["split_mode"] = mode

    training_cfg = cfg.setdefault("training", {})
    training_cfg["split_mode"] = mode
    validation_cfg = cfg.setdefault("validation", {})
    validation_cfg["split_mode"] = mode
    return mode


def resolve_iid_reference_split_config(
    cfg: Mapping[str, Any],
) -> IIDReferenceSplitConfig:
    """Resolve group-aware IID reference split settings."""
    validation_cfg = cfg.get("validation", {}) or {}
    iid_cfg = validation_cfg.get("iid_reference", {}) or {}

    val_fraction = float(iid_cfg.get("val_fraction", 0.15))
    test_groups = int(iid_cfg.get("test_groups", 30))
    grouped_cfg = cfg.get("evaluation", {}).get("grouped", {}) or {}
    spectra_per_group_map = grouped_cfg.get("spectra_per_group", {}) or {}
    spectra_per_group = int(
        iid_cfg.get(
            "spectra_per_group",
            spectra_per_group_map.get("test", 100),
        )
    )
    random_seed = int(iid_cfg.get("random_seed", validation_cfg.get("random_seed", 42)))

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            "validation.iid_reference.val_fraction must be in (0, 1); "
            f"got {val_fraction}"
        )
    if test_groups <= 0:
        raise ValueError(
            "validation.iid_reference.test_groups must be positive; "
            f"got {test_groups}"
        )
    if spectra_per_group <= 0:
        raise ValueError(
            "validation.iid_reference.spectra_per_group must be positive; "
            f"got {spectra_per_group}"
        )

    return IIDReferenceSplitConfig(
        val_fraction=val_fraction,
        test_groups=test_groups,
        spectra_per_group=spectra_per_group,
        random_seed=random_seed,
    )
