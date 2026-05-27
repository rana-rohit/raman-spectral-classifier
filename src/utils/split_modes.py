"""
Split-mode configuration helpers.

The pipeline supports the original holdout evaluation path plus an IID
reference-only path used for standard Raman classification experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


HOLDOUT = "holdout"
IID_REFERENCE = "iid_reference"
VALID_SPLIT_MODES = {HOLDOUT, IID_REFERENCE}


@dataclass(frozen=True)
class IIDReferenceSplitConfig:
    train_fraction: float
    val_fraction: float
    test_fraction: float
    random_seed: int


def resolve_split_mode(cfg: Mapping[str, Any]) -> str:
    """Return the active split mode, defaulting to the legacy holdout path."""
    training_cfg = cfg.get("training", {}) or {}
    validation_cfg = cfg.get("validation", {}) or {}
    mode = (
        cfg.get("split_mode")
        or training_cfg.get("split_mode")
        or validation_cfg.get("split_mode")
        or HOLDOUT
    )
    mode = str(mode).strip().lower()
    if mode not in VALID_SPLIT_MODES:
        raise ValueError(
            f"Unknown split_mode={mode!r}. Expected one of: "
            f"{sorted(VALID_SPLIT_MODES)}"
        )
    return mode


def resolve_iid_reference_split_config(cfg: Mapping[str, Any]) -> IIDReferenceSplitConfig:
    """Resolve IID reference split fractions and seed from config."""
    validation_cfg = cfg.get("validation", {}) or {}
    iid_cfg = validation_cfg.get("iid_reference", {}) or {}

    val_fraction = float(iid_cfg.get("val_fraction", 0.15))
    test_fraction = float(iid_cfg.get("test_fraction", 0.15))
    train_fraction = float(
        iid_cfg.get("train_fraction", 1.0 - val_fraction - test_fraction)
    )
    random_seed = int(iid_cfg.get("random_seed", validation_cfg.get("random_seed", 42)))

    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "validation.iid_reference split fractions must sum to 1.0; "
            f"got train={train_fraction}, val={val_fraction}, "
            f"test={test_fraction}, total={total}"
        )
    if min(train_fraction, val_fraction, test_fraction) <= 0.0:
        raise ValueError(
            "validation.iid_reference split fractions must all be positive; "
            f"got train={train_fraction}, val={val_fraction}, test={test_fraction}"
        )

    return IIDReferenceSplitConfig(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        random_seed=random_seed,
    )

