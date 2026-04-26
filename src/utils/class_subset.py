from __future__ import annotations

from typing import Sequence

import torch


def subset_mask(targets: torch.Tensor, class_ids: Sequence[int]) -> torch.Tensor:
    """
    Deprecated in 5-class setup.
    """
    return torch.ones_like(targets, dtype=torch.bool)


def remap_targets_to_subset(
    targets: torch.Tensor,
    class_ids: Sequence[int],
) -> torch.Tensor:
    """
    No-op in 5-class setup (targets already remapped).
    """
    return targets.long()


def slice_logits_to_subset(
    logits: torch.Tensor,
    class_ids: Sequence[int],
) -> torch.Tensor:
    """
    No-op in 5-class setup (kept for compatibility).
    """
    return logits


def prepare_subset_eval_logits(
    main_logits: torch.Tensor,
    targets: torch.Tensor,
    class_ids: Sequence[int],
    aux_logits: torch.Tensor | None = None,
    aux_blend: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified for 5-class training setup.

    Assumes:
    - model outputs already match class space
    - targets already remapped to [0..N-1]
    """

    if aux_logits is not None:
        if aux_logits.size(-1) != main_logits.size(-1):
            raise ValueError("Aux logits must match main logits size")

        blend = float(aux_blend)
        main_logits = (1.0 - blend) * main_logits + blend * aux_logits

    return main_logits, targets.long()

import numpy as np

def filter_and_remap_classes(X, y, keep_classes):
    keep_classes = np.array(sorted(keep_classes))

    mask = np.isin(y, keep_classes)

    if mask.shape[0] != X.shape[0]:
        raise ValueError("Mask and data size mismatch")

    X_filtered = X[mask]
    y_filtered = y[mask].astype(int)

    if X_filtered.shape[0] == 0:
        raise ValueError("No samples found for selected classes")

    class_map = {cls: i for i, cls in enumerate(keep_classes)}

    try:
        y_remapped = np.fromiter((class_map[label] for label in y_filtered), dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"Unexpected label encountered during remapping: {e}")

    return X_filtered, y_remapped