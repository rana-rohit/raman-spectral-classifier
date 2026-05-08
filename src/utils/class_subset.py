"""
src/utils/class_subset.py

Ontology-aware utilities for:
- subset filtering
- sparse clinical label remapping
- compact transfer-space conversion

Semantic spaces:

1. Clinical sparse space:
   [0,2,3,5,6]

2. Compact transfer space:
   [0,1,2,3,4]

This module implements the transition between them.
"""

from __future__ import annotations

from typing import Sequence

import torch
import numpy as np
from metadata.ontology import (
    COMPACT_LABEL_MAP,
    INVERSE_COMPACT_LABEL_MAP,
)

def subset_mask(targets: torch.Tensor, class_ids: Sequence[int]) -> torch.Tensor:
    if not class_ids:
        return torch.zeros_like(targets, dtype=torch.bool)
    if targets.numel() > 0 and int(targets.min()) >= 0 and int(targets.max()) < len(class_ids):
        return torch.ones_like(targets, dtype=torch.bool)
    valid = torch.as_tensor(class_ids, dtype=targets.dtype, device=targets.device)
    return (targets[:, None] == valid[None, :]).any(dim=1)


def remap_targets_to_subset(
    targets: torch.Tensor,
    class_ids: Sequence[int],
) -> torch.Tensor:
    if not class_ids:
        return targets.long()

    # Compact transfer-space labels are already contiguous:
    # [0,1,2,3,4]
    # In this case no sparse->compact remapping is needed.
    if targets.numel() > 0 and int(targets.min()) >= 0 and int(targets.max()) < len(class_ids):
        return targets.long()

    mapping = {int(cls): idx for idx, cls in enumerate(class_ids)}
    unknown = set(
        targets.detach().cpu().tolist()
    ) - set(mapping.keys())

    if unknown:
        raise ValueError(
            f"Unknown sparse labels encountered: {sorted(unknown)}"
        )
    mapped = [mapping[int(label)] for label in targets.detach().cpu().tolist()]
    return torch.as_tensor(mapped, dtype=torch.long, device=targets.device)


def slice_logits_to_subset(
    logits: torch.Tensor,
    class_ids: Sequence[int],
) -> torch.Tensor:
    if logits.size(-1) == len(class_ids):
        return logits
    idx = torch.as_tensor(class_ids, dtype=torch.long, device=logits.device)
    return logits.index_select(dim=-1, index=idx)


def prepare_subset_eval_logits(
    main_logits: torch.Tensor,
    targets: torch.Tensor,
    class_ids: Sequence[int],
    aux_logits: torch.Tensor | None = None,
    aux_blend: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    main_logits = slice_logits_to_subset(main_logits, class_ids)
    targets = remap_targets_to_subset(targets, class_ids)
    if aux_logits is not None:
        if aux_logits.size(-1) != main_logits.size(-1):
            raise ValueError("Aux logits must match main logits size")

        blend = float(aux_blend)
        main_logits = (1.0 - blend) * main_logits + blend * aux_logits

    return main_logits, targets.long()

# ------------------------------------------------------------
# Sparse clinical-space -> compact transfer-space
#
# Example:
#
# sparse labels:
#   [0,2,3,5,6]
#
# compact labels:
#   [0,1,2,3,4]
#
# Used during:
# - transfer learning
# - finetuning
# - clinical OOD evaluation
# ------------------------------------------------------------
def filter_and_remap_classes(X, y, keep_classes):
    keep_classes = np.array(sorted(keep_classes))
    if len(np.unique(keep_classes)) != len(keep_classes):
        raise ValueError("Duplicate class IDs detected in keep_classes")
        
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

    if y_remapped.min() < 0 or y_remapped.max() >= len(keep_classes):
        raise ValueError(
            f"Remapped labels must be in [0, {len(keep_classes) - 1}], "
            f"got [{y_remapped.min()}, {y_remapped.max()}]"
        )

    return X_filtered, y_remapped


def class_maps(keep_classes):
    keep_classes = [int(cls) for cls in sorted(keep_classes)]
    if keep_classes == [0, 2, 3, 5, 6]:
        class_map = COMPACT_LABEL_MAP
        inverse_class_map = INVERSE_COMPACT_LABEL_MAP
    else:
        class_map = {
            cls: idx
            for idx, cls in enumerate(keep_classes)
        }
        inverse_class_map = {
            idx: cls
            for cls, idx in class_map.items()
        }
    return class_map, inverse_class_map
