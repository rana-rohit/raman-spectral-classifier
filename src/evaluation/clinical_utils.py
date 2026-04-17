from __future__ import annotations

from typing import Sequence

import torch

from src.utils.class_subset import prepare_subset_eval_logits


DEFAULT_CLINICAL_CLASSES = (0, 2, 3, 5, 6)


def clinical_subset_eval(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_classes: Sequence[int] = DEFAULT_CLINICAL_CLASSES,
    aux_logits: torch.Tensor | None = None,
    aux_blend: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    return prepare_subset_eval_logits(
        main_logits=logits,
        targets=targets,
        class_ids=valid_classes,
        aux_logits=aux_logits,
        aux_blend=aux_blend,
    )
