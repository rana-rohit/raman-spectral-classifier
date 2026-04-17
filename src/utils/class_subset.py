from __future__ import annotations

from typing import Sequence

import torch


def subset_mask(targets: torch.Tensor, class_ids: Sequence[int]) -> torch.Tensor:
    class_tensor = torch.tensor(
        [int(cls) for cls in class_ids],
        device=targets.device,
        dtype=targets.dtype,
    )
    return (targets.unsqueeze(-1) == class_tensor).any(dim=-1)


def remap_targets_to_subset(
    targets: torch.Tensor,
    class_ids: Sequence[int],
) -> torch.Tensor:
    class_ids = [int(cls) for cls in class_ids]
    mapped = torch.full_like(targets, fill_value=-1)
    for idx, cls in enumerate(class_ids):
        mapped = torch.where(
            targets == cls,
            torch.full_like(mapped, fill_value=idx),
            mapped,
        )
    if (mapped < 0).any():
        raise ValueError("Targets contain labels outside the requested subset.")
    return mapped.long()


def slice_logits_to_subset(
    logits: torch.Tensor,
    class_ids: Sequence[int],
) -> torch.Tensor:
    indices = torch.tensor(
        [int(cls) for cls in class_ids],
        device=logits.device,
        dtype=torch.long,
    )
    return logits.index_select(dim=-1, index=indices)


def prepare_subset_eval_logits(
    main_logits: torch.Tensor,
    targets: torch.Tensor,
    class_ids: Sequence[int],
    aux_logits: torch.Tensor | None = None,
    aux_blend: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare logits/targets for evaluation on a clinically valid class subset.

    ``main_logits`` always originate from the 30-class head. When an auxiliary
    shared-class head is present, its 5-class logits can be blended into the
    sliced main logits to bias evaluation toward the clinically relevant subset.
    """
    mask = subset_mask(targets, class_ids)
    filtered_targets = targets[mask]

    if filtered_targets.numel() == 0:
        empty_logits = main_logits.new_zeros((0, len(class_ids)))
        empty_targets = targets.new_zeros((0,), dtype=torch.long)
        return empty_logits, empty_targets

    subset_logits = slice_logits_to_subset(main_logits[mask], class_ids)
    if aux_logits is not None:
        filtered_aux = aux_logits[mask]
        if filtered_aux.size(-1) != len(class_ids):
            raise ValueError(
                "Auxiliary logits width does not match the requested subset size."
            )
        blend = float(aux_blend)
        subset_logits = (1.0 - blend) * subset_logits + blend * filtered_aux

    mapped_targets = remap_targets_to_subset(filtered_targets, class_ids)
    return subset_logits, mapped_targets
