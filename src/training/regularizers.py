from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class L2SPRegularizer:
    """
    L2-SP regularization anchors finetuning weights to the pretrained solution.
    """

    def __init__(
        self,
        reference_state: dict[str, torch.Tensor],
        lambda_: float,
        exclude_patterns: Sequence[str] | None = None,
    ) -> None:
        self.reference_state = reference_state
        self.lambda_ = float(lambda_)
        self.exclude_patterns = list(exclude_patterns or [])

    def __call__(self, model: nn.Module) -> torch.Tensor:
        penalty = None
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.reference_state:
                continue
            if any(pattern in name for pattern in self.exclude_patterns):
                continue

            reference = self.reference_state[name].to(param.device)
            value = torch.mean((param - reference) ** 2)
            penalty = value if penalty is None else penalty + value

        if penalty is None:
            first_param = next(model.parameters())
            penalty = torch.tensor(0.0, device=first_param.device)
        return self.lambda_ * penalty
