"""
src/training/losses.py

Loss functions for spectral classification.

- StandardCrossEntropy: baseline, well-understood
- LabelSmoothingCrossEntropy: prevents overconfidence, used for Transformer
- FocalLoss: down-weights easy samples (useful if classes vary in difficulty)

All losses accept (logits, targets) with shapes (B, C) and (B,).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.
    smoothing=0.0 is identical to standard cross-entropy.
    smoothing=0.1 is standard for Transformer training.

    Reference: "Rethinking the Inception Architecture" (Szegedy et al.)
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.smoothing  = smoothing
        self.reduction  = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth targets: (1 - eps) for true class, eps/(C-1) for others
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss for hard-example mining.
    gamma=0 reduces to cross-entropy. gamma=2 is standard.
    Useful if some of the 30 classes are consistently harder to separate.
    """

    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt      = torch.exp(-ce_loss)
        focal   = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


def get_loss(name: str, **kwargs) -> nn.Module:
    """Factory — returns loss by config name."""
    registry = {
        "cross_entropy":           nn.CrossEntropyLoss,
        "label_smoothing":         LabelSmoothingCrossEntropy,
        "focal":                   FocalLoss,
    }
    if name not in registry:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(registry)}")
    return registry[name](**kwargs)