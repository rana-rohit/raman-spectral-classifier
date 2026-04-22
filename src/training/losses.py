"""
src/training/losses.py

Loss helpers for spectral classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(
                log_probs,
                self.smoothing / (n_classes - 1),
            )
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def coral_loss(source, target):
    """
    CORAL loss: aligns covariance of source and target features
    source: (N, D)
    target: (M, D)
    """

    # ensure enough samples
    if source.size(0) < 2 or target.size(0) < 2:
        return source.new_tensor(0.0)

    # center features
    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)

    # covariance with safe denominator
    cov_s = (source.T @ source) / (source.size(0) - 1)
    cov_t = (target.T @ target) / (target.size(0) - 1)

    # Frobenius norm
    loss = torch.mean((cov_s - cov_t) ** 2) / (source.size(1) ** 2)
    return loss
    
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    loss_type: str = "mse_probs",
    temperature: float = 1.0,
) -> torch.Tensor:
    if logits_a.numel() == 0 or logits_b.numel() == 0:
        return logits_a.new_tensor(0.0)

    probs_a = torch.softmax(logits_a / temperature, dim=-1)
    probs_b = torch.softmax(logits_b / temperature, dim=-1)

    if loss_type == "mse_probs":
        return torch.mean((probs_a - probs_b) ** 2)
    if loss_type == "kl_probs":
        return 0.5 * (
            F.kl_div(probs_a.log(), probs_b, reduction="batchmean")
            + F.kl_div(probs_b.log(), probs_a, reduction="batchmean")
        )
    raise ValueError(f"Unknown consistency loss '{loss_type}'")


def get_loss(name: str, **kwargs) -> nn.Module:
    registry = {
        "cross_entropy": nn.CrossEntropyLoss,
        "label_smoothing": LabelSmoothingCrossEntropy,
        "focal": FocalLoss,
    }
    if name not in registry:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(registry)}")
    return registry[name](**kwargs)
