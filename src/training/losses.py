"""
src/training/losses.py

Loss helpers for spectral classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.register_buffer(
            "weight",
            weight.float() if weight is not None else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(
                log_probs, self.smoothing / max(1, (n_classes - 1))
            )
            smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)
        if self.weight is not None:
            sample_weight = self.weight.gather(0, targets)
            loss = loss * sample_weight
        else:
            sample_weight = None

        if self.reduction == "mean":
            if sample_weight is not None:
                return loss.sum() / sample_weight.sum().clamp_min(1e-8)
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
    loss = torch.mean((cov_s - cov_t) ** 2)
    return loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        reduction: str = "mean",
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer(
            "weight",
            weight.float() if weight is not None else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss.detach())
        focal = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss

    Expected input:
        features: (B, V, D) or (B, D)
            B = batch size
            V = number of views
            D = embedding dimension

        labels: (B,)
    """

    def __init__(
        self,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        device = features.device

        # If features is (B, D), treat V as 1
        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        B, V, D = features.shape

        features = F.normalize(features, dim=-1)
        features = features.view(B * V, D)

        labels = labels.repeat_interleave(V)
        sim_matrix = torch.matmul(features, features.T)
        sim_matrix = sim_matrix / self.temperature
        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(B * V, device=device)

        mask = mask * logits_mask

        exp_logits = torch.exp(sim_matrix) * logits_mask

        log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Only compute average over samples that have at least one positive pair in the batch
        mask_pos_count = mask.sum(dim=1)
        pos_mask = mask_pos_count > 0
        if pos_mask.sum() == 0:
            return features.new_tensor(0.0, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1)[pos_mask] / mask_pos_count[
            pos_mask
        ]
        loss = -mean_log_prob_pos.mean()

        return loss


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
            F.kl_div(torch.log(probs_a + 1e-8), probs_b, reduction="batchmean")
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
