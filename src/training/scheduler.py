"""
src/training/scheduler.py

Learning rate schedulers.

- get_scheduler(): factory that returns the right scheduler from config
- WarmupCosineScheduler: linear warmup + cosine decay (required for Transformer)
"""

import math
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    ReduceLROnPlateau,
    _LRScheduler,
)


class WarmupCosineScheduler(_LRScheduler):
    """
    Linear LR warmup for `warmup_epochs`, then cosine annealing to `eta_min`.
    Transformers are sensitive to high LR at initialisation — warmup
    prevents the unstable early training behaviour.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.eta_min       = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            scale = (epoch + 1) / max(1, self.warmup_epochs)
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            scale = self.eta_min / self.base_lrs[0] + 0.5 * (
                1 - self.eta_min / self.base_lrs[0]
            ) * (1 + math.cos(math.pi * progress))

        return [base_lr * scale for base_lr in self.base_lrs]


def get_scheduler(name: str, optimizer, cfg: dict):
    """
    Factory — returns scheduler from config.

    cfg should contain scheduler-specific params, e.g.:
      cosine:    {T_max: 100, eta_min: 1e-6}
      step:      {step_size: 30, gamma: 0.1}
      plateau:   {patience: 5, factor: 0.5}
      warmup_cosine: {warmup_epochs: 10, total_epochs: 100}
    """
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("T_max", 100),
            eta_min=cfg.get("eta_min", 1e-6),
        )
    elif name == "step":
        return StepLR(
            optimizer,
            step_size=cfg.get("step_size", 30),
            gamma=cfg.get("gamma", 0.1),
        )
    elif name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",                    # We track val accuracy
            patience=cfg.get("patience", 5),
            factor=cfg.get("factor", 0.5),
            verbose=True,
        )
    elif name == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=cfg.get("warmup_epochs", 10),
            total_epochs=cfg.get("total_epochs", 100),
            eta_min=cfg.get("eta_min", 1e-6),
        )
    else:
        raise ValueError(f"Unknown scheduler '{name}'")