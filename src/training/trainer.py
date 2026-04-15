"""
src/training/trainer.py

Core training engine. Handles the full training loop for all model types.

Features:
- Train + validation loops with metric logging
- Early stopping on val accuracy (configurable patience)
- Checkpoint saving (latest + best)
- Device-agnostic (CPU / CUDA / MPS)
- Structured logging via ExperimentLogger

Usage:
    trainer = Trainer(model, loaders, optimizer, scheduler, loss_fn, cfg, logger)
    trainer.fit()
    results = trainer.evaluate(loaders["test"], split_name="test")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.losses import get_loss
from src.training.scheduler import get_scheduler
from src.utils.checkpoint import save_checkpoint
from src.utils.logging import ExperimentLogger
from src.evaluation.metrics import compute_metrics


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self._best      = -float("inf")
        self._counter   = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        if metric > self._best + self.min_delta:
            self._best   = metric
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    @property
    def best(self) -> float:
        return self._best


class Trainer:
    """
    Encapsulates the full training lifecycle for one model.
    """

    def __init__(
        self,
        model: nn.Module,
        loaders: Dict[str, DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: nn.Module,
        cfg: dict,
        logger: ExperimentLogger,
        exp_dir: str,
        n_classes: int = 30,
    ) -> None:
        self.model      = model
        self.loaders    = loaders
        self.optimizer  = optimizer
        self.scheduler  = scheduler
        self.loss_fn    = loss_fn
        self.cfg        = cfg
        self.logger     = logger
        self.exp_dir    = Path(exp_dir)
        self.n_classes  = n_classes

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        print(f"  Device: {self.device}")

        self.early_stopping = EarlyStopping(
            patience=cfg.get("early_stopping_patience", 10)
        )

    def fit(self) -> Dict:
        max_epochs = self.cfg.get("max_epochs", 100)

        for epoch in range(1, max_epochs + 1):
            train_metrics = self._train_one_epoch(epoch)
            self.logger.log(epoch, "train", train_metrics)

            val_metrics = self._eval_one_epoch(self.loaders["val"])
            self.logger.log(epoch, "val", val_metrics)

            val_acc = val_metrics["accuracy"]

            sched = self.scheduler
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            if isinstance(sched, ReduceLROnPlateau):
                sched.step(val_acc)
            else:
                sched.step()

            is_best = val_acc >= self.early_stopping.best
            save_checkpoint(
                path=str(self.exp_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"),
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                config=self.cfg,
                is_best=is_best,
            )

            if self.early_stopping.step(val_acc):
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best val_acc={self.early_stopping.best:.4f})")
                break

        # =========================
        # FINETUNE PHASE
        # =========================

        print("\n[Finetune Phase] Adapting model to new domain...")

        finetune_loader = self.loaders.get("finetune", None)

        if finetune_loader is not None:

            for name, param in self.model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=1e-4,
                weight_decay=1e-4
            )

            finetune_epochs = 30

            for epoch in range(1, finetune_epochs + 1):
                self.model.train()
                total_loss = 0.0

                for x, y in finetune_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    self.optimizer.zero_grad(set_to_none=True)
                    logits = self.model(x)
                    loss = self.loss_fn(logits, y)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    total_loss += loss.item()

                print(f"[Finetune] Epoch {epoch}/{finetune_epochs} | Loss: {total_loss:.4f}")

                for g in self.optimizer.param_groups:
                    g["lr"] *= 0.95

            self.model.eval()
            val_metrics = self._eval_one_epoch(self.loaders["finetune"])
            f1 = val_metrics.get("f1_macro", val_metrics.get("f1", 0.0))
            print(f"[Finetune] Post-adapt Val F1: {f1:.4f}")

        return self.logger.best