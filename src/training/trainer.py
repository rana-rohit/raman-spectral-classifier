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

        # Device resolution: CUDA > MPS > CPU
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

    # ------------------------------------------------------------------ #
    # Main fit loop
    # ------------------------------------------------------------------ #

    def fit(self) -> Dict:
        max_epochs = self.cfg.get("max_epochs", 100)

        for epoch in range(1, max_epochs + 1):
            train_metrics = self._train_one_epoch(epoch)
            self.logger.log(epoch, "train", train_metrics)

            val_metrics = self._eval_one_epoch(self.loaders["val"])
            self.logger.log(epoch, "val", val_metrics)

            val_acc = val_metrics["accuracy"]

            # Scheduler step
            sched = self.scheduler
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            if isinstance(sched, ReduceLROnPlateau):
                sched.step(val_acc)
            else:
                sched.step()

            # Checkpoint
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

            # Early stopping
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
            # Lower learning rate for finetuning
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=3e-4,
                weight_decay=1e-4
            )

            finetune_epochs = 10

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

            val_metrics = self._eval_one_epoch(self.loaders["val"])
            f1 = val_metrics.get("f1") or val_metrics.get("macro_f1")
            print(f"[Finetune] Post-adapt Val F1: {f1:.4f}")

        return self.logger.best

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        loader: DataLoader,
        split_name: str = "test",
        class_names: Optional[list] = None,
    ) -> Dict:
        """Run evaluation on any DataLoader and log final results."""
        metrics = self._eval_one_epoch(loader, return_preds=True)
        self.logger.log_final(split_name, metrics)
        return metrics

    def evaluate_ood(self) -> Dict[str, Dict]:
        """Evaluate on all OOD splits. Returns dict[split_name -> metrics]."""
        results = {}
        for ood_name, ood_loader in self.loaders.get("ood", {}).items():
            metrics = self._eval_one_epoch(ood_loader)
            self.logger.log_final(ood_name, metrics)
            results[ood_name] = metrics
        return results

    # ------------------------------------------------------------------ #
    # Internal loops
    # ------------------------------------------------------------------ #

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for x, y in self.loaders["train"]:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss   = self.loss_fn(logits, y)
            loss.backward()

            # Gradient clipping — essential for Transformer stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(dim=-1) == y).sum().item()
            total      += len(y)

        return {
            "loss":     total_loss / total,
            "accuracy": correct / total,
            "lr":       self.optimizer.param_groups[0]["lr"],
            "epoch_time": time.time() - t0,
        }

    @torch.no_grad()
    def _eval_one_epoch(
        self,
        loader: DataLoader,
        return_preds: bool = False,
    ) -> Dict[str, float]:
        self.model.eval()
        all_logits, all_targets = [], []

        for x, y in loader:
            x = x.to(self.device)
            logits = self.model(x)
            all_logits.append(logits.cpu())
            all_targets.append(y)

        all_logits  = torch.cat(all_logits,  dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        loss = self.loss_fn(all_logits, all_targets).item()
        metrics = compute_metrics(all_logits, all_targets, self.n_classes)
        metrics["loss"] = loss

        return metrics


# ------------------------------------------------------------------ #
# Factory
# ------------------------------------------------------------------ #

def build_trainer(
    model: nn.Module,
    loaders: Dict,
    cfg: dict,
    exp_dir: str,
    n_classes: int = 30,
) -> Trainer:
    """
    Build a fully configured Trainer from config dict.
    This is the function called by train.py.
    """
    train_cfg = cfg.get("training", {})
    model_name = cfg.get("model", {}).get("name", "model")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
    )

    scheduler = get_scheduler(
        name=train_cfg.get("scheduler", "cosine"),
        optimizer=optimizer,
        cfg=train_cfg.get("scheduler_cfg", {"T_max": train_cfg.get("max_epochs", 100)}),
    )

    loss_fn = get_loss(
        name=train_cfg.get("loss", "cross_entropy"),
        **train_cfg.get("loss_kwargs", {}),
    )

    logger = ExperimentLogger(
        exp_dir=exp_dir,
        model_name=model_name,
        config=cfg,
    )

    return Trainer(
        model=model,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        cfg=train_cfg,
        logger=logger,
        exp_dir=exp_dir,
        n_classes=n_classes,
    )