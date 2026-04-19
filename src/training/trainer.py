"""
src/training/trainer.py

Core training engine shared across all model families.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics
from src.training.losses import consistency_loss, get_loss
from src.training.regularizers import L2SPRegularizer
from src.training.scheduler import get_scheduler
from src.utils.checkpoint import save_checkpoint
from src.utils.class_subset import subset_mask, remap_targets_to_subset
from src.utils.logging import ExperimentLogger


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best = -float("inf")
        self._counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        if metric > self._best + self.min_delta:
            self._best = metric
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
        reference_state: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.train_cfg = cfg.get("training", {})
        self.logger = logger
        self.exp_dir = Path(exp_dir)
        self.n_classes = n_classes
        self.reference_state = reference_state

        self.monitor_metric = self.train_cfg.get("monitor_metric", "f1_macro")
        self.gradient_clip = self.train_cfg.get("gradient_clip", 1.0)

        self.aux_cfg = cfg.get("multitask", {}).get("auxiliary_shared_head", {})
        self.shared_class_ids = self.aux_cfg.get(
            "classes",
            cfg.get("dataset", {}).get("shared_classes", []),
        )
        self.aux_loss_weight = (
            self.aux_cfg.get("loss_weight", 0.0)
            if self.aux_cfg.get("enabled", False)
            else 0.0
        )

        self.consistency_cfg = self.train_cfg.get("consistency", {})
        self.consistency_enabled = self.consistency_cfg.get("enabled", False)
        self.consistency_weight = self.consistency_cfg.get("loss_weight", 0.0)
        self.supervised_on_both_views = self.consistency_cfg.get("supervised_on_both_views", True)

        self.l2sp = None
        l2sp_cfg = self.train_cfg.get("l2sp", {})
        if l2sp_cfg.get("enabled", False) and reference_state is not None:
            self.l2sp = L2SPRegularizer(
                reference_state=reference_state,
                lambda_=l2sp_cfg.get("lambda", 0.0),
                exclude_patterns=l2sp_cfg.get("exclude_patterns", []),
            )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)
        print(f"  Device: {self.device}")
        self.early_stopping = EarlyStopping(
            patience=self.train_cfg.get("early_stopping_patience", 10)
        )

    def fit(self) -> Dict:
        max_epochs = self.train_cfg.get("max_epochs", 100)

        for epoch in range(1, max_epochs + 1):
            train_metrics = self._train_one_epoch(epoch)
            self.logger.log(epoch, "train", train_metrics)

            val_metrics = self._eval_one_epoch(self.loaders["val"])
            self.logger.log(epoch, "val", val_metrics)

            monitor_value = val_metrics.get(self.monitor_metric)
            if monitor_value is None:
                raise KeyError(
                    f"Monitor metric '{self.monitor_metric}' not found in validation metrics. "
                    f"Available: {sorted(val_metrics.keys())}"
                )

            from torch.optim.lr_scheduler import ReduceLROnPlateau

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(monitor_value)
            else:
                self.scheduler.step()

            is_best = monitor_value >= self.early_stopping.best
            save_checkpoint(
                path=str(self.exp_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"),
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics={
                    **train_metrics,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    "monitor_metric": self.monitor_metric,
                    "monitor_value": monitor_value,
                },
                config=self.cfg,
                is_best=is_best,
            )

            if self.early_stopping.step(monitor_value):
                print(
                    f"\n  Early stopping at epoch {epoch} "
                    f"(best val_{self.monitor_metric}={self.early_stopping.best:.4f})"
                )
                break

        return self.logger.best

    def evaluate(
        self,
        loader: DataLoader,
        split_name: str = "test",
        class_names: Optional[list] = None,
    ) -> Dict:
        del class_names
        metrics = self._eval_one_epoch(loader)
        self.logger.log_final(split_name, metrics)
        return metrics

    def evaluate_ood(self) -> Dict[str, Dict]:
        results = {}
        for ood_name, ood_loader in self.loaders.get("ood", {}).items():
            metrics = self._eval_one_epoch(ood_loader)
            self.logger.log_final(ood_name, metrics)
            results[ood_name] = metrics
        return results

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        del epoch
        self.model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_aux_loss = 0.0
        total_consistency_loss = 0.0
        total_l2sp_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()
        all_logits, all_targets = [], []

        for batch in self.loaders["train"]:
            x1, x2, y = self._parse_batch(batch)
            self.optimizer.zero_grad(set_to_none=True)

            outputs1 = self._normalize_outputs(self.model(x1))
            outputs2 = None
            if x2 is not None:
                outputs2 = self._normalize_outputs(self.model(x2))

            main_loss1 = self.loss_fn(outputs1["main_logits"], y)
            aux_loss1 = self._compute_aux_loss(outputs1, y)

            if outputs2 is not None and self.supervised_on_both_views:
                main_loss2 = self.loss_fn(outputs2["main_logits"], y)
                aux_loss2 = self._compute_aux_loss(outputs2, y)
                main_loss = 0.5 * (main_loss1 + main_loss2)
                aux_loss = 0.5 * (aux_loss1 + aux_loss2)
            else:
                main_loss = main_loss1
                aux_loss = aux_loss1

            consistency_term = self._compute_consistency_loss(outputs1, outputs2, y)
            l2sp_term = self.l2sp(self.model) if self.l2sp is not None else self._zero_loss()

            loss = (
                main_loss
                + self.aux_loss_weight * aux_loss
                + self.consistency_weight * consistency_term
                + l2sp_term
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
            self.optimizer.step()

            batch_size = len(y)
            total_loss += loss.item() * batch_size
            total_main_loss += main_loss.item() * batch_size
            total_aux_loss += aux_loss.item() * batch_size
            total_consistency_loss += consistency_term.item() * batch_size
            total_l2sp_loss += l2sp_term.item() * batch_size
            correct += (outputs1["main_logits"].argmax(dim=-1) == y).sum().item()
            total += batch_size
            all_logits.append(outputs1["main_logits"].detach().cpu())
            all_targets.append(y.detach().cpu())

        metrics = compute_metrics(
            torch.cat(all_logits, dim=0),
            torch.cat(all_targets, dim=0),
            self.n_classes,
        )
        metrics.update(
            {
                "loss": total_loss / total,
                "main_loss": total_main_loss / total,
                "aux_loss": total_aux_loss / total,
                "consistency_loss": total_consistency_loss / total,
                "l2sp_loss": total_l2sp_loss / total,
                "accuracy": correct / total,
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_time": time.time() - t0,
            }
        )
        return metrics

    @torch.no_grad()
    def _eval_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_logits, all_targets = [], []

        for batch in loader:
            x, _, y = self._parse_batch(batch)
            outputs = self._normalize_outputs(self.model(x))
            all_logits.append(outputs["main_logits"].cpu())
            all_targets.append(y.cpu())

        logits = torch.cat(all_logits, dim=0)
        targets = torch.cat(all_targets, dim=0)
        loss = self.loss_fn(logits, targets).item()
        metrics = compute_metrics(logits, targets, self.n_classes)
        metrics["loss"] = loss
        return metrics

    def _normalize_outputs(self, outputs) -> dict[str, torch.Tensor | None]:
        if torch.is_tensor(outputs):
            return {
                "main_logits": outputs,
                "aux_logits": None,
                "features": None,
            }
        if isinstance(outputs, dict):
            return {
                "main_logits": outputs["main_logits"],
                "aux_logits": outputs.get("aux_logits"),
                "features": outputs.get("features"),
            }
        raise TypeError(f"Unsupported model output type: {type(outputs)!r}")

    def _parse_batch(
        self,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if isinstance(batch, dict):
            x1 = batch["x1"].to(self.device)
            x2 = batch.get("x2")
            x2 = x2.to(self.device) if x2 is not None else None
            y = batch["y"].to(self.device)
            return x1, x2, y

        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch

            # 🔥 Handle consistency case: x = (x1, x2)
            if isinstance(x, (tuple, list)) and len(x) == 2:
                x1, x2 = x
                return x1.to(self.device), x2.to(self.device), y.to(self.device)

            # 🔹 Standard case: single input
            return x.to(self.device), None, y.to(self.device)

        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    def _compute_aux_loss(
        self,
        outputs: dict[str, torch.Tensor | None],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        aux_logits = outputs.get("aux_logits")
        if aux_logits is None or not self.shared_class_ids:
            return self._zero_loss()

        mask = subset_mask(targets, self.shared_class_ids)
        if not mask.any():
            return self._zero_loss()

        aux_targets = remap_targets_to_subset(targets[mask], self.shared_class_ids)
        return self.loss_fn(aux_logits[mask], aux_targets)

    def _compute_consistency_loss(
        self,
        outputs1: dict[str, torch.Tensor | None],
        outputs2: dict[str, torch.Tensor | None] | None,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if not self.consistency_enabled or outputs2 is None:
            return self._zero_loss()

        loss_type = self.consistency_cfg.get("loss_type", "mse_probs")
        temperature = self.consistency_cfg.get("temperature", 1.0)
        total = consistency_loss(
            outputs1["main_logits"],
            outputs2["main_logits"],
            loss_type=loss_type,
            temperature=temperature,
        )

        if outputs1["aux_logits"] is not None and outputs2["aux_logits"] is not None:
            mask = subset_mask(targets, self.shared_class_ids)
            if mask.any():
                total = total + consistency_loss(
                    outputs1["aux_logits"][mask],
                    outputs2["aux_logits"][mask],
                    loss_type=loss_type,
                    temperature=temperature,
                )
        return total

    def _zero_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device)


def build_trainer(
    model: nn.Module,
    loaders: Dict,
    cfg: dict,
    exp_dir: str,
    n_classes: int = 30,
    reference_state: dict[str, torch.Tensor] | None = None,
) -> Trainer:
    train_cfg = cfg.get("training", {})
    model_name = cfg.get("model", {}).get("name", "model")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters were found for the requested phase.")

    optimizer = torch.optim.AdamW(
        trainable_params,
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
        cfg=cfg,
        logger=logger,
        exp_dir=exp_dir,
        n_classes=n_classes,
        reference_state=reference_state,
    )
