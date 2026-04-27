"""
src/training/trainer.py

Core training engine shared across all model families.
"""

from __future__ import annotations

import time
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Function

from src.evaluation.metrics import compute_metrics
from src.training.losses import consistency_loss, get_loss, coral_loss
from src.training.regularizers import L2SPRegularizer
from src.training.scheduler import get_scheduler
from src.utils.checkpoint import save_checkpoint
from src.utils.class_subset import subset_mask, remap_targets_to_subset
from src.utils.logging import ExperimentLogger
from src.data.augmentation import AugmentationPipeline
from itertools import cycle

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

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

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
        aug_cfg = cfg.get("augmentation", {})
        self.augment = AugmentationPipeline.from_config(aug_cfg)
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
        self.dann_cfg = self.train_cfg.get("dann", {})
        self.dann_enabled = self.dann_cfg.get("enabled", False)
        self.dann_weight = self.dann_cfg.get("weight", 0.5)
        self.coral_cfg = self.train_cfg.get("coral", {})
        self.coral_enabled = self.coral_cfg.get("enabled", False)
        self.coral_weight = self.coral_cfg.get("weight", 0.5)
        self.monitor_metric = self.train_cfg.get("monitor_metric", "f1_macro")
        self.gradient_clip = self.train_cfg.get("gradient_clip", 1.0)

        self.clinical_loader = self.loaders.get("clinical_train", None)

        if self.clinical_loader is not None:
            self.clinical_iter = cycle(self.clinical_loader)
        else:
            self.clinical_iter = None

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
        
        # Optionally freeze BatchNorm running statistics.
        # Useful during pretraining with multi-forward passes, but should
        # be disabled during finetuning to allow domain adaptation.
        if self.train_cfg.get("freeze_bn", True):
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()                 # freeze running mean/var
                    m.requires_grad_(True)   # still allow gamma/beta to learn

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
        self.augment.set_epoch(epoch)
        self.model.train()
        total_loss = 0.0
        total_main_loss = 0.0
        total_aux_loss = 0.0
        total_consistency_loss = 0.0
        total_l2sp_loss = 0.0
        total_coral_loss = 0.0
        total_domain_loss = 0.0
        total_domain_samples = 0
        correct = 0
        total = 0
        t0 = time.time()
        all_logits, all_targets = [], []

        for batch in self.loaders["train"]:

            clin_batch = None
            if self.clinical_iter is not None:
                clin_batch = next(self.clinical_iter)

            x1, x2, y = self._parse_batch(batch)
            x_clin = None

            if clin_batch is not None:
                x_clin, _, y_clin = self._parse_batch(clin_batch, augment=False)

            self.optimizer.zero_grad(set_to_none=True)

            outputs1 = self._normalize_outputs(self.model(x1))
            coral_term = self._zero_loss()

            outputs_clin = None

            if x_clin is not None:

                outputs_clin = self._normalize_outputs(self.model(x_clin))

                if self.coral_enabled:
                    feat_ref = outputs1.get("features")
                    feat_clin = outputs_clin.get("features")
                    
                    if feat_ref is not None and feat_clin is not None:

                        min_bs = min(feat_ref.size(0), feat_clin.size(0))
                        feat_ref = feat_ref[:min_bs]
                        feat_clin = feat_clin[:min_bs]

                        coral_term = coral_loss(feat_ref, feat_clin)
                            
            if x_clin is not None:
                outputs_target = outputs_clin
                loss_src = self.loss_fn(outputs1["main_logits"], y)
                loss_tgt = self.loss_fn(outputs_target["main_logits"], y_clin)

                main_loss = 0.5 * loss_src + 0.5 * loss_tgt
            else:
                main_loss = self.loss_fn(outputs1["main_logits"], y)
            
            if x_clin is not None:
                aux_loss = self._compute_aux_loss(outputs_clin, y_clin)
            else:
                aux_loss = self._compute_aux_loss(outputs1, y)

            outputs2 = None
            if x2 is not None:
                outputs2 = self._normalize_outputs(self.model(x2))

            consistency_term = self._compute_consistency_loss(outputs1, outputs2, y)

            l2sp_term = self.l2sp(self.model) if self.l2sp is not None else self._zero_loss()
            
            coral_weight = self.coral_weight
            
            loss = (
                main_loss
                + self.aux_loss_weight * aux_loss
                + self.consistency_weight * consistency_term
                + l2sp_term
                + coral_weight * coral_term
            )
            
            # DANN
            domain_loss = torch.tensor(0.0).to(self.device)
            if self.dann_enabled and outputs_clin is not None:

                feat_ref = outputs1.get("features")
                feat_clin = outputs_clin.get("features")

                if feat_ref is not None and feat_clin is not None:

                    min_bs = min(feat_ref.size(0), feat_clin.size(0))
                    feat_ref = feat_ref[:min_bs]
                    feat_clin = feat_clin[:min_bs]

                    feat_all = torch.cat([feat_ref, feat_clin], dim=0)

                    domain_labels = torch.cat([
                        torch.zeros(feat_ref.size(0), dtype=torch.long, device=self.device),
                        torch.ones(feat_clin.size(0), dtype=torch.long, device=self.device)
                    ])

                    p = epoch / self.train_cfg.get("max_epochs", 100)
                    lambda_ = 2. / (1. + np.exp(-10 * p)) - 1

                    feat_all = GradReverse.apply(feat_all, lambda_)

                    domain_logits = self.model.domain_classifier(feat_all)

                    domain_loss = nn.CrossEntropyLoss()(domain_logits, domain_labels)
                    
                    if epoch == 1 and len(all_logits) == 0:
                        print(f"DOMAIN LOSS SAMPLE: {domain_loss.item():.4f}")

                    domain_batch_size = domain_labels.size(0)

                    total_domain_loss += domain_loss.item() * domain_batch_size
                    total_domain_samples += domain_batch_size

                    loss = loss + self.dann_weight * domain_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
            self.optimizer.step()

            batch_size = len(y)
            total_loss += loss.item() * batch_size
            total_main_loss += main_loss.item() * batch_size
            total_aux_loss += aux_loss.item() * batch_size
            total_consistency_loss += consistency_term.item() * batch_size
            total_l2sp_loss += l2sp_term.item() * batch_size
            total_coral_loss += coral_term.item() * batch_size
            total += batch_size
            if x_clin is not None:
                correct += (outputs_target["main_logits"].argmax(dim=-1) == y_clin).sum().item()
                all_logits.append(outputs_target["main_logits"].detach().cpu())
                all_targets.append(y_clin.detach().cpu())
            else:
                correct += (outputs1["main_logits"].argmax(dim=-1) == y).sum().item()
                all_logits.append(outputs1["main_logits"].detach().cpu())
                all_targets.append(y.detach().cpu())
        
        avg_domain_loss = total_domain_loss / max(total_domain_samples, 1)

        print(f"[Epoch {epoch}] Avg Domain Loss: {avg_domain_loss:.4f}")

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
                "coral_loss": total_coral_loss / total,
                "domain_loss": avg_domain_loss,
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

    def _parse_batch(self, batch, augment=True):
        if isinstance(batch, dict):
            x1 = batch["x1"]

            if self.model.training and augment:
                x1 = x1.cpu().numpy()
                x1_aug = []
                for sample in x1:
                    augmented = self.augment(sample)
                    x1_aug.append(augmented)

                x1 = torch.from_numpy(np.array(x1_aug)).float().contiguous()

            x1 = x1.to(self.device)

            y = batch["y"].to(self.device)   

            x2 = None

            return x1, x2, y

        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x, y = batch

            if isinstance(x, (tuple, list)) and len(x) == 2:
                x1, _ = x 

                if self.model.training and augment:
                    x1 = x1.cpu().numpy()
                    x1_aug = []

                    for sample in x1:
                        augmented = self.augment(sample)

                        x1_aug.append(augmented)

                    x1 = torch.from_numpy(np.array(x1_aug)).float().contiguous()

                return x1.to(self.device), None, y.to(self.device)

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
