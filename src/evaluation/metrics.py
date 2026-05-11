"""
src/evaluation/metrics.py

Complete metric suite for spectral classification research.

Computes:
- Accuracy
- Macro F1 (handles class imbalance correctly)
- Per-class F1 (for understanding which classes are hard)
- ROC-AUC (one-vs-rest, macro)
- Matthews Correlation Coefficient
- Confusion matrix

Supports metrics for:

- isolate-space pretraining
- compact transfer-space evaluation
- clinical OOD transfer analysis

Transfer-learning experiments operate in:
compact transfer-space labels:
[0,1,2,3,4]

NOTE:
Current metrics operate at spectrum-level.

Future upgrades may include:
- patient-level aggregation
- LOOCV evaluation
- uncertainty calibration
- ensemble consensus metrics
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute all classification metrics from raw logits and targets.

    Args:
        logits:    (N, C) raw model outputs
        targets:   (N,) integer class labels
        n_classes: Total number of classes in the model output
        prefix:    Optional prefix for metric keys (e.g. "val_")

    Returns: Dict of metric_name -> float value
    """
    if logits.numel() == 0 or targets.numel() == 0:
        metrics = {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "mcc": 0.0,
            "roc_auc": 0.0,
        }
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        return metrics

    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    preds = logits.argmax(dim=-1).detach().cpu().numpy()
    y     = targets.detach().cpu().numpy()
    # --------------------------------------------------------
    # Compact transfer-space integrity check
    # Transfer evaluation must operate on:  
    # [0,1,2,3,4]
    # Sparse clinical labels:
    # [0,2,3,5,6]
    # should never appear after remapping.
    # --------------------------------------------------------

    if n_classes == 5:
        unexpected = (
            set(np.unique(y).tolist())
            - set([0,1,2,3,4])
        )
        if unexpected:
            raise AssertionError(
                "Metrics received invalid compact "
                f"transfer labels: {sorted(unexpected)}"
            )
    # --------------------------------------------------------
    # Only evaluate classes actually present
    # in predictions or targets.
    # Prevents empty-class instability during:
    # - transfer folds
    # - ablations
    # - small validation subsets
    # --------------------------------------------------------

    present_classes = np.unique(np.concatenate([y, preds]))

    # Core metrics
    accuracy = (preds == y).mean()
    if np.isnan(accuracy):
        raise ValueError(
            "Accuracy computation produced NaN"
        )
    macro_f1 = _macro_f1(y, preds, present_classes)
    mcc      = _matthews_corrcoef(y, preds)

    metrics = {
        "accuracy":  float(accuracy),
        "f1_macro":  float(macro_f1),
        "mcc":       float(mcc),
    }

    # ROC-AUC (one-vs-rest, macro over present classes)
    try:
        auc = _roc_auc_ovr(y, probs, present_classes)
        metrics["roc_auc"] = float(auc) if not np.isnan(auc) else 0.0
    except Exception:
        metrics["roc_auc"] = float("nan")

    # Per-class F1 (only for present classes)
    per_class = _per_class_f1(y, preds, present_classes)
    for cls, f1 in per_class.items():
        metrics[f"f1_class_{cls}"] = float(f1)

    # Apply prefix
    if prefix:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

    return metrics

# --------------------------------------------------------
# Confusion matrix operates ONLY on
# present classes for cleaner visualization.
#
# Semantic interpretation depends on:
#
# - isolate-space
# - compact transfer-space
#
# determined externally by evaluator.
# --------------------------------------------------------
def compute_confusion_matrix(
    logits: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
) -> tuple[np.ndarray, list[int]]:
    """Returns (n_classes, n_classes) confusion matrix (rows=true, cols=pred)."""
    if logits.numel() == 0 or targets.numel() == 0:
        return np.zeros((0, 0), dtype=int), []

    preds = logits.argmax(dim=-1).detach().cpu().numpy()
    y     = targets.detach().cpu().numpy()
    
    present = sorted(np.unique(np.concatenate([y, preds])).tolist())
    n = len(present)
    idx_map = {cls: i for i, cls in enumerate(present)}
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(y, preds):
        cm[idx_map[true], idx_map[pred]] += 1
    return cm, present

# --------------------------------------------------------
# Domain-transfer degradation metric
#
# Measures performance drop from:
#
# reference-domain ->
# clinical OOD domain
#
# Larger gap indicates:
# weaker domain generalization.
# --------------------------------------------------------
def compute_transfer_gap(
    source_metrics: Dict[str, float],
    ood_metrics: Dict[str, float],
    metric: str = "accuracy",
) -> float:
    """
    transfer_gap = source_accuracy - ood_accuracy
    Large gap = poor generalisation.
    """
    return source_metrics[metric] - ood_metrics[metric]


# ------------------------------------------------------------------ #
# Internal helpers (pure numpy, no sklearn dependency)
# ------------------------------------------------------------------ #

def _macro_f1(y_true, y_pred, classes) -> float:
    f1s = []
    for cls in classes:
        tp = ((y_pred == cls) & (y_true == cls)).sum()
        fp = ((y_pred == cls) & (y_true != cls)).sum()
        fn = ((y_pred != cls) & (y_true == cls)).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            prec = tp / (tp + fp)
            rec  = tp / (tp + fn)
            if prec + rec == 0:
                f1 = 0.0
            else:
                f1 = 2 * prec * rec / (prec + rec)
        f1s.append(f1)
    return float(np.mean(f1s))


def _per_class_f1(y_true, y_pred, classes) -> Dict[int, float]:
    result = {}
    for cls in classes:
        tp = ((y_pred == cls) & (y_true == cls)).sum()
        fp = ((y_pred == cls) & (y_true != cls)).sum()
        fn = ((y_pred != cls) & (y_true == cls)).sum()
        if tp + fp == 0 or tp + fn == 0:
            f1 = 0.0
        else:
            prec = tp / (tp + fp)
            rec  = tp / (tp + fn)
            if prec + rec == 0:
                f1 = 0.0
            else:
                f1 = 2 * prec * rec / (prec + rec)
        result[int(cls)] = float(f1)
    return result


def _roc_auc_ovr(y_true, probs, classes) -> float:
    """One-vs-rest macro AUC using the trapezoidal rule."""
    aucs = []
    for cls in classes:
        cls = int(cls)
        binary_y = (y_true == cls).astype(float)
        scores   = probs[:, cls]
        auc = _binary_auc(binary_y, scores)
        aucs.append(auc)
    return float(np.mean(aucs))


def _binary_auc(y_true, scores) -> float:
    """Compute binary AUC via trapezoidal rule."""
    desc_idx = np.argsort(-scores)
    y_sorted = y_true[desc_idx]
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    tp_cumsum = np.cumsum(y_sorted)
    fp_cumsum = np.cumsum(1 - y_sorted)
    tpr = tp_cumsum / pos
    fpr = fp_cumsum / neg
    # Prepend (0,0)
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy >= 2.0 compat
    return float(_trapz(tpr, fpr))


def _matthews_corrcoef(y_true, y_pred) -> float:
    """Multiclass MCC using the exact formula."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    K = len(classes)
    if K < 2:
        return float("nan")
    # Build confusion matrix
    cm = np.zeros((K, K), dtype=float)
    idx = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        if p in idx:
            cm[idx[t], idx[p]] += 1

    t_sum = cm.sum(axis=1)
    p_sum = cm.sum(axis=0)
    n_correct = np.trace(cm)
    n_samples = cm.sum()

    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)

    denom = np.sqrt(cov_ypyp * cov_ytyt)
    if denom == 0:
        return 0.0
    return float(cov_ytyp / denom)

def majority_vote_predictions(
    preds,
    targets,
    sample_ids,
    spectra_per_group,
):
    """
    Aggregate spectrum-level predictions into
    patient/isolate-level predictions.

    Parameters
    ----------
    preds : np.ndarray
        Spectrum-level predicted labels.

    targets : np.ndarray
        Spectrum-level ground-truth labels.

    sample_ids : list[str]
        Sample identifiers.

    spectra_per_group : int
        Number of spectra belonging to one isolate/patient.

    Returns
    -------
    patient_preds : np.ndarray
    patient_targets : np.ndarray
    """

    preds = np.asarray(preds)
    targets = np.asarray(targets)

    patient_preds = []
    patient_targets = []

    n_groups = len(preds) // spectra_per_group

    for i in range(n_groups):
        start = i * spectra_per_group
        end = start + spectra_per_group

        group_preds = preds[start:end]
        group_targets = targets[start:end]

        voted_pred = Counter(group_preds).most_common(1)[0][0]
        voted_target = Counter(group_targets).most_common(1)[0][0]

        patient_preds.append(voted_pred)
        patient_targets.append(voted_target)

    return np.asarray(patient_preds), np.asarray(patient_targets)