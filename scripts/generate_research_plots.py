"""
scripts/generate_research_plots.py

Unified research-grade visualization pipeline.

ONE COMMAND → ONE EXPERIMENT → COMPLETE VISUALIZATION PACKAGE.

Usage:
  python scripts/generate_research_plots.py --exp_dir /path/to/experiment
  python scripts/generate_research_plots.py --exp_dir /path/to/experiment --embeddings
  python scripts/generate_research_plots.py --exp_dir /path/to/experiment --dpi 300

Design rules:
  * If n_classes > 10: disable cell annotations, enlarge figure.
  * If stage == clinical_transfer: emphasize grouped metrics + OOD robustness.
  * Embeddings are DISABLED by default (CPU-expensive).
  * ROC/PR: only macro + micro + top-5 classes. No 30-class spaghetti.
  * All figures exported as both PNG and PDF at high DPI.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# GLOBAL STYLE CONFIGURATION
# ============================================================

# Curated publication palette — avoids generic primary colors
PALETTE = {
    "primary":    "#2D5F8A",   # Steel blue
    "secondary":  "#E07B54",   # Warm coral
    "accent":     "#5DAE8B",   # Sage green
    "highlight":  "#C4A35A",   # Gold
    "muted":      "#8B7D9C",   # Lavender grey
    "dark":       "#2C3E50",   # Charcoal
    "light_bg":   "#FAFBFC",   # Near-white
    "grid":       "#E8ECF0",   # Subtle grid
}

# Semantic color mapping for treatment classes
TREATMENT_COLORS = {
    "Meropenem":    "#2D5F8A",
    "Ciprofloxacin":"#E07B54",
    "TZP":          "#5DAE8B",
    "Vancomycin":   "#C4A35A",
    "Ceftriaxone":  "#8B7D9C",
    "Penicillin":   "#D4726A",
    "Daptomycin":   "#6B9BC3",
    "Caspofungin":  "#9B8EC4",
}

# Multi-model comparison palette
MODEL_COLORS = [
    "#2D5F8A", "#E07B54", "#5DAE8B", "#C4A35A",
    "#8B7D9C", "#D4726A", "#6B9BC3", "#9B8EC4",
]

SPLIT_DISPLAY = {
    "test":          "Reference Test",
    "val":           "Validation",
    "train":         "Training",
    "2018clinical":  "Clinical 2018 (OOD)",
    "2019clinical":  "Clinical 2019 (OOD)",
}


def _apply_global_style():
    """Apply consistent publication-quality style to all plots."""
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          11,
        "axes.titlesize":     14,
        "axes.titleweight":   "bold",
        "axes.labelsize":     12,
        "axes.labelweight":   "medium",
        "axes.linewidth":     0.8,
        "axes.edgecolor":     "#555555",
        "axes.facecolor":     PALETTE["light_bg"],
        "figure.facecolor":   "white",
        "figure.dpi":         150,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    9,
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "#CCCCCC",
        "grid.alpha":         0.3,
        "grid.linewidth":     0.5,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.15,
    })
    sns.set_style("whitegrid", {
        "grid.color":     PALETTE["grid"],
        "axes.facecolor": PALETTE["light_bg"],
    })


# ============================================================
# STAGE & LABEL SEMANTICS
# ============================================================

STAGE_DISPLAY = {
    "pretrain_30class":           "Stage 1 — Isolate Pretraining (30-class)",
    "pretrain_treatment_8class":  "Stage 2 — Treatment Pretraining (8-class)",
    "transfer_5class":            "Stage 3 — Clinical Transfer (5-class)",
}


def _stage_title(stage: str) -> str:
    return STAGE_DISPLAY.get(stage, stage)


def _resolve_labels(cfg: dict, n_classes: int) -> List[str]:
    """Resolve human-readable class labels from config and ontology."""
    stage = cfg.get("task", {}).get("stage", "unknown")

    if stage == "pretrain_30class":
        try:
            from metadata.ontology import ISOLATES
            return [ISOLATES[i]["strain"] for i in range(n_classes)]
        except Exception:
            return [f"Isolate {i}" for i in range(n_classes)]

    if stage == "pretrain_treatment_8class":
        try:
            from metadata.ontology import GLOBAL_TREATMENTS
            return [GLOBAL_TREATMENTS[i] for i in range(n_classes)]
        except Exception:
            return [f"Treatment {i}" for i in range(n_classes)]

    if stage == "transfer_5class":
        ids = cfg.get("task", {}).get("clinical_sparse_global_ids", list(range(n_classes)))
        try:
            from metadata.ontology import GLOBAL_TREATMENTS
            return [GLOBAL_TREATMENTS[int(i)] for i in ids]
        except Exception:
            return [f"Clinical {i}" for i in range(n_classes)]

    return [f"Class {i}" for i in range(n_classes)]


def _split_display(split: str) -> str:
    return SPLIT_DISPLAY.get(split, split)


# ============================================================
# FILE I/O UTILITIES
# ============================================================

def _load_config(exp_dir: Path) -> dict:
    """Load experiment config from YAML or JSON."""
    yaml_path = exp_dir / "config.yaml"
    json_path = exp_dir / "config.json"
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path, "r") as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            pass
    if json_path.exists():
        with open(json_path, "r") as f:
            return json.load(f)
    return {}


def _find_eval_results(exp_dir: Path) -> List[Path]:
    return sorted(exp_dir.glob("*_eval_results.json"))


def _load_predictions(pred_dir: Path, split: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    logits_path = pred_dir / f"{split}_logits.npy"
    probs_path  = pred_dir / f"{split}_probabilities.npy"
    tgts_path   = pred_dir / f"{split}_targets.npy"
    if not (logits_path.exists() and probs_path.exists() and tgts_path.exists()):
        return None
    return np.load(logits_path), np.load(probs_path), np.load(tgts_path)


def _load_embeddings(emb_dir: Path, split: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    feat_path = emb_dir / f"{split}_features.npy"
    tgt_path  = emb_dir / f"{split}_targets.npy"
    if not (feat_path.exists() and tgt_path.exists()):
        return None
    return np.load(feat_path), np.load(tgt_path)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, path: Path, dpi: int) -> None:
    """Save figure as PNG + PDF, then close."""
    _ensure_dir(path.parent)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(str(pdf_path), dpi=min(dpi, 300), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# FIGURE 1: CONFUSION MATRICES (adaptive)
# ============================================================

def plot_confusion_matrix(
    targets: np.ndarray,
    preds: np.ndarray,
    labels: List[str],
    title: str,
    out_path: Path,
    dpi: int,
    normalize: bool = True,
) -> None:
    """
    Adaptive confusion matrix.
    - n_classes <= 10: annotated, square, 10x9 figure
    - n_classes > 10:  no annotations, enlarged, auto-scaled
    """
    n = len(labels)
    cm = confusion_matrix(targets, preds, labels=list(range(n)))

    if normalize:
        cm_plot = cm.astype(float)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_plot = cm_plot / row_sums
    else:
        cm_plot = cm

    # Adaptive sizing
    large = n > 10
    fig_w = max(10, n * 0.5) if large else 10
    fig_h = max(9, n * 0.45) if large else 9
    annot = not large
    fmt = ".2f" if normalize else ".0f"

    # Custom blue colormap with better contrast
    cmap = LinearSegmentedColormap.from_list(
        "research_blues",
        ["#FFFFFF", "#D6E8F5", "#7FB3D8", "#2D5F8A", "#1A3A5C"],
    )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        cm_plot,
        ax=ax,
        cmap=cmap,
        annot=annot,
        fmt=fmt if annot else "",
        cbar=True,
        square=True,
        linewidths=0.3 if not large else 0.1,
        linecolor="#DDDDDD",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": max(7, 11 - n // 5)} if annot else {},
    )

    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    norm_tag = "Normalized" if normalize else "Counts"
    ax.set_title(f"{title}\n({norm_tag})", pad=15)

    rotation = 45 if n > 6 else 0
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha="right" if rotation else "center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    _save_fig(fig, out_path, dpi)


# ============================================================
# FIGURE 2: PER-CLASS FAILURE HEATMAP
# ============================================================

def plot_failure_heatmap(
    targets: np.ndarray,
    preds: np.ndarray,
    labels: List[str],
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    """
    Grid heatmap showing Precision, Recall, F1 per class,
    sorted by F1-score (worst first). Color encodes performance.
    """
    report = classification_report(targets, preds, output_dict=True, zero_division=0)
    classes = [str(i) for i in range(len(labels))]

    data = []
    for i, cls_key in enumerate(classes):
        if cls_key not in report:
            continue
        data.append({
            "class": labels[i],
            "Precision": report[cls_key]["precision"],
            "Recall":    report[cls_key]["recall"],
            "F1-Score":  report[cls_key]["f1-score"],
            "Support":   int(report[cls_key]["support"]),
        })

    if not data:
        return

    # Sort by F1 ascending (worst first for attention)
    data.sort(key=lambda x: x["F1-Score"])

    class_names = [d["class"] for d in data]
    metric_names = ["Precision", "Recall", "F1-Score"]
    heatmap_data = np.array([[d[m] for m in metric_names] for d in data])

    # Red-yellow-green diverging colormap
    cmap = LinearSegmentedColormap.from_list(
        "failure_map",
        ["#D4726A", "#EECC77", "#5DAE8B"],
    )

    n = len(class_names)
    fig_h = max(4, n * 0.35 + 2)
    fig, ax = plt.subplots(figsize=(6, fig_h))

    annot_labels = np.array([
        [f"{heatmap_data[i, j]:.2f}" for j in range(len(metric_names))]
        for i in range(n)
    ])

    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=cmap,
        annot=annot_labels if n <= 15 else False,
        fmt="",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=metric_names,
        yticklabels=class_names,
        annot_kws={"size": 10, "weight": "bold"},
    )

    ax.set_title(f"{title}\n(sorted by F1 ↑)", pad=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    _save_fig(fig, out_path, dpi)


# ============================================================
# FIGURE 3: ROC CURVES (clean, no spaghetti)
# ============================================================

def plot_roc_clean(
    probs: np.ndarray,
    targets: np.ndarray,
    labels: List[str],
    title: str,
    out_path: Path,
    dpi: int,
    max_per_class: int = 5,
) -> None:
    """
    Clean ROC plot:
    - Always shows macro and micro averages.
    - Per-class curves only for top-k by AUC (default 5).
    - Skips entirely if binary (< 2 classes).
    """
    n_classes = len(labels)
    if n_classes < 2:
        return

    y_bin = label_binarize(targets, classes=list(range(n_classes)))
    if y_bin.ndim == 1:
        y_bin = np.column_stack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(8, 7))

    # Micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro,
            color=PALETTE["primary"], linewidth=2.5,
            label=f"Micro-average (AUC = {auc_micro:.3f})")

    # Per-class AUCs
    class_aucs = []
    for i in range(n_classes):
        try:
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], probs[:, i])
            auc_i = auc(fpr_i, tpr_i)
            class_aucs.append((i, auc_i, fpr_i, tpr_i))
        except Exception:
            pass

    # Macro-average AUC
    if class_aucs:
        macro_auc = np.mean([a[1] for a in class_aucs])
        ax.plot([], [], ' ', label=f"Macro-average AUC = {macro_auc:.3f}")

    # Plot only top-k classes by AUC
    class_aucs.sort(key=lambda x: x[1], reverse=True)
    show_k = min(max_per_class, len(class_aucs))
    colors = sns.color_palette("husl", show_k)

    for idx, (i, auc_val, fpr_i, tpr_i) in enumerate(class_aucs[:show_k]):
        ax.plot(fpr_i, tpr_i,
                color=colors[idx], linewidth=1.2, alpha=0.85,
                label=f"{labels[i]} ({auc_val:.3f})")

    if len(class_aucs) > show_k:
        ax.plot([], [], ' ', label=f"... +{len(class_aucs) - show_k} classes omitted")

    # Diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, pad=12)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])

    _save_fig(fig, out_path, dpi)


# ============================================================
# FIGURE 4: GROUPED VS SPECTRUM COMPARISON
# ============================================================

def plot_grouped_vs_spectrum(
    results: dict,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    """
    Double-bar chart: spectrum-level vs patient/group-level accuracy and F1.
    Shows the gain from majority voting aggregation.
    """
    splits, spec_acc, grp_acc, spec_f1, grp_f1 = [], [], [], [], []

    for split_name, data in results.get("splits", {}).items():
        metrics = data.get("metrics", {})
        group_metrics = data.get("group_metrics", {})
        if not group_metrics or "accuracy" not in group_metrics:
            continue
        splits.append(_split_display(split_name))
        spec_acc.append(metrics.get("accuracy", 0.0))
        grp_acc.append(group_metrics.get("accuracy", 0.0))
        spec_f1.append(metrics.get("f1_macro", 0.0))
        grp_f1.append(group_metrics.get("f1_macro", 0.0))

    if not splits:
        return

    x = np.arange(len(splits))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Accuracy panel
    ax = axes[0]
    bars1 = ax.bar(x - width / 2, spec_acc, width, label="Spectrum-level",
                   color=PALETTE["primary"], alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, grp_acc, width, label="Patient/Group-level",
                   color=PALETTE["accent"], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 1.05)

    # Annotate gain
    for i in range(len(splits)):
        gain = grp_acc[i] - spec_acc[i]
        if abs(gain) > 0.001:
            sign = "+" if gain > 0 else ""
            ax.annotate(f"{sign}{gain:.1%}",
                       xy=(x[i] + width / 2, grp_acc[i]),
                       xytext=(0, 5), textcoords="offset points",
                       ha="center", fontsize=7, color=PALETTE["dark"])

    # F1 panel
    ax = axes[1]
    ax.bar(x - width / 2, spec_f1, width, label="Spectrum-level",
           color=PALETTE["primary"], alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, grp_f1, width, label="Patient/Group-level",
           color=PALETTE["accent"], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=20, ha="right")
    ax.set_title("Macro F1", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    for i in range(len(splits)):
        gain = grp_f1[i] - spec_f1[i]
        if abs(gain) > 0.001:
            sign = "+" if gain > 0 else ""
            ax.annotate(f"{sign}{gain:.1%}",
                       xy=(x[i] + width / 2, grp_f1[i]),
                       xytext=(0, 5), textcoords="offset points",
                       ha="center", fontsize=7, color=PALETTE["dark"])

    fig.suptitle(f"Grouped vs Spectrum Analysis — {title}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, out_path, dpi)


# ============================================================
# FIGURE 5: OOD DEGRADATION ANALYSIS
# ============================================================

def plot_ood_degradation(
    results: dict,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    """
    Bar chart showing in-domain (test) vs each OOD split,
    with transfer gap annotations.
    """
    splits_data = results.get("splits", {})
    test_data = splits_data.get("test", {})
    if not test_data:
        return

    test_metrics = test_data.get("metrics", {})
    test_acc = test_metrics.get("accuracy", 0)
    test_f1 = test_metrics.get("f1_macro", 0)

    ood_splits = []
    for sname, sdata in splits_data.items():
        if sname == "test" or sname in ("train", "val"):
            continue
        m = sdata.get("metrics", {})
        gap = sdata.get("transfer_gap", None)
        ood_splits.append({
            "name": _split_display(sname),
            "accuracy": m.get("accuracy", 0),
            "f1_macro": m.get("f1_macro", 0),
            "gap": gap,
        })

    if not ood_splits:
        return

    names = ["Reference Test"] + [s["name"] for s in ood_splits]
    accs  = [test_acc] + [s["accuracy"] for s in ood_splits]
    f1s   = [test_f1] + [s["f1_macro"] for s in ood_splits]

    x = np.arange(len(names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 5))

    colors_acc = [PALETTE["primary"]] + [PALETTE["secondary"]] * len(ood_splits)
    colors_f1  = [PALETTE["accent"]]  + [PALETTE["highlight"]] * len(ood_splits)

    ax.bar(x - width / 2, accs, width, label="Accuracy",
           color=colors_acc, alpha=0.85, edgecolor="white")
    ax.bar(x + width / 2, f1s, width, label="Macro F1",
           color=colors_f1, alpha=0.85, edgecolor="white")

    # Annotate transfer gaps
    for i, ood in enumerate(ood_splits):
        if ood["gap"] is not None:
            ax.annotate(f"Gap: {ood['gap']:.1%}",
                       xy=(x[i + 1], max(accs[i + 1], f1s[i + 1])),
                       xytext=(0, 8), textcoords="offset points",
                       ha="center", fontsize=8, color="#CC3333", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"OOD Degradation Analysis — {title}", pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.1)

    # Reference line
    ax.axhline(y=test_acc, color=PALETTE["primary"], linestyle="--", alpha=0.3, linewidth=1)

    _save_fig(fig, out_path, dpi)


# ============================================================
# FIGURE 6: STAGE TRANSITION (multi-stage lineage)
# ============================================================

def plot_stage_transition(
    stage_summaries: List[Dict],
    out_path: Path,
    dpi: int,
) -> None:
    """
    Line + marker plot showing metrics evolving across
    Stage 1 → Stage 2 → Stage 3.
    """
    if len(stage_summaries) < 2:
        return

    stages = [s["display"] for s in stage_summaries]
    accs   = [s.get("accuracy", 0) for s in stage_summaries]
    f1s    = [s.get("f1_macro", 0) for s in stage_summaries]
    x = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x, accs, marker="o", markersize=10, linewidth=2.5,
            color=PALETTE["primary"], label="Accuracy", zorder=3)
    ax.plot(x, f1s, marker="s", markersize=10, linewidth=2.5,
            color=PALETTE["secondary"], label="Macro F1", zorder=3)

    # Annotate values
    for i in range(len(stages)):
        ax.annotate(f"{accs[i]:.2%}", xy=(x[i], accs[i]),
                   xytext=(0, 12), textcoords="offset points",
                   ha="center", fontsize=9, color=PALETTE["primary"], fontweight="bold")
        ax.annotate(f"{f1s[i]:.2%}", xy=(x[i], f1s[i]),
                   xytext=(0, -18), textcoords="offset points",
                   ha="center", fontsize=9, color=PALETTE["secondary"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Stage Transition — Metrics Across Training Pipeline", pad=15)
    ax.legend(loc="best", fontsize=10)
    ax.set_ylim(0, 1.1)

    # Subtle stage transition arrows
    for i in range(len(stages) - 1):
        ax.annotate("", xy=(x[i + 1] - 0.15, 0.05), xytext=(x[i] + 0.15, 0.05),
                   arrowprops=dict(arrowstyle="->", color=PALETTE["muted"], lw=1.5))

    _save_fig(fig, out_path, dpi)


# ============================================================
# FIGURE 7: MODEL / ARCHITECTURE COMPARISON
# ============================================================

def plot_model_comparison(
    model_data: List[Dict],
    split: str,
    out_path: Path,
    dpi: int,
) -> None:
    """
    Dot plot comparing metrics across different model architectures
    discovered in sibling experiment directories.
    """
    if len(model_data) < 2:
        return

    names = [m["name"] for m in model_data]
    accs  = [m.get("accuracy", 0) for m in model_data]
    f1s   = [m.get("f1_macro", 0) for m in model_data]
    mccs  = [m.get("mcc", 0) for m in model_data]

    metrics_dict = {"Accuracy": accs, "Macro F1": f1s, "MCC": mccs}
    metric_names = list(metrics_dict.keys())

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))

    x = np.arange(len(names))
    total_width = 0.7
    bar_width = total_width / len(metric_names)
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]

    for idx, (mname, values) in enumerate(metrics_dict.items()):
        offset = (idx - len(metric_names) / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                     label=mname, color=colors[idx], alpha=0.85, edgecolor="white")
        # Annotate
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontweight="medium")
    ax.set_ylabel("Score")
    ax.set_title(f"Model Architecture Comparison — {_split_display(split)}", pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.15)

    _save_fig(fig, out_path, dpi)


# ============================================================
# FIGURE 8: LEARNING CURVES
# ============================================================

def plot_learning_curves(
    metrics_path: Path,
    title: str,
    out_dir: Path,
    dpi: int,
) -> None:
    """Plot loss and accuracy/F1 learning curves from metrics.json."""
    if not metrics_path.exists():
        return

    with open(metrics_path, "r") as f:
        try:
            rows = json.load(f)
        except json.JSONDecodeError:
            return

    if not rows:
        return

    # Group by split
    split_data: Dict[str, Dict[str, List]] = {}
    for row in rows:
        split = row.get("split", "train")
        if split not in split_data:
            split_data[split] = {"epoch": [], "loss": [], "accuracy": [], "f1_macro": []}
        split_data[split]["epoch"].append(row.get("epoch", 0))
        for key in ["loss", "accuracy", "f1_macro"]:
            split_data[split][key].append(row.get(key, None))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"train": PALETTE["primary"], "val": PALETTE["secondary"], "source_val": PALETTE["accent"]}

    # Loss
    ax = axes[0]
    for split, data in split_data.items():
        losses = [v for v in data["loss"] if v is not None]
        epochs = data["epoch"][:len(losses)]
        if losses:
            ax.plot(epochs, losses, label=f"{split} loss",
                   color=colors.get(split, PALETTE["muted"]), linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss", fontweight="bold")
    ax.legend(fontsize=8)

    # Accuracy + F1
    ax = axes[1]
    for split, data in split_data.items():
        c = colors.get(split, PALETTE["muted"])
        accs = [v for v in data["accuracy"] if v is not None]
        epochs_acc = data["epoch"][:len(accs)]
        if accs:
            ax.plot(epochs_acc, accs, label=f"{split} acc", color=c, linewidth=1.8)
        f1s = [v for v in data["f1_macro"] if v is not None]
        epochs_f1 = data["epoch"][:len(f1s)]
        if f1s:
            ax.plot(epochs_f1, f1s, label=f"{split} F1",
                   color=c, linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Learning Curves", fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle(f"Training Progress — {title}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, out_dir / "learning_curves.png", dpi)


# ============================================================
# FIGURE 9: OPTIONAL EMBEDDINGS (t-SNE / PCA)
# ============================================================

def plot_embedding_tsne(
    features: np.ndarray,
    targets: np.ndarray,
    labels: List[str],
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    """t-SNE 2D projection of latent embeddings."""
    from sklearn.manifold import TSNE

    perplexity = min(30, max(5, len(features) // 10))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(features)

    n = len(labels)
    palette = sns.color_palette("husl", n)

    fig, ax = plt.subplots(figsize=(8, 7))
    for i, label in enumerate(labels):
        mask = targets == i
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  s=8, label=label, alpha=0.7, color=palette[i])

    ax.set_title(f"t-SNE — {title}", pad=12)
    ax.set_xticks([])
    ax.set_yticks([])

    # Smart legend placement
    ncol = max(1, n // 8)
    ax.legend(fontsize=7, ncol=ncol, loc="best", markerscale=2)

    _save_fig(fig, out_path, dpi)


# ============================================================
# METRIC SUMMARY EXPORT
# ============================================================

def write_metrics_csv(results: dict, out_path: Path) -> None:
    """Export a clean CSV summary of all splits and metrics."""
    rows = []
    for split_name, data in results.get("splits", {}).items():
        metrics = data.get("metrics", {})
        group_metrics = data.get("group_metrics", {})
        row = {"split": split_name}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                row[f"spectrum_{k}"] = v
        for k, v in group_metrics.items():
            if isinstance(v, (int, float)):
                row[f"grouped_{k}"] = v
        if "transfer_gap" in data:
            row["transfer_gap"] = data["transfer_gap"]
        rows.append(row)

    if not rows:
        return

    _ensure_dir(out_path.parent)
    all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(
    exp_dir: Path,
    model_name: str,
    stage: str,
    results: dict,
    figures_generated: List[str],
    out_path: Path,
) -> None:
    """Write a human-readable experiment summary in Markdown."""
    _ensure_dir(out_path.parent)

    lines = [
        f"# Experiment Visualization Report",
        "",
        f"**Experiment:** `{exp_dir.name}`",
        f"**Model:** `{model_name}`",
        f"**Stage:** {_stage_title(stage)}",
        "",
        "## Metrics Summary",
        "",
    ]

    for split_name, data in results.get("splits", {}).items():
        metrics = data.get("metrics", {})
        group_metrics = data.get("group_metrics", {})
        lines.append(f"### {_split_display(split_name)}")
        lines.append(f"- Accuracy: **{metrics.get('accuracy', 'N/A'):.4f}**" if isinstance(metrics.get('accuracy'), float) else f"- Accuracy: N/A")
        lines.append(f"- Macro F1: **{metrics.get('f1_macro', 'N/A'):.4f}**" if isinstance(metrics.get('f1_macro'), float) else f"- Macro F1: N/A")
        lines.append(f"- MCC: **{metrics.get('mcc', 'N/A'):.4f}**" if isinstance(metrics.get('mcc'), float) else f"- MCC: N/A")
        if group_metrics and isinstance(group_metrics.get("accuracy"), float):
            lines.append(f"- Grouped Accuracy: **{group_metrics['accuracy']:.4f}**")
            lines.append(f"- Grouped F1: **{group_metrics.get('f1_macro', 0):.4f}**")
        if "transfer_gap" in data:
            lines.append(f"- Transfer Gap: **{data['transfer_gap']:.4f}**")
        lines.append("")

    lines.append("## Generated Figures")
    lines.append("")
    for fig_name in sorted(figures_generated):
        lines.append(f"- `{fig_name}`")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# FIGURE 10: PUBLICATION COMPOSITE PANEL
# ============================================================

def plot_publication_panel(
    targets: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    labels: List[str],
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    """
    Generate a 2x2 publication-ready composite panel containing:
      A) Normalized Confusion Matrix
      B) Sorted Per-Class Failure Heatmap
      C) Clean ROC curves (micro + macro + top-3 classes)
      D) Precision-Recall curves (micro + macro + top-3 classes)
    """
    n = len(labels)
    fig = plt.figure(figsize=(15, 13))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # ----------------------------------------------------
    # PANEL A: Confusion Matrix
    # ----------------------------------------------------
    ax_cm = fig.add_subplot(gs[0, 0])
    cm = confusion_matrix(targets, preds, labels=list(range(n)))
    cm_plot = cm.astype(float)
    row_sums = cm_plot.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_plot = cm_plot / row_sums

    cmap_blues = LinearSegmentedColormap.from_list(
        "research_blues",
        ["#FFFFFF", "#D6E8F5", "#7FB3D8", "#2D5F8A", "#1A3A5C"],
    )
    sns.heatmap(
        cm_plot,
        ax=ax_cm,
        cmap=cmap_blues,
        annot=n <= 10,
        fmt=".2f" if n <= 10 else "",
        cbar=True,
        square=True,
        linewidths=0.3 if n <= 10 else 0.1,
        linecolor="#DDDDDD",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": max(7, 10 - n // 5)} if n <= 10 else {},
    )
    ax_cm.set_xlabel("Predicted", fontweight="bold")
    ax_cm.set_ylabel("True", fontweight="bold")
    ax_cm.set_title("A) Confusion Matrix (Normalized)", fontweight="bold")
    rotation = 45 if n > 6 else 0
    ax_cm.set_xticklabels(ax_cm.get_xticklabels(), rotation=rotation, ha="right" if rotation else "center")
    ax_cm.set_yticklabels(ax_cm.get_yticklabels(), rotation=0)

    # ----------------------------------------------------
    # PANEL B: Failure Heatmap
    # ----------------------------------------------------
    ax_hm = fig.add_subplot(gs[0, 1])
    report = classification_report(targets, preds, output_dict=True, zero_division=0)
    classes = [str(i) for i in range(n)]
    data = []
    for i, cls_key in enumerate(classes):
        if cls_key not in report:
            continue
        data.append({
            "class": labels[i],
            "Precision": report[cls_key]["precision"],
            "Recall":    report[cls_key]["recall"],
            "F1-Score":  report[cls_key]["f1-score"],
        })
    data.sort(key=lambda x: x["F1-Score"])
    class_names = [d["class"] for d in data]
    metric_names = ["Precision", "Recall", "F1-Score"]
    heatmap_data = np.array([[d[m] for m in metric_names] for d in data])

    cmap_red_green = LinearSegmentedColormap.from_list(
        "failure_map",
        ["#D4726A", "#EECC77", "#5DAE8B"],
    )
    sns.heatmap(
        heatmap_data,
        ax=ax_hm,
        cmap=cmap_red_green,
        annot=len(class_names) <= 15,
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        linewidths=0.5,
        linecolor="white",
        xticklabels=metric_names,
        yticklabels=class_names,
        annot_kws={"size": 9, "weight": "bold"},
    )
    ax_hm.set_yticklabels(ax_hm.get_yticklabels(), rotation=0)
    ax_hm.set_title("B) Failure Analysis Heatmap (sorted by F1 ↑)", fontweight="bold")

    # ----------------------------------------------------
    # PANEL C: ROC Curves
    # ----------------------------------------------------
    ax_roc = fig.add_subplot(gs[1, 0])
    y_bin = label_binarize(targets, classes=list(range(n)))
    if y_bin.ndim == 1:
        y_bin = np.column_stack([1 - y_bin, y_bin])
    
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    ax_roc.plot(fpr_micro, tpr_micro,
                color=PALETTE["primary"], linewidth=2.0,
                label=f"Micro-avg (AUC = {auc_micro:.2f})")

    class_aucs = []
    for i in range(n):
        try:
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], probs[:, i])
            auc_i = auc(fpr_i, tpr_i)
            class_aucs.append((i, auc_i, fpr_i, tpr_i))
        except Exception:
            pass

    if class_aucs:
        macro_auc = np.mean([a[1] for a in class_aucs])
        ax_roc.plot([], [], ' ', label=f"Macro-avg AUC = {macro_auc:.2f}")

    class_aucs.sort(key=lambda x: x[1], reverse=True)
    show_k = min(3, len(class_aucs))
    colors = sns.color_palette("husl", show_k)
    for idx, (i, auc_val, fpr_i, tpr_i) in enumerate(class_aucs[:show_k]):
        ax_roc.plot(fpr_i, tpr_i,
                    color=colors[idx], linewidth=1.0, alpha=0.8,
                    label=f"{labels[i]} ({auc_val:.2f})")

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("C) Receiver Operating Characteristic (ROC)", fontweight="bold")
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.05])

    # ----------------------------------------------------
    # PANEL D: Precision-Recall Curves
    # ----------------------------------------------------
    ax_pr = fig.add_subplot(gs[1, 1])
    precision_micro, recall_micro, _ = precision_recall_curve(y_bin.ravel(), probs.ravel())
    ap_micro = average_precision_score(y_bin, probs, average="micro")
    ax_pr.plot(recall_micro, precision_micro,
               color=PALETTE["secondary"], linewidth=2.0,
               label=f"Micro-avg (AP = {ap_micro:.2f})")

    class_aps = []
    for i in range(n):
        try:
            p_i, r_i, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
            ap_i = average_precision_score(y_bin[:, i], probs[:, i])
            class_aps.append((i, ap_i, r_i, p_i))
        except Exception:
            pass

    if class_aps:
        macro_ap = np.mean([a[1] for a in class_aps])
        ax_pr.plot([], [], ' ', label=f"Macro-avg AP = {macro_ap:.2f}")

    class_aps.sort(key=lambda x: x[1], reverse=True)
    show_k = min(3, len(class_aps))
    colors_pr = sns.color_palette("husl", show_k)
    for idx, (i, ap_val, r_i, p_i) in enumerate(class_aps[:show_k]):
        ax_pr.plot(r_i, p_i,
                   color=colors_pr[idx], linewidth=1.0, alpha=0.8,
                   label=f"{labels[i]} ({ap_val:.2f})")

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("D) Precision-Recall Curve (PRC)", fontweight="bold")
    ax_pr.legend(loc="lower left", fontsize=8)
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.05])

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.96)
    _save_fig(fig, out_path, dpi)


# ============================================================
# SIBLING EXPERIMENT DISCOVERY (for model comparison)
# ============================================================

def _discover_sibling_experiments(exp_dir: Path) -> List[Dict]:
    """
    Scan sibling directories for comparable experiments.
    Returns list of {name, accuracy, f1_macro, mcc} dicts.
    """
    parent = exp_dir.parent
    if not parent.exists():
        return []

    siblings = []
    for sibling in parent.iterdir():
        if not sibling.is_dir():
            continue
        eval_files = list(sibling.glob("*_eval_results.json"))
        if not eval_files:
            continue
        try:
            with open(eval_files[0], "r") as f:
                data = json.load(f)
            model = data.get("model", sibling.name)
            test_metrics = data.get("splits", {}).get("test", {}).get("metrics", {})
            if not test_metrics:
                # Try summary
                test_metrics = data.get("summary", {}).get("test", {})
            if test_metrics:
                siblings.append({
                    "name": model,
                    "dir": str(sibling),
                    "accuracy": test_metrics.get("accuracy", 0),
                    "f1_macro": test_metrics.get("f1_macro", 0),
                    "mcc": test_metrics.get("mcc", 0),
                })
        except Exception:
            continue

    return siblings


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def run_all_plots(
    exp_dir: str,
    dpi: int = 500,
    no_staging: bool = False,
    no_embeddings: bool = True,
    **kwargs,
) -> None:
    """
    Public API for backward compatibility with analyze_experiment.py.
    Generates the complete research-grade visualization package.
    """
    generate_research_plots(
        exp_dir=exp_dir,
        dpi=dpi,
        embeddings=not no_embeddings,
    )


def generate_research_plots(
    exp_dir: str,
    dpi: int = 500,
    embeddings: bool = False,
) -> None:
    """
    MAIN ENTRY POINT.

    Discovers evaluation artifacts, generates all high-value
    research figures, and writes summary reports.
    """
    _apply_global_style()

    exp_path = Path(exp_dir)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_path}")

    print(f"\n{'='*60}")
    print(f"  RESEARCH VISUALIZATION PIPELINE")
    print(f"  Experiment: {exp_path.name}")
    print(f"{'='*60}")

    cfg = _load_config(exp_path)
    model_name = cfg.get("model", {}).get("name", exp_path.name)
    stage = cfg.get("task", {}).get("stage", "unknown")

    # Discover evaluation results
    eval_paths = _find_eval_results(exp_path)
    if not eval_paths:
        print("[WARNING] No *_eval_results.json found. Attempting to plot from predictions only.")
        # Create minimal results from predictions if available
        eval_paths = []

    # Output directory
    out_root = exp_path / "research_plots"
    subdirs = {
        "confusion":    out_root / "confusion_matrices",
        "class":        out_root / "class_analysis",
        "roc":          out_root / "roc_curves",
        "grouped":      out_root / "grouped_analysis",
        "ood":          out_root / "ood_analysis",
        "stages":       out_root / "stage_comparisons",
        "models":       out_root / "model_comparisons",
        "embeddings":   out_root / "embeddings",
        "summaries":    out_root / "summaries",
        "panels":       out_root / "publication_panels",
        "learning":     out_root / "learning_curves",
    }
    for d in subdirs.values():
        _ensure_dir(d)

    figures_generated = []
    pred_dir = exp_path / "predictions"
    emb_dir  = exp_path / "embeddings"

    # --------------------------------------------------------
    # Process each evaluation result file
    # --------------------------------------------------------
    stage_summaries = []

    for eval_path in eval_paths:
        eval_stage = eval_path.stem.replace("_eval_results", "")
        stage_display = _stage_title(eval_stage)

        print(f"\n  Processing: {eval_stage}")
        with open(eval_path, "r") as f:
            results = json.load(f)

        # Determine n_classes from results
        first_split = next(iter(results.get("splits", {})), None)
        if first_split is None:
            continue
        first_split_data = results["splits"][first_split]
        n_classes = first_split_data.get("n_classes", 0)
        if n_classes == 0:
            # Infer from confusion matrix
            cm = first_split_data.get("confusion_matrix", [])
            n_classes = len(cm) if cm else 5

        labels = _resolve_labels(cfg, n_classes)

        # Metrics CSV
        csv_path = subdirs["summaries"] / f"metrics_{eval_stage}.csv"
        write_metrics_csv(results, csv_path)
        figures_generated.append(str(csv_path.relative_to(out_root)))

        # ---- Per-split figures ----
        splits = list(results.get("splits", {}).keys())
        for split in splits:
            split_data = results["splits"][split]

            # Try loading prediction arrays
            preds_bundle = _load_predictions(pred_dir, split)
            if preds_bundle is not None:
                logits, probs, targets = preds_bundle
                preds = probs.argmax(axis=-1)

                # Confusion matrices (normalized + raw)
                for norm, tag in [(True, "normalized"), (False, "raw")]:
                    cm_path = subdirs["confusion"] / f"confusion_{split}_{tag}.png"
                    plot_confusion_matrix(
                        targets, preds, labels,
                        f"{stage_display} — {_split_display(split)}",
                        cm_path, dpi, normalize=norm,
                    )
                    figures_generated.append(str(cm_path.relative_to(out_root)))
                    print(f"    [OK] Confusion matrix ({split}, {tag})")

                # Per-class failure heatmap
                hm_path = subdirs["class"] / f"failure_heatmap_{split}.png"
                plot_failure_heatmap(
                    targets, preds, labels,
                    f"{stage_display} — {_split_display(split)}",
                    hm_path, dpi,
                )
                figures_generated.append(str(hm_path.relative_to(out_root)))
                print(f"    [OK] Failure heatmap ({split})")

                # Clean ROC
                roc_path = subdirs["roc"] / f"roc_{split}.png"
                plot_roc_clean(
                    probs, targets, labels,
                    f"ROC — {stage_display} — {_split_display(split)}",
                    roc_path, dpi,
                )
                figures_generated.append(str(roc_path.relative_to(out_root)))
                print(f"    [OK] ROC curve ({split})")

                # Publication panel composite
                panel_path = subdirs["panels"] / f"publication_panel_{split}.png"
                plot_publication_panel(
                    targets, preds, probs, labels,
                    f"Composite Analysis — {stage_display} — {_split_display(split)}",
                    panel_path, dpi,
                )
                figures_generated.append(str(panel_path.relative_to(out_root)))
                print(f"    [OK] Publication panel ({split})")

            # Grouped confusion from eval results
            group_metrics = split_data.get("group_metrics", {})
            if group_metrics and "targets" in group_metrics and "predictions" in group_metrics:
                group_labels = labels[:n_classes]
                g_tgts = np.array(group_metrics["targets"])
                g_preds = np.array(group_metrics["predictions"])
                if len(g_tgts) > 0:
                    gcm_path = subdirs["confusion"] / f"grouped_confusion_{split}_normalized.png"
                    plot_confusion_matrix(
                        g_tgts, g_preds, group_labels,
                        f"{stage_display} — {_split_display(split)} (Grouped)",
                        gcm_path, dpi, normalize=True,
                    )
                    figures_generated.append(str(gcm_path.relative_to(out_root)))
                    print(f"    [OK] Grouped confusion matrix ({split})")

            # Embeddings (optional)
            if embeddings:
                emb_data = _load_embeddings(emb_dir, split)
                if emb_data is not None:
                    features, emb_targets = emb_data
                    tsne_path = subdirs["embeddings"] / f"tsne_{split}.png"
                    plot_embedding_tsne(
                        features, emb_targets, labels,
                        f"{stage_display} — {_split_display(split)}",
                        tsne_path, dpi,
                    )
                    figures_generated.append(str(tsne_path.relative_to(out_root)))
                    print(f"    [OK] t-SNE embedding ({split})")

        # ---- Cross-split figures ----

        # Grouped vs Spectrum
        grp_path = subdirs["grouped"] / f"grouped_vs_spectrum_{eval_stage}.png"
        plot_grouped_vs_spectrum(results, stage_display, grp_path, dpi)
        figures_generated.append(str(grp_path.relative_to(out_root)))
        print(f"    [OK] Grouped vs Spectrum")

        # OOD Degradation
        ood_path = subdirs["ood"] / f"ood_degradation_{eval_stage}.png"
        plot_ood_degradation(results, stage_display, ood_path, dpi)
        figures_generated.append(str(ood_path.relative_to(out_root)))
        print(f"    [OK] OOD Degradation")

        # Collect stage summary for stage-transition plot
        test_metrics = results.get("splits", {}).get("test", {}).get("metrics", {})
        if test_metrics:
            stage_summaries.append({
                "stage": eval_stage,
                "display": stage_display,
                "accuracy": test_metrics.get("accuracy", 0),
                "f1_macro": test_metrics.get("f1_macro", 0),
            })

    # ---- Cross-stage figures ----
    if len(stage_summaries) >= 2:
        trans_path = subdirs["stages"] / "stage_transition.png"
        plot_stage_transition(stage_summaries, trans_path, dpi)
        figures_generated.append(str(trans_path.relative_to(out_root)))
        print(f"\n    [OK] Stage transition comparison")

    # ---- Model comparison (sibling discovery) ----
    siblings = _discover_sibling_experiments(exp_path)
    if len(siblings) >= 2:
        comp_path = subdirs["models"] / "model_comparison_test.png"
        plot_model_comparison(siblings, "test", comp_path, dpi)
        figures_generated.append(str(comp_path.relative_to(out_root)))
        print(f"    [OK] Model comparison ({len(siblings)} architectures)")

    # ---- Learning curves ----
    metrics_json = exp_path / "metrics.json"
    if metrics_json.exists():
        plot_learning_curves(
            metrics_json,
            f"{model_name} — {_stage_title(stage)}",
            subdirs["learning"],
            dpi,
        )
        figures_generated.append("learning_curves/learning_curves.png")
        print(f"    [OK] Learning curves")

    # ---- Summary report ----
    summary_path = subdirs["summaries"] / "visualization_report.md"
    write_summary_md(
        exp_path, model_name, stage,
        results if eval_paths else {},
        figures_generated,
        summary_path,
    )

    # ---- Cleanup ----
    plt.close("all")

    print(f"\n{'='*60}")
    print(f"  VISUALIZATION COMPLETE")
    print(f"  Output: {out_root}")
    print(f"  Figures: {len(figures_generated)}")
    print(f"{'='*60}\n")


# ============================================================
# CLI ENTRY POINT
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate research-grade visualization package for one experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_research_plots.py --exp_dir experiments/resnet_s3_transfer
  python scripts/generate_research_plots.py --exp_dir experiments/resnet_s3_transfer --embeddings
  python scripts/generate_research_plots.py --exp_dir experiments/resnet_s3_transfer --dpi 300
        """,
    )
    p.add_argument("--exp_dir", required=True, help="Path to experiment directory")
    p.add_argument("--dpi", type=int, default=500, help="Export DPI (default: 500)")
    p.add_argument("--embeddings", action="store_true",
                   help="Generate t-SNE/UMAP embedding plots (disabled by default)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    generate_research_plots(
        exp_dir=args.exp_dir,
        dpi=args.dpi,
        embeddings=args.embeddings,
    )


if __name__ == "__main__":
    main()
