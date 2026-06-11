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
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
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
    "clinical_all":   "Clinical All (OOD)",
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
        "savefig.dpi":        500,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
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
            # pyrefly: ignore [missing-import]
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
    direct_matches = sorted(exp_dir.glob("*_eval_results.json"))
    if direct_matches:
        return direct_matches

    recursive_matches = sorted(exp_dir.rglob("*_eval_results.json"))
    if recursive_matches:
        return recursive_matches

    analysis_dir = exp_dir / "analysis"
    if analysis_dir.exists():
        analysis_matches = sorted(analysis_dir.glob("*_eval_results.json"))
        if analysis_matches:
            return analysis_matches

        analysis_recursive_matches = sorted(analysis_dir.rglob("*_eval_results.json"))
        if analysis_recursive_matches:
            return analysis_recursive_matches

    parent_analysis_dir = exp_dir.parent / "analysis"
    if parent_analysis_dir.exists():
        parent_matches = sorted(parent_analysis_dir.glob("*_eval_results.json"))
        if parent_matches:
            return parent_matches

        parent_recursive_matches = sorted(parent_analysis_dir.rglob("*_eval_results.json"))
        if parent_recursive_matches:
            return parent_recursive_matches

    return []


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
    try:
        fig.tight_layout()
    except Exception:
        pass
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
        cm_plot = np.rint((cm_plot / row_sums) * 100).astype(int)
    else:
        cm_plot = cm

    # Adaptive sizing
    large = n > 10
    fig_w = max(10, n * 0.5) if large else 10
    fig_h = max(9, n * 0.45) if large else 9
    annot = not large
    if annot:
        if normalize:
            annot_data = np.array([[str(int(v)) if v > 0 else "" for v in row] for row in cm_plot])
        else:
            annot_data = np.array([["" if int(v) == 0 else str(int(v)) for v in row] for row in cm_plot])
    else:
        annot_data = False

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
        annot=annot_data,
        fmt="",
        cbar=True,
        vmin=0 if normalize else None,
        vmax=100 if normalize else None,
        square=True,
        linewidths=0.3 if not large else 0.1,
        linecolor="#DDDDDD",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": max(9, 13 - n // 5), "weight": "bold"} if annot else {},
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
    if n_classes < 2 or probs is None or len(targets) == 0:
        return
    if probs.ndim != 2 or probs.shape[1] < n_classes:
        return
    if len(np.unique(targets)) < 2:
        return

    y_bin = label_binarize(targets, classes=list(range(n_classes)))
    if y_bin.ndim == 1:
        y_bin = np.column_stack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(12, 8))

    # Micro-average
    try:
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs[:, :n_classes].ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        ax.plot(fpr_micro, tpr_micro,
                color=PALETTE["primary"], linewidth=2.5,
                label=f"Micro-average (AUC = {auc_micro:.3f})")
    except Exception:
        plt.close(fig)
        return

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
    splits = []
    spec_acc, grp_acc = [], []
    spec_f1, grp_f1 = [], []
    spec_mcc, grp_mcc = [], []

    for split_name, data in results.get("splits", {}).items():
        metrics = (
            data.get("metrics")
            or data.get("spectrum_metrics")
            or {}
        )

        group_metrics = (
            data.get("group_metrics")
            or data.get("patient_metrics")
            or {}
        )
        if not group_metrics or "accuracy" not in group_metrics:
            continue
        splits.append(_split_display(split_name))
        spec_acc.append(metrics.get("accuracy", 0.0))
        grp_acc.append(group_metrics.get("accuracy", 0.0))
        spec_f1.append(metrics.get("f1_macro", 0.0))
        grp_f1.append(group_metrics.get("f1_macro", 0.0))
        spec_mcc.append(metrics.get("mcc", 0.0))
        grp_mcc.append(group_metrics.get("mcc", 0.0))

    if not splits:
        return

    metric_names = ["Accuracy", "Macro F1", "MCC"]
    
    if not spec_acc:
        return

    spectrum_vals = [
        np.mean(spec_acc),
        np.mean(spec_f1),
        np.mean(spec_mcc),
    ]

    patient_vals = [
        np.mean(grp_acc),
        np.mean(grp_f1),
        np.mean(grp_mcc),
    ]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    bars1 = ax.bar(
        x - width/2,
        spectrum_vals,
        width,
        label="Spectrum-level",
        color=PALETTE["primary"]
    )

    bars2 = ax.bar(
        x + width/2,
        patient_vals,
        width,
        label="Patient-level",
        color=PALETTE["accent"]
    )
    
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width()/2,
            h + 0.01,
            f"{h*100:.1f}%",
            ha="center",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ymin = min(
        min(spectrum_vals),
        min(patient_vals)
    )

    ax.set_ylim(max(0, ymin - 0.1), 1.02)
    ax.set_ylabel("Score")
    ax.set_title(
        f"Patient vs Spectrum Performance\n{title}",
        pad=15
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

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
    if any("patient_accuracy" in m for m in model_data):
        metrics_dict["Patient Acc."] = [m.get("patient_accuracy", 0) for m in model_data]
    if any("patient_f1_macro" in m for m in model_data):
        metrics_dict["Patient F1"] = [m.get("patient_f1_macro", 0) for m in model_data]
    metric_names = list(metrics_dict.keys())

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))

    x = np.arange(len(names))
    total_width = 0.7
    bar_width = total_width / len(metric_names)
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["highlight"], PALETTE["muted"]]

    for idx, (mname, values) in enumerate(metrics_dict.items()):
        offset = (idx - len(metric_names) / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width,
                     label=mname, color=colors[idx % len(colors)], alpha=0.85, edgecolor="white")
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
    from sklearn.metrics import silhouette_score

    perplexity = min(30, max(5, len(features) // 10))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    coords = tsne.fit_transform(features)
    sil_score = None

    try:
        sil_score = silhouette_score(features, targets)
    except Exception:
        pass

    n = len(labels)
    palette = sns.color_palette("husl", n)

    fig, ax = plt.subplots(figsize=(12, 8))
    for i, label in enumerate(labels):
        mask = targets == i
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                  s=12, label=label, alpha=0.7, color=palette[i])

    if sil_score is not None:
        ax.set_title(
            f"t-SNE — {title}\nSilhouette Score = {sil_score:.3f}",
            pad=12,
        )
    else:
        ax.set_title(
            f"t-SNE — {title}",
            pad=12,
        )
    ax.set_xticks([])
    ax.set_yticks([])

    # Legend below plot
    ncol = min(5, max(3, n // 6))
    ax.legend(
        fontsize=7,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=True,
        markerscale=2,
    )
    fig.subplots_adjust(bottom=0.22)
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
# PATIENT-CV AND AGGREGATED RESULT SUPPORT
# ============================================================

METRIC_KEYS = ["accuracy", "f1_macro", "mcc", "precision_macro", "recall_macro"]
METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "f1_macro": "Macro-F1",
    "mcc": "MCC",
    "precision_macro": "Precision",
    "recall_macro": "Recall",
}


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fold_sort_key(path: Path) -> Tuple[int, str]:
    digits = "".join(ch for ch in path.name if ch.isdigit())
    return (int(digits) if digits else 999, path.name)


def _discover_fold_dirs(exp_dir: Path) -> List[Path]:
    fold_dirs = [
        d for d in exp_dir.iterdir()
        if d.is_dir() and (d.name.startswith("fold_") or "_fold" in d.name)
    ]
    return sorted(fold_dirs, key=_fold_sort_key)


def _find_aggregated_results(exp_dir: Path) -> Optional[Path]:
    candidates = [
        exp_dir / "aggregated_cv_results.json",
        exp_dir / "analysis" / "aggregated_cv_results.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    nested = sorted(exp_dir.rglob("aggregated_cv_results.json"))
    return nested[0] if nested else None


def _metric_value(metrics: dict, key: str) -> Optional[float]:
    aliases = {
        "precision_macro": ["precision_macro", "precision"],
        "recall_macro": ["recall_macro", "recall"],
        "f1_macro": ["f1_macro", "macro_f1"],
    }
    for candidate in aliases.get(key, [key]):
        value = metrics.get(candidate)
        if isinstance(value, (int, float)) and not np.isnan(value):
            return float(value)
    return None


def _compute_metrics_from_labels(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    if len(targets) == 0 or len(preds) == 0:
        return {}
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "precision_macro": float(precision_score(targets, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(targets, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(targets, preds)),
    }


def _patient_vote_local(
    probabilities: np.ndarray,
    targets: np.ndarray,
    patient_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if probabilities.size == 0 or len(targets) == 0 or len(patient_ids) != len(targets):
        return np.array([]), np.array([])
    preds, tgts = [], []
    for pid in sorted(set(patient_ids.tolist())):
        mask = patient_ids == pid
        pid_targets = targets[mask]
        if len(pid_targets) == 0 or len(np.unique(pid_targets)) != 1:
            continue
        preds.append(int(np.argmax(probabilities[mask].mean(axis=0))))
        tgts.append(int(pid_targets[0]))
    return np.asarray(preds), np.asarray(tgts)


def _metrics_from_detailed_split(split_data: dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    targets = np.asarray(split_data.get("targets") or [])
    preds = np.asarray(split_data.get("predictions") or [])
    spectrum_metrics = _compute_metrics_from_labels(targets, preds)

    grouped_preds = split_data.get("grouped_predictions")
    grouped_targets = split_data.get("grouped_targets")
    if grouped_preds is not None and grouped_targets is not None:
        patient_metrics = _compute_metrics_from_labels(np.asarray(grouped_targets), np.asarray(grouped_preds))
    else:
        probs = np.asarray(split_data.get("probabilities") or [])
        pids = split_data.get("patient_ids")
        if pids is None:
            patient_metrics = {}
        else:
            patient_preds, patient_targets = _patient_vote_local(probs, targets, np.asarray(pids))
            patient_metrics = _compute_metrics_from_labels(patient_targets, patient_preds)
    return spectrum_metrics, patient_metrics


def _add_metric_row(rows: List[Dict], fold: str, split: str, level: str, metrics: dict, n_value: Optional[int] = None) -> None:
    row = {"fold": fold, "split": split, "level": level}
    if n_value is not None:
        row["n"] = int(n_value)
    for key in METRIC_KEYS:
        value = _metric_value(metrics, key)
        if value is not None:
            row[key] = value
    if any(key in row for key in METRIC_KEYS):
        rows.append(row)


def collect_fold_metric_rows(fold_dirs: List[Path]) -> List[Dict]:
    rows: List[Dict] = []
    for fold_dir in fold_dirs:
        eval_files = sorted(fold_dir.glob("*_eval_results.json"))
        detailed_path = fold_dir / "detailed_predictions.json"
        detailed = _load_json(detailed_path) if detailed_path.exists() else {}

        if eval_files:
            data = _load_json(eval_files[0])
            for split_name, split_data in data.get("splits", {}).items():
                _add_metric_row(
                    rows,
                    fold_dir.name,
                    split_name,
                    "Spectrum",
                    split_data.get("metrics", {}),
                    split_data.get("n_samples"),
                )
                group_metrics = split_data.get("group_metrics", {})
                if group_metrics:
                    if "precision_macro" not in group_metrics and "targets" in group_metrics and "predictions" in group_metrics:
                        computed = _compute_metrics_from_labels(
                            np.asarray(group_metrics["targets"]),
                            np.asarray(group_metrics["predictions"]),
                        )
                        group_metrics = {**group_metrics, **computed}
                    _add_metric_row(
                        rows,
                        fold_dir.name,
                        split_name,
                        "Patient",
                        group_metrics,
                        group_metrics.get("n_groups"),
                    )

        for split_name, split_data in detailed.items():
            spectrum_metrics, patient_metrics = _metrics_from_detailed_split(split_data)
            known = {(r["fold"], r["split"], r["level"]) for r in rows}
            if (fold_dir.name, split_name, "Spectrum") not in known:
                _add_metric_row(rows, fold_dir.name, split_name, "Spectrum", spectrum_metrics, len(split_data.get("targets") or []))
            if patient_metrics and (fold_dir.name, split_name, "Patient") not in known:
                n_patients = len(set(split_data.get("patient_ids") or [])) if split_data.get("patient_ids") else None
                _add_metric_row(rows, fold_dir.name, split_name, "Patient", patient_metrics, n_patients)
    return rows


def _write_rows_csv(rows: List[Dict], out_path: Path) -> None:
    if not rows:
        return
    _ensure_dir(out_path.parent)
    fieldnames = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_cross_fold_stats(rows: List[Dict]) -> List[Dict]:
    stats_rows: List[Dict] = []
    groups = sorted({(r["split"], r["level"]) for r in rows})
    for split, level in groups:
        subset = [r for r in rows if r["split"] == split and r["level"] == level]
        for metric in METRIC_KEYS:
            values = np.asarray([r[metric] for r in subset if metric in r], dtype=float)
            if values.size == 0:
                continue
            stats_rows.append({
                "split": split,
                "level": level,
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "n_folds": int(values.size),
                "formatted": _format_mean_std(metric, float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0),
            })
    return stats_rows


def _format_mean_std(metric: str, mean: float, std: float) -> str:
    if metric == "mcc":
        return f"{mean:.2f} +/- {std:.2f}"
    return f"{mean * 100:.1f} +/- {std * 100:.1f}"


def plot_fold_metric_comparison(rows: List[Dict], metric: str, level: str, out_path: Path, dpi: int) -> None:
    subset = [r for r in rows if r.get("level") == level and metric in r]
    if not subset:
        return
    folds = sorted({r["fold"] for r in subset}, key=lambda x: _fold_sort_key(Path(x)))
    splits = sorted({r["split"] for r in subset})
    x = np.arange(len(folds))

    fig, ax = plt.subplots(figsize=(max(8, len(folds) * 1.2), 5))
    for idx, split in enumerate(splits):
        values = []
        for fold in folds:
            match = next((r for r in subset if r["fold"] == fold and r["split"] == split), None)
            values.append(match.get(metric, np.nan) if match else np.nan)
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2.0,
            markersize=7,
            color=MODEL_COLORS[idx % len(MODEL_COLORS)],
            label=_split_display(split),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(folds, rotation=0)
    ax.set_ylabel(METRIC_DISPLAY[metric])
    ax.set_title(f"Fold-wise {level}-level {METRIC_DISPLAY[metric]}", pad=12)
    ax.set_ylim(-0.05 if metric == "mcc" else 0, 1.05)
    ax.legend(loc="best", fontsize=8)
    _save_fig(fig, out_path, dpi)


def plot_cross_fold_statistics(stats_rows: List[Dict], out_path: Path, dpi: int) -> None:
    if not stats_rows:
        return
    preferred_split = next(
        (split for split in ["clinical_all", "test", "2018clinical", "2019clinical"]
         if any(r["split"] == split for r in stats_rows)),
        stats_rows[0]["split"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, level in zip(axes, ["Spectrum", "Patient"]):
        subset = [r for r in stats_rows if r["split"] == preferred_split and r["level"] == level]
        if not subset:
            ax.axis("off")
            ax.set_title(f"{level}-level unavailable")
            continue
        values = [_metric_value({"value": r["mean"]}, "value") for r in subset]
        errors = [r["std"] for r in subset]
        labels = [METRIC_DISPLAY[r["metric"]] for r in subset]
        x = np.arange(len(labels))
        ax.bar(x, values, yerr=errors, capsize=4, color=PALETTE["primary"] if level == "Spectrum" else PALETTE["accent"], alpha=0.88, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(-0.05 if any(r["metric"] == "mcc" for r in subset) else 0, 1.08)
        ax.set_ylabel("Mean score")
        ax.set_title(f"{level}-level ({_split_display(preferred_split)})", fontweight="bold")
        for i, r in enumerate(subset):
            ax.text(i, r["mean"] + r["std"] + 0.02, _format_mean_std(r["metric"], r["mean"], r["std"]), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Cross-fold Mean +/- SD", fontsize=14, fontweight="bold", y=1.03)
    _save_fig(fig, out_path, dpi)


def write_cross_fold_report(rows: List[Dict], stats_rows: List[Dict], out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    lines = ["# Patient-CV Cross-Fold Summary", ""]
    if not rows:
        lines.append("No fold-level metrics were available.")
    else:
        lines.append(f"Folds detected: {len(sorted({r['fold'] for r in rows}))}")
        lines.append("")
        lines.append("## Mean +/- SD")
        lines.append("")
        for row in stats_rows:
            label = f"{row['level']} {_split_display(row['split'])} {METRIC_DISPLAY[row['metric']]}"
            lines.append(f"- {label}: **{row['formatted']}** (min {row['min']:.4f}, max {row['max']:.4f})")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_confusion_matrix_from_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    out_path: Path,
    dpi: int,
    normalize: bool,
) -> None:
    targets, preds = [], []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            if count > 0:
                targets.extend([i] * count)
                preds.extend([j] * count)
    if not targets:
        return
    plot_confusion_matrix(np.asarray(targets), np.asarray(preds), labels, title, out_path, dpi, normalize=normalize)


def _n_classes_from_aggregate(aggregate: dict) -> int:
    for split_data in aggregate.get("splits", {}).values():
        for key in ["spectrum_confusion_matrix", "patient_confusion_matrix"]:
            cm = split_data.get(key)
            if cm:
                return len(cm)
    return 5


def _bundle_from_detailed(fold_dirs: List[Path], split: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    probs_list, targets_list, preds_list, pid_list = [], [], [], []
    for fold_dir in fold_dirs:
        detailed_path = fold_dir / "detailed_predictions.json"
        if not detailed_path.exists():
            continue
        detailed = _load_json(detailed_path)
        split_names = [split]
        if split == "clinical_all":
            split_names = [name for name in detailed.keys() if "clinical" in name]
        for split_name in split_names:
            data = detailed.get(split_name)
            if not data:
                continue
            if data.get("probabilities") is None or data.get("targets") is None:
                continue
            probs_list.append(np.asarray(data["probabilities"]))
            targets_list.append(np.asarray(data["targets"]))
            preds_list.append(np.asarray(data.get("predictions") or np.asarray(data["probabilities"]).argmax(axis=-1)))
            pids = data.get("patient_ids")
            if pids is not None:
                pid_list.append(np.asarray(pids))
    if not probs_list:
        return None
    probs = np.concatenate(probs_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    preds = np.concatenate(preds_list, axis=0)
    pids = np.concatenate(pid_list, axis=0) if pid_list else np.array([])
    return probs, targets, preds, pids


def process_aggregated_cv_results(
    aggregate_path: Path,
    fold_dirs: List[Path],
    labels: List[str],
    subdirs: Dict[str, Path],
    out_root: Path,
    dpi: int,
) -> List[str]:
    figures: List[str] = []
    aggregate = _load_json(aggregate_path)
    rows = []
    for split_name, split_data in aggregate.get("splits", {}).items():
        _add_metric_row(rows, "aggregated", split_name, "Spectrum", split_data.get("spectrum_metrics", {}), split_data.get("spectrum_metrics", {}).get("n_samples"))
        _add_metric_row(rows, "aggregated", split_name, "Patient", split_data.get("patient_metrics", {}), split_data.get("patient_metrics", {}).get("n_patients"))
        
        # --------------------------------------------------
        # Grouped vs Spectrum comparison
        # --------------------------------------------------

        grouped_path = (
            subdirs["grouped"]
            / f"clinical_cv_grouped_vs_spectrum_{split_name}.png"
        )
        
        print(f"[DEBUG] Creating grouped plot: {grouped_path}")

        plot_grouped_vs_spectrum(
            {
                "splits": {
                    split_name: {
                        "metrics": split_data.get("spectrum_metrics", {}),
                        "group_metrics": split_data.get("patient_metrics", {}),
                    }
                }
            },
            f"Aggregated {_split_display(split_name)}",
            grouped_path,
            dpi,
        )
        
        print(f"[DEBUG] Exists after plotting: {grouped_path.exists()}")
        
        if grouped_path.exists():
            figures.append(str(grouped_path.relative_to(out_root)))

        for level, cm_key, stem in [
            ("Spectrum", "spectrum_confusion_matrix", "spectrum"),
            ("Patient", "patient_confusion_matrix", "patient"),
        ]:
            cm = np.asarray(split_data.get(cm_key) or [])
            if cm.size == 0:
                continue
            for normalize, tag in [(False, "raw"), (True, "normalized")]:
                path = subdirs["confusion"] / f"clinical_cv_{split_name}_{stem}_{tag}.png"
                plot_confusion_matrix_from_matrix(
                    cm,
                    labels[: cm.shape[0]],
                    f"Aggregated {_split_display(split_name)} {level}-level",
                    path,
                    dpi,
                    normalize=normalize,
                )
                if path.exists():
                    figures.append(str(path.relative_to(out_root)))

        bundle = _bundle_from_detailed(fold_dirs, split_name)
        if bundle is not None:
            probs, targets, preds, pids = bundle
            roc_path = subdirs["roc"] / f"clinical_cv_roc_{split_name}.png"
            plot_roc_clean(probs, targets, labels[: probs.shape[1]], f"ROC - Aggregated {_split_display(split_name)}", roc_path, dpi)
            if roc_path.exists():
                figures.append(str(roc_path.relative_to(out_root)))
            if split_name == "clinical_all" and len(pids) == len(targets):
                patient_preds, patient_targets = _patient_vote_local(probs, targets, pids)
                for normalize, tag in [(False, "raw"), (True, "normalized")]:
                    path = subdirs["confusion"] / f"clinical_all_patient_from_probabilities_{tag}.png"
                    plot_confusion_matrix(patient_targets, patient_preds, labels[: probs.shape[1]], "Clinical All Patient-level", path, dpi, normalize=normalize)
                    if path.exists():
                        figures.append(str(path.relative_to(out_root)))

    csv_path = subdirs["summaries"] / "aggregated_cv_metrics.csv"
    _write_rows_csv(rows, csv_path)
    if csv_path.exists():
        figures.append(str(csv_path.relative_to(out_root)))

    report_path = subdirs["summaries"] / "clinical_all_summary.md"
    write_clinical_all_summary(aggregate, report_path)
    if report_path.exists():
        figures.append(str(report_path.relative_to(out_root)))
    return figures


def write_clinical_all_summary(aggregate: dict, out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    lines = ["# Aggregated Clinical CV Summary", ""]
    split_data = aggregate.get("splits", {}).get("clinical_all")
    if not split_data:
        lines.append("`clinical_all` was not present in aggregated CV results.")
    else:
        for level, key in [("Spectrum", "spectrum_metrics"), ("Patient", "patient_metrics")]:
            metrics = split_data.get(key, {})
            lines.append(f"## {level}-level clinical_all")
            for metric in METRIC_KEYS:
                value = _metric_value(metrics, metric)
                if value is not None:
                    lines.append(f"- {METRIC_DISPLAY[metric]}: **{value:.4f}**")
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_domain_shift_outputs(metric_rows: List[Dict], subdirs: Dict[str, Path], out_root: Path, dpi: int) -> List[str]:
    figures: List[str] = []
    splits = {r["split"] for r in metric_rows}
    required = {"test", "2018clinical", "2019clinical", "clinical_all"}
    if not required.issubset(splits):
        return figures

    rows = [r for r in metric_rows if r["split"] in required and r["level"] in ("Spectrum", "Patient")]
    csv_path = subdirs["ood"] / "domain_shift_summary.csv"
    _write_rows_csv(rows, csv_path)
    if csv_path.exists():
        figures.append(str(csv_path.relative_to(out_root)))

    levels = [level for level in ["Spectrum", "Patient"] if any(r["level"] == level for r in rows)]
    fig, axes = plt.subplots(1, len(levels), figsize=(7 * len(levels), 5), squeeze=False)
    split_order = ["test", "2018clinical", "2019clinical", "clinical_all"]
    for ax, level in zip(axes[0], levels):
        level_rows = [r for r in rows if r["level"] == level]
        x = np.arange(len(split_order))
        width = 0.25
        for idx, metric in enumerate(["accuracy", "f1_macro", "mcc"]):
            values = []
            for split in split_order:
                match = next((r for r in level_rows if r["split"] == split), None)
                values.append(match.get(metric, np.nan) if match else np.nan)
            ax.bar(x + (idx - 1) * width, values, width, label=METRIC_DISPLAY[metric], color=MODEL_COLORS[idx], alpha=0.88, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([_split_display(s) for s in split_order], rotation=20, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Score")
        ax.set_title(f"{level}-level domain generalization", fontweight="bold")
        ax.legend(fontsize=8)
    fig.suptitle("Stage 3 In-domain vs Clinical Domain Shift", fontsize=14, fontweight="bold", y=1.03)
    plot_path = subdirs["ood"] / "stage3_domain_shift_comparison.png"
    _save_fig(fig, plot_path, dpi)
    if plot_path.exists():
        figures.append(str(plot_path.relative_to(out_root)))

    report_path = subdirs["ood"] / "domain_shift_interpretation.md"
    lines = ["# Stage 3 Domain Generalization Interpretation", ""]
    for level in levels:
        test_row = next((r for r in rows if r["level"] == level and r["split"] == "test"), None)
        clinical_row = next((r for r in rows if r["level"] == level and r["split"] == "clinical_all"), None)
        if not test_row or not clinical_row:
            continue
        lines.append(f"## {level}-level")
        for metric in ["accuracy", "f1_macro", "mcc"]:
            if metric in test_row and metric in clinical_row:
                gap = test_row[metric] - clinical_row[metric]
                lines.append(f"- {METRIC_DISPLAY[metric]} clinical degradation: **{gap:.4f}**")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    figures.append(str(report_path.relative_to(out_root)))
    return figures


def collect_aggregate_metric_rows(aggregate: dict) -> List[Dict]:
    rows: List[Dict] = []
    for split_name, split_data in aggregate.get("splits", {}).items():
        _add_metric_row(
            rows,
            "aggregated",
            split_name,
            "Spectrum",
            split_data.get("spectrum_metrics", {}),
            split_data.get("spectrum_metrics", {}).get("n_samples"),
        )
        _add_metric_row(
            rows,
            "aggregated",
            split_name,
            "Patient",
            split_data.get("patient_metrics", {}),
            split_data.get("patient_metrics", {}).get("n_patients"),
        )
    return rows


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
    cm_plot = np.rint((cm_plot / row_sums) * 100).astype(int)
    cm_annot = np.array([[str(int(v)) if v > 0 else "" for v in row] for row in cm_plot])

    cmap_blues = LinearSegmentedColormap.from_list(
        "research_blues",
        ["#FFFFFF", "#D6E8F5", "#7FB3D8", "#2D5F8A", "#1A3A5C"],
    )
    sns.heatmap(
        cm_plot,
        ax=ax_cm,
        cmap=cmap_blues,
        annot=cm_annot if n <= 10 else False,
        fmt="",
        cbar=True,
        vmin=0,
        vmax=100,
        square=True,
        linewidths=0.3 if n <= 10 else 0.1,
        linecolor="#DDDDDD",
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": max(9, 13 - n // 5), "weight": "bold"} if n <= 10 else {},
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
    roc_available = probs is not None and probs.ndim == 2 and probs.shape[1] >= n and len(np.unique(targets)) >= 2
    if roc_available:
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs[:, :n].ravel())
        auc_micro = auc(fpr_micro, tpr_micro)
        ax_roc.plot(fpr_micro, tpr_micro,
                    color=PALETTE["primary"], linewidth=2.0,
                    label=f"Micro-avg (AUC = {auc_micro:.2f})")
    else:
        ax_roc.text(0.5, 0.5, "ROC unavailable", ha="center", va="center", fontsize=12)

    class_aucs = []
    if roc_available:
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
    if roc_available:
        precision_micro, recall_micro, _ = precision_recall_curve(y_bin.ravel(), probs[:, :n].ravel())
        ap_micro = average_precision_score(y_bin, probs[:, :n], average="micro")
        ax_pr.plot(recall_micro, precision_micro,
                   color=PALETTE["secondary"], linewidth=2.0,
                   label=f"Micro-avg (AP = {ap_micro:.2f})")
    else:
        ax_pr.text(0.5, 0.5, "PR unavailable", ha="center", va="center", fontsize=12)

    class_aps = []
    if roc_available:
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
    Supports standard experiments and patient-CV aggregate parents.
    """
    own_experiment = exp_dir.parent if exp_dir.name == "analysis" else exp_dir
    parent = exp_dir.parent.parent if exp_dir.name == "analysis" else exp_dir.parent
    if not parent.exists():
        return []

    siblings = []
    for sibling in parent.iterdir():
        if not sibling.is_dir():
            continue
        if sibling.resolve() in {exp_dir.resolve(), own_experiment.resolve()} or sibling.name == "analysis":
            continue
        if sibling.name.startswith("fold_") or "_fold" in sibling.name:
            continue
        try:
            aggregate_path = _find_aggregated_results(sibling)
            if aggregate_path is not None:
                aggregate = _load_json(aggregate_path)
                split_data = (
                    aggregate.get("splits", {}).get("clinical_all")
                    or aggregate.get("splits", {}).get("test")
                    or next(iter(aggregate.get("splits", {}).values()), {})
                )
                spec = split_data.get("spectrum_metrics", {})
                pat = split_data.get("patient_metrics", {})
                if spec:
                    row = {
                        "name": _load_config(sibling).get("model", {}).get("name", sibling.name),
                        "dir": str(sibling),
                        "accuracy": spec.get("accuracy", 0),
                        "f1_macro": spec.get("f1_macro", 0),
                        "mcc": spec.get("mcc", 0),
                    }
                    if pat:
                        row.update({
                            "patient_accuracy": pat.get("accuracy", 0),
                            "patient_f1_macro": pat.get("f1_macro", 0),
                            "patient_mcc": pat.get("mcc", 0),
                        })
                    siblings.append(row)
                    continue

            eval_files = list(sibling.glob("*_eval_results.json"))
            if not eval_files:
                eval_files = [p for p in sibling.rglob("*_eval_results.json") if "research_plots" not in str(p)]
            if not eval_files:
                continue
            data = _load_json(eval_files[0])
            model = data.get("model", _load_config(sibling).get("model", {}).get("name", sibling.name))
            split_data = (
                data.get("splits", {}).get("clinical_all")
                or data.get("splits", {}).get("test")
                or next(iter(data.get("splits", {}).values()), {})
            )
            test_metrics = split_data.get("metrics", {}) or data.get("summary", {}).get("test", {})
            group_metrics = split_data.get("group_metrics", {})
            if test_metrics:
                row = {
                    "name": model,
                    "dir": str(sibling),
                    "accuracy": test_metrics.get("accuracy", 0),
                    "f1_macro": test_metrics.get("f1_macro", 0),
                    "mcc": test_metrics.get("mcc", 0),
                }
                if group_metrics:
                    row.update({
                        "patient_accuracy": group_metrics.get("accuracy", 0),
                        "patient_f1_macro": group_metrics.get("f1_macro", 0),
                        "patient_mcc": group_metrics.get("mcc", 0),
                    })
                siblings.append(row)
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
    fold_dirs = _discover_fold_dirs(exp_path)
    fold_names = {d.name for d in fold_dirs}
    aggregate_path = _find_aggregated_results(exp_path)
    if not cfg and fold_dirs:
        cfg = _load_config(fold_dirs[0])
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
        "crossfold":    out_root / "cross_fold",
    }
    for d in subdirs.values():
        _ensure_dir(d)

    figures_generated = []
    pred_dir = exp_path / "predictions"
    emb_dir  = exp_path / "embeddings"
    domain_metric_rows: List[Dict] = []
    labels_for_aggregate: Optional[List[str]] = None

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
        if labels_for_aggregate is None:
            labels_for_aggregate = labels

        # ---- Per-split figures ----
        splits = list(results.get("splits", {}).keys())
        artifact_root = eval_path.parent if eval_path.parent != exp_path else exp_path
        artifact_prefix = f"{artifact_root.name}_" if artifact_root.name in fold_names else ""
        pred_dir_loop = artifact_root / "predictions"
        emb_dir_loop = artifact_root / "embeddings"

        # Metrics CSV
        csv_path = subdirs["summaries"] / f"{artifact_prefix}metrics_{eval_stage}.csv"
        write_metrics_csv(results, csv_path)
        if csv_path.exists():
            figures_generated.append(str(csv_path.relative_to(out_root)))

        for split in splits:
            split_data = results["splits"][split]
            _add_metric_row(domain_metric_rows, artifact_root.name, split, "Spectrum", split_data.get("metrics", {}), split_data.get("n_samples"))
            if split_data.get("group_metrics"):
                _add_metric_row(domain_metric_rows, artifact_root.name, split, "Patient", split_data.get("group_metrics", {}), split_data.get("group_metrics", {}).get("n_groups"))

            # Try loading prediction arrays
            preds_bundle = _load_predictions(pred_dir_loop, split)
            if preds_bundle is not None:
                logits, probs, targets = preds_bundle
                preds = probs.argmax(axis=-1)

                # Confusion matrices (normalized + raw)
                for norm, tag in [(True, "normalized"), (False, "raw")]:
                    cm_path = subdirs["confusion"] / f"{artifact_prefix}confusion_{split}_{tag}.png"
                    plot_confusion_matrix(
                        targets, preds, labels,
                        f"{stage_display} — {_split_display(split)}",
                        cm_path, dpi, normalize=norm,
                    )
                    if cm_path.exists():
                        figures_generated.append(str(cm_path.relative_to(out_root)))
                        print(f"    [OK] Confusion matrix ({artifact_prefix}{split}, {tag})")

                # Per-class failure heatmap
                hm_path = subdirs["class"] / f"{artifact_prefix}failure_heatmap_{split}.png"
                plot_failure_heatmap(
                    targets, preds, labels,
                    f"{stage_display} — {_split_display(split)}",
                    hm_path, dpi,
                )
                if hm_path.exists():
                    figures_generated.append(str(hm_path.relative_to(out_root)))
                    print(f"    [OK] Failure heatmap ({artifact_prefix}{split})")

                # Clean ROC
                roc_path = subdirs["roc"] / f"{artifact_prefix}roc_{split}.png"
                plot_roc_clean(
                    probs, targets, labels,
                    f"ROC — {stage_display} — {_split_display(split)}",
                    roc_path, dpi,
                )
                if roc_path.exists():
                    figures_generated.append(str(roc_path.relative_to(out_root)))
                    print(f"    [OK] ROC curve ({artifact_prefix}{split})")

                # Publication panel composite
                panel_path = subdirs["panels"] / f"{artifact_prefix}publication_panel_{split}.png"
                plot_publication_panel(
                    targets, preds, probs, labels,
                    f"Composite Analysis — {stage_display} — {_split_display(split)}",
                    panel_path, dpi,
                )
                if panel_path.exists():
                    figures_generated.append(str(panel_path.relative_to(out_root)))
                    print(f"    [OK] Publication panel ({artifact_prefix}{split})")

            # Grouped confusion from eval results
            group_metrics = split_data.get("group_metrics", {})
            if group_metrics and "targets" in group_metrics and "predictions" in group_metrics:
                group_labels = labels[:n_classes]
                g_tgts = np.array(group_metrics["targets"])
                g_preds = np.array(group_metrics["predictions"])
                if len(g_tgts) > 0:
                    gcm_path = subdirs["confusion"] / f"{artifact_prefix}grouped_confusion_{split}_normalized.png"
                    plot_confusion_matrix(
                        g_tgts, g_preds, group_labels,
                        f"{stage_display} — {_split_display(split)} (Grouped)",
                        gcm_path, dpi, normalize=True,
                    )
                    if gcm_path.exists():
                        figures_generated.append(str(gcm_path.relative_to(out_root)))
                        print(f"    [OK] Grouped confusion matrix ({artifact_prefix}{split})")

            # Embeddings (optional)
            if embeddings:
                emb_data = _load_embeddings(emb_dir_loop, split)
                if emb_data is not None:
                    features, emb_targets = emb_data
                    tsne_path = subdirs["embeddings"] / f"{artifact_prefix}tsne_{split}.png"
                    plot_embedding_tsne(
                        features, emb_targets, labels,
                        f"{stage_display} — {_split_display(split)}",
                        tsne_path, dpi,
                    )
                    if tsne_path.exists():
                        figures_generated.append(str(tsne_path.relative_to(out_root)))
                        print(f"    [OK] t-SNE embedding ({artifact_prefix}{split})")

        # ---- Cross-split figures ----

        # Grouped vs Spectrum
        grp_path = subdirs["grouped"] / f"{artifact_prefix}grouped_vs_spectrum_{eval_stage}.png"
        plot_grouped_vs_spectrum(results, stage_display, grp_path, dpi)
        if grp_path.exists():
            figures_generated.append(str(grp_path.relative_to(out_root)))
            print(f"    [OK] Grouped vs Spectrum")

        # OOD Degradation
        ood_path = subdirs["ood"] / f"{artifact_prefix}ood_degradation_{eval_stage}.png"
        plot_ood_degradation(results, stage_display, ood_path, dpi)
        if ood_path.exists():
            figures_generated.append(str(ood_path.relative_to(out_root)))
            print(f"    [OK] OOD Degradation")

        # Collect stage summary for stage-transition plot
        test_metrics = results.get("splits", {}).get("test", {}).get("metrics", {})
        if test_metrics and artifact_root == exp_path:
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
        if trans_path.exists():
            figures_generated.append(str(trans_path.relative_to(out_root)))
            print(f"\n    [OK] Stage transition comparison")

    # ---- Patient-CV fold-wise analysis ----
    if fold_dirs:
        print(f"\n  Patient-CV detected: {len(fold_dirs)} folds")
        fold_metric_rows = collect_fold_metric_rows(fold_dirs)
        domain_metric_rows.extend(fold_metric_rows)

        fold_csv = subdirs["crossfold"] / "fold_metrics_summary.csv"
        _write_rows_csv(fold_metric_rows, fold_csv)
        if fold_csv.exists():
            figures_generated.append(str(fold_csv.relative_to(out_root)))

        stats_rows = compute_cross_fold_stats(fold_metric_rows)
        stats_csv = subdirs["crossfold"] / "cross_fold_statistics.csv"
        _write_rows_csv(stats_rows, stats_csv)
        if stats_csv.exists():
            figures_generated.append(str(stats_csv.relative_to(out_root)))

        for metric, stem in [("accuracy", "accuracy"), ("f1_macro", "macro_f1"), ("mcc", "mcc")]:
            for level, level_stem in [("Spectrum", "spectrum"), ("Patient", "patient")]:
                path = subdirs["crossfold"] / f"fold_{level_stem}_{stem}_comparison.png"
                plot_fold_metric_comparison(fold_metric_rows, metric, level, path, dpi)
                if path.exists():
                    figures_generated.append(str(path.relative_to(out_root)))
                    print(f"    [OK] Fold {level.lower()} {METRIC_DISPLAY[metric]} comparison")

        stats_plot = subdirs["crossfold"] / "cross_fold_mean_std.png"
        plot_cross_fold_statistics(stats_rows, stats_plot, dpi)
        if stats_plot.exists():
            figures_generated.append(str(stats_plot.relative_to(out_root)))

        report_path = subdirs["crossfold"] / "cross_fold_summary.md"
        write_cross_fold_report(fold_metric_rows, stats_rows, report_path)
        if report_path.exists():
            figures_generated.append(str(report_path.relative_to(out_root)))

    # ---- Aggregated CV / clinical_all outputs ----
    if aggregate_path is not None:
        aggregate = _load_json(aggregate_path)
        if labels_for_aggregate is None:
            labels_for_aggregate = _resolve_labels(cfg, _n_classes_from_aggregate(aggregate))
        aggregate_rows = collect_aggregate_metric_rows(aggregate)
        domain_metric_rows = [r for r in domain_metric_rows if r.get("fold") != "aggregated"] + aggregate_rows
        figures_generated.extend(process_aggregated_cv_results(
            aggregate_path,
            fold_dirs,
            labels_for_aggregate,
            subdirs,
            out_root,
            dpi,
        ))
        print(f"    [OK] Aggregated CV outputs")

    # ---- Stage-3 domain generalization outputs ----
    domain_figs = write_domain_shift_outputs(domain_metric_rows, subdirs, out_root, dpi)
    if domain_figs:
        figures_generated.extend(domain_figs)
        print(f"    [OK] Stage 3 domain-shift analysis")

    # ---- Model comparison (sibling discovery) ----
    siblings = _discover_sibling_experiments(exp_path)
    if len(siblings) >= 2:
        comp_path = subdirs["models"] / "model_comparison_test.png"
        plot_model_comparison(siblings, "test", comp_path, dpi)
        if comp_path.exists():
            figures_generated.append(str(comp_path.relative_to(out_root)))
            print(f"    [OK] Model comparison ({len(siblings)} architectures)")

    # ---- Learning curves ----
    metrics_json = exp_path / "metrics.json"

    if not metrics_json.exists() and fold_dirs:
        candidate = fold_dirs[0] / "metrics.json"
        if candidate.exists():
            metrics_json = candidate

    if metrics_json.exists():
        plot_learning_curves(
            metrics_json,
            f"{model_name} — {_stage_title(stage)}",
            subdirs["learning"],
            dpi,
        )
        if (subdirs["learning"] / "learning_curves.png").exists():
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
