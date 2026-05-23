"""
scripts/plots.py

Generate publication-quality plots and reports from saved evaluation artifacts.
This script does NOT retrain models and only consumes saved predictions,
metrics, and embeddings produced by scripts/evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate post-training plots")
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--split", default=None, help="Only plot a specific split")
    p.add_argument("--dpi", type=int, default=500)
    p.add_argument("--no-embeddings", action="store_true")
    p.add_argument("--no-staging", action="store_true", help="Write outputs directly to exp dir")
    return p.parse_args()


def _load_config_any(exp_dir: Path) -> dict:
    yaml_path = exp_dir / "config.yaml"
    json_path = exp_dir / "config.json"
    if yaml_path.exists():
        return load_config(str(yaml_path))
    if json_path.exists():
        with open(json_path, "r") as f:
            return json.load(f)
    raise FileNotFoundError("No config.yaml or config.json found in exp dir")


def _find_eval_results(exp_dir: Path) -> Optional[Path]:
    candidates = list(exp_dir.glob("*_eval_results.json"))
    if candidates:
        return candidates[0]
    return None


def _stage_title(cfg: dict) -> str:
    stage = cfg.get("task", {}).get("stage", "unknown")
    stage_map = {
        "pretrain_30class": "Stage 1 (Isolate Space)",
        "pretrain_treatment_8class": "Stage 2 (Treatment Space)",
        "transfer_5class": "Stage 3 (Clinical Transfer)",
    }
    return stage_map.get(stage, stage)


def _class_labels(cfg: dict, n_classes: int) -> List[str]:
    stage = cfg.get("task", {}).get("stage", "unknown")
    if stage == "pretrain_30class":
        try:
            from metadata.ontology import ISOLATES
            return [ISOLATES[i] for i in range(n_classes)]
        except Exception:
            return [f"Class {i}" for i in range(n_classes)]
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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_tree_contents(src: str, dst: str) -> None:
    for root, _, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel_root) if rel_root != "." else dst
        os.makedirs(target_root, exist_ok=True)
        for fname in files:
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(target_root, fname)
            shutil.copy2(src_path, dst_path)


def _save_figure(fig, base_path: Path, dpi: int) -> None:
    _ensure_dir(base_path.parent)
    fig.savefig(str(base_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    if base_path.suffix.lower() == ".png":
        pdf_path = base_path.with_suffix(".pdf")
        fig.savefig(str(pdf_path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _load_predictions(pred_dir: Path, split: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    logits_path = pred_dir / f"{split}_logits.npy"
    probs_path = pred_dir / f"{split}_probabilities.npy"
    targets_path = pred_dir / f"{split}_targets.npy"
    if not (logits_path.exists() and probs_path.exists() and targets_path.exists()):
        print(f"[plots] Missing predictions for split '{split}'.")
        return None
    logits = np.load(logits_path)
    probs = np.load(probs_path)
    targets = np.load(targets_path)
    return logits, probs, targets


def plot_confusion(split: str, targets: np.ndarray, preds: np.ndarray, labels: List[str], title: str, out_dir: Path, dpi: int) -> None:
    cm = confusion_matrix(targets, preds, labels=list(range(len(labels))))
    sns.set_style("white")
    for normalize in (False, True):
        if normalize:
            cm_plot = cm.astype(float)
            row_sums = cm_plot.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            cm_plot = cm_plot / row_sums
        else:
            cm_plot = cm
        fig, ax = plt.subplots(figsize=(10, 9))
        sns.heatmap(
            cm_plot,
            ax=ax,
            cmap="Blues",
            annot=True,
            fmt=".0f" if not normalize else ".2f",
            cbar=True,
            square=True,
            xticklabels=labels,
            yticklabels=labels,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        n_title = "Normalized" if normalize else "Raw"
        ax.set_title(f"{title} — {split} ({n_title})")
        base = out_dir / f"confusion_matrix_{split}_{'normalized' if normalize else 'raw'}.png"
        _save_figure(fig, base, dpi)


def plot_roc_pr(split: str, probs: np.ndarray, targets: np.ndarray, labels: List[str], title: str, out_dir: Path, dpi: int) -> None:
    n_classes = len(labels)
    y_bin = label_binarize(targets, classes=list(range(n_classes)))
    sns.set_style("whitegrid")

    # ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    ax.plot(fpr_micro, tpr_micro, label=f"micro (AUC={auc(fpr_micro, tpr_micro):.3f})", linewidth=2)
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})", alpha=0.8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {title} — {split}")
    ax.legend(fontsize=8, loc="lower right")
    _save_figure(fig, out_dir / f"roc_curve_{split}.png", dpi)

    # PR
    fig, ax = plt.subplots(figsize=(8, 6))
    prec_micro, rec_micro, _ = precision_recall_curve(y_bin.ravel(), probs.ravel())
    ax.plot(rec_micro, prec_micro, label="micro", linewidth=2)
    for i, label in enumerate(labels):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
        ax.plot(rec, prec, label=label, alpha=0.8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {title} — {split}")
    ax.legend(fontsize=8, loc="lower left")
    _save_figure(fig, out_dir / f"pr_curve_{split}.png", dpi)


def plot_learning_curves(metrics_path: Path, title: str, out_dir: Path, dpi: int) -> None:
    if not metrics_path.exists():
        print("[plots] metrics.json not found, skipping learning curves")
        return
    with open(metrics_path, "r") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    if df.empty:
        return
    sns.set_style("whitegrid")

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 5))
    for split in ["train", "val", "source_val"]:
        sub = df[df["split"] == split]
        if not sub.empty and "loss" in sub:
            ax.plot(sub["epoch"], sub["loss"], label=f"{split} loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curve — {title}")
    ax.legend()
    _save_figure(fig, out_dir / "loss_curve.png", dpi)

    # Accuracy + F1
    fig, ax = plt.subplots(figsize=(8, 5))
    for split in ["train", "val", "source_val"]:
        sub = df[df["split"] == split]
        if not sub.empty and "accuracy" in sub:
            ax.plot(sub["epoch"], sub["accuracy"], label=f"{split} acc")
        if not sub.empty and "f1_macro" in sub:
            ax.plot(sub["epoch"], sub["f1_macro"], linestyle="--", label=f"{split} f1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title(f"Learning Curves — {title}")
    ax.legend()
    _save_figure(fig, out_dir / "learning_curve.png", dpi)


def plot_grouped_vs_spectrum(results: dict, title: str, out_dir: Path, dpi: int) -> None:
    splits = []
    spectrum_acc = []
    grouped_acc = []
    spectrum_f1 = []
    grouped_f1 = []
    for split_name, data in results.get("splits", {}).items():
        metrics = data.get("metrics", {})
        group_metrics = data.get("group_metrics", {})
        if not group_metrics:
            continue
        splits.append(split_name)
        spectrum_acc.append(metrics.get("accuracy", 0.0))
        grouped_acc.append(group_metrics.get("accuracy", 0.0))
        spectrum_f1.append(metrics.get("f1_macro", 0.0))
        grouped_f1.append(group_metrics.get("f1_macro", 0.0))

    if not splits:
        print("[plots] No grouped metrics found, skipping grouped vs spectrum plot")
        return

    x = np.arange(len(splits))
    width = 0.35
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, spectrum_acc, width, label="Spectrum Acc")
    ax.bar(x + width/2, grouped_acc, width, label="Grouped Acc")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Grouped vs Spectrum Accuracy — {title}")
    ax.legend()
    _save_figure(fig, out_dir / "grouped_vs_spectrum_accuracy.png", dpi)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, spectrum_f1, width, label="Spectrum F1")
    ax.bar(x + width/2, grouped_f1, width, label="Grouped F1")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=30, ha="right")
    ax.set_ylabel("F1 Macro")
    ax.set_title(f"Grouped vs Spectrum F1 — {title}")
    ax.legend()
    _save_figure(fig, out_dir / "grouped_vs_spectrum_f1.png", dpi)


def plot_per_class(split: str, targets: np.ndarray, preds: np.ndarray, labels: List[str], title: str, out_dir: Path, dpi: int) -> None:
    report = classification_report(targets, preds, output_dict=True, zero_division=0)
    classes = [str(i) for i in range(len(labels))]
    f1 = [report[c]["f1-score"] for c in classes]
    prec = [report[c]["precision"] for c in classes]
    rec = [report[c]["recall"] for c in classes]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x - 0.25, prec, 0.25, label="Precision")
    ax.bar(x, rec, 0.25, label="Recall")
    ax.bar(x + 0.25, f1, 0.25, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Per-Class Performance — {title} — {split}")
    ax.legend()
    _save_figure(fig, out_dir / f"per_class_{split}.png", dpi)


def plot_embeddings(exp_dir: Path, split: str, labels: List[str], title: str, out_dir: Path, dpi: int) -> None:
    emb_dir = exp_dir / "embeddings"
    feat_path = emb_dir / f"{split}_features.npy"
    tgt_path = emb_dir / f"{split}_targets.npy"
    if not (feat_path.exists() and tgt_path.exists()):
        print(f"[plots] No embeddings found for split '{split}'")
        return
    features = np.load(feat_path)
    targets = np.load(tgt_path)

    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(features) // 10)), random_state=42)
    emb_tsne = tsne.fit_transform(features)

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, label in enumerate(labels):
        mask = targets == i
        ax.scatter(emb_tsne[mask, 0], emb_tsne[mask, 1], s=10, label=label, alpha=0.7)
    ax.set_title(f"t-SNE — {title} — {split}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(fontsize=7, ncol=2)
    _save_figure(fig, out_dir / f"tsne_{split}.png", dpi)

    try:
        import importlib
        import importlib.util

        if importlib.util.find_spec("umap") is None:
            raise ImportError("umap not installed")
        umap_mod = importlib.import_module("umap")
        reducer = umap_mod.UMAP(n_components=2, random_state=42)
        emb_umap = reducer.fit_transform(features)
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, label in enumerate(labels):
            mask = targets == i
            ax.scatter(emb_umap[mask, 0], emb_umap[mask, 1], s=10, label=label, alpha=0.7)
        ax.set_title(f"UMAP — {title} — {split}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, ncol=2)
        _save_figure(fig, out_dir / f"umap_{split}.png", dpi)
    except Exception:
        print("[plots] UMAP not available; skipping UMAP plot")


def write_reports(exp_dir: Path, results: dict, title: str, reports_dir: Path) -> None:
    _ensure_dir(reports_dir)
    rows = []
    for split_name, data in results.get("splits", {}).items():
        metrics = data.get("metrics", {})
        group_metrics = data.get("group_metrics", {})
        row = {"split": split_name}
        row.update({f"spectrum_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        row.update({f"grouped_{k}": v for k, v in group_metrics.items() if isinstance(v, (int, float))})
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(reports_dir / "metrics_summary.csv", index=False)

    summary = {
        "title": title,
        "stage": results.get("task"),
        "splits": list(results.get("splits", {}).keys()),
    }
    with open(reports_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    cfg = _load_config_any(exp_dir)
    title = _stage_title(cfg)

    staging_dir = None
    if not args.no_staging:
        staging_dir = tempfile.mkdtemp(prefix="plot_staging_")
        print(f"[plots] Using staging directory: {staging_dir}")

    eval_path = _find_eval_results(exp_dir)
    if eval_path is None:
        raise FileNotFoundError("No *_eval_results.json found in exp dir")
    with open(eval_path, "r") as f:
        results = json.load(f)

    base_dir = Path(staging_dir) if staging_dir else exp_dir
    plots_dir = base_dir / "plots"
    reports_dir = base_dir / "reports"
    pred_dir = exp_dir / "predictions"
    _ensure_dir(plots_dir)
    _ensure_dir(reports_dir)

    write_reports(exp_dir, results, title, reports_dir)
    plot_learning_curves(exp_dir / "metrics.json", title, plots_dir, args.dpi)

    splits = list(results.get("splits", {}).keys())
    if args.split:
        splits = [s for s in splits if s == args.split]

    report_lines = []
    for split in splits:
        loaded = _load_predictions(pred_dir, split)
        if loaded is None:
            continue
        logits, probs, targets = loaded
        preds = probs.argmax(axis=-1)
        labels = _class_labels(cfg, probs.shape[1])

        plot_confusion(split, targets, preds, labels, title, plots_dir, args.dpi)
        plot_roc_pr(split, probs, targets, labels, title, plots_dir, args.dpi)
        plot_per_class(split, targets, preds, labels, title, plots_dir, args.dpi)

        report = classification_report(
            targets,
            preds,
            target_names=labels,
            zero_division=0,
        )
        report_lines.append(f"=== {split.upper()} ===\n{report}\n")

        if not args.no_embeddings:
            emb_dir = plots_dir / "embeddings"
            _ensure_dir(emb_dir)
            plot_embeddings(exp_dir, split, labels, title, emb_dir, args.dpi)

    plot_grouped_vs_spectrum(results, title, plots_dir, args.dpi)

    if report_lines:
        with open(reports_dir / "classification_report.txt", "w") as f:
            f.write("\n".join(report_lines))

    if staging_dir is not None:
        _copy_tree_contents(staging_dir, str(exp_dir))
        shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"[plots] Completed plot generation in {exp_dir / 'plots'}")


if __name__ == "__main__":
    main()
