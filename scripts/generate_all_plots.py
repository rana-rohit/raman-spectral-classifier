"""
scripts/generate_all_plots.py

Unified experiment plotting pipeline.

Usage:
  python scripts/generate_all_plots.py --exp_dir /path/to/experiment
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate all plots + reports for one experiment")
    p.add_argument("--exp_dir", required=True, help="Path to experiment directory")
    p.add_argument("--dpi", type=int, default=500)
    p.add_argument("--no-staging", action="store_true", help="Write outputs directly to exp dir")
    p.add_argument("--no-embeddings", action="store_true", help="Skip t-SNE/UMAP plots")
    return p.parse_args()


def _load_config_any(exp_dir: Path) -> dict:
    yaml_path = exp_dir / "config.yaml"
    json_path = exp_dir / "config.json"
    if yaml_path.exists():
        import yaml
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f) or {}
    if json_path.exists():
        with open(json_path, "r") as f:
            return json.load(f)
    return {}


def _find_eval_results(exp_dir: Path) -> List[Path]:
    return sorted(exp_dir.glob("*_eval_results.json"))


def _stage_from_path(path: Path) -> str:
    return path.stem.replace("_eval_results", "")


def _stage_title(stage: str) -> str:
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
        return None
    logits = np.load(logits_path)
    probs = np.load(probs_path)
    targets = np.load(targets_path)
    return logits, probs, targets


def _load_embeddings(emb_dir: Path, split: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    feat_path = emb_dir / f"{split}_features.npy"
    tgt_path = emb_dir / f"{split}_targets.npy"
    if not (feat_path.exists() and tgt_path.exists()):
        return None
    return np.load(feat_path), np.load(tgt_path)


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


def plot_grouped_confusion(split: str, group_targets: List[int], group_preds: List[int], labels: List[str], title: str, out_dir: Path, dpi: int) -> None:
    if not group_targets or not group_preds:
        return
    cm = confusion_matrix(group_targets, group_preds, labels=list(range(len(labels))))
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
            cmap="Greens",
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
        ax.set_title(f"{title} — {split} (Grouped, {n_title})")
        base = out_dir / f"grouped_confusion_{split}_{'normalized' if normalize else 'raw'}.png"
        _save_figure(fig, base, dpi)


def plot_roc_pr(split: str, probs: np.ndarray, targets: np.ndarray, labels: List[str], title: str, out_dir: Path, dpi: int) -> None:
    n_classes = len(labels)
    y_bin = label_binarize(targets, classes=list(range(n_classes)))
    sns.set_style("whitegrid")

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
        return

    x = np.arange(len(splits))
    width = 0.35
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, spectrum_acc, width, label="Spectrum Acc")
    ax.bar(x + width / 2, grouped_acc, width, label="Grouped Acc")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Grouped vs Spectrum Accuracy — {title}")
    ax.legend()
    _save_figure(fig, out_dir / "grouped_vs_spectrum_accuracy.png", dpi)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, spectrum_f1, width, label="Spectrum F1")
    ax.bar(x + width / 2, grouped_f1, width, label="Grouped F1")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=30, ha="right")
    ax.set_ylabel("F1 Macro")
    ax.set_title(f"Grouped vs Spectrum F1 — {title}")
    ax.legend()
    _save_figure(fig, out_dir / "grouped_vs_spectrum_f1.png", dpi)


def plot_stage_comparison(stage_summaries: List[Dict], out_dir: Path, dpi: int) -> None:
    if len(stage_summaries) < 2:
        return
    sns.set_style("whitegrid")
    stages = [s["stage"] for s in stage_summaries]
    acc = [s.get("accuracy", 0.0) for s in stage_summaries]
    f1 = [s.get("f1_macro", 0.0) for s in stage_summaries]
    x = np.arange(len(stages))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, acc, marker="o", label="Accuracy")
    ax.plot(x, f1, marker="o", label="Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("Score")
    ax.set_title("Stage Comparison (Test Split)")
    ax.legend()
    _save_figure(fig, out_dir / "stage_comparison.png", dpi)


def plot_embeddings(emb_dir: Path, split: str, labels: List[str], title: str, out_dir: Path, dpi: int) -> None:
    emb = _load_embeddings(emb_dir, split)
    if emb is None:
        return
    features, targets = emb

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
        print("[generate_all_plots] UMAP not available; skipping UMAP plot")


def write_metrics_table(results: dict, out_path: Path) -> None:
    rows = []
    for split_name, data in results.get("splits", {}).items():
        metrics = data.get("metrics", {})
        group_metrics = data.get("group_metrics", {})
        row = {"split": split_name}
        row.update({f"spectrum_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        row.update({f"grouped_{k}": v for k, v in group_metrics.items() if isinstance(v, (int, float))})
        rows.append(row)
    if not rows:
        return
    import csv
    _ensure_dir(out_path.parent)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_summary_md(exp_dir: Path, model_name: str, stages: List[str], splits: List[str], out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    with open(out_path, "w") as f:
        f.write("**Experiment Summary**\n\n")
        f.write(f"Experiment: {exp_dir.name}\n\n")
        f.write(f"Model: {model_name}\n\n")
        f.write(f"Stages: {', '.join(stages) if stages else 'unknown'}\n\n")
        f.write(f"Splits: {', '.join(splits) if splits else 'unknown'}\n\n")
        f.write("Outputs:\n")
        f.write("- plots/ (confusion, grouped, roc, per_class, comparison, summaries, embeddings)\n")
        f.write("- reports/experiment_summary.md\n")
        f.write("- tables/metrics_summary.csv\n")


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    cfg = _load_config_any(exp_dir)
    model_name = cfg.get("model", {}).get("name", exp_dir.name)

    eval_paths = _find_eval_results(exp_dir)
    if not eval_paths:
        raise FileNotFoundError("No *_eval_results.json found in experiment directory")

    staging_dir = None
    if not args.no_staging:
        staging_dir = tempfile.mkdtemp(prefix="plot_staging_")
        print(f"[generate_all_plots] Using staging directory: {staging_dir}")

    base_dir = Path(staging_dir) if staging_dir else exp_dir

    plots_dir = base_dir / "plots"
    reports_dir = base_dir / "reports"
    tables_dir = base_dir / "tables"

    # Plot subfolders
    confusion_dir = plots_dir / "confusion"
    grouped_dir = plots_dir / "grouped"
    roc_dir = plots_dir / "roc"
    comparison_dir = plots_dir / "comparison"
    per_class_dir = plots_dir / "per_class"
    summaries_dir = plots_dir / "summaries"
    embeddings_dir = plots_dir / "embeddings"

    for d in [plots_dir, reports_dir, tables_dir, confusion_dir, grouped_dir, roc_dir, comparison_dir, per_class_dir, summaries_dir, embeddings_dir]:
        _ensure_dir(d)

    stages = []
    all_splits = set()
    stage_summaries = []

    pred_dir = exp_dir / "predictions"
    emb_dir = exp_dir / "embeddings"

    for eval_path in eval_paths:
        stage = _stage_from_path(eval_path)
        stages.append(stage)
        title = _stage_title(stage)
        with open(eval_path, "r") as f:
            results = json.load(f)

        splits = list(results.get("splits", {}).keys())
        all_splits.update(splits)

        # Metrics table per stage
        write_metrics_table(results, tables_dir / f"metrics_summary_{stage}.csv")

        # Grouped vs spectrum
        plot_grouped_vs_spectrum(results, title, summaries_dir, args.dpi)

        # Per split plots
        for split in splits:
            preds_bundle = _load_predictions(pred_dir, split)
            if preds_bundle is None:
                continue
            logits, probs, targets = preds_bundle
            preds = probs.argmax(axis=-1)
            labels = _class_labels(cfg, probs.shape[1])

            plot_confusion(split, targets, preds, labels, title, confusion_dir, args.dpi)
            plot_roc_pr(split, probs, targets, labels, title, roc_dir, args.dpi)
            plot_per_class(split, targets, preds, labels, title, per_class_dir, args.dpi)

            group_metrics = results.get("splits", {}).get(split, {}).get("group_metrics", {})
            if group_metrics:
                plot_grouped_confusion(
                    split,
                    group_metrics.get("targets", []),
                    group_metrics.get("predictions", []),
                    labels,
                    title,
                    grouped_dir,
                    args.dpi,
                )

            if not args.no_embeddings:
                plot_embeddings(emb_dir, split, labels, title, embeddings_dir, args.dpi)

        # Stage summary for comparison plots
        test_metrics = results.get("splits", {}).get("test", {}).get("metrics", {})
        if test_metrics:
            stage_summaries.append({
                "stage": stage,
                "accuracy": test_metrics.get("accuracy", 0.0),
                "f1_macro": test_metrics.get("f1_macro", 0.0),
            })

    # Stage comparison plot (if multiple stages exist)
    plot_stage_comparison(stage_summaries, comparison_dir, args.dpi)

    # Unified summary report
    write_summary_md(exp_dir, model_name, stages, sorted(all_splits), reports_dir / "experiment_summary.md")

    if staging_dir is not None:
        _copy_tree_contents(staging_dir, str(exp_dir))
        shutil.rmtree(staging_dir, ignore_errors=True)

    plt.close("all")
    print(f"[generate_all_plots] Completed plot generation in {exp_dir / 'plots'}")


if __name__ == "__main__":
    main()
