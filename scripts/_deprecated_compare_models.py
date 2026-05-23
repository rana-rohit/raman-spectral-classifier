"""
scripts/compare_models.py

Generate model comparison plots from saved evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare multiple experiments")
    p.add_argument("--exp-dirs", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out-dir", default="experiments/comparisons")
    p.add_argument("--dpi", type=int, default=500)
    p.add_argument("--no-staging", action="store_true", help="Write outputs directly to out dir")
    return p.parse_args()


def _copy_tree_contents(src: str, dst: str) -> None:
    for root, _, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel_root) if rel_root != "." else dst
        os.makedirs(target_root, exist_ok=True)
        for fname in files:
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(target_root, fname)
            shutil.copy2(src_path, dst_path)


def _find_eval_results(exp_dir: Path) -> Path:
    candidates = list(exp_dir.glob("*_eval_results.json"))
    if not candidates:
        raise FileNotFoundError(f"No *_eval_results.json in {exp_dir}")
    return candidates[0]


def _load_predictions(exp_dir: Path, split: str):
    pred_dir = exp_dir / "predictions"
    probs_path = pred_dir / f"{split}_probabilities.npy"
    targets_path = pred_dir / f"{split}_targets.npy"
    if not (probs_path.exists() and targets_path.exists()):
        return None
    return np.load(probs_path), np.load(targets_path)


def main() -> None:
    args = parse_args()
    staging_dir = None
    if not args.no_staging:
        staging_dir = tempfile.mkdtemp(prefix="compare_staging_")
        print(f"[compare] Using staging directory: {staging_dir}")

    out_dir = Path(staging_dir) if staging_dir else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = []
    acc = []
    f1 = []
    grouped_acc = []
    grouped_f1 = []

    for exp in args.exp_dirs:
        exp_dir = Path(exp)
        eval_path = _find_eval_results(exp_dir)
        with open(eval_path, "r") as f:
            results = json.load(f)
        model_name = results.get("model", exp_dir.name)
        split_data = results.get("splits", {}).get(args.split, {})
        metrics = split_data.get("metrics", {})
        group_metrics = split_data.get("group_metrics", {})

        models.append(model_name)
        acc.append(metrics.get("accuracy", 0.0))
        f1.append(metrics.get("f1_macro", 0.0))
        grouped_acc.append(group_metrics.get("accuracy", 0.0))
        grouped_f1.append(group_metrics.get("f1_macro", 0.0))

    sns.set_style("whitegrid")
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, acc, width, label="Spectrum Acc")
    ax.bar(x + width/2, grouped_acc, width, label="Grouped Acc")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Model Comparison — {args.split}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_accuracy.png", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "comparison_accuracy.pdf", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, f1, width, label="Spectrum F1")
    ax.bar(x + width/2, grouped_f1, width, label="Grouped F1")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("F1 Macro")
    ax.set_title(f"Model Comparison — {args.split}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_f1.png", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "comparison_f1.pdf", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ROC comparison (micro-average)
    fig, ax = plt.subplots(figsize=(8, 6))
    for exp in args.exp_dirs:
        exp_dir = Path(exp)
        preds = _load_predictions(exp_dir, args.split)
        if preds is None:
            continue
        probs, targets = preds
        n_classes = probs.shape[1]
        y_bin = np.eye(n_classes)[targets]
        fpr, tpr, _ = roc_curve(y_bin.ravel(), probs.ravel())
        ax.plot(fpr, tpr, label=f"{exp_dir.name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Comparison — {args.split}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_roc.png", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / "comparison_roc.pdf", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if staging_dir is not None:
        _copy_tree_contents(staging_dir, args.out_dir)
        shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"[compare] Saved comparison plots in {args.out_dir}")


if __name__ == "__main__":
    main()
