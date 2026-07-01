"""
scripts/lime_explain.py

Generate LIME explanations for trained Raman spectroscopy models.

Works for all stages:
    - Stage 1: pretrain_30class    (isolate-space)
    - Stage 2: pretrain_treatment_8class (treatment-space)
    - Stage 3: transfer_5class     (compact clinical transfer)

Works for all splits:
    - IID:     test (reference domain)
    - OOD:     2018clinical, 2019clinical
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocessing import SpectralPreprocessor
from src.data.registry import DataRegistry
from src.models.registry import get_model
from src.utils.checkpoint import load_best_model
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.xai.lime_explainer import SpectralLimeExplainer
from src.xai.predict_wrapper import build_predict_fn
from src.xai.xai_visualization import (plot_lime_comparison,
                                       plot_lime_explanation)


# Stage-aware label semantics
def _resolve_class_names(cfg: dict, n_classes: int) -> list[str]:

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
        ids = cfg.get("task", {}).get(
            "clinical_sparse_global_ids",
            list(range(n_classes)),
        )
        try:
            from metadata.ontology import GLOBAL_TREATMENTS

            return [GLOBAL_TREATMENTS[int(i)] for i in ids]
        except Exception:
            return [f"Clinical {i}" for i in range(n_classes)]

    return [f"Class {i}" for i in range(n_classes)]


def _stage_display_name(stage: str) -> str:
    return {
        "pretrain_30class": "Stage 1 — Isolate Space (30 classes)",
        "pretrain_treatment_8class": "Stage 2 — Treatment Space (8 classes)",
        "transfer_5class": "Stage 3 — Clinical Transfer (5 classes)",
    }.get(stage, stage)


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate LIME explanations for Raman spectral models"
    )
    p.add_argument(
        "--exp-dir",
        required=True,
        help="Path to trained experiment directory",
    )
    p.add_argument(
        "--split",
        default=None,
        help="Split to explain (test, 2018clinical, 2019clinical)",
    )
    p.add_argument(
        "--per-class",
        type=int,
        default=2,
        help="Number of samples to explain per class",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Number of LIME perturbation samples",
    )
    p.add_argument(
        "--n-features",
        type=int,
        default=20,
        help="Number of top features for LIME",
    )
    p.add_argument(
        "--n-background",
        type=int,
        default=500,
        help="Number of background samples for LIME distribution",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return p.parse_args()


# Main
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    exp_dir = Path(args.exp_dir)
    cfg = load_config(str(exp_dir / "config.yaml"))

    # --------------------------------------------------------
    # Resolve stage and label semantics
    # --------------------------------------------------------
    task_cfg = cfg["task"]
    stage = task_cfg["stage"]

    if stage == "pretrain_30class":
        clinical_sparse_ids = []
        n_classes = 30
    elif stage == "pretrain_treatment_8class":
        clinical_sparse_ids = []
        n_classes = 8
    elif stage == "transfer_5class":
        clinical_sparse_ids = task_cfg["clinical_sparse_global_ids"]
        n_classes = len(clinical_sparse_ids)
    else:
        raise ValueError(f"Unknown stage: {stage}")

    cfg["model"]["n_classes"] = n_classes
    cfg["seed"] = int(args.seed)

    class_names = _resolve_class_names(cfg, n_classes)
    stage_display = _stage_display_name(stage)

    print("\n" + "=" * 60)
    print("LIME EXPLAINABILITY")
    print("=" * 60)
    print(f"  Stage:         {stage_display}")
    print(f"  Classes:       {n_classes}")
    print(f"  Class Names:   {class_names}")
    print(f"  Experiment:    {exp_dir}")
    print(f"  Per-class:     {args.per_class}")
    print(f"  LIME samples:  {args.n_samples}")
    print(f"  LIME features: {args.n_features}")
    print("=" * 60 + "\n")

    # --------------------------------------------------------
    # Load data — reference split for background distribution
    # --------------------------------------------------------
    print("[LIME] Loading data...")
    registry = DataRegistry(data_root="data/raw", cfg=cfg)
    registry.load("reference")

    X_ref, y_ref = registry.get_arrays("reference")
    preprocessor = SpectralPreprocessor.from_config(cfg["preprocessing"])
    preprocessor.fit(X_ref)

    # --------------------------------------------------------
    # Build background sample from reference data (RAW spectra)
    # --------------------------------------------------------
    n_bg = min(args.n_background, len(X_ref))
    rng = np.random.default_rng(args.seed)
    bg_indices = rng.choice(len(X_ref), size=n_bg, replace=False)
    X_background = np.array(X_ref[bg_indices])

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    print("[LIME] Loading model...")
    model_name = cfg["model"]["name"]
    model = get_model(model_name, cfg)
    checkpoint = load_best_model(str(exp_dir), model)

    # Verify checkpoint stage integrity
    ckpt_cfg = checkpoint.get("config", {})
    ckpt_stage = ckpt_cfg.get("task", {}).get("stage", None)
    if ckpt_stage and ckpt_stage != stage:
        raise ValueError(
            f"Checkpoint stage mismatch: expected {stage}, got {ckpt_stage}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Build prediction wrapper
    # --------------------------------------------------------
    predict_fn = build_predict_fn(
        model=model,
        preprocessor=preprocessor,
        device=str(device),
        batch_size=256,
    )

    # --------------------------------------------------------
    # Load wavenumbers (optional, for axis labels)
    # --------------------------------------------------------
    wavenumbers = None
    wave_path = Path("data/raw/wavenumbers.npy")
    if wave_path.exists():
        try:
            loaded_w = np.load(wave_path)
            if len(loaded_w) == X_background.shape[1]:
                wavenumbers = loaded_w
                print(f"[LIME] Loaded wavenumbers: {wavenumbers.shape}")
            else:
                print(
                    f"[LIME] Warning: wavenumbers length ({len(loaded_w)}) "
                    f"does not match signal length ({X_background.shape[1]}). "
                    "Proceeding with default spectral indexing."
                )
        except Exception as e:
            print(f"[LIME] Warning: failed to load wavenumbers: {e}")

    # --------------------------------------------------------
    # Build LIME explainer
    # --------------------------------------------------------
    print("[LIME] Initializing LIME explainer...")
    explainer = SpectralLimeExplainer(
        predict_fn=predict_fn,
        training_data=X_background,
        wavenumbers=wavenumbers,
        class_names=class_names,
        n_features=args.n_features,
        n_samples=args.n_samples,
        random_state=args.seed,
    )

    # --------------------------------------------------------
    # Determine which split to explain
    # --------------------------------------------------------
    split = args.split or cfg.get("xai", {}).get("split")
    if split is None:
        raise ValueError("No split specified. Use --split test or --split 2018clinical")

    # Load the split data (raw arrays for LIME input)
    if split == "reference":
        X_split, y_split = X_ref, y_ref
    elif split in ["test"]:
        registry.load(split)
        X_split, y_split = registry.get_arrays(split, allow_holdout=True)
    else:
        # OOD splits (2018clinical, 2019clinical, etc.)
        registry.load(split)
        X_split, y_split = registry.get_arrays(split)

    print(
        f"[LIME] Split '{split}': {len(X_split)} samples, {len(np.unique(y_split))} classes"
    )

    # --------------------------------------------------------
    # Resolve label remapping for non-isolate stages
    # --------------------------------------------------------
    split_cfg = cfg.get("splits", {}).get(split, {})
    label_space = split_cfg.get("label_space", "")

    if stage == "pretrain_treatment_8class":
        if label_space == "isolate_space":
            from metadata.ontology import ISOLATE_TO_TREATMENT

            y_mapped = np.array(
                [ISOLATE_TO_TREATMENT[int(lbl)] for lbl in y_split],
                dtype=np.int64,
            )
            y_split = y_mapped

    elif stage == "transfer_5class" and clinical_sparse_ids:
        # Check if labels need mapping (isolate-space → treatment → compact)
        split_cfg = cfg.get("splits", {}).get(split, {})
        label_space = split_cfg.get("label_space", "")

        if label_space == "isolate_space":
            from metadata.ontology import ISOLATE_TO_TREATMENT

            y_treatment = np.array(
                [ISOLATE_TO_TREATMENT[int(lbl)] for lbl in y_split],
                dtype=np.int64,
            )
            mask = np.isin(y_treatment, clinical_sparse_ids)
            X_split = np.array(X_split[mask])
            y_treatment = y_treatment[mask]
            from src.utils.class_subset import class_maps

            cmap, _ = class_maps(clinical_sparse_ids)
            y_split = np.array([cmap[int(lbl)] for lbl in y_treatment])
        else:
            # Clinical data: already in treatment space
            mask = np.isin(y_split, clinical_sparse_ids)
            X_split = np.array(X_split[mask])
            y_filtered = y_split[mask]
            from src.utils.class_subset import class_maps

            cmap, _ = class_maps(clinical_sparse_ids)
            y_split = np.array([cmap[int(lbl)] for lbl in y_filtered])

    # --------------------------------------------------------
    # Generate explanations
    # --------------------------------------------------------
    print(f"\n[LIME] Generating explanations for split '{split}'...")

    output_dir = exp_dir / "lime" / split
    output_dir.mkdir(parents=True, exist_ok=True)

    class_counts = {i: 0 for i in range(n_classes)}
    all_explanations_by_class: dict[int, list] = {i: [] for i in range(n_classes)}

    unique_labels = np.unique(y_split)

    indices = rng.permutation(len(X_split))

    for idx in indices:
        label = int(y_split[idx])

        if label not in class_counts:
            continue
        if class_counts[label] >= args.per_class:
            continue

        raw_spectrum = np.array(X_split[idx])
        sample_num = class_counts[label]

        print(
            f"  Explaining: class {label} ({class_names[label] if label < len(class_names) else '?'}) "
            f"— sample {sample_num + 1}/{args.per_class}"
        )

        explanation = explainer.explain_sample(
            spectrum=raw_spectrum,
            label=label,
        )

        safe_name = (
            (class_names[label].replace(" ", "_").replace("/", "_").replace(".", ""))
            if label < len(class_names)
            else f"class_{label}"
        )

        plot_path = output_dir / f"{safe_name}_sample_{sample_num}.png"

        plot_lime_explanation(
            explanation=explanation,
            save_path=plot_path,
            stage_label=stage_display,
            split_label=split,
        )

        all_explanations_by_class[label].append(explanation)
        class_counts[label] += 1

        target_classes = set(unique_labels.tolist()) & set(class_counts.keys())
        if all(class_counts.get(c, 0) >= args.per_class for c in target_classes):
            break

    print("\n[LIME] Generating comparison plots...")
    for label, explanations in all_explanations_by_class.items():
        if len(explanations) < 2:
            continue

        safe_name = (
            (class_names[label].replace(" ", "_").replace("/", "_").replace(".", ""))
            if label < len(class_names)
            else f"class_{label}"
        )

        comparison_path = output_dir / f"{safe_name}_comparison.png"

        plot_lime_comparison(
            explanations=explanations,
            save_path=comparison_path,
            title=f"LIME Comparison — {class_names[label] if label < len(class_names) else f'Class {label}'} ({split})",
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    total = sum(class_counts.values())
    print("\n" + "=" * 60)
    print("LIME EXPLANATION COMPLETE")
    print("=" * 60)
    print(f"  Stage:       {stage_display}")
    print(f"  Split:       {split}")
    print(f"  Explained:   {total} samples")
    print(f"  Output:      {output_dir}")
    print(f"  Per-class:   {dict(class_counts)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
