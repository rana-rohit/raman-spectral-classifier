"""
scripts/compare_models_xai.py

Publication-quality multi-model XAI comparison for Raman spectroscopy.

Scientific question:
    "Do different deep learning architectures rely on similar Raman
     spectral regions when correctly classifying the same antimicrobial
     treatment category?"

Pipeline:
    1. Locate all Stage 3 experiment folders (patient-CV runs).
    2. Load detailed_predictions.json from every fold; aggregate.
    3. Compute per-model correct-classification masks.
    4. For each treatment class, find patients correctly classified by
       ALL models (intersection across architectures).
    5. Select the best class (largest common intersection).
    6. Randomly sample N patients from that intersection.
    7. For every selected patient × every model: generate LIME explanation.
    8. Extract top-K important Raman peaks from each explanation.
    Note: Data augmentation (if enabled) is applied during preprocessing as configured in `configs/data/augmentation.yaml`.
    9. Compute consensus peak frequency (±5 cm⁻¹ tolerance).
   10. Output:
         - common_patient_summary.json
         - patient_<id>_comparison.png  (6-panel LIME figure)
         - consensus_peaks.csv
         - consensus_peak_frequency.png
         - xai_consensus_report.txt

Usage:
    python scripts/compare_models_xai.py \
        --results-root experiments/ \
        --n-patients 5 \
        --top-k-peaks 10

Requirements:
    - Python 3.10+
    - Existing repository modules (no modifications needed)
    - All Stage 3 patient-CV experiments must have been run
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys

# Reconfigure stdout/stderr to support UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.preprocessing import SpectralPreprocessor
from src.data.registry import DataRegistry
from src.models.registry import get_model
from src.utils.checkpoint import load_best_model
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.xai.lime_explainer import SpectralLimeExplainer
from src.xai.predict_wrapper import build_predict_fn


STAGE3_EXPERIMENT_NAMES: list[str] = [
    "cnn_s3_transfer_ts_iid_patient_cv",
    "cnn_transformer_s3_transfer_ts_iid_patient_cv",
    "inception1d_s3_transfer_ts_iid_patient_cv",
    "resnet_s3_transfer_ts_iid_patient_cv",
    "tcn_s3_transfer_ts_iid_patient_cv",
    "transformer_s3_transfer_ts_iid_patient_cv",
]

# Human-readable display names for each model (used in figure panels)
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "cnn_s3_transfer_ts_iid_patient_cv": "CNN",
    "cnn_transformer_s3_transfer_ts_iid_patient_cv": "CNN-Transformer",
    "inception1d_s3_transfer_ts_iid_patient_cv": "Inception1D",
    "resnet_s3_transfer_ts_iid_patient_cv": "ResNet1D",
    "tcn_s3_transfer_ts_iid_patient_cv": "TCN",
    "transformer_s3_transfer_ts_iid_patient_cv": "Transformer",
}

# Canonical 2×3 layout order for the 6-panel figure
PANEL_ORDER: list[str] = [
    "cnn_s3_transfer_ts_iid_patient_cv",
    "resnet_s3_transfer_ts_iid_patient_cv",
    "inception1d_s3_transfer_ts_iid_patient_cv",
    "tcn_s3_transfer_ts_iid_patient_cv",
    "transformer_s3_transfer_ts_iid_patient_cv",
    "cnn_transformer_s3_transfer_ts_iid_patient_cv",
]


# Compact label -> global treatment ID
COMPACT_TO_GLOBAL: dict[int, int] = {0: 0, 1: 2, 2: 3, 3: 5, 4: 6}

# Compact label -> treatment name
COMPACT_CLASS_NAMES: dict[int, str] = {
    0: "Meropenem",
    1: "TZP",
    2: "Vancomycin",
    3: "Penicillin",
    4: "Daptomycin",
}

N_CLASSES: int = 5
CLINICAL_SPARSE_IDS: list[int] = [0, 2, 3, 5, 6]


# CLI
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Multi-model LIME explainability comparison for "
            "Stage 3 Raman spectral classification"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--results-root",
        type=str,
        required=True,
        help="Root directory containing all experiment folders",
    )
    p.add_argument(
        "--n-patients",
        type=int,
        default=5,
        help="Number of patients to sample for XAI comparison (default: 5)",
    )
    p.add_argument(
        "--top-k-peaks",
        type=int,
        default=10,
        help="Number of top LIME peaks to extract per explanation (default: 10)",
    )
    p.add_argument(
        "--lime-n-samples",
        type=int,
        default=2000,
        help="Number of LIME perturbation samples (default: 2000)",
    )
    p.add_argument(
        "--lime-n-features",
        type=int,
        default=20,
        help="Number of top LIME features (default: 20)",
    )
    p.add_argument(
        "--lime-n-background",
        type=int,
        default=500,
        help="Number of background samples for LIME (default: 500)",
    )
    p.add_argument(
        "--peak-tolerance",
        type=float,
        default=5.0,
        help="Peak matching tolerance in cm⁻¹ (default: ±5)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="paper_xai_analysis",
        help="Output directory (default: paper_xai_analysis)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return p.parse_args()


# STEP 1: Locate experiment folders
def locate_experiments(results_root: Path) -> dict[str, Path]:
    """
    Locate all Stage 3 experiment directories under results_root.

    Returns
    -------
    dict mapping experiment name -> directory path
    """
    found: dict[str, Path] = {}

    for name in STAGE3_EXPERIMENT_NAMES:
        candidate = results_root / name
        if candidate.is_dir():
            found[name] = candidate
        else:
    
            for child in results_root.iterdir():
                if child.is_dir() and child.name == name:
                    found[name] = child
                    break


    print("\n[STEP 1] Locating Stage 3 experiment folders...")
    for name in STAGE3_EXPERIMENT_NAMES:
        status = "✓ FOUND" if name in found else "✗ MISSING"
        path_str = str(found[name]) if name in found else "—"
        print(f"  {status}  {name}")
        if name in found:
            print(f"          {path_str}")

    missing = set(STAGE3_EXPERIMENT_NAMES) - set(found.keys())
    if missing:
        raise FileNotFoundError(
            f"Missing experiment directories: {sorted(missing)}. "
            f"Searched under: {results_root}"
        )

    return found


# STEP 2: Load and aggregate detailed_predictions.json
def load_fold_predictions(exp_dir: Path) -> dict[str, Any]:
    """
    Load detailed_predictions.json from every fold in an experiment
    and aggregate predictions across folds.

    Returns
    -------
    dict with keys for each clinical split, containing aggregated:
        - predictions: np.ndarray
        - targets: np.ndarray
        - patient_ids: np.ndarray
        - probabilities: np.ndarray
        - grouped_predictions: np.ndarray or None
        - grouped_targets: np.ndarray or None
    """
    # Find fold directories
    fold_dirs = sorted(
        [
            d
            for d in exp_dir.iterdir()
            if d.is_dir() and ("fold_" in d.name or "_fold" in d.name)
        ]
    )

    if not fold_dirs:
        # Maybe folds are nested
        fold_dirs = sorted(exp_dir.glob("**/fold_*"))

    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {exp_dir}")

    # Load all fold data
    all_fold_data: list[dict] = []
    for fd in fold_dirs:
        pred_file = fd / "detailed_predictions.json"
        if not pred_file.exists():
            print(f"  WARNING: {pred_file} not found, skipping")
            continue
        with open(pred_file, "r", encoding="utf-8") as f:
            all_fold_data.append(json.load(f))

    if not all_fold_data:
        raise FileNotFoundError(
            f"No detailed_predictions.json files loaded from {exp_dir}"
        )

    # Identify all clinical splits
    all_splits: set[str] = set()
    for fd in all_fold_data:
        all_splits.update(fd.keys())
    clinical_splits = sorted([s for s in all_splits if "clinical" in s])

    # Aggregate across folds for each split
    aggregated: dict[str, dict] = {}

    for split_name in clinical_splits:
        preds_list, targets_list, pids_list = [], [], []
        probs_list, gpreds_list, gtargets_list = [], [], []

        for fd in all_fold_data:
            if split_name not in fd:
                continue
            data = fd[split_name]
            preds_list.append(np.array(data["predictions"]))
            targets_list.append(np.array(data["targets"]))

            pids = data.get("patient_ids")
            if pids is not None:
                pids_list.append(np.array(pids))

            probs = data.get("probabilities")
            if probs is not None:
                probs_list.append(np.array(probs))

            gp = data.get("grouped_predictions")
            if gp is not None:
                gpreds_list.append(np.array(gp))

            gt = data.get("grouped_targets")
            if gt is not None:
                gtargets_list.append(np.array(gt))

        if not preds_list:
            continue

        aggregated[split_name] = {
            "predictions": np.concatenate(preds_list),
            "targets": np.concatenate(targets_list),
            "patient_ids": np.concatenate(pids_list) if pids_list else None,
            "probabilities": np.concatenate(probs_list) if probs_list else None,
            "grouped_predictions": np.concatenate(gpreds_list) if gpreds_list else None,
            "grouped_targets": np.concatenate(gtargets_list) if gtargets_list else None,
        }

    return aggregated


# STEPS 3–7: Find common correctly-classified patients
def compute_correct_patients(
    all_model_preds: dict[str, dict[str, dict]],
) -> dict[str, dict[int, set[str]]]:
    """
    For each model, compute which patients were correctly classified
    in each treatment class.

    Returns
    -------
    dict[model_name -> dict[class_id -> set of patient_ids]]
    """
    model_correct: dict[str, dict[int, set[str]]] = {}

    for model_name, splits_data in all_model_preds.items():
        class_patients: dict[int, set[str]] = defaultdict(set)

        for split_name, data in splits_data.items():
            predictions = data["predictions"]
            targets = data["targets"]
            patient_ids = data["patient_ids"]

            if patient_ids is None:
                print(f"  WARNING: No patient IDs for {model_name}/{split_name}")
                continue

            # Build patient-level predictions via majority vote on probabilities
            # Use spectrum-level correct mask to identify which patients
            # have all their spectra correctly classified
            correct_mask = predictions == targets

            # Group by patient
            patient_spectra: dict[str, list[tuple[int, int, bool]]] = defaultdict(list)
            for i, pid in enumerate(patient_ids):
                pid_str = str(pid)
                patient_spectra[pid_str].append(
                    (int(predictions[i]), int(targets[i]), bool(correct_mask[i]))
                )

            # A patient is "correctly classified" if the majority vote
            # of their spectra predictions matches their target label.
            # (Patient-level majority vote is the clinically relevant metric.)
            for pid_str, spectra_info in patient_spectra.items():
                target_label = spectra_info[0][1]  # All spectra share same target
                # Majority vote: count predictions per class
                pred_counts: Counter = Counter()
                for pred, _, _ in spectra_info:
                    pred_counts[pred] += 1
                majority_pred = pred_counts.most_common(1)[0][0]

                if majority_pred == target_label:
                    class_patients[target_label].add(pid_str)

        model_correct[model_name] = dict(class_patients)

    return model_correct


def find_common_patients(
    model_correct: dict[str, dict[int, set[str]]],
) -> tuple[dict[int, set[str]], int, set[str]]:
    """
    Compute the intersection of correctly classified patients across
    ALL models for each class.

    Returns
    -------
    class_common : dict[class_id -> set of common patient_ids]
    best_class : int  (class with largest intersection)
    best_patients : set[str]  (the common patients for best_class)
    """
    model_names = list(model_correct.keys())
    all_classes: set[int] = set()
    for mc in model_correct.values():
        all_classes.update(mc.keys())

    class_common: dict[int, set[str]] = {}

    for cls in sorted(all_classes):
        # Start with the first model's patients, intersect with all others
        sets_for_class = []
        for model_name in model_names:
            patients = model_correct[model_name].get(cls, set())
            sets_for_class.append(patients)

        if sets_for_class:
            common = sets_for_class[0].copy()
            for s in sets_for_class[1:]:
                common &= s
            class_common[cls] = common
        else:
            class_common[cls] = set()

    # Find the best class = largest common intersection
    best_class = max(class_common, key=lambda c: len(class_common[c]))
    best_patients = class_common[best_class]

    return class_common, best_class, best_patients


# STEP 8: Sample patients
def sample_patients(
    patient_set: set[str],
    n_patients: int,
    seed: int,
) -> list[str]:
    """Randomly sample N patients from the common intersection."""
    sorted_patients = sorted(patient_set)
    rng = np.random.default_rng(seed)

    if n_patients >= len(sorted_patients):
        print(
            f"  NOTE: Requested {n_patients} patients but only "
            f"{len(sorted_patients)} available. Using all."
        )
        return sorted_patients

    selected = rng.choice(sorted_patients, size=n_patients, replace=False)
    return sorted(selected.tolist())


# Model loading helper (reuses existing repository logic)
def load_model_and_explainer(
    exp_dir: Path,
    X_background: np.ndarray,
    preprocessor: SpectralPreprocessor,
    wavenumbers: Optional[np.ndarray],
    class_names: list[str],
    lime_n_features: int,
    lime_n_samples: int,
    seed: int,
) -> tuple[torch.nn.Module, SpectralLimeExplainer]:
    """
    Load a trained model from an experiment directory and build a
    LIME explainer around it.

    Reuses:
        - src.utils.config.load_config
        - src.models.registry.get_model
        - src.utils.checkpoint.load_best_model
        - src.xai.predict_wrapper.build_predict_fn
        - src.xai.lime_explainer.SpectralLimeExplainer
    """
    fold_dirs = sorted(exp_dir.glob("fold_*"))

    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {exp_dir}")

    model_dir = fold_dirs[0]

    print(f"  Using fold directory: {model_dir.name}")

    cfg = load_config(str(model_dir / "config.yaml"))

    # Configure for Stage 3 transfer
    task_cfg = cfg["task"]
    clinical_sparse_ids = task_cfg.get(
        "clinical_sparse_global_ids", CLINICAL_SPARSE_IDS
    )
    n_classes = len(clinical_sparse_ids)
    cfg["model"]["n_classes"] = n_classes
    cfg["seed"] = seed

    # Build and load model
    model_name = cfg["model"]["name"]
    model = get_model(model_name, cfg)
    checkpoint = load_best_model(str(model_dir), model)

    # Verify checkpoint stage
    ckpt_stage = checkpoint.get("config", {}).get("task", {}).get("stage")
    if ckpt_stage and ckpt_stage != "transfer_5class":
        raise ValueError(
            f"Expected transfer_5class checkpoint, got {ckpt_stage} " f"in {exp_dir}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Build prediction wrapper
    predict_fn = build_predict_fn(
        model=model,
        preprocessor=preprocessor,
        device=str(device),
        batch_size=256,
    )

    # Build LIME explainer
    explainer = SpectralLimeExplainer(
        predict_fn=predict_fn,
        training_data=X_background,
        wavenumbers=wavenumbers,
        class_names=class_names,
        n_features=lime_n_features,
        n_samples=lime_n_samples,
        random_state=seed,
    )

    return model, explainer


# Patient spectrum retrieval
def get_patient_spectra(
    all_model_preds: dict[str, dict[str, dict]],
    patient_id: str,
) -> tuple[Optional[int], Optional[str]]:
    """
    Look up the target class and clinical split for a given patient.

    Returns
    -------
    (target_class, split_name) or (None, None) if not found
    """
    for model_name in all_model_preds:
        for split_name, data in all_model_preds[model_name].items():
            pids = data["patient_ids"]
            if pids is None:
                continue
            for i, pid in enumerate(pids):
                if str(pid) == patient_id:
                    return int(data["targets"][i]), split_name
    return None, None


# Peak extraction from LIME explanations
def extract_peaks(
    explanation,
    wavenumbers: Optional[np.ndarray],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Extract top-K important Raman spectral peaks from a LIME explanation.

    Returns a list of dicts with keys:
        - wavenumber: float (cm⁻¹) or spectral index
        - weight: float (LIME importance weight)
        - rank: int (1 = most important)
    """
    importance = explanation.importance
    abs_importance = np.abs(importance)

    # Get indices of top-K highest absolute importance
    if len(abs_importance) <= top_k:
        top_indices = np.argsort(abs_importance)[::-1]
    else:
        top_indices = np.argpartition(abs_importance, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(abs_importance[top_indices])[::-1]]

    peaks = []
    for rank, idx in enumerate(top_indices, start=1):
        wn = float(wavenumbers[idx]) if wavenumbers is not None else float(idx)
        peaks.append(
            {
                "wavenumber": wn,
                "weight": float(importance[idx]),
                "rank": rank,
            }
        )

    return peaks


# Consensus analysis
def compute_consensus(
    all_peaks: dict[str, dict[str, list[dict]]],
    tolerance: float,
    model_names: list[str],
) -> list[dict[str, Any]]:
    """
    Compute consensus peak frequency across models.

    Parameters
    ----------
    all_peaks : dict[patient_id -> dict[model_name -> list of peak dicts]]
    tolerance : float (cm⁻¹)
    model_names : list of model names

    Returns
    -------
    List of consensus peak dicts sorted by frequency (descending),
    each with keys: wavenumber, frequency, models
    """
    # Collect all peaks across all patients and models
    # For each (patient, model) pair, collect peak wavenumbers
    all_wavenumbers_by_model: dict[str, list[float]] = defaultdict(list)

    for patient_id, model_peaks in all_peaks.items():
        for model_name, peaks in model_peaks.items():
            for peak in peaks:
                all_wavenumbers_by_model[model_name].append(peak["wavenumber"])

    # Merge all peaks into a single sorted list
    all_wns: list[float] = []
    for wns in all_wavenumbers_by_model.values():
        all_wns.extend(wns)
    all_wns.sort()

    if not all_wns:
        return []

    # Cluster nearby peaks within tolerance
    clusters: list[list[float]] = []
    current_cluster: list[float] = [all_wns[0]]

    for wn in all_wns[1:]:
        if wn - current_cluster[-1] <= tolerance:
            current_cluster.append(wn)
        else:
            clusters.append(current_cluster)
            current_cluster = [wn]
    clusters.append(current_cluster)

    # For each cluster, count how many unique models contributed
    consensus_peaks: list[dict[str, Any]] = []

    for cluster in clusters:
        center = float(np.mean(cluster))

        # Check which models have a peak within tolerance of the center
        contributing_models: set[str] = set()
        for model_name, wns in all_wavenumbers_by_model.items():
            for wn in wns:
                if abs(wn - center) <= tolerance:
                    contributing_models.add(model_name)
                    break

        display_names = sorted(
            [MODEL_DISPLAY_NAMES.get(m, m) for m in contributing_models]
        )

        consensus_peaks.append(
            {
                "wavenumber": round(center, 1),
                "frequency": len(contributing_models),
                "models": display_names,
                "n_models": len(model_names),
            }
        )

    # Sort by frequency (descending), then by wavenumber
    consensus_peaks.sort(key=lambda x: (-x["frequency"], x["wavenumber"]))

    # Deduplicate nearby consensus peaks (merge if within tolerance)
    deduped: list[dict[str, Any]] = []
    for peak in consensus_peaks:
        merged = False
        for existing in deduped:
            if abs(existing["wavenumber"] - peak["wavenumber"]) <= tolerance:
                # Keep the one with higher frequency
                if peak["frequency"] > existing["frequency"]:
                    existing.update(peak)
                merged = True
                break
        if not merged:
            deduped.append(peak)

    return deduped


# Publication-quality plotting
# Color scheme — consistent with existing xai_visualization.py
_POSITIVE_COLOR = "#22C55E"
_POSITIVE_FILL_COLOR = "#86EFAC"
_POSITIVE_GLOW_COLOR = "#DCFCE7"
_NEGATIVE_COLOR = "#F44336"
_SPECTRUM_COLOR = "#212121"
_BACKGROUND_COLOR = "#FAFAFA"
_GRID_COLOR = "#E0E0E0"


def plot_patient_comparison(
    explanations: dict[str, Any],
    patient_id: str,
    class_name: str,
    wavenumbers: Optional[np.ndarray],
    save_path: Path,
    dpi: int = 300,
) -> None:
    """
    Generate a 2×3 panel figure showing LIME explanations for the SAME
    patient across all 6 models.

    Layout:
        CNN             ResNet1D
        Inception1D     TCN
        Transformer     CNN-Transformer
    """
    try:
        from scipy.signal import savgol_filter as _savgol
    except ImportError:
        _savgol = None

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(16, 14),
        facecolor="white",
        constrained_layout=False,
    )

    fig.suptitle(
        f"LIME Explanation Comparison — Patient {patient_id}\n"
        f"Treatment: {class_name} | Stage 3 Clinical Transfer",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Determine shared x-axis range across all panels
    x_min, x_max = float("inf"), float("-inf")
    for exp_name in PANEL_ORDER:
        if exp_name not in explanations:
            continue
        exp = explanations[exp_name]
        wn = exp.wavenumbers
        if wn is not None:
            x_min = min(x_min, wn.min())
            x_max = max(x_max, wn.max())

    for panel_idx, exp_name in enumerate(PANEL_ORDER):
        row, col = divmod(panel_idx, 2)
        ax = axes[row, col]
        display_name = MODEL_DISPLAY_NAMES.get(exp_name, exp_name)

        if exp_name not in explanations:
            ax.text(
                0.5,
                0.5,
                f"{display_name}\n(not available)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_facecolor(_BACKGROUND_COLOR)
            continue

        exp = explanations[exp_name]
        spectrum = exp.spectrum
        importance = exp.importance
        x_axis = (
            exp.wavenumbers if exp.wavenumbers is not None else np.arange(len(spectrum))
        )

        # Sort by wavenumber if available
        if exp.wavenumbers is not None:
            order = np.argsort(x_axis)
            x_axis = x_axis[order]
            spectrum = spectrum[order]
            importance = importance[order]

        ax.set_facecolor(_BACKGROUND_COLOR)

        # Smooth spectrum for display
        display_spectrum = spectrum.copy()
        if _savgol is not None and len(spectrum) >= 11:
            try:
                display_spectrum = _savgol(spectrum, 11, 3)
            except Exception:
                pass

        # Plot spectrum
        ax.plot(
            x_axis,
            display_spectrum,
            color=_SPECTRUM_COLOR,
            linewidth=0.7,
            alpha=0.85,
            zorder=3,
        )

        # Overlay importance regions
        max_imp = float(np.abs(importance).max() or 1.0)
        positive = np.maximum(importance, 0)
        negative = np.minimum(importance, 0)

        # Positive (supports prediction)
        pos_indices = np.flatnonzero(positive > 0)
        for idx in pos_indices:
            x_center = float(x_axis[idx])
            strength = float(positive[idx] / max_imp)
            band_width = 18.0 + 17.0 * strength
            glow_alpha = 0.14 + 0.10 * strength
            fill_alpha = 0.07 + 0.05 * strength
            left = x_center - band_width / 2.0
            right = x_center + band_width / 2.0
            ax.axvspan(
                left,
                right,
                facecolor=_POSITIVE_GLOW_COLOR,
                alpha=glow_alpha,
                edgecolor="none",
                zorder=1,
            )
            ax.axvspan(
                left,
                right,
                facecolor=_POSITIVE_FILL_COLOR,
                alpha=fill_alpha,
                edgecolor="none",
                zorder=1.1,
            )

        # Negative (opposes prediction)
        neg_indices = np.flatnonzero(negative < 0)
        for idx in neg_indices:
            x_center = float(x_axis[idx])
            strength = float(abs(negative[idx]) / max_imp)
            band_width = 18.0 + 17.0 * strength
            band_alpha = 0.07 + 0.07 * strength
            left = x_center - band_width / 2.0
            right = x_center + band_width / 2.0
            ax.axvspan(
                left,
                right,
                facecolor=_NEGATIVE_COLOR,
                alpha=band_alpha,
                edgecolor="none",
                zorder=1,
            )

        # Labels and styling
        ax.set_title(
            f"{display_name}  (conf: {exp.confidence:.1%})",
            fontsize=11,
            fontweight="bold",
            pad=6,
        )
        ax.grid(True, alpha=0.25, color=_GRID_COLOR)

        if row == 2:
            x_label = "Wavenumber (cm⁻¹)" if exp.wavenumbers is not None else "Index"
            ax.set_xlabel(x_label, fontsize=10)
        if col == 0:
            ax.set_ylabel("Intensity (a.u.)", fontsize=10)

        # Enforce shared x-axis range
        if x_min < x_max:
            ax.set_xlim(x_min, x_max)

        ax.tick_params(labelsize=8)

    fig.subplots_adjust(
        top=0.92,
        bottom=0.06,
        left=0.06,
        right=0.98,
        hspace=0.28,
        wspace=0.15,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_consensus_bar_chart(
    consensus_peaks: list[dict],
    save_path: Path,
    n_models: int,
    n_patients: int,
    dpi: int = 300,
) -> None:
    """
    Publication-quality bar chart: Peak Position vs Number of Models.
    """
    if not consensus_peaks:
        print("  WARNING: No consensus peaks to plot.")
        return

    # Filter to peaks appearing in at least 2 models
    filtered = [p for p in consensus_peaks if p["frequency"] >= 2]
    if not filtered:
        filtered = consensus_peaks[:20]

    # Sort by consensus strength
    filtered.sort(
        key=lambda x: (x["frequency"], x["wavenumber"]),
        reverse=True,
    )

    wavenumbers = [p["wavenumber"] for p in filtered]
    frequencies = [p["frequency"] for p in filtered]
    labels = [f"{wn:.0f} cm⁻¹" for wn in wavenumbers]

    # Color gradient based on frequency
    colors = []
    for freq in frequencies:
        ratio = freq / n_models
        if ratio >= 0.9:
            colors.append("#22C55E")  # High consensus — green
        elif ratio >= 0.6:
            colors.append("#3B82F6")  # Medium — blue
        elif ratio >= 0.4:
            colors.append("#F59E0B")  # Low-medium — amber
        else:
            colors.append("#94A3B8")  # Low — slate

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="white")

    bars = ax.bar(
        range(len(labels)),
        frequencies,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
        width=0.7,
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_xlabel("Peak Position (cm⁻¹)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Models", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Consensus Raman Peaks Across {n_patients} Patients and {n_models} Models",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0, n_models + 0.5)
    ax.set_yticks(range(n_models + 1))
    ax.axhline(
        y=n_models,
        color="#22C55E",
        linewidth=1.0,
        linestyle="--",
        alpha=0.5,
        label=f"All {n_models} models",
    )
    ax.grid(True, axis="y", alpha=0.3, color=_GRID_COLOR)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # Add frequency labels on bars
    for bar, wn, freq in zip(bars, wavenumbers, frequencies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"{freq}/{n_models}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# Main pipeline
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  MULTI-MODEL XAI COMPARISON FOR RAMAN SPECTROSCOPY")
    print("=" * 70)
    print(f"  Results root:   {results_root}")
    print(f"  Output dir:     {output_dir}")
    print(f"  N patients:     {args.n_patients}")
    print(f"  Top K peaks:    {args.top_k_peaks}")
    print(f"  Peak tolerance: ±{args.peak_tolerance} cm⁻¹")
    print(f"  LIME samples:   {args.lime_n_samples}")
    print(f"  LIME features:  {args.lime_n_features}")
    print(f"  Seed:           {args.seed}")
    print("=" * 70 + "\n")

    # ------------------------------------------------------------------
    # STEP 1: Locate experiments
    # ------------------------------------------------------------------
    experiments = locate_experiments(results_root)

    # ------------------------------------------------------------------
    # STEP 2: Load and aggregate predictions
    # ------------------------------------------------------------------
    print("\n[STEP 2] Loading detailed_predictions.json from all folds...")
    all_model_preds: dict[str, dict[str, dict]] = {}

    for exp_name, exp_dir in experiments.items():
        display = MODEL_DISPLAY_NAMES.get(exp_name, exp_name)
        print(f"\n  Loading: {display} ({exp_name})")
        try:
            preds = load_fold_predictions(exp_dir)
            all_model_preds[exp_name] = preds
            for split_name, data in preds.items():
                n_samples = len(data["predictions"])
                n_patients = (
                    len(set(data["patient_ids"].tolist()))
                    if data["patient_ids"] is not None
                    else 0
                )
                print(f"    {split_name}: {n_samples} spectra, {n_patients} patients")
        except Exception as e:
            print(f"    ERROR: {e}")
            raise

    # ------------------------------------------------------------------
    # STEPS 3–6: Compute correct masks and find common patients
    # ------------------------------------------------------------------
    print("\n[STEP 3] Computing per-model correct classification masks...")
    model_correct = compute_correct_patients(all_model_preds)

    for model_name, class_patients in model_correct.items():
        display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        total = sum(len(v) for v in class_patients.values())
        print(f"  {display}: {total} correctly classified patients total")

    print("\n[STEP 5-6] Computing intersections across all models...")
    class_common, best_class, best_patients = find_common_patients(model_correct)

    print("\n  Class Rankings (patients correctly classified by ALL models):")
    print("  " + "-" * 50)
    for cls in sorted(class_common.keys()):
        class_name = COMPACT_CLASS_NAMES.get(cls, f"Class {cls}")
        n_common = len(class_common[cls])
        marker = " ← BEST" if cls == best_class else ""
        print(f"    Class {cls} ({class_name}): {n_common} common patients{marker}")

    # ------------------------------------------------------------------
    # STEP 7: Select best class
    # ------------------------------------------------------------------
    best_class_name = COMPACT_CLASS_NAMES.get(best_class, f"Class {best_class}")
    print(f"\n[STEP 7] Best class: {best_class} ({best_class_name})")
    print(f"  Common patients across ALL models: {len(best_patients)}")

    if len(best_patients) == 0:
        print("\n  ERROR: No patients are correctly classified by ALL models.")
        print("  Consider checking individual model predictions or relaxing criteria.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # STEP 8: Sample patients
    # ------------------------------------------------------------------
    print(f"\n[STEP 8] Sampling {args.n_patients} patients...")
    selected_patients = sample_patients(best_patients, args.n_patients, args.seed)
    print(f"  Selected: {selected_patients}")

    # ------------------------------------------------------------------
    # OUTPUT 1: Save common patient summary
    # ------------------------------------------------------------------
    summary = {
        "class_rankings": {
            str(cls): {
                "class_name": COMPACT_CLASS_NAMES.get(cls, f"Class {cls}"),
                "n_common_patients": len(patients),
                "common_patient_ids": sorted(patients),
            }
            for cls, patients in class_common.items()
        },
        "best_class": {
            "class_id": best_class,
            "class_name": best_class_name,
            "n_common_patients": len(best_patients),
        },
        "selected_patients": selected_patients,
        "models": [MODEL_DISPLAY_NAMES.get(n, n) for n in STAGE3_EXPERIMENT_NAMES],
        "n_models": len(STAGE3_EXPERIMENT_NAMES),
    }

    summary_path = output_dir / "common_patient_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OUTPUT 1] Saved: {summary_path}")

    # ------------------------------------------------------------------
    # XAI PHASE: Load data for LIME
    # ------------------------------------------------------------------
    print("\n[XAI] Loading reference data for LIME background distribution...")

    # Load config from the first experiment to get preprocessing settings
    first_exp = experiments[STAGE3_EXPERIMENT_NAMES[0]]
    config_candidates = sorted(first_exp.rglob("config.yaml"))

    if not config_candidates:
        raise FileNotFoundError(f"No config.yaml found under {first_exp}")

    cfg = load_config(str(config_candidates[0]))

    print(f"  Using config: {config_candidates[0]}")

    # Load reference data
    registry = DataRegistry(data_root="data/raw", cfg=cfg)
    registry.load("reference")
    X_ref, y_ref = registry.get_arrays("reference")

    # Fit preprocessor on reference data
    preprocessor = SpectralPreprocessor.from_config(cfg["preprocessing"])
    preprocessor.fit(X_ref)

    # Build background sample
    n_bg = min(args.lime_n_background, len(X_ref))
    rng = np.random.default_rng(args.seed)
    bg_indices = rng.choice(len(X_ref), size=n_bg, replace=False)
    X_background = np.array(X_ref[bg_indices])
    print(f"  Background samples: {X_background.shape}")

    # Load wavenumbers
    wavenumbers = None
    wave_path = Path("data/raw/wavenumbers.npy")
    if wave_path.exists():
        try:
            loaded_w = np.load(wave_path)
            if len(loaded_w) == X_background.shape[1]:
                wavenumbers = loaded_w
                print(f"  Wavenumbers loaded: {wavenumbers.shape}")
            else:
                print(
                    f"  WARNING: wavenumber length ({len(loaded_w)}) ≠ "
                    f"signal length ({X_background.shape[1]})"
                )
        except Exception as e:
            print(f"  WARNING: failed to load wavenumbers: {e}")

    # Load clinical split data to get raw spectra for LIME input
    print("\n[XAI] Loading clinical split data...")
    clinical_splits_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for split_name in ["2018clinical", "2019clinical"]:
        try:
            registry.load(split_name)
            X_split, y_split = registry.get_arrays(split_name)
            print(f"  {split_name}: {X_split.shape}")

            # Remap labels to compact transfer space
            from metadata.ontology import ISOLATE_TO_TREATMENT
            from src.utils.class_subset import class_maps

            split_cfg = cfg.get("splits", {}).get(split_name, {})
            label_space = split_cfg.get("label_space", "")

            if label_space == "isolate_space":
                y_treatment = np.array(
                    [ISOLATE_TO_TREATMENT[int(lbl)] for lbl in y_split],
                    dtype=np.int64,
                )
                mask = np.isin(y_treatment, CLINICAL_SPARSE_IDS)
                X_split = np.array(X_split[mask])
                y_treatment = y_treatment[mask]
                cmap, _ = class_maps(CLINICAL_SPARSE_IDS)
                y_split = np.array([cmap[int(lbl)] for lbl in y_treatment])
            else:
                mask = np.isin(y_split, CLINICAL_SPARSE_IDS)
                X_split = np.array(X_split[mask])
                y_filtered = y_split[mask]
                cmap, _ = class_maps(CLINICAL_SPARSE_IDS)
                y_split = np.array([cmap[int(lbl)] for lbl in y_filtered])

            # Generate patient IDs for this split
            from metadata.patient_ids import generate_patient_ids

            patient_ids = generate_patient_ids(y_split, split_name)

            clinical_splits_data[split_name] = (X_split, y_split, np.array(patient_ids))
            print(
                f"    After filtering: {X_split.shape}, {len(set(patient_ids))} patients"
            )
        except Exception as e:
            print(f"  WARNING: Could not load {split_name}: {e}")

    # Class names for LIME
    from metadata.ontology import GLOBAL_TREATMENTS

    class_names = [GLOBAL_TREATMENTS[gid] for gid in CLINICAL_SPARSE_IDS]

    # ------------------------------------------------------------------
    # XAI: Generate LIME explanations for each patient × each model
    # ------------------------------------------------------------------
    print("\n[XAI] Generating LIME explanations...")
    print("  This may take several minutes per patient per model.\n")

    # Storage for all peaks
    all_peaks: dict[str, dict[str, list[dict]]] = {}  # patient -> model -> peaks
    all_explanations: dict[str, dict[str, Any]] = {}  # patient -> model -> explanation

    for patient_idx, patient_id in enumerate(selected_patients):
        print(f"\n  Patient {patient_idx + 1}/{len(selected_patients)}: {patient_id}")
        print("  " + "-" * 50)

        # Find the raw spectrum for this patient
        target_class, split_name = get_patient_spectra(all_model_preds, patient_id)
        if target_class is None or split_name is None:
            print(f"    WARNING: Could not find patient {patient_id} in predictions")
            continue

        # Get raw spectra from clinical data
        patient_spectrum = None
        for clin_split, (X_clin, y_clin, pids_clin) in clinical_splits_data.items():
            matching = np.where(pids_clin == patient_id)[0]
            if len(matching) > 0:
                # Use the first spectrum for this patient
                patient_spectrum = np.array(X_clin[matching[0]])
                print(f"    Found spectrum in {clin_split} (index {matching[0]})")
                break

        if patient_spectrum is None:
            print(f"    WARNING: No raw spectrum found for patient {patient_id}")
            continue

        patient_explanations: dict[str, Any] = {}
        patient_peaks: dict[str, list[dict]] = {}

        for exp_name in PANEL_ORDER:
            display_name = MODEL_DISPLAY_NAMES.get(exp_name, exp_name)
            print(f"    Explaining with {display_name}...", end=" ", flush=True)

            try:
                exp_dir = experiments[exp_name]
                model, explainer = load_model_and_explainer(
                    exp_dir=exp_dir,
                    X_background=X_background,
                    preprocessor=preprocessor,
                    wavenumbers=wavenumbers,
                    class_names=class_names,
                    lime_n_features=args.lime_n_features,
                    lime_n_samples=args.lime_n_samples,
                    seed=args.seed,
                )

                explanation = explainer.explain_sample(
                    spectrum=patient_spectrum,
                    label=target_class,
                )

                patient_explanations[exp_name] = explanation

                # Extract peaks
                peaks = extract_peaks(explanation, wavenumbers, args.top_k_peaks)
                patient_peaks[exp_name] = peaks

                print(
                    f"done (pred: {explanation.predicted_label}, "
                    f"conf: {explanation.confidence:.1%})"
                )

                # Clean up model to free memory
                del model, explainer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback

                traceback.print_exc()

        all_explanations[patient_id] = patient_explanations
        all_peaks[patient_id] = patient_peaks

        # ------------------------------------------------------------------
        # OUTPUT 2: Per-patient comparison figure
        # ------------------------------------------------------------------
        safe_pid = re.sub(r"[^\w\-]", "_", patient_id)
        fig_path = output_dir / f"patient_{safe_pid}_comparison.png"
        plot_patient_comparison(
            explanations=patient_explanations,
            patient_id=patient_id,
            class_name=COMPACT_CLASS_NAMES.get(target_class, f"Class {target_class}"),
            wavenumbers=wavenumbers,
            save_path=fig_path,
        )

    # ------------------------------------------------------------------
    # OUTPUT 3: Consensus peaks CSV
    # ------------------------------------------------------------------
    print("\n[CONSENSUS] Computing cross-model peak consensus...")
    consensus = compute_consensus(
        all_peaks,
        tolerance=args.peak_tolerance,
        model_names=list(STAGE3_EXPERIMENT_NAMES),
    )

    csv_path = output_dir / "consensus_peaks.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Peak", "Frequency", "Models"])
        for peak in consensus:
            writer.writerow(
                [
                    f"{peak['wavenumber']:.1f}",
                    f"{peak['frequency']}/{peak['n_models']}",
                    ", ".join(sorted(peak["models"])),
                ]
            )
    print(f"\n[OUTPUT 3] Saved: {csv_path}")

    # ------------------------------------------------------------------
    # OUTPUT 4: Consensus peak frequency bar chart
    # ------------------------------------------------------------------
    chart_path = output_dir / "consensus_peak_frequency.png"
    plot_consensus_bar_chart(
        consensus_peaks=consensus,
        save_path=chart_path,
        n_models=len(STAGE3_EXPERIMENT_NAMES),
        n_patients=len(selected_patients),
    )
    print(f"[OUTPUT 4] Saved: {chart_path}")

    # ------------------------------------------------------------------
    # OUTPUT 5: XAI consensus report
    # ------------------------------------------------------------------
    report_path = output_dir / "xai_consensus_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("XAI CONSENSUS REPORT\n")
        f.write("Multi-Model LIME Explainability Comparison\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Best class: {best_class} ({best_class_name})\n\n")
        f.write(
            f"Common patients (correctly classified by ALL {len(STAGE3_EXPERIMENT_NAMES)} models): "
            f"{len(best_patients)}\n\n"
        )

        f.write("Selected patients:\n")
        for pid in selected_patients:
            f.write(f"  {pid}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("Class Rankings\n")
        f.write("-" * 70 + "\n\n")
        for cls in sorted(class_common.keys()):
            class_name = COMPACT_CLASS_NAMES.get(cls, f"Class {cls}")
            n_common = len(class_common[cls])
            marker = " ← BEST" if cls == best_class else ""
            f.write(
                f"  Class {cls} ({class_name}): {n_common} common patients{marker}\n"
            )

        f.write("\n" + "-" * 70 + "\n")
        f.write("Models Used\n")
        f.write("-" * 70 + "\n\n")
        for exp_name in STAGE3_EXPERIMENT_NAMES:
            display = MODEL_DISPLAY_NAMES.get(exp_name, exp_name)
            f.write(f"  {display} ({exp_name})\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("Most Consistent Peaks\n")
        f.write("-" * 70 + "\n\n")

        n_models = len(STAGE3_EXPERIMENT_NAMES)
        for peak in consensus:
            if peak["frequency"] >= 2:
                f.write(
                    f"  {peak['wavenumber']:.0f} cm⁻¹ → "
                    f"{peak['frequency']}/{n_models} models "
                    f"({', '.join(peak['models'])})\n"
                )

        f.write("\n" + "-" * 70 + "\n")
        f.write("Per-Patient Peak Details\n")
        f.write("-" * 70 + "\n\n")

        for patient_id in selected_patients:
            f.write(f"\n  Patient: {patient_id}\n")
            f.write("  " + "~" * 40 + "\n")

            if patient_id not in all_peaks:
                f.write("    (no data)\n")
                continue

            for exp_name in PANEL_ORDER:
                display = MODEL_DISPLAY_NAMES.get(exp_name, exp_name)
                if exp_name in all_peaks[patient_id]:
                    peaks = all_peaks[patient_id][exp_name]
                    peak_strs = [f"{p['wavenumber']:.0f}" for p in peaks[:5]]
                    f.write(f"    {display:20s}: {', '.join(peak_strs)} cm⁻¹\n")
                else:
                    f.write(f"    {display:20s}: (not available)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"[OUTPUT 5] Saved: {report_path}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Output directory:   {output_dir.resolve()}")
    print(f"  Best class:         {best_class} ({best_class_name})")
    print(f"  Common patients:    {len(best_patients)}")
    print(f"  Selected patients:  {len(selected_patients)}")
    print(f"  Consensus peaks:    {len([p for p in consensus if p['frequency'] >= 2])}")
    print(f"\n  Outputs:")
    print(f"    1. {summary_path}")
    print(f"    2. {output_dir}/patient_*_comparison.png")
    print(f"    3. {csv_path}")
    print(f"    4. {chart_path}")
    print(f"    5. {report_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
