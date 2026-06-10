"""
scripts/aggregate_folds.py

Post-processing aggregation script for 5-fold patient-aware cross-validation.
Loads predictions from each fold, concatenates them, and computes overall
spectrum-level and patient-level classification metrics, confusion matrices,
and generates plots.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import compute_confusion_matrix, patient_vote_predictions
from src.evaluation.visualization import save_confusion_matrix_figure
from metadata.ontology import GLOBAL_TREATMENTS, INVERSE_COMPACT_LABEL_MAP


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate 5-fold cross validation results")
    p.add_argument("--run-dir", required=True, help="Directory containing fold subdirectories")
    p.add_argument("--save-dir", default=None, help="Directory to save aggregated results (defaults to run-dir)")
    return p.parse_args()


def _compute_aggregate(split_name, records, save_dir):
    import torch

    logits = np.concatenate([record["logits"] for record in records], axis=0)
    probs = np.concatenate([record["probabilities"] for record in records], axis=0)
    preds = np.concatenate([record["predictions"] for record in records], axis=0)
    targets = np.concatenate([record["targets"] for record in records], axis=0)
    patient_ids = np.concatenate([record["patient_ids"] for record in records], axis=0)

    if len(patient_ids) != len(targets):
        raise RuntimeError(
            f"{split_name}: patient_id count ({len(patient_ids)}) does not match "
            f"target count ({len(targets)})"
        )

    patient_seen_counts = {
        str(pid): int(np.sum(patient_ids == pid))
        for pid in sorted(set(patient_ids.tolist()))
    }
    patient_folds = {}
    for record in records:
        fold_name = record.get("fold_name", "unknown")
        for pid in sorted(set(record["patient_ids"].tolist())):
            patient_folds.setdefault(str(pid), set()).add(str(fold_name))
    repeated_patients = {
        pid: sorted(folds)
        for pid, folds in patient_folds.items()
        if len(folds) > 1
    }
    if repeated_patients:
        example_pid = sorted(repeated_patients)[0]
        raise RuntimeError(
            f"{split_name}: patient {example_pid} appears in multiple folds "
            f"{repeated_patients[example_pid]}; each patient must be held out exactly once"
        )

    spec_accuracy = accuracy_score(targets, preds)
    spec_precision = precision_score(targets, preds, average="macro", zero_division=0)
    spec_recall = recall_score(targets, preds, average="macro", zero_division=0)
    spec_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    spec_mcc = matthews_corrcoef(targets, preds)

    cm_spec, present_classes = compute_confusion_matrix(
        torch.as_tensor(logits),
        torch.as_tensor(targets),
        n_classes=logits.shape[-1],
    )

    pat_preds, pat_targets, unique_pids = patient_vote_predictions(
        probabilities=probs,
        targets=targets,
        patient_ids=patient_ids,
    )

    if len(unique_pids) != len(set(unique_pids)):
        raise RuntimeError(f"{split_name}: duplicate patient IDs after aggregation")

    pat_accuracy = accuracy_score(pat_targets, pat_preds)
    pat_precision = precision_score(pat_targets, pat_preds, average="macro", zero_division=0)
    pat_recall = recall_score(pat_targets, pat_preds, average="macro", zero_division=0)
    pat_f1 = f1_score(pat_targets, pat_preds, average="macro", zero_division=0)
    pat_mcc = matthews_corrcoef(pat_targets, pat_preds)

    cm_pat, _ = compute_confusion_matrix(
        torch.as_tensor(np.eye(logits.shape[-1])[pat_preds]),
        torch.as_tensor(pat_targets),
        n_classes=logits.shape[-1],
    )

    print(
        f"  Spectrum-Level Accuracy: {spec_accuracy:.4f} | "
        f"Precision: {spec_precision:.4f} | Recall: {spec_recall:.4f} | "
        f"F1 Macro: {spec_f1:.4f} | MCC: {spec_mcc:.4f}"
    )
    print(
        f"  Patient-Level Accuracy:  {pat_accuracy:.4f} | "
        f"Precision: {pat_precision:.4f} | Recall: {pat_recall:.4f} | "
        f"F1 Macro: {pat_f1:.4f} | MCC: {pat_mcc:.4f}"
    )
    print(f"  Total Spectra: {len(targets)} | Total Patients: {len(pat_targets)}")

    class_labels = []
    for c in present_classes:
        global_idx = INVERSE_COMPACT_LABEL_MAP.get(int(c), int(c))
        treatment_name = GLOBAL_TREATMENTS.get(global_idx, f"Group {c}")
        class_labels.append(treatment_name)

    fig_dir = save_dir / "confusion_matrices" / split_name
    fig_dir.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix_figure(
        targets=targets,
        predictions=preds,
        class_labels=class_labels,
        save_path=fig_dir / "aggregated_spectrum_confusion_normalized.png",
        title=f"Aggregated {split_name} Spectrum-Level Confusion (Normalized)",
        normalize=True,
    )

    save_confusion_matrix_figure(
        targets=pat_targets,
        predictions=pat_preds,
        class_labels=class_labels,
        save_path=fig_dir / "aggregated_patient_confusion_normalized.png",
        title=f"Aggregated {split_name} Patient-Level Confusion (Normalized)",
        normalize=True,
    )

    return {
        "spectrum_metrics": {
            "accuracy": float(spec_accuracy),
            "precision_macro": float(spec_precision),
            "recall_macro": float(spec_recall),
            "f1_macro": float(spec_f1),
            "mcc": float(spec_mcc),
            "n_samples": int(len(targets)),
        },
        "patient_metrics": {
            "accuracy": float(pat_accuracy),
            "precision_macro": float(pat_precision),
            "recall_macro": float(pat_recall),
            "f1_macro": float(pat_f1),
            "mcc": float(pat_mcc),
            "n_patients": int(len(pat_targets)),
        },
        "present_classes": [int(x) for x in present_classes],
        "spectrum_confusion_matrix": cm_spec.tolist(),
        "patient_confusion_matrix": cm_pat.tolist(),
        "patient_ids": [str(pid) for pid in unique_pids],
        "patient_spectrum_counts": patient_seen_counts,
    }


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    save_dir = Path(args.save_dir) if args.save_dir else run_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n============================================================")
    print(f"AGGREGATING PATIENT CV RESULTS IN:")
    print(f"  {run_dir.resolve()}")
    print(f"============================================================\n")

    # Find fold directories
    fold_dirs = sorted([
        d for d in run_dir.iterdir()
        if d.is_dir() and ("_fold" in d.name or "fold_" in d.name)
    ])

    if not fold_dirs:
        # Check if the run-dir itself contains folds inside subdirectories
        fold_dirs = sorted([
            d for d in run_dir.glob("**/fold_*")
            if d.is_dir()
        ])
        if not fold_dirs:
            fold_dirs = sorted([
                d for d in run_dir.glob("**/*_fold*")
                if d.is_dir()
            ])

    if not fold_dirs:
        raise FileNotFoundError(
            f"No fold subdirectories found in {run_dir}. "
            f"Expected subdirectories containing '_fold' or 'fold_' in name."
        )

    print(f"Found {len(fold_dirs)} fold subdirectories:")
    for fd in fold_dirs:
        print(f"  - {fd.name}")

    # Load predictions from all folds
    fold_data = []
    for fd in fold_dirs:
        pred_file = fd / "detailed_predictions.json"
        if not pred_file.exists():
            print(f"Warning: {pred_file} not found. Skipping fold {fd.name}.")
            continue
        with open(pred_file, "r") as f:
            fold_data.append(json.load(f))

    if not fold_data:
        raise FileNotFoundError("No detailed_predictions.json files loaded.")

    # Identify clinical splits present in the results
    all_splits = set()
    for fd in fold_data:
        all_splits.update(fd.keys())
    
    preferred_splits = [
        "test",
        "2018clinical",
        "2019clinical",
        "clinical_val",
    ]
    clinical_splits = sorted([s for s in preferred_splits if s in all_splits])
    if not clinical_splits:
        print("No clinical splits found in predictions. Using all splits.")
        clinical_splits = sorted(list(all_splits))

    aggregated_results = {
        "run_dir": str(run_dir.resolve()),
        "folds_found": [fd.name for fd in fold_dirs],
        "splits": {},
    }

    clinical_all_records = []

    # Process each split independently, then once more as clinical_all.
    for split_name in clinical_splits:
        print(f"\nProcessing split: {split_name}...")
        records = []

        for fd_idx, fd in enumerate(fold_data):
            if split_name not in fd:
                continue
            data = fd[split_name]
            pids = data.get("patient_ids")
            if pids is None:
                raise RuntimeError(
                    f"Patient IDs missing for {split_name} in fold {fold_dirs[fd_idx].name}. "
                    "Patient-aware aggregation requires real patient IDs."
                )
            record = {
                "logits": np.array(data["logits"]),
                "probabilities": np.array(data["probabilities"]),
                "predictions": np.array(data["predictions"]),
                "targets": np.array(data["targets"]),
                "patient_ids": np.array(pids),
                "fold_name": fold_dirs[fd_idx].name,
            }
            records.append(record)

            if split_name in {"2018clinical", "2019clinical", "clinical_val"}:
                clinical_all_records.append(record)

        if not records:
            print(f"No data found for split {split_name} across folds.")
            continue

        aggregated_results["splits"][split_name] = _compute_aggregate(
            split_name,
            records,
            save_dir,
        )

    if clinical_all_records:
        print("\nProcessing split: clinical_all...")
        aggregated_results["splits"]["clinical_all"] = _compute_aggregate(
            "clinical_all",
            clinical_all_records,
            save_dir,
        )

    # Save results json
    out_json = save_dir / "aggregated_cv_results.json"
    with open(out_json, "w") as f:
        json.dump(aggregated_results, f, indent=2)

    print(f"\n============================================================")
    print(f"AGGREGATION COMPLETE. Saved results to:")
    print(f"  - {out_json.resolve()}")
    for split_name in aggregated_results["splits"]:
        fig_dir = save_dir / "confusion_matrices" / split_name
        print(f"  - {fig_dir.resolve()}/aggregated_spectrum_confusion_normalized.png")
        print(f"  - {fig_dir.resolve()}/aggregated_patient_confusion_normalized.png")
    print(f"============================================================\n")


if __name__ == "__main__":
    main()
