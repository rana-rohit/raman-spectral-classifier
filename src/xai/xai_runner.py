"""
src/xai/xai_runner.py

Single orchestration layer for Raman XAI outputs.

This module keeps the entrypoint logic in one place so running
scripts/xai.py generates both saliency and LIME artifacts in a
stable directory layout:

    experiments/<run>/xai/<stage>/<split>/
        saliency/<class>/...
        lime/<class>/...

The underlying explainers remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metadata.ontology import GLOBAL_TREATMENTS, ISOLATES
from src.data.augmentation import AugmentationPipeline
from src.data.dataloader import build_all_loaders
from src.data.preprocessing import SpectralPreprocessor
from src.data.registry import DataRegistry
from src.models.registry import get_model
from src.utils.checkpoint import load_best_model
from src.utils.class_subset import class_maps
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.split_modes import IID_REFERENCE, resolve_split_mode
from src.xai.predict_wrapper import build_predict_fn
from src.xai.saliency import compute_saliency, compute_smoothgrad


def _safe_name(value: str) -> str:
    return str(value).replace(" ", "_").replace("/", "_").replace(".", "")


def _display_label(stage: str, label: int, clinical_sparse_ids: list[int]) -> str:
    if stage == "pretrain_30class":
        if 0 <= label < len(ISOLATES):
            return ISOLATES[label]["strain"]
        return f"Class {label}"

    if stage == "pretrain_treatment_8class":
        return GLOBAL_TREATMENTS.get(label, f"Class {label}")

    if stage == "transfer_5class":
        if 0 <= label < len(clinical_sparse_ids):
            global_id = int(clinical_sparse_ids[label])
            return GLOBAL_TREATMENTS.get(global_id, f"Class {label}")
        return f"Class {label}"

    return f"Class {label}"


def _artifact_stem(
    sample_index: int,
    true_label: str,
    predicted_label: str,
    suffix: str,
) -> str:
    return (
        f"sample_{sample_index:03d}_true_{_safe_name(true_label)}"
        f"_pred_{_safe_name(predicted_label)}_{suffix}"
    )


def _stage_display_name(stage: str) -> str:
    return {
        "pretrain_30class": "Stage 1 - Isolate Space (30 classes)",
        "pretrain_treatment_8class": "Stage 2 - Treatment Space (8 classes)",
        "transfer_5class": "Stage 3 - Clinical Transfer (5 classes)",
    }.get(stage, stage)


def _resolve_stage_context(cfg: dict, seed: int) -> tuple[str, str, int, list[int]]:
    task_cfg = cfg["task"]
    stage = task_cfg["stage"]
    label_space = task_cfg["label_space"]

    if stage == "pretrain_30class":
        clinical_sparse_ids: list[int] = []
        n_classes = 30
    elif stage == "pretrain_treatment_8class":
        clinical_sparse_ids = []
        n_classes = 8
    elif stage == "transfer_5class":
        clinical_sparse_ids = list(task_cfg["clinical_sparse_global_ids"])
        n_classes = len(clinical_sparse_ids)
    else:
        raise ValueError(f"Unknown XAI stage: {stage}")

    cfg["model"]["n_classes"] = n_classes
    cfg["seed"] = int(seed)
    return stage, label_space, n_classes, clinical_sparse_ids


def _validate_label_space(stage: str, label_space: str, n_classes: int) -> None:
    expected = {
        "pretrain_30class": ("isolate_space", 30),
        "pretrain_treatment_8class": ("global_treatment_space", 8),
        "transfer_5class": ("sparse_global_treatment_space", 5),
    }
    if stage not in expected:
        raise ValueError(f"Unsupported XAI stage: {stage}")

    expected_label_space, expected_classes = expected[stage]
    assert label_space == expected_label_space, (
        f"Stage/label-space mismatch: stage={stage} expects {expected_label_space}, "
        f"found {label_space}"
    )
    assert n_classes == expected_classes, (
        f"Stage/class-count mismatch: stage={stage} expects {expected_classes} classes, "
        f"found {n_classes}"
    )


def _class_names(
    stage: str, n_classes: int, clinical_sparse_ids: list[int]
) -> list[str]:
    if stage == "pretrain_30class":
        try:
            return [ISOLATES[i]["strain"] for i in range(n_classes)]
        except Exception:
            return [f"Isolate {i}" for i in range(n_classes)]

    if stage == "pretrain_treatment_8class":
        return [GLOBAL_TREATMENTS[i] for i in range(n_classes)]

    if stage == "transfer_5class":
        return [GLOBAL_TREATMENTS[int(i)] for i in clinical_sparse_ids]

    return [f"Class {i}" for i in range(n_classes)]


def _label_folder_name(stage: str, label: int, clinical_sparse_ids: list[int]) -> str:
    if stage == "pretrain_30class":
        return f"isolate_{label}"

    if stage == "pretrain_treatment_8class":
        return f"{label}_{_safe_name(GLOBAL_TREATMENTS[label])}"

    if stage == "transfer_5class":
        global_id = int(clinical_sparse_ids[label])
        return f"{global_id}_{_safe_name(GLOBAL_TREATMENTS[global_id])}"

    return f"class_{label}"


def _resolve_loader(loaders: dict, split: str):
    if split in loaders:
        return loaders[split]
    if split in loaders.get("ood", {}):
        return loaders["ood"][split]
    raise ValueError(f"Unknown XAI split: {split}")


def _load_split_arrays(
    registry: DataRegistry, split: str
) -> tuple[np.ndarray, np.ndarray]:
    if split == "reference":
        return registry.get_arrays("reference")
    if split == "test":
        registry.load(split)
        return registry.get_arrays(split, allow_holdout=True)

    registry.load(split)
    return registry.get_arrays(split)


def _validate_split_labels(stage: str, y_split: np.ndarray, n_classes: int) -> None:
    y_split = np.asarray(y_split)
    if y_split.size == 0:
        raise ValueError(f"XAI split for stage {stage} produced no samples")

    if y_split.min() < 0 or y_split.max() >= n_classes:
        raise ValueError(
            f"Label remap produced invalid labels for stage {stage}: "
            f"min={int(y_split.min())}, max={int(y_split.max())}, n_classes={n_classes}"
        )


def _validate_probability_matrix(
    probs: np.ndarray, n_classes: int, context: str
) -> None:
    if probs.ndim != 2:
        raise ValueError(
            f"{context}: expected 2D probability matrix, got shape {probs.shape}"
        )
    if probs.shape[1] != n_classes:
        raise ValueError(
            f"{context}: expected {n_classes} probability columns, got {probs.shape[1]}"
        )
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"{context}: probability rows do not sum to 1")


def _remap_lime_labels(
    cfg: dict,
    stage: str,
    split: str,
    X_split: np.ndarray,
    y_split: np.ndarray,
    clinical_sparse_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    split_cfg = cfg.get("splits", {}).get(split, {})
    label_space = split_cfg.get("label_space", "")

    if stage == "pretrain_treatment_8class" and label_space == "isolate_space":
        from metadata.ontology import ISOLATE_TO_TREATMENT

        y_split = np.array(
            [ISOLATE_TO_TREATMENT[int(lbl)] for lbl in y_split],
            dtype=np.int64,
        )
        return X_split, y_split

    if stage == "transfer_5class":
        if label_space == "isolate_space":
            from metadata.ontology import ISOLATE_TO_TREATMENT

            y_treatment = np.array(
                [ISOLATE_TO_TREATMENT[int(lbl)] for lbl in y_split],
                dtype=np.int64,
            )
            mask = np.isin(y_treatment, clinical_sparse_ids)
            X_split = np.array(X_split[mask])
            y_treatment = y_treatment[mask]
            cmap, _ = class_maps(clinical_sparse_ids)
            y_split = np.array([cmap[int(lbl)] for lbl in y_treatment], dtype=np.int64)
            return X_split, y_split

        mask = np.isin(y_split, clinical_sparse_ids)
        X_split = np.array(X_split[mask])
        y_filtered = y_split[mask]
        cmap, _ = class_maps(clinical_sparse_ids)
        y_split = np.array([cmap[int(lbl)] for lbl in y_filtered], dtype=np.int64)
        return X_split, y_split

    return X_split, y_split


def _load_wavenumbers(signal_length: int) -> np.ndarray | None:
    wave_path = Path("data/raw/wavenumbers.npy")
    if not wave_path.exists():
        print("[XAI] Wavenumbers not found; falling back to index-based axes")
        return None

    try:
        loaded = np.load(wave_path)
    except Exception:
        print(
            "[XAI] Warning: failed to load wavenumbers; falling back to index-based axes"
        )
        return None

    if len(loaded) != signal_length:
        print(
            f"[XAI] Warning: wavenumber length {len(loaded)} does not match signal length {signal_length}; "
            "falling back to index-based axes"
        )
        return None

    print(f"[XAI] Loaded wavenumbers from {wave_path} ({len(loaded)} points)")
    return loaded


def _plot_saliency(
    signal: np.ndarray,
    saliency: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(signal, label="Signal")
    ax.plot(saliency, label="Saliency", alpha=0.7)
    ax.set_xlabel("Wavelength Index")
    ax.set_ylabel("Intensity / Importance")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _extract_batch_xy(batch):
    if isinstance(batch, dict):
        x = batch.get("x1")
        if x is None:
            x = batch.get("x")
        if x is None:
            x = next(value for value in batch.values() if torch.is_tensor(value))
        y = batch.get("y")
        return x, y

    return batch


def _run_saliency(
    loader,
    model: torch.nn.Module,
    stage: str,
    clinical_sparse_ids: list[int],
    output_root: Path,
    device: torch.device,
    per_class_limit: int = 2,
) -> dict[int, int]:
    print("[XAI] Generating saliency maps...")

    if not hasattr(model, "forward_features") or not hasattr(model, "forward_logits"):
        print(
            "[XAI] Saliency skipped: model does not expose forward_features()/forward_logits()"
        )
        return {}

    method_root = output_root / "saliency"
    method_root.mkdir(parents=True, exist_ok=True)
    print(f"[XAI] Output root validated: {output_root}")
    print(f"[XAI] Saliency directory: {method_root}")

    if stage == "transfer_5class":
        class_counts = {i: 0 for i in range(len(clinical_sparse_ids))}
    elif stage == "pretrain_treatment_8class":
        class_counts = {i: 0 for i in range(8)}
    else:
        class_counts = {i: 0 for i in range(30)}

    for batch in loader:
        x, y = _extract_batch_xy(batch)

        for i in range(x.shape[0]):
            label = int(y[i].item())
            if label not in class_counts:
                continue

            limit = 5 if stage == "transfer_5class" and label == 2 else per_class_limit
            if class_counts[label] >= limit:
                continue

            xi = x[i : i + 1].to(device)
            with torch.no_grad():
                outputs = model(xi)
                logits = (
                    outputs["main_logits"] if isinstance(outputs, dict) else outputs
                )
                if logits.ndim != 2 or logits.shape[1] != len(class_counts):
                    raise ValueError(
                        f"Saliency model output shape mismatch: expected (1, {len(class_counts)}), got {tuple(logits.shape)}"
                    )
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                _validate_probability_matrix(probs, len(class_counts), "saliency")
                predicted_class = int(np.argmax(probs[0]))
                confidence = float(probs[0, predicted_class])

            saliency = compute_saliency(model, xi)
            smooth_saliency = compute_smoothgrad(model, xi)
            signal = xi[0].mean(dim=0).detach().cpu().numpy()
            true_name = _display_label(stage, label, clinical_sparse_ids)
            pred_name = _display_label(stage, predicted_class, clinical_sparse_ids)
            base_title = (
                f"{_stage_display_name(stage)} | Saliency | Split: {output_root.name} | "
                f"True: {true_name} | Predicted: {pred_name} | Confidence: {confidence:.2f}"
            )

            folder_name = _label_folder_name(stage, label, clinical_sparse_ids)
            save_dir = method_root / folder_name
            save_dir.mkdir(parents=True, exist_ok=True)

            sample_num = class_counts[label] + 1
            save_path_sal = (
                save_dir
                / f"{_artifact_stem(sample_num, true_name, pred_name, 'saliency')}.png"
            )
            save_path_sg = (
                save_dir
                / f"{_artifact_stem(sample_num, true_name, pred_name, 'smoothgrad')}.png"
            )

            _plot_saliency(signal, saliency, save_path_sal, base_title)
            _plot_saliency(
                signal,
                smooth_saliency,
                save_path_sg,
                base_title.replace("Saliency", "SmoothGrad"),
            )

            class_counts[label] += 1

        if stage == "transfer_5class":
            done = all(
                class_counts[i] >= (5 if i == 2 else per_class_limit)
                for i in class_counts
            )
        else:
            done = all(count >= per_class_limit for count in class_counts.values())

        if done:
            break

    summary_path = method_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "method": "saliency",
                "stage": stage,
                "class_counts": class_counts,
                "output_root": str(method_root),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[XAI] Saliency outputs saved to {method_root}")
    return class_counts


def _run_lime(
    cfg: dict,
    stage: str,
    label_space: str,
    split: str,
    exp_dir: Path,
    model_name: str,
    X_ref: np.ndarray,
    X_split: np.ndarray,
    y_split: np.ndarray,
    clinical_sparse_ids: list[int],
    n_classes: int,
    preprocessor: SpectralPreprocessor,
    model: torch.nn.Module,
    device: torch.device,
    output_root: Path,
    args: argparse.Namespace,
) -> dict[int, int]:
    print("[XAI] Generating LIME explanations...")

    from src.xai.lime_explainer import SpectralLimeExplainer
    from src.xai.xai_visualization import (plot_lime_comparison,
                                           plot_lime_explanation)

    class_names = _class_names(stage, n_classes, clinical_sparse_ids)
    method_root = output_root / "lime"
    method_root.mkdir(parents=True, exist_ok=True)
    print(f"[XAI] Output root validated: {output_root}")
    print(f"[XAI] LIME directory: {method_root}")

    n_background = min(args.lime_background, len(X_ref))
    rng = np.random.default_rng(args.seed)
    bg_indices = rng.choice(len(X_ref), size=n_background, replace=False)
    X_background = np.array(X_ref[bg_indices])

    if n_background < max(200, args.lime_features * 10):
        warnings.warn(
            f"LIME background sample size ({n_background}) may be too small for {args.lime_features} features; explanations may be unstable",
            RuntimeWarning,
        )

    predict_fn = build_predict_fn(
        model=model,
        preprocessor=preprocessor,
        device=str(device),
        batch_size=256,
    )

    wavenumbers = _load_wavenumbers(X_background.shape[1])
    explainer = SpectralLimeExplainer(
        predict_fn=predict_fn,
        training_data=X_background,
        wavenumbers=wavenumbers,
        class_names=class_names,
        n_features=args.lime_features,
        n_samples=args.lime_samples,
        random_state=args.seed,
    )

    X_split, y_split = _remap_lime_labels(
        cfg=cfg,
        stage=stage,
        split=split,
        X_split=np.asarray(X_split),
        y_split=np.asarray(y_split),
        clinical_sparse_ids=clinical_sparse_ids,
    )
    _validate_split_labels(stage, y_split, n_classes)

    preview_probs = predict_fn(X_background[:1])
    _validate_probability_matrix(preview_probs, n_classes, "LIME predict_fn preview")

    print(f"[XAI] Stage: {_stage_display_name(stage)}")
    print(f"[XAI] Label space: {label_space}")
    print(f"[XAI] Split: {split}")
    print(f"[XAI] Number of classes: {n_classes}")
    print(f"[XAI] Model: {model_name}")
    print(f"[XAI] Checkpoint: {exp_dir / 'best_model.pt'}")

    class_counts = {i: 0 for i in range(n_classes)}
    explanations_by_class: dict[int, list] = {i: [] for i in range(n_classes)}
    indices = rng.permutation(len(X_split))

    for idx in indices:
        label = int(y_split[idx])
        if label not in class_counts:
            continue
        if class_counts[label] >= args.lime_per_class:
            continue

        raw_spectrum = np.array(X_split[idx])
        explanation = explainer.explain_sample(
            spectrum=raw_spectrum,
            label=label,
        )

        _validate_probability_matrix(
            explanation.probabilities[np.newaxis, :],
            n_classes,
            "LIME explanation probabilities",
        )

        true_name = _display_label(stage, label, clinical_sparse_ids)
        pred_name = explanation.predicted_label
        sample_num = class_counts[label] + 1
        title = (
            f"{_stage_display_name(stage)} | LIME | Split: {split} | "
            f"True: {true_name} | Predicted: {pred_name} | Confidence: {explanation.confidence:.2f}"
        )

        folder_name = _label_folder_name(stage, label, clinical_sparse_ids)
        sample_dir = method_root / folder_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        plot_path = (
            sample_dir
            / f"{_artifact_stem(sample_num, true_name, pred_name, 'lime')}.png"
        )
        plot_lime_explanation(
            explanation=explanation,
            save_path=plot_path,
            title=title,
            stage_label=_stage_display_name(stage),
            split_label=split,
        )

        explanations_by_class[label].append(explanation)
        class_counts[label] += 1

        target_classes = set(np.unique(y_split).tolist()) & set(class_counts.keys())
        if all(class_counts.get(c, 0) >= args.lime_per_class for c in target_classes):
            break

    for label, explanations in explanations_by_class.items():
        if len(explanations) < 2:
            continue

        folder_name = _label_folder_name(stage, label, clinical_sparse_ids)
        sample_dir = method_root / folder_name
        comparison_path = sample_dir / f"class_{label:03d}_comparison.png"

        plot_lime_comparison(
            explanations=explanations,
            save_path=comparison_path,
            title=f"{_stage_display_name(stage)} | LIME Comparison | Split: {split} | Class: {class_names[label] if label < len(class_names) else f'Class {label}'}",
        )

    summary_path = method_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "method": "lime",
                "stage": stage,
                "split": split,
                "class_counts": class_counts,
                "output_root": str(method_root),
                "per_class": args.lime_per_class,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[XAI] LIME outputs saved to {method_root}")
    return class_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate saliency and LIME explanations for trained Raman models",
    )
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=["all", "saliency", "lime"],
        help="Which XAI methods to run. Default runs both.",
    )
    parser.add_argument(
        "--saliency-per-class",
        type=int,
        default=2,
        help="Number of saliency examples to save per class.",
    )
    parser.add_argument(
        "--lime-per-class",
        type=int,
        default=2,
        help="Number of LIME examples to save per class.",
    )
    parser.add_argument(
        "--lime-samples",
        type=int,
        default=2000,
        help="Number of LIME perturbation samples.",
    )
    parser.add_argument(
        "--lime-features",
        type=int,
        default=20,
        help="Number of top LIME features.",
    )
    parser.add_argument(
        "--lime-background",
        type=int,
        default=500,
        help="Number of background reference samples for LIME.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    exp_dir = Path(args.exp_dir)
    cfg = load_config(str(exp_dir / "config.yaml"))

    stage, label_space, n_classes, clinical_sparse_ids = _resolve_stage_context(
        cfg, args.seed
    )
    _validate_label_space(stage, label_space, n_classes)

    print("\n" + "=" * 60)
    print("XAI EXPLAINABILITY")
    print("=" * 60)
    print(f"  Stage:         {_stage_display_name(stage)}")
    print(f"  Label space:   {label_space}")
    print(f"  Classes:       {n_classes}")
    print(f"  Experiment:    {exp_dir}")
    print(f"  Methods:       {args.methods}")
    print("=" * 60 + "\n")

    xai_split = args.split or cfg.get("xai", {}).get("split")
    if xai_split is None:
        raise ValueError("No XAI split specified")

    methods = set(args.methods)
    if "all" in methods:
        methods = {"saliency", "lime"}

    print("[XAI] Loading data...")
    print(f"[XAI] Active stage: {_stage_display_name(stage)}")
    print(f"[XAI] Label space: {label_space}")
    print(f"[XAI] Split: {xai_split}")
    print(f"[XAI] Number of classes: {n_classes}")
    print(f"[XAI] Model name: {cfg['model']['name']}")
    registry = DataRegistry(data_root="data/raw", cfg=cfg)
    if resolve_split_mode(cfg) == IID_REFERENCE:
        registry.load("reference")
        print("[XAI] Data source: IID reference split")
    else:
        registry.load_all()
        print("[XAI] Data source: full registry with OOD splits")

    X_ref, y_ref = registry.get_arrays("reference")
    X_split, y_split = _load_split_arrays(registry, xai_split)

    preprocessor = SpectralPreprocessor.from_config(cfg["preprocessing"])
    preprocessor.fit(X_ref)

    augmentation = AugmentationPipeline.from_config(cfg["augmentation"])
    if len(augmentation.steps) == 0 or augmentation.p == 0:
        augmentation = None

    cfg["batch_size"] = cfg.get("training", {}).get("batch_size", 256)
    cfg["num_workers"] = cfg.get("training", {}).get("num_workers", 4)
    cfg["consistency"] = cfg.get("training", {}).get("consistency", {})

    loaders = build_all_loaders(
        registry,
        preprocessor,
        augmentation,
        cfg,
        clinical_sparse_ids=clinical_sparse_ids,
        n_classes=n_classes,
    )
    loader = _resolve_loader(loaders, xai_split)

    print("[XAI] Loading model...")
    model_name = cfg["model"]["name"]
    model = get_model(model_name, cfg)
    checkpoint = load_best_model(str(exp_dir), model)

    checkpoint_cfg = checkpoint.get("config", {})
    checkpoint_stage = checkpoint_cfg.get("task", {}).get("stage")
    checkpoint_label_space = checkpoint_cfg.get("task", {}).get("label_space")
    checkpoint_model_space = checkpoint_cfg.get("model", {}).get("semantic_space")

    assert checkpoint_stage == stage, (
        "Checkpoint stage mismatch:\n"
        f"Expected: {stage}\n"
        f"Found: {checkpoint_stage}"
    )
    assert checkpoint_label_space == label_space, (
        "Checkpoint label-space mismatch:\n"
        f"Expected: {label_space}\n"
        f"Found: {checkpoint_label_space}"
    )
    assert checkpoint_model_space == cfg["model"]["semantic_space"], (
        "Checkpoint model semantic-space mismatch:\n"
        f"Expected: {cfg['model']['semantic_space']}\n"
        f"Found: {checkpoint_model_space}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"[XAI] Checkpoint path: {exp_dir / 'best_model.pt'}")

    xai_root = exp_dir / "xai" / stage / xai_split
    xai_root.mkdir(parents=True, exist_ok=True)
    (xai_root / "lime").mkdir(parents=True, exist_ok=True)
    (xai_root / "saliency").mkdir(parents=True, exist_ok=True)
    print(f"[XAI] Output tree ready: {xai_root}")
    print(f"[XAI]   - {xai_root / 'lime'}")
    print(f"[XAI]   - {xai_root / 'saliency'}")

    outputs: dict[str, dict[int, int]] = {}
    if "saliency" in methods:
        outputs["saliency"] = _run_saliency(
            loader=loader,
            model=model,
            stage=stage,
            clinical_sparse_ids=clinical_sparse_ids,
            output_root=xai_root,
            device=device,
            per_class_limit=args.saliency_per_class,
        )

    if "lime" in methods:
        outputs["lime"] = _run_lime(
            cfg=cfg,
            stage=stage,
            label_space=label_space,
            split=xai_split,
            exp_dir=exp_dir,
            model_name=model_name,
            X_ref=X_ref,
            X_split=X_split,
            y_split=y_split,
            clinical_sparse_ids=clinical_sparse_ids,
            n_classes=n_classes,
            preprocessor=preprocessor,
            model=model,
            device=device,
            output_root=xai_root,
            args=args,
        )

    manifest_path = xai_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stage": stage,
                "split": xai_split,
                "methods": sorted(methods),
                "output_root": str(xai_root),
                "outputs": {
                    name: {str(k): v for k, v in counts.items()}
                    for name, counts in outputs.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[XAI] Saved outputs to {xai_root}")


if __name__ == "__main__":
    main()
