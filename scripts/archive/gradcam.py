"""
scripts/gradcam.py

Generate Grad-CAM visualizations for trained models without retraining.
Outputs figures into <exp_dir>/plots/gradcam.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.augmentation import AugmentationPipeline
from src.data.dataloader import build_all_loaders
from src.data.preprocessing import SpectralPreprocessor
from src.data.registry import DataRegistry
from src.interpretability.gradcam1d import GradCAM1D
from src.models.registry import get_model
from src.utils.checkpoint import load_best_model
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.split_modes import IID_REFERENCE, resolve_split_mode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Grad-CAM plots")
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--split", default=None)
    p.add_argument("--per-class", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _stage_title(cfg: dict) -> str:
    stage = cfg.get("task", {}).get("stage", "unknown")
    stage_map = {
        "pretrain_30class": "Stage 1 (Isolate Space)",
        "pretrain_treatment_8class": "Stage 2 (Treatment Space)",
        "transfer_5class": "Stage 3 (Clinical Transfer)",
    }
    return stage_map.get(stage, stage)


def _class_labels(cfg: dict, n_classes: int) -> list[str]:
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
        ids = cfg.get("task", {}).get(
            "clinical_sparse_global_ids", list(range(n_classes))
        )
        try:
            from metadata.ontology import GLOBAL_TREATMENTS

            return [GLOBAL_TREATMENTS[int(i)] for i in ids]
        except Exception:
            return [f"Clinical {i}" for i in range(n_classes)]
    return [f"Class {i}" for i in range(n_classes)]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    exp_dir = Path(args.exp_dir)
    cfg = load_config(str(exp_dir / "config.yaml"))

    task_cfg = cfg["task"]
    stage = task_cfg["stage"]
    task_cfg["label_space"]

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

    registry = DataRegistry(data_root="data/raw", cfg=cfg)
    if resolve_split_mode(cfg) == IID_REFERENCE:
        registry.load("reference")
    else:
        registry.load_all()

    X_ref, _ = registry.get_arrays("reference")
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

    split = args.split or cfg.get("xai", {}).get("split")
    if split is None:
        raise ValueError("No split specified for Grad-CAM")

    if split in loaders:
        loader = loaders[split]
    elif split in loaders.get("ood", {}):
        loader = loaders["ood"][split]
    else:
        raise ValueError(f"Unknown split: {split}")

    model_name = cfg["model"]["name"]
    model = get_model(model_name, cfg)
    _ = load_best_model(str(exp_dir), model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    gcam = GradCAM1D(model)
    labels = _class_labels(cfg, n_classes)
    title = _stage_title(cfg)

    plot_dir = exp_dir / "plots" / "gradcam"
    plot_dir.mkdir(parents=True, exist_ok=True)

    wavenumbers = None
    wave_path = Path("data/raw/wavenumbers.npy")
    if wave_path.exists():
        wavenumbers = np.load(wave_path)

    class_counts = {i: 0 for i in range(n_classes)}

    for batch in loader:
        x, y = batch
        for i in range(x.shape[0]):
            label = int(y[i].item())
            if class_counts[label] >= args.per_class:
                continue
            xi = x[i : i + 1].to(device)
            cam = gcam.compute(xi, target_class=label, signal_length=xi.shape[-1])
            signal = xi[0].mean(dim=0).detach().cpu().numpy()

            fig, ax = plt.subplots(figsize=(10, 4))
            x_axis = wavenumbers if wavenumbers is not None else np.arange(len(signal))
            ax.plot(x_axis, signal, label="Signal", color="black", linewidth=1.0)
            ax.fill_between(x_axis, 0, cam, color="red", alpha=0.35, label="Grad-CAM")
            ax.set_title(f"Grad-CAM — {title} — {labels[label]} ({split})")
            ax.set_xlabel("Wavenumber" if wavenumbers is not None else "Index")
            ax.set_ylabel("Intensity / Importance")
            ax.legend(loc="upper right")
            fig.tight_layout()

            out_path = (
                plot_dir / f"gradcam_{split}_class_{label}_{class_counts[label]}.png"
            )
            fig.savefig(out_path, dpi=500, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            class_counts[label] += 1
        if all(c >= args.per_class for c in class_counts.values()):
            break

    print(f"[gradcam] Saved Grad-CAM plots in {plot_dir}")


if __name__ == "__main__":
    main()
