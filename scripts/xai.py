import os
import sys
import torch
from pathlib import Path
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.augmentation import AugmentationPipeline
from src.data.dataloader import build_all_loaders
from src.data.preprocessing import SpectralPreprocessor
from src.data.registry import DataRegistry
from src.models.registry import get_model
from src.utils.checkpoint import load_best_model
from src.utils.config import load_config
from src.utils.seed import set_seed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_saliency(signal, saliency, save_path):
    """Inline saliency plot (migrated from scripts/plot_saliency.py)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(signal, label="Signal")
    ax.plot(saliency, label="Saliency", alpha=0.7)
    ax.set_xlabel("Wavelength Index")
    ax.set_ylabel("Intensity / Importance")
    ax.set_title("Saliency Map")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


from src.xai.saliency import compute_saliency, compute_smoothgrad
from metadata.ontology import (
    COMPACT_TO_GLOBAL,
    GLOBAL_TREATMENTS,
    CLINICAL_LABELS,
)

# -----------------------------
# ARGUMENTS
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default=None)
    return p.parse_args()


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # Load config (same as train.py)
    cfg = load_config(
        os.path.join(
            args.exp_dir,
            "config.yaml",
        )
    )

    task_cfg = cfg["task"]
    stage = task_cfg["stage"]
    label_space = task_cfg["label_space"]
    if stage == "pretrain_30class":
        clinical_sparse_ids = []
        n_classes = 30
    elif stage == "pretrain_treatment_8class":
        clinical_sparse_ids = []
        n_classes = 8
    elif stage == "transfer_5class":
        clinical_sparse_ids = (
            task_cfg["clinical_sparse_global_ids"]
        )
        n_classes = len(clinical_sparse_ids)

    else:
        raise ValueError(
            f"Unknown XAI stage: {stage}"
        )
    
    cfg["model"]["n_classes"] = n_classes
    # --------------------------------------------------------
    # Semantic-space integrity checks
    # --------------------------------------------------------
    exp_dir = args.exp_dir

    # -----------------------------
    # DATA
    # -----------------------------
    print("[XAI] Loading data...")

    registry = DataRegistry(data_root="data/raw", cfg=cfg)
    registry.load_all()

    # Fit preprocessor on FULL reference set (matches pretrained backbone)
    X_ref, _ = registry.get_arrays("reference")
    preprocessor = SpectralPreprocessor.from_config(cfg["preprocessing"])
    preprocessor.fit(X_ref)

    augmentation = AugmentationPipeline.from_config(cfg["augmentation"])
    if len(augmentation.steps) == 0 or augmentation.p == 0:
        augmentation = None

    cfg["batch_size"] = (
        cfg.get("training", {})
        .get("batch_size", 256)
    )

    cfg["num_workers"] = (
        cfg.get("training", {})
        .get("num_workers", 4)
    )

    cfg["consistency"] = (
        cfg.get("training", {})
        .get("consistency", {})
    )

    loaders = build_all_loaders(
        registry,
        preprocessor,
        augmentation,
        cfg,
        clinical_sparse_ids=clinical_sparse_ids,
        n_classes=n_classes,
    )

    xai_split = (
        args.split
        or cfg.get("xai", {}).get("split")
    )

    if xai_split is None:
        raise ValueError(
            "No XAI split specified"
        )

    if xai_split in loaders:
        loader = loaders[xai_split]

    elif xai_split in loaders.get("ood", {}):
        loader = loaders["ood"][xai_split]

    else:
        raise ValueError(
            f"Unknown XAI split: {xai_split}"
        )

    # -----------------------------
    # MODEL
    # -----------------------------
    print("[XAI] Loading model...")

    model_name = cfg["model"]["name"]

    model = get_model(
        model_name,
        cfg,
    )
    checkpoint = load_best_model(
        exp_dir,
        model,
    )
    checkpoint_cfg = checkpoint.get(
        "config",
        {},
    )

    checkpoint_stage = (
        checkpoint_cfg
        .get("task", {})
        .get("stage", None)
    )

    checkpoint_label_space = (
        checkpoint_cfg
        .get("task", {})
        .get("label_space", None)
    )

    checkpoint_model_space = (
        checkpoint_cfg
        .get("model", {})
        .get("semantic_space", None)
    )

    assert checkpoint_stage == stage, (
        "Checkpoint stage mismatch:\n"
        f"Expected: {stage}\n"
        f"Found: {checkpoint_stage}"
    )

    assert (
        checkpoint_label_space
        == label_space
    ), (
        "Checkpoint label-space mismatch:\n"
        f"Expected: {label_space}\n"
        f"Found: {checkpoint_label_space}"
    )

    assert (
        checkpoint_model_space
        == cfg["model"]["semantic_space"]
    ), (
        "Checkpoint model semantic-space mismatch:\n"
        f"Expected: "
        f"{cfg['model']['semantic_space']}\n"
        f"Found: {checkpoint_model_space}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # -----------------------------
    # XAI
    # -----------------------------
    print("[XAI] Generating saliency maps...")

    xai_root = Path(exp_dir) / "xai"
    xai_root.mkdir(parents=True, exist_ok=True)

    class_counts = {i: 0 for i in range(n_classes)}

    for x, y in loader:
        for i in range(x.shape[0]):

            label = y[i].item()

            if stage == "transfer_5class":
                limit = 5 if label == 2 else 2
            else:
                limit = 2

            if class_counts[label] >= limit:
                continue

            xi = x[i:i+1].to(device)

            saliency = compute_saliency(model, xi)
            smooth_saliency = compute_smoothgrad(model, xi)
            signal = xi[0].mean(dim=0).detach().cpu().numpy()

            if stage == "transfer_5class":
                global_id = COMPACT_TO_GLOBAL[label]
                treatment_name = (
                    GLOBAL_TREATMENTS[global_id]
                )

                clinical_info = (
                    CLINICAL_LABELS[global_id]
                )

            elif stage == "pretrain_treatment_8class":

                global_id = label

                treatment_name = (
                    GLOBAL_TREATMENTS[global_id]
                )

            else:

                global_id = label

                treatment_name = (
                    f"isolate_{label}"
                )

            safe_name = (
                f"{global_id}_{treatment_name}"
                .replace(" ", "_")
                .replace("/", "_")
            )

            save_dir = xai_root / safe_name
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path_sal = save_dir / f"sample_{class_counts[label]}_saliency.png"
            save_path_sg = save_dir / f"sample_{class_counts[label]}_smoothgrad.png"

            plot_saliency(signal, saliency, save_path_sal)
            plot_saliency(signal, smooth_saliency, save_path_sg)

            class_counts[label] += 1

        if stage == "transfer_5class":
            done = all(
                class_counts[i] >= (
                    5 if i == 2 else 2
                )
                for i in class_counts
            )
        else:
            done = all(
                class_counts[i] >= 2
                for i in class_counts
            )

        if done:
            break

    print(f"[XAI] Saved to {xai_root}")


if __name__ == "__main__":
    main()