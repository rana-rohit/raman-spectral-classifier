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
from src.utils.class_subset import filter_and_remap_classes

from src.xai.saliency import compute_saliency, compute_smoothgrad
from scripts.plot_saliency import plot_saliency


# -----------------------------
# ARGUMENTS
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--exp-dir", required=True)
    p.add_argument("--exp-name", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # Load config (same as train.py)
    cfg = load_config(
        "configs/data/splits.yaml",
        "configs/data/preprocessing.yaml",
        "configs/data/augmentation.yaml",
        "configs/training/base.yaml",
        f"configs/model/{args.model}.yaml",
    )

    shared_classes = cfg["dataset"]["shared_classes"]
    n_classes = cfg["dataset"]["n_classes_clinical"]
    cfg["model"]["n_classes"] = n_classes

    exp_dir = os.path.join(args.exp_dir, args.exp_name)

    # -----------------------------
    # DATA
    # -----------------------------
    print("[XAI] Loading data...")

    registry = DataRegistry(data_root="data/raw", cfg=cfg)
    registry.load_all()

    X_ref, y_ref = registry.get_arrays("reference")
    X_ref, _ = filter_and_remap_classes(X_ref, y_ref, shared_classes)

    preprocessor = SpectralPreprocessor.from_config(cfg["preprocessing"])
    preprocessor.fit(X_ref)

    augmentation = AugmentationPipeline.from_config(cfg["augmentation"])
    if len(augmentation.steps) == 0 or augmentation.p == 0:
        augmentation = None

    loader_cfg = {
        "batch_size": cfg.get("training", {}).get("batch_size", 256),
        "num_workers": cfg.get("training", {}).get("num_workers", 4),
        "validation": cfg["validation"],
        "seed": args.seed,
    }

    loaders = build_all_loaders(
        registry,
        preprocessor,
        augmentation,
        loader_cfg,
        shared_classes=shared_classes,
    )

    # -----------------------------
    # MODEL
    # -----------------------------
    print("[XAI] Loading model...")

    model = get_model(args.model, cfg)
    load_best_model(exp_dir, model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # -----------------------------
    # XAI
    # -----------------------------
    print("[XAI] Generating saliency maps...")

    xai_root = Path(exp_dir) / "xai"
    xai_root.mkdir(parents=True, exist_ok=True)

    loader = loaders["ood"]["2018clinical"]

    class_counts = {i: 0 for i in range(n_classes)}

    for x, y in loader:
        for i in range(x.shape[0]):

            label = y[i].item()

            limit = 5 if label == 2 else 2

            if class_counts[label] >= limit:
                continue

            xi = x[i:i+1].to(device)

            saliency = compute_saliency(model, xi)
            smooth_saliency = compute_smoothgrad(model, xi)
            signal = xi[0].mean(dim=0).detach().cpu().numpy()

            save_dir = xai_root / f"class_{label}"
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path_sal = save_dir / f"sample_{class_counts[label]}_saliency.png"
            save_path_sg = save_dir / f"sample_{class_counts[label]}_smoothgrad.png"

            plot_saliency(signal, saliency, save_path_sal)
            plot_saliency(signal, smooth_saliency, save_path_sg)

            class_counts[label] += 1

        if all(class_counts[i] >= (5 if i == 2 else 2) for i in class_counts):
            break

    print(f"[XAI] Saved to {xai_root}")


if __name__ == "__main__":
    main()