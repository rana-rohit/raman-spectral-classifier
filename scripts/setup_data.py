"""
scripts/setup_data.py

Bootstraps the entire data pipeline:
  1. Loads all splits via DataRegistry
  2. Fits the preprocessor on the reference split ONLY
  3. Verifies transforms are consistent across all splits
  4. Prints a comprehensive summary

Run once before training to verify everything is wired correctly:
    python scripts/setup_data.py

This script does NOT touch the test or clinical splits for training —
it only inspects their shapes and confirms loading works.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import numpy as np

from src.utils.seed import set_seed
from src.data.registry import DataRegistry
from src.data.preprocessing import SpectralPreprocessor
from src.data.augmentation import AugmentationPipeline
from src.data.dataset import make_train_val_split, SpectralDataset
from src.data.dataloader import build_all_loaders


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    set_seed(42)

    print("=" * 60)
    print("  Spectral Classifier — Data Pipeline Bootstrap")
    print("=" * 60)

    # ---- Load configs ----
    splits_cfg  = load_yaml("configs/data/splits.yaml")
    prep_cfg    = load_yaml("configs/data/preprocessing.yaml")
    aug_cfg     = load_yaml("configs/data/augmentation.yaml")

    # ---- Build registry and load all splits ----
    print("\n[1/5] Loading all splits from disk...")
    registry = DataRegistry(data_root="data/raw", cfg=splits_cfg)
    registry.load_all()
    registry.summary()

    # ---- Fit preprocessor on reference ONLY ----
    print("[2/5] Fitting preprocessor on reference split...")
    X_ref, y_ref = registry.get_arrays("reference")
    preprocessor = SpectralPreprocessor.from_config(prep_cfg["preprocessing"])
    X_ref_clean  = preprocessor.fit_transform(X_ref)
    print(f"      {preprocessor}")
    print(f"      Reference: raw  mean={X_ref.mean():.4f}  std={X_ref.std():.4f}")
    print(f"      Reference: proc mean={X_ref_clean.mean():.4f}  std={X_ref_clean.std():.4f}")

    # ---- Verify transforms on other splits ----
    print("\n[3/5] Verifying transforms across all splits...")
    for split_name in registry.available_splits():
        X, y = registry.get_arrays(split_name)
        X_proc = preprocessor.transform(X)
        print(f"      {split_name:>16s}  raw_mean={X.mean():.4f}  "
              f"proc_mean={X_proc.mean():.4f}  proc_std={X_proc.std():.4f}")

    # ---- Build augmentation pipeline ----
    print("\n[4/5] Building augmentation pipeline...")
    augmentation = AugmentationPipeline.from_config(aug_cfg["augmentation"])

    if len(augmentation.steps) == 0 or augmentation.p == 0:
        augmentation = None

    if augmentation is None:
        print("      Steps: []")
        print("      Apply probability: 0.0")
    else:
        print(f"      Steps: {[type(s).__name__ for s in augmentation.steps]}")
        print(f"      Apply probability: {augmentation.p}")

    # ---- Build all DataLoaders and smoke-test ----
    print("\n[5/5] Building DataLoaders and smoke-testing batch shapes...")
    loader_cfg = {
        "batch_size": 256,
        "num_workers": 0,   # 0 for bootstrap (avoids multiprocessing overhead)
        "validation": splits_cfg["validation"],
    }
    loaders = build_all_loaders(registry, preprocessor, augmentation, loader_cfg)

    for name, loader in loaders.items():
        if name == "ood":
            for ood_name, ood_loader in loader.items():
                x_batch, y_batch = next(iter(ood_loader))
                print(f"      OOD {ood_name:>14s}: x={tuple(x_batch.shape)}  "
                      f"y={tuple(y_batch.shape)}  "
                      f"classes={sorted(y_batch.unique().tolist())}")
        else:
            x_batch, y_batch = next(iter(loader))
            print(f"      {name:>20s}: x={tuple(x_batch.shape)}  "
                  f"y={tuple(y_batch.shape)}")

    print("\n[OK] Data pipeline verified. Ready to train.\n")


if __name__ == "__main__":
    main()