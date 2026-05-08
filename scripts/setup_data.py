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

from src.utils.seed import set_seed
from src.data.registry import DataRegistry
from src.data.preprocessing import SpectralPreprocessor
from src.data.augmentation import AugmentationPipeline
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
    splits_cfg = load_yaml("configs/data/splits.yaml")
    prep_cfg = load_yaml("configs/data/preprocessing.yaml")
    aug_cfg = load_yaml("configs/data/augmentation.yaml")

    # ---- Build registry and load all splits ----
    print("\n[1/5] Loading all splits from disk...")

    registry = DataRegistry(
        data_root="data/raw",
        cfg=splits_cfg,
    )

    registry.load_all()
    registry.summary()

    # ---- Task semantics ----
    task_cfg = splits_cfg["task"]

    stage = task_cfg["stage"]
    label_space = task_cfg["label_space"]

    clinical_sparse_ids = task_cfg.get(
        "clinical_sparse_global_ids",
        None,
    )

    if stage == "pretrain_30class":

        n_classes = 30

    elif stage == "pretrain_treatment_8class":

        n_classes = 8

    elif stage == "transfer_5class":

        n_classes = len(clinical_sparse_ids)

    else:

        raise ValueError(
            f"Unknown setup_data stage: {stage}"
        )

    # --------------------------------------------------------
    # Semantic-space integrity checks
    # --------------------------------------------------------

    if stage == "pretrain_30class":

        assert label_space == "isolate_space"

    elif stage == "pretrain_treatment_8class":

        assert label_space == "global_treatment_space"

    elif stage == "transfer_5class":

        assert (
            label_space
            == "sparse_global_treatment_space"
        )

        assert clinical_sparse_ids == [0, 2, 3, 5, 6], (
            "transfer_5class must use verified sparse "
            "global treatment IDs"
        )

    print("\n  Semantic configuration")
    print(f"      stage          : {stage}")
    print(f"      label_space    : {label_space}")
    print(f"      n_classes      : {n_classes}")

    if clinical_sparse_ids is not None:
        print(f"      sparse IDs     : {clinical_sparse_ids}")

    # ---- Fit preprocessor on reference ONLY ----
    print("\n[2/5] Fitting preprocessor on reference split...")

    X_ref, _ = registry.get_arrays("reference")

    # Always fit preprocessor on FULL reference set
    preprocessor = SpectralPreprocessor.from_config(
        prep_cfg["preprocessing"]
    )

    X_ref_clean = preprocessor.fit_transform(X_ref)

    print(f"      {preprocessor}")
    print(
        f"      Reference: raw  mean={X_ref.mean():.4f}  "
        f"std={X_ref.std():.4f}"
    )

    print(
        f"      Reference: proc mean={X_ref_clean.mean():.4f}  "
        f"std={X_ref_clean.std():.4f}"
    )

    # ---- Verify transforms on other splits ----
    print("\n[3/5] Verifying transforms across all splits...")

    for split_name in registry.available_splits():

        if split_name.lower() == "test":
            print(f"      {split_name:>16s}  (skipped HOLDOUT)")
            continue

        X, y = registry.get_arrays(split_name)

        X_proc = preprocessor.transform(X)

        print(
            f"      {split_name:>16s}  "
            f"raw_mean={X.mean():.4f}  "
            f"proc_mean={X_proc.mean():.4f}  "
            f"proc_std={X_proc.std():.4f}"
        )

    # ---- Build augmentation pipeline ----
    print("\n[4/5] Building augmentation pipeline...")

    augmentation = AugmentationPipeline.from_config(
        aug_cfg["augmentation"]
    )

    if len(augmentation.steps) == 0 or augmentation.p == 0:
        augmentation = None

    if augmentation is None:

        print("      Steps: []")
        print("      Apply probability: 0.0")

    else:

        print(
            f"      Steps: "
            f"{[type(s).__name__ for s in augmentation.steps]}"
        )

        print(f"      Apply probability: {augmentation.p}")

    # ---- Build all DataLoaders and smoke-test ----
    print(
        "\n[5/5] Building DataLoaders and "
        "smoke-testing batch shapes..."
    )

    loader_cfg = {
        "batch_size": 256,
        "num_workers": 0,
        "validation": splits_cfg["validation"],
    }

    loaders = build_all_loaders(
        registry,
        preprocessor,
        augmentation,
        loader_cfg,
        clinical_sparse_ids=clinical_sparse_ids,
        n_classes=n_classes,
    )

    for name, loader in loaders.items():

        if name == "ood":

            for ood_name, ood_loader in loader.items():

                x_batch, y_batch = next(iter(ood_loader))

                classes = sorted(
                    set(ood_loader.dataset.y.tolist())
                )

                print(
                    f"      OOD {ood_name:>14s}: "
                    f"x={tuple(x_batch.shape)}  "
                    f"y={tuple(y_batch.shape)}  "
                    f"classes={classes}"
                )

                actual_classes = set(classes)

                if stage == "transfer_5class":

                    expected_classes = set(range(n_classes))

                    assert hasattr(
                        ood_loader.dataset,
                        "inverse_class_map",
                    ), (
                        f"{ood_name} missing inverse_class_map"
                    )

                elif stage == "pretrain_treatment_8class":

                    expected_classes = set(range(8))

                else:

                    expected_classes = set(range(30))

                if not actual_classes.issubset(expected_classes):

                    raise ValueError(
                        f"OOD split {ood_name} "
                        f"has invalid classes: {classes}"
                    )

                if len(actual_classes) < len(expected_classes):

                    print(
                        f"[WARN] OOD split {ood_name} "
                        f"missing classes: {classes}"
                    )

        else:

            x_batch, y_batch = next(iter(loader))

            classes = sorted(
                set(loader.dataset.y.tolist())
            )

            print(
                f"      {name:>20s}: "
                f"x={tuple(x_batch.shape)}  "
                f"y={tuple(y_batch.shape)}  "
                f"classes={classes}"
            )

            actual_classes = set(classes)

            if stage == "transfer_5class":

                expected_classes = set(range(n_classes))

            elif stage == "pretrain_treatment_8class":

                expected_classes = set(range(8))

            else:

                expected_classes = set(range(30))

            if not actual_classes.issubset(expected_classes):

                raise ValueError(
                    f"Loader {name} "
                    f"has invalid classes: {classes}"
                )

            if len(actual_classes) < len(expected_classes):

                print(
                    f"[WARN] Loader {name} "
                    f"missing some classes: {classes}"
                )

    print("\n[OK] Data pipeline verified. Ready to train.\n")


if __name__ == "__main__":
    main()