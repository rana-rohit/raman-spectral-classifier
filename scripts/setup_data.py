"""
scripts/setup_data.py

Bootstraps the entire data pipeline:
  1. Loads all splits via DataRegistry
  2. Fits the preprocessor on the reference split ONLY
  3. Verifies transforms are consistent across all splits
  4. Prints a comprehensive summary

Run once before training to verify everything is wired correctly:
    python scripts/setup_data.py

This script does NOT touch the test or clinical splits for training -
it only inspects their shapes and confirms loading works.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse

from src.utils.seed import set_seed
from src.data.registry import DataRegistry
from src.data.preprocessing import SpectralPreprocessor
from src.data.augmentation import AugmentationPipeline
from src.data.dataloader import build_all_loaders
from src.utils.config import apply_overrides, load_config
from src.utils.split_modes import (
    IID_REFERENCE,
    canonicalize_split_mode_config,
    resolve_split_mode,
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "s1_isolate",
            "s2_treatment",
            "s3_transfer",
        ],
    )
    parser.add_argument(
        "--split-mode",
        choices=["holdout", "iid_reference"],
        default=None,
        help="Evaluation protocol to activate for this runtime check.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Dotted key=value config overrides, matching scripts/train.py.",
    )

    args, dotlist_overrides = parser.parse_known_args()
    args.override = list(args.override) + dotlist_overrides
    return args


def _active_registry_splits(cfg: dict, registry: DataRegistry) -> list[str]:
    stage = cfg.get("task", {}).get("stage")
    split_mode = resolve_split_mode(cfg)
    active = ["reference"]
    if split_mode != IID_REFERENCE:
        active.append("test")

    include_finetune = (
        cfg.get("training", {})
        .get("finetune", {})
        .get("enabled", False)
        or cfg.get("data", {}).get("include_finetune_split", False)
    )
    if include_finetune:
        active.append("finetune")

    if stage == "transfer_5class":
        active.extend(registry.ood_split_names())
    elif (
        stage == "pretrain_treatment_8class"
        and cfg.get("evaluation", {}).get("clinical_ood", {}).get("enabled", False)
    ):
        active.extend(registry.ood_split_names())

    return [
        split_name
        for split_name in active
        if split_name in registry.available_splits()
    ]


def main():
    set_seed(42)
    args = parse_args()
    print("=" * 60)
    print("  Spectral Classifier - Data Pipeline Bootstrap")
    print("=" * 60)

    cfg = load_config(
        "configs/data/splits.yaml",
        "configs/data/preprocessing.yaml",
        "configs/data/augmentation.yaml",
        "configs/training/base.yaml",
        f"configs/stages/{args.stage}.yaml",
    )
    cfg = apply_overrides(dict(cfg), args.override)
    split_mode = canonicalize_split_mode_config(
        cfg,
        split_mode=args.split_mode,
    )

    task_cfg = cfg["task"]

    stage = task_cfg["stage"]
    label_space = task_cfg["label_space"]

    clinical_sparse_ids = task_cfg.get(
        "clinical_sparse_global_ids",
        [],
    )

    print("\n[1/5] Loading semantically active splits from disk...")
    print(f"      Split Mode: {split_mode}")

    registry = DataRegistry(
        data_root="data/raw",
        cfg=cfg,
    )

    active_splits = _active_registry_splits(cfg, registry)
    for split_name in active_splits:
        registry.load(split_name)
    registry.summary(active_splits)

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

        assert len(clinical_sparse_ids) == 5, (
            "transfer_5class must use verified sparse "
            "global treatment IDs"
        )

    print("\n  Semantic configuration")
    print(f"      stage          : {stage}")
    print(f"      label_space    : {label_space}")
    print(f"      n_classes      : {n_classes}")

    if clinical_sparse_ids is not None:
        print(f"      sparse IDs     : {clinical_sparse_ids}")

    print("\n[2/5] Fitting preprocessor on reference split...")

    X_ref, _ = registry.get_arrays("reference")

    preprocessor = SpectralPreprocessor.from_config(
        cfg["preprocessing"]
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

    print("\n[3/5] Verifying transforms across all splits...")

    for split_name in active_splits:

        if split_name.lower() == "test":
            print(f"      {split_name:>16s}  (skipped HOLDOUT)")
            continue

        X, _ = registry.get_arrays(split_name)

        X_proc = preprocessor.transform(X)

        print(
            f"      {split_name:>16s}  "
            f"raw_mean={X.mean():.4f}  "
            f"proc_mean={X_proc.mean():.4f}  "
            f"proc_std={X_proc.std():.4f}"
        )

    print("\n[4/5] Building augmentation pipeline...")

    augmentation = AugmentationPipeline.from_config(
        cfg["augmentation"]
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

    print(
        "\n[5/5] Building DataLoaders and "
        "smoke-testing batch shapes..."
    )

    cfg["batch_size"] = 256
    cfg["num_workers"] = 0

    loaders = build_all_loaders(
        registry,
        preprocessor,
        augmentation,
        cfg,
        clinical_sparse_ids=clinical_sparse_ids,
        n_classes=n_classes,
    )

    from src.utils.logging import print_split_provenance
    print_split_provenance(loaders, cfg, context="setup")

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
