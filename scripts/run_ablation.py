"""
scripts/run_ablation.py

Runs systematic ablation experiments for the research paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.augmentation import AugmentationPipeline
from src.data.dataloader import build_all_loaders
from src.data.preprocessing import SpectralPreprocessor
from src.data.registry import DataRegistry
from src.evaluation.evaluator import ModelEvaluator, compare_models
from src.models.registry import get_model
from src.training.finetuner import finetune
from src.training.trainer import build_trainer
from src.utils.checkpoint import load_best_model
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Run ablation experiments")

    p.add_argument(
        "--ablation",
        required=True,
        choices=[
            "patch_size",
            "augmentation",
            "few_shot",
            "all",
        ],
    )

    p.add_argument("--pretrained", default=None)
    p.add_argument("--exp-dir", default="experiments/ablations")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage", required=True, choices=["s1_isolate", "s2_treatment", "s3_transfer"])

    return p.parse_args()


def _build_loaders_for_cfg(cfg: dict, seed: int) -> dict:

    registry = DataRegistry(data_root="data/raw", cfg=cfg)
    registry.load_all()

    task_cfg = cfg["task"]

    stage = task_cfg["stage"]
    label_space = task_cfg["label_space"]

    # --------------------------------------------------------
    # Stage-aware semantic-space handling
    # --------------------------------------------------------

    if stage == "pretrain_30class":
        clinical_sparse_ids = None
        n_classes = 30

    elif stage == "pretrain_treatment_8class":
        clinical_sparse_ids = None
        n_classes = 8

    elif stage == "transfer_5class":
        clinical_sparse_ids = task_cfg.get("clinical_sparse_global_ids", None)
        n_classes = len(clinical_sparse_ids)

    else:
        raise ValueError(f"Unknown ablation stage: {stage}")

    cfg = dict(cfg)
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["n_classes"] = n_classes

    # --------------------------------------------------------
    # Semantic-space integrity checks
    # --------------------------------------------------------

    if stage == "pretrain_30class":

        assert label_space == "isolate_space"
        assert cfg["model"]["semantic_space"] == "isolate_space"
        assert n_classes == 30

    elif stage == "pretrain_treatment_8class":

        assert label_space == "global_treatment_space"
        assert cfg["model"]["semantic_space"] == "global_treatment_space"
        assert n_classes == 8

    elif stage == "transfer_5class":

        assert label_space == "sparse_global_treatment_space"
        assert cfg["model"]["semantic_space"] == "compact_transfer_space"
        assert n_classes == 5

        assert clinical_sparse_ids == [0, 2, 3, 5, 6], (
            "transfer_5class must use verified sparse global treatment IDs"
        )

    # --------------------------------------------------------
    # Fit preprocessor on FULL reference set
    # --------------------------------------------------------

    X_ref, y_ref = registry.get_arrays("reference")

    preprocessor = SpectralPreprocessor.from_config(cfg["preprocessing"])
    preprocessor.fit(X_ref)

    augmentation = AugmentationPipeline.from_config(cfg["augmentation"])

    if len(augmentation.steps) == 0 or augmentation.p == 0:
        augmentation = None

    cfg = dict(cfg)
    cfg["seed"] = seed
    return build_all_loaders(
        registry,
        preprocessor,
        augmentation,
        cfg,
        clinical_sparse_ids=clinical_sparse_ids,
        n_classes=n_classes,
    )


def run_single_experiment(
    cfg: dict,
    exp_dir: str,
    n_classes: int,
    seed: int,
) -> dict:

    model_name = cfg["model"]["name"]

    loaders = _build_loaders_for_cfg(cfg, seed)

    model = get_model(model_name, cfg)

    trainer = build_trainer(
        model,
        loaders,
        cfg,
        exp_dir,
        n_classes,
    )

    trainer.fit()

    checkpoint = load_best_model(exp_dir, model)

    checkpoint_cfg = checkpoint.get("config", {})
    checkpoint_stage = checkpoint_cfg.get("task", {}).get("stage", None)
    checkpoint_label_space = checkpoint_cfg.get("task", {}).get(
        "label_space",
        None,
    )

    checkpoint_model_space = checkpoint_cfg.get("model", {}).get(
        "semantic_space",
        None,
    )

    assert checkpoint_stage == cfg["task"]["stage"], (
        "Checkpoint stage mismatch:\n"
        f"Expected: {cfg['task']['stage']}\n"
        f"Found: {checkpoint_stage}"
    )

    assert checkpoint_label_space == cfg["task"]["label_space"], (
        "Checkpoint label-space mismatch:\n"
        f"Expected: {cfg['task']['label_space']}\n"
        f"Found: {checkpoint_label_space}"
    )

    assert checkpoint_model_space == cfg["model"]["semantic_space"], (
        "Checkpoint model semantic-space mismatch:\n"
        f"Expected: {cfg['model']['semantic_space']}\n"
        f"Found: {checkpoint_model_space}"
    )

    evaluator = ModelEvaluator(
        model=model,
        model_name=model_name,
        n_classes=n_classes,
        device=str(next(model.parameters()).device),
        cfg=cfg,
    )

    results = evaluator.evaluate_all(loaders)

    results["stage"] = cfg["task"]["stage"]

    stage = cfg["task"]["stage"]

    evaluator.save(os.path.join(exp_dir, f"{stage}_results.json"))

    return results





def ablation_patch_size(
    base_cfg,
    exp_root,
    n_classes,
    seed,
):

    print("\n" + "=" * 60)
    print("  ABLATION: Transformer patch_size in {10, 20, 25, 50}")
    print("=" * 60)

    all_results = []

    for patch_size in [10, 20, 25, 50]:

        import copy

        cfg = copy.deepcopy(base_cfg)

        cfg["model"]["name"] = "transformer"
        cfg["model"]["patch_size"] = patch_size

        exp_dir = os.path.join(exp_root, f"transformer_patch{patch_size}")

        print(f"\n  Running: patch_size={patch_size}")

        results = run_single_experiment(
            cfg,
            exp_dir,
            n_classes,
            seed,
        )

        results["model"] = f"patch_size={patch_size}"

        all_results.append(results)

    _print_ablation_summary(all_results, exp_root, "patch_size")

    return all_results


def ablation_few_shot(
    base_cfg,
    loaders,
    exp_root,
    n_classes,
    pretrained_dir,
):

    assert base_cfg["task"]["stage"] == "transfer_5class", (
        "Few-shot ablation only valid for transfer_5class"
    )

    print("\n" + "=" * 60)
    print("  ABLATION: Few-shot learning curve (10, 25, 50, 100 shots/class)")
    print("=" * 60)

    shot_results = {}

    for n_shots in [10, 25, 50, 100]:

        print(f"\n  Running: {n_shots} shots/class")

        import copy

        cfg = copy.deepcopy(base_cfg)

        model_name = cfg["model"]["name"]
        model = get_model(model_name, cfg)

        exp_dir = os.path.join(exp_root, f"few_shot_{n_shots}")

        results = finetune(
            model=model,
            pretrained_exp_dir=pretrained_dir,
            loaders=loaders,
            cfg=cfg,
            exp_dir=exp_dir,
            n_shots_per_class=n_shots,
            n_classes=n_classes,
        )

        shot_results[n_shots] = results

    print("\n  Few-shot learning curve:")

    print(
        f"  {'Shots/class':<14} "
        f"{'Val acc':<12} "
        f"{'Test acc':<12} "
        f"{'2018clin':<12} "
        f"{'2019clin':<12}"
    )

    print("  " + "-" * 60)

    for n_shots, res in shot_results.items():

        val = res.get("val", {}).get("accuracy", float("nan"))
        test = res.get("test", {}).get("accuracy", float("nan"))

        ood_accs = [
            v.get("accuracy", float("nan"))
            for v in res.get("ood", {}).values()
        ]

        ood_str = "  ".join(f"{a:.4f}" for a in ood_accs)

        print(
            f"  {n_shots:<14} "
            f"{val:.4f}       "
            f"{test:.4f}       "
            f"{ood_str}"
        )

    out_path = os.path.join(exp_root, "few_shot_curve.json")

    with open(out_path, "w") as f:

        json.dump(
            {
                k: {
                    sk: sv
                    for sk, sv in v.items()
                    if sk != "best_metrics"
                }
                for k, v in shot_results.items()
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n  Saved to {out_path}")

    return shot_results


def ablation_augmentation(
    base_cfg,
    exp_root,
    n_classes,
    seed,
):

    print("\n" + "=" * 60)
    print("  ABLATION: Remove one augmentation type at a time")
    print("=" * 60)

    aug_types = [
        "gaussian_noise",
        "baseline_shift",
        "multiplicative_intensity",
        "baseline_drift",
        "peak_broadening",
        "nonlinear_warp",
    ]

    all_results = []

    for disabled in ["none"] + aug_types:

        import copy

        cfg = copy.deepcopy(base_cfg)

        cfg["model"]["name"] = "cnn"

        label = "all_aug" if disabled == "none" else f"no_{disabled}"

        exp_dir = os.path.join(exp_root, f"augmentation_{label}")

        if disabled != "none":
            cfg["augmentation"]["steps"][disabled]["enabled"] = False

        print(f"\n  Running: {label}")

        results = run_single_experiment(
            cfg,
            exp_dir,
            n_classes,
            seed,
        )

        results["model"] = label

        all_results.append(results)

    _print_ablation_summary(all_results, exp_root, "augmentation")

    return all_results


def _print_ablation_summary(
    all_results,
    exp_root,
    ablation_name,
):

    stages = {r.get("stage", None) for r in all_results}

    assert len(stages) == 1, (
        "Cannot compare models across different stages"
    )

    split_names = list(all_results[0].get("splits", {}).keys())

    table = compare_models(
        all_results,
        split_names=split_names,
        save_path=os.path.join(exp_root, f"{ablation_name}_table.txt"),
    )

    print(f"\n  {ablation_name.upper()} RESULTS:")
    print(table)


def main():

    args = parse_args()

    set_seed(args.seed)

    cfg = load_config(
        "configs/data/splits.yaml",
        "configs/data/preprocessing.yaml",
        "configs/data/augmentation.yaml",
        "configs/training/base.yaml",
        f"configs/stages/{args.stage}.yaml",
        "configs/model/resnet1d.yaml",
    )

    task_cfg = cfg["task"]
    stage = task_cfg["stage"]
    label_space = task_cfg["label_space"]
    semantic_space = cfg["model"]["semantic_space"]

    if stage == "pretrain_30class":
        assert label_space == "isolate_space"
        assert semantic_space == "isolate_space"

    elif stage == "pretrain_treatment_8class":
        assert label_space == "global_treatment_space"
        assert semantic_space == "global_treatment_space"

    elif stage == "transfer_5class":
        assert label_space == "sparse_global_treatment_space"
        assert semantic_space == "compact_transfer_space"

    if stage == "pretrain_30class":
        n_classes = 30

    elif stage == "pretrain_treatment_8class":
        n_classes = 8

    elif stage == "transfer_5class":
        n_classes = 5

    else:
        raise ValueError(f"Unknown ablation stage: {stage}")

    exp_root = args.exp_dir

    os.makedirs(exp_root, exist_ok=True)



    if args.ablation in {"patch_size", "all"}:
        ablation_patch_size(
            dict(cfg),
            exp_root,
            n_classes,
            args.seed,
        )

    if args.ablation in {"augmentation", "all"}:
        ablation_augmentation(
            dict(cfg),
            exp_root,
            n_classes,
            args.seed,
        )

    if args.ablation in {"few_shot", "all"}:

        if not args.pretrained:
            print("--pretrained is required for few_shot ablation.")
            sys.exit(1)

        loaders = _build_loaders_for_cfg(dict(cfg), args.seed)

        ablation_few_shot(
            dict(cfg),
            loaders,
            exp_root,
            n_classes,
            args.pretrained,
        )


if __name__ == "__main__":
    main()
