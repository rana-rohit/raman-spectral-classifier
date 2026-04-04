"""
scripts/run_ablation.py

Runs systematic ablation experiments for the research paper.

Ablations implemented:
  1. hybrid_handoff  — CNN blocks before Transformer (1, 2, 3)
  2. patch_size      — Transformer patch size (10, 20, 25, 50)
  3. augmentation    — Remove one augmentation type at a time
  4. few_shot        — Fine-tuning learning curve (10, 25, 50, 100 shots)

Usage:
    python scripts/run_ablation.py --ablation hybrid_handoff
    python scripts/run_ablation.py --ablation few_shot --pretrained experiments/hybrid_best
    python scripts/run_ablation.py --ablation all
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.seed import set_seed
from src.utils.config import load_config, save_config
from src.data.registry import DataRegistry
from src.data.preprocessing import SpectralPreprocessor
from src.data.augmentation import AugmentationPipeline
from src.data.dataloader import build_all_loaders
from src.models.registry import get_model
from src.training.trainer import build_trainer
from src.training.finetuner import finetune
from src.evaluation.evaluator import ModelEvaluator, compare_models


def parse_args():
    p = argparse.ArgumentParser(description="Run ablation experiments")
    p.add_argument("--ablation",   required=True,
                   choices=["hybrid_handoff", "patch_size", "augmentation", "few_shot", "all"])
    p.add_argument("--pretrained", default=None,
                   help="Pretrained experiment dir (required for few_shot ablation)")
    p.add_argument("--exp-dir",    default="experiments/ablations")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def run_single_experiment(cfg: dict, exp_dir: str, loaders: dict, n_classes: int) -> dict:
    """Train one model with given config, return evaluation results."""
    model_name = cfg["model"]["name"]
    model = get_model(model_name, cfg)
    trainer = build_trainer(model, loaders, cfg, exp_dir, n_classes)
    trainer.fit()
    evaluator = ModelEvaluator(model, model_name, n_classes)
    results = evaluator.evaluate_all(loaders)
    evaluator.save(os.path.join(exp_dir, "results.json"))
    return results


def ablation_hybrid_handoff(base_cfg, loaders, exp_root, n_classes):
    """Ablate: how many CNN blocks before the Transformer?"""
    print("\n" + "="*60)
    print("  ABLATION: Hybrid handoff_blocks ∈ {1, 2, 3}")
    print("="*60)

    all_results = []
    for n_blocks in [1, 2, 3]:
        import copy
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["name"] = "hybrid"
        cfg["model"]["handoff_blocks"] = n_blocks
        exp_dir = os.path.join(exp_root, f"hybrid_handoff_{n_blocks}")
        print(f"\n  Running: handoff_blocks={n_blocks}")
        results = run_single_experiment(cfg, exp_dir, loaders, n_classes)
        results["model"] = f"hybrid_handoff={n_blocks}"
        all_results.append(results)

    _print_ablation_summary(all_results, loaders, exp_root, "hybrid_handoff")
    return all_results


def ablation_patch_size(base_cfg, loaders, exp_root, n_classes):
    """Ablate: Transformer patch size."""
    print("\n" + "="*60)
    print("  ABLATION: Transformer patch_size ∈ {10, 20, 25, 50}")
    print("="*60)

    all_results = []
    for patch_size in [10, 20, 25, 50]:
        import copy
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["name"] = "transformer"
        cfg["model"]["patch_size"] = patch_size
        exp_dir = os.path.join(exp_root, f"transformer_patch{patch_size}")
        print(f"\n  Running: patch_size={patch_size}")
        results = run_single_experiment(cfg, exp_dir, loaders, n_classes)
        results["model"] = f"patch_size={patch_size}"
        all_results.append(results)

    _print_ablation_summary(all_results, loaders, exp_root, "patch_size")
    return all_results


def ablation_few_shot(base_cfg, loaders, exp_root, n_classes, pretrained_dir):
    """Ablate: fine-tuning learning curve with varying shots per class."""
    print("\n" + "="*60)
    print("  ABLATION: Few-shot learning curve (10, 25, 50, 100 shots/class)")
    print("="*60)

    from src.utils.checkpoint import load_best_model

    shot_results = {}
    for n_shots in [10, 25, 50, 100]:
        print(f"\n  Running: {n_shots} shots/class")
        import copy
        cfg = copy.deepcopy(base_cfg)

        # Re-load pretrained model fresh for each run
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

    # Print learning curve summary
    print("\n  Few-shot learning curve:")
    print(f"  {'Shots/class':<14} {'Val acc':<12} {'Test acc':<12} "
          f"{'2018clin':<12} {'2019clin':<12}")
    print("  " + "─" * 60)
    for n_shots, res in shot_results.items():
        val  = res.get("val",  {}).get("accuracy", float("nan"))
        test = res.get("test", {}).get("accuracy", float("nan"))
        ood_accs = [v.get("accuracy", float("nan"))
                    for v in res.get("ood", {}).values()]
        ood_str = "  ".join(f"{a:.4f}" for a in ood_accs)
        print(f"  {n_shots:<14} {val:.4f}       {test:.4f}       {ood_str}")

    # Save
    out_path = os.path.join(exp_root, "few_shot_curve.json")
    with open(out_path, "w") as f:
        json.dump({k: {sk: sv for sk, sv in v.items()
                       if sk != "best_metrics"}
                   for k, v in shot_results.items()}, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")
    return shot_results


def ablation_augmentation(base_cfg, loaders, exp_root, n_classes):
    """Ablate: disable one augmentation type at a time."""
    print("\n" + "="*60)
    print("  ABLATION: Remove one augmentation type at a time")
    print("="*60)

    aug_types = ["gaussian_noise", "baseline_shift", "amplitude_scale", "spectral_shift"]
    all_results = []

    # Baseline: all augmentations enabled (should already exist from main training)
    for disabled in ["none"] + aug_types:
        import copy
        cfg = copy.deepcopy(base_cfg)
        cfg["model"]["name"] = "cnn"

        label = "all_aug" if disabled == "none" else f"no_{disabled}"
        exp_dir = os.path.join(exp_root, f"augmentation_{label}")

        if disabled != "none":
            cfg["augmentation"]["steps"][disabled]["enabled"] = False

        print(f"\n  Running: {label}")
        results = run_single_experiment(cfg, exp_dir, loaders, n_classes)
        results["model"] = label
        all_results.append(results)

    _print_ablation_summary(all_results, loaders, exp_root, "augmentation")
    return all_results


def _print_ablation_summary(all_results, loaders, exp_root, ablation_name):
    split_names = ["test"] + list(loaders.get("ood", {}).keys())
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
        "configs/model/hybrid.yaml",   # Default; overridden per ablation
    )

    registry = DataRegistry(data_root="data/raw", cfg=cfg)
    registry.load_all()
    X_ref, _ = registry.get_arrays("reference")
    preprocessor = SpectralPreprocessor.from_config(cfg["preprocessing"])
    preprocessor.fit(X_ref)
    augmentation = AugmentationPipeline.from_config(cfg["augmentation"])
    loaders = build_all_loaders(
        registry, preprocessor, augmentation,
        {"batch_size": 256, "num_workers": 4, "validation": cfg["validation"]},
    )
    n_classes = cfg["dataset"]["n_classes_full"]

    ablation = args.ablation
    exp_root  = args.exp_dir
    os.makedirs(exp_root, exist_ok=True)

    if ablation == "hybrid_handoff" or ablation == "all":
        ablation_hybrid_handoff(dict(cfg), loaders, exp_root, n_classes)

    if ablation == "patch_size" or ablation == "all":
        ablation_patch_size(dict(cfg), loaders, exp_root, n_classes)

    if ablation == "augmentation" or ablation == "all":
        ablation_augmentation(dict(cfg), loaders, exp_root, n_classes)

    if ablation == "few_shot" or ablation == "all":
        if not args.pretrained:
            print("--pretrained is required for few_shot ablation.")
            sys.exit(1)
        ablation_few_shot(dict(cfg), loaders, exp_root, n_classes, args.pretrained)


if __name__ == "__main__":
    main()