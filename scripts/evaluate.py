"""
scripts/evaluate.py

Standalone evaluation: load a trained checkpoint and run the full
evaluation suite including OOD splits and pairwise McNemar's tests.

Usage:
    # Evaluate one model
    python scripts/evaluate.py --exp-dir experiments/cnn_20240101_120000

    # Compare multiple trained models (generates the paper results table)
    python scripts/evaluate.py \\
        --compare \\
        experiments/cnn_20240101_120000 \\
        experiments/resnet1d_20240101_130000 \\
        experiments/transformer_20240101_140000 \\
        experiments/hybrid_20240101_150000
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.utils.checkpoint import load_best_model
from src.data.registry import DataRegistry
from src.data.preprocessing import SpectralPreprocessor
from src.data.augmentation import AugmentationPipeline
from src.data.dataloader import build_all_loaders
from src.models.registry import get_model
from src.evaluation.evaluator import ModelEvaluator, compare_models


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained spectral classifiers")
    p.add_argument("--exp-dir",  default=None, help="Single experiment directory")
    p.add_argument("--compare",  nargs="+", default=None,
                   help="Multiple experiment directories for comparison")
    p.add_argument("--split",    default="test",
                   help="Primary evaluation split (default: test)")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def evaluate_one(exp_dir: str, loaders: dict, n_classes: int) -> dict:
    """Load checkpoint from exp_dir and evaluate on all splits."""
    cfg_path = os.path.join(exp_dir, "config.yaml")
    cfg = load_config(cfg_path)

    model_name = cfg.get("model", {}).get("name", "unknown")
    model = get_model(model_name, dict(cfg))

    print(f"\nLoading best checkpoint from {exp_dir}...")
    checkpoint = load_best_model(exp_dir, model)
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")

    evaluator = ModelEvaluator(
        model=model,
        model_name=model_name,
        n_classes=n_classes,
    )
    results = evaluator.evaluate_all(loaders)
    evaluator.save(os.path.join(exp_dir, "eval_results.json"))
    return results


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load shared data infrastructure from base configs
    cfg = load_config(
        "configs/data/splits.yaml",
        "configs/data/preprocessing.yaml",
        "configs/data/augmentation.yaml",
    )

    registry = DataRegistry(data_root="data/raw", cfg=cfg)
    registry.load_all()

    X_ref, _ = registry.get_arrays("reference")
    preprocessor = SpectralPreprocessor.from_config(cfg["preprocessing"])
    preprocessor.fit(X_ref)

    loaders = build_all_loaders(
        registry, preprocessor,
        AugmentationPipeline.from_config(cfg["augmentation"]),
        {"batch_size": 512, "num_workers": 4, "validation": cfg["validation"]},
    )

    n_classes = cfg["dataset"]["n_classes_full"]

    # ---- Single model evaluation ----
    if args.exp_dir and not args.compare:
        evaluate_one(args.exp_dir, loaders, n_classes)

    # ---- Multi-model comparison ----
    elif args.compare:
        all_results = []
        for exp_dir in args.compare:
            results = evaluate_one(exp_dir, loaders, n_classes)
            all_results.append(results)

        split_names = [args.split] + list(loaders["ood"].keys())
        table = compare_models(
            all_results,
            split_names=split_names,
            save_path="experiments/comparison_table.txt",
        )
        print("\n" + "="*60)
        print("  MODEL COMPARISON TABLE")
        print("="*60)
        print(table)

        # McNemar's tests between all pairs on the test split
        print("\n  McNemar's pairwise significance tests (test split):\n")
        test_key = args.split
        for i in range(len(all_results)):
            for j in range(i + 1, len(all_results)):
                ra, rb = all_results[i], all_results[j]
                na, nb = ra["model"], rb["model"]
                preds_a = ra["splits"].get(test_key, {}).get("predictions", [])
                preds_b = rb["splits"].get(test_key, {}).get("predictions", [])
                tgts    = ra["splits"].get(test_key, {}).get("targets", [])

                if preds_a and preds_b and tgts:
                    stat = ModelEvaluator.mcnemar_test(preds_a, preds_b, tgts)
                    sig  = "***" if stat["significant"] else "n.s."
                    print(f"    {na} vs {nb}: "
                          f"p={stat['p_value']:.4f} {sig}")
    else:
        print("Provide --exp-dir or --compare. See --help.")


if __name__ == "__main__":
    main()