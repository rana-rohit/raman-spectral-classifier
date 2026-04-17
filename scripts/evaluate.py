"""
scripts/evaluate.py

Standalone evaluation for trained experiments.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.augmentation import AugmentationPipeline
from src.data.dataloader import build_all_loaders
from src.data.preprocessing import SpectralPreprocessor
from src.data.registry import DataRegistry
from src.evaluation.evaluator import ModelEvaluator, compare_models
from src.models.registry import get_model
from src.utils.checkpoint import load_best_model
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained spectral classifiers")
    p.add_argument("--exp-dir", default=None)
    p.add_argument("--compare", nargs="+", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_eval_loaders(cfg: dict, seed: int) -> tuple[dict, int]:
    registry = DataRegistry(data_root="data/raw", cfg=cfg)
    registry.load_all()

    X_ref, _ = registry.get_arrays("reference")
    preprocessor = SpectralPreprocessor.from_config(cfg["preprocessing"])
    preprocessor.fit(X_ref)

    augmentation = AugmentationPipeline.from_config(cfg["augmentation"])
    if len(augmentation.steps) == 0 or augmentation.p == 0:
        augmentation = None

    loaders = build_all_loaders(
        registry,
        preprocessor,
        augmentation,
        {
            "batch_size": cfg.get("training", {}).get("batch_size", 512),
            "num_workers": cfg.get("training", {}).get("num_workers", 4),
            "validation": cfg["validation"],
            "seed": seed,
            "consistency": cfg.get("training", {}).get("consistency", {}),
        },
    )
    return loaders, cfg["dataset"]["n_classes_full"]


def evaluate_one(exp_dir: str, seed: int) -> dict:
    cfg_path = os.path.join(exp_dir, "config.yaml")
    cfg = load_config(cfg_path)
    loaders, n_classes = build_eval_loaders(cfg, seed)

    model_name = cfg.get("model", {}).get("name", "unknown")
    model = get_model(model_name, dict(cfg))

    print(f"\nLoading best checkpoint from {exp_dir}...")
    checkpoint = load_best_model(exp_dir, model)
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")

    evaluator = ModelEvaluator(
        model=model,
        model_name=model_name,
        n_classes=n_classes,
        device=str(next(model.parameters()).device),
        cfg=dict(cfg),
    )
    results = evaluator.evaluate_all(loaders)
    evaluator.save(os.path.join(exp_dir, "eval_results.json"))
    return results


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.exp_dir and not args.compare:
        evaluate_one(args.exp_dir, args.seed)
    elif args.compare:
        all_results = [evaluate_one(exp_dir, args.seed) for exp_dir in args.compare]
        split_names = [args.split] + [
            name for name in all_results[0].get("splits", {}).keys()
            if name != args.split
        ]
        table = compare_models(
            all_results,
            split_names=split_names,
            save_path="experiments/comparison_table.txt",
        )
        print("\n" + "=" * 60)
        print("  MODEL COMPARISON TABLE")
        print("=" * 60)
        print(table)

        print("\n  McNemar's pairwise significance tests (test split):\n")
        test_key = args.split
        for i in range(len(all_results)):
            for j in range(i + 1, len(all_results)):
                ra, rb = all_results[i], all_results[j]
                na, nb = ra["model"], rb["model"]
                preds_a = ra["splits"].get(test_key, {}).get("predictions", [])
                preds_b = rb["splits"].get(test_key, {}).get("predictions", [])
                tgts = ra["splits"].get(test_key, {}).get("targets", [])
                if preds_a and preds_b and tgts:
                    stat = ModelEvaluator.mcnemar_test(preds_a, preds_b, tgts)
                    sig = "***" if stat["significant"] else "n.s."
                    print(f"    {na} vs {nb}: p={stat['p_value']:.4f} {sig}")
    else:
        print("Provide --exp-dir or --compare. See --help.")


if __name__ == "__main__":
    main()
