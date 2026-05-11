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

    task_cfg = cfg["task"]
    clinical_sparse_ids = task_cfg.get(
        "clinical_sparse_global_ids",
        [],
    )
    stage = task_cfg["stage"]
    label_space = task_cfg["label_space"]
    
    if stage == "pretrain_30class":
        clinical_sparse_ids = []
        n_classes = cfg["dataset"]["n_classes_full"]

    elif stage == "pretrain_treatment_8class":
        clinical_sparse_ids = []
        n_classes = 8

    elif stage == "transfer_5class":
        clinical_sparse_ids = task_cfg.get(
            "clinical_sparse_global_ids",
            None,
        )
        n_classes = len(clinical_sparse_ids)

    else:
        raise ValueError(
            f"Unknown evaluation stage: {stage}"
        )
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
        assert len(clinical_sparse_ids) == 5, (
            "transfer_5class requires "
            "5 sparse clinical treatment IDs"
        )

    print("\n==================================================")
    print(" EVALUATION TASK")
    print("==================================================")
    print(f"Task Name    : {task_cfg['name']}")
    print(f"Stage        : {stage}")
    print(f"Label Space  : {label_space}")
    print(f"Num Classes  : {n_classes}")

    if len(clinical_sparse_ids) > 0:
        print(f"Clinical IDs : {clinical_sparse_ids}")

    print("==================================================")
    
    # Fit preprocessor on FULL reference set (matches pretrained backbone)
    X_ref, _ = registry.get_arrays("reference")
    preprocessor = SpectralPreprocessor.from_config(cfg["preprocessing"])
    preprocessor.fit(X_ref)

    augmentation = AugmentationPipeline.from_config(cfg["augmentation"])
    if len(augmentation.steps) == 0 or augmentation.p == 0:
        augmentation = None
    
    cfg["batch_size"] = (
        cfg.get("training", {})
        .get("batch_size", 512)
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
    return loaders, n_classes


def evaluate_one(exp_dir: str, seed: int) -> dict:
    cfg_path = os.path.join(exp_dir, "config.yaml")
    cfg = load_config(cfg_path)
    task_cfg = cfg["task"]
    stage = task_cfg["stage"]
    label_space = task_cfg["label_space"]
    loaders, n_classes = build_eval_loaders(cfg, seed)

    model_name = cfg.get("model", {}).get("name", "unknown")
    model = get_model(model_name, dict(cfg))

    print(f"\nLoading best checkpoint from {exp_dir}...")
    checkpoint = load_best_model(exp_dir, model)
    checkpoint_cfg = checkpoint.get("config", {})

    checkpoint_stage = (
        checkpoint_cfg
        .get("task", {})
        .get("stage", None)
    )

    assert checkpoint_stage == stage, (
        "Checkpoint stage mismatch:\n"
        f"Expected: {stage}\n"
        f"Found: {checkpoint_stage}"
    )
    checkpoint_label_space = (
        checkpoint_cfg
        .get("task", {})
        .get("label_space", None)
    )

    assert checkpoint_label_space == label_space, (
        "Checkpoint label-space mismatch:\n"
        f"Expected: {label_space}\n"
        f"Found: {checkpoint_label_space}"
    )

    checkpoint_model_space = (
        checkpoint_cfg
        .get("model", {})
        .get("semantic_space", None)
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
    
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}")

    cfg = dict(cfg)
    cfg["experiment"] = {
        "save_dir": exp_dir
    }

    evaluator = ModelEvaluator(
        model=model,
        model_name=model_name,
        n_classes=n_classes,
        device=str(next(model.parameters()).device),
        cfg=cfg,
    )
    results = evaluator.evaluate_all(loaders)
    results["task"] = task_cfg["name"]
    eval_path = os.path.join(exp_dir, f"{stage}_eval_results.json")
    evaluator.save(eval_path)

    print(f"\nSaved evaluation results to:\n  {eval_path}")
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
