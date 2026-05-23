"""
scripts/evaluate.py

Standalone evaluation for trained experiments.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
import shutil

import torch
import matplotlib.pyplot as plt

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


def _load_config_any(exp_dir: str) -> dict:
    cfg_yaml = os.path.join(exp_dir, "config.yaml")
    cfg_json = os.path.join(exp_dir, "config.json")
    if os.path.exists(cfg_yaml):
        return load_config(cfg_yaml)
    if os.path.exists(cfg_json):
        import json
        with open(cfg_json, "r") as f:
            return json.load(f)
    raise FileNotFoundError(
        f"No config.yaml or config.json found in {exp_dir}"
    )


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained spectral classifiers")
    p.add_argument("--exp-dir", default=None)
    p.add_argument("--compare", nargs="+", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-save-outputs",
        action="store_true",
        help="Skip saving predictions/probabilities/features for plotting",
    )
    return p.parse_args()


def _is_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def _resolve_eval_num_workers(cfg: dict) -> int:
    eval_cfg = cfg.get("evaluation", {})
    if "num_workers" in eval_cfg:
        return int(eval_cfg["num_workers"])

    if os.name == "nt" or _is_notebook():
        return 0

    train_cfg = cfg.get("training", {})
    return int(cfg.get("num_workers", train_cfg.get("num_workers", 4)))


def _copy_tree_contents(src: str, dst: str) -> None:
    for root, _, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel_root) if rel_root != "." else dst
        os.makedirs(target_root, exist_ok=True)
        for fname in files:
            src_path = os.path.join(root, fname)
            dst_path = os.path.join(target_root, fname)
            shutil.copy2(src_path, dst_path)


def build_eval_loaders(cfg: dict, seed: int) -> tuple[dict, int, DataRegistry]:
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

    from src.utils.logging import print_stage_header, print_label_space_info
    print_stage_header(stage, task_cfg['name'])
    print_label_space_info(label_space, clinical_sparse_ids)
    
    # Fit preprocessor on FULL reference set (matches pretrained backbone)
    X_ref, _ = registry.get_arrays("reference")
    preprocessor = SpectralPreprocessor.from_config(cfg["preprocessing"])
    preprocessor.fit(X_ref)

    augmentation = AugmentationPipeline.from_config(cfg["augmentation"])
    if len(augmentation.steps) == 0 or augmentation.p == 0:
        augmentation = None

    num_workers = _resolve_eval_num_workers(cfg)
    cfg["num_workers"] = num_workers
    if num_workers == 0:
        print("[EvaluationRunner] Using num_workers=0 for evaluation safety.")
    
    loaders = build_all_loaders(
        registry,
        preprocessor,
        augmentation,
        cfg,
        clinical_sparse_ids=clinical_sparse_ids,
        n_classes=n_classes,
    )

    from src.utils.logging import print_split_provenance
    print_split_provenance(loaders, cfg, context="evaluation")
    return loaders, n_classes, registry


def _save_outputs(base_dir: str, split_name: str, artifact) -> None:
    import numpy as np
    from pathlib import Path

    out_dir = Path(base_dir) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    logits = artifact.logits
    probs = artifact.probabilities
    preds = artifact.predictions
    targets = artifact.targets

    np.save(out_dir / f"{split_name}_logits.npy", logits)
    np.save(out_dir / f"{split_name}_probabilities.npy", probs)
    np.save(out_dir / f"{split_name}_predictions.npy", preds)
    np.save(out_dir / f"{split_name}_targets.npy", targets)

    if artifact.features is not None:
        emb_dir = Path(base_dir) / "embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        np.save(emb_dir / f"{split_name}_features.npy", artifact.features)
        np.save(emb_dir / f"{split_name}_targets.npy", targets)


def evaluate_one(
    exp_dir: str,
    seed: int,
    save_outputs: bool = True,
    include_predictions: bool = False,
    use_staging: bool = True,
) -> dict:
    cfg = _load_config_any(exp_dir)
    task_cfg = cfg["task"]
    stage = task_cfg["stage"]
    label_space = task_cfg["label_space"]
    cfg.setdefault("evaluation", {})["include_predictions"] = bool(include_predictions)

    print("[EvaluationRunner] Building loaders...")
    loaders, n_classes, registry = build_eval_loaders(cfg, seed)

    model_name = cfg.get("model", {}).get("name", "unknown")
    model = get_model(model_name, dict(cfg))

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
    if checkpoint_label_space is None:
        print("[EvaluationRunner] Warning: checkpoint missing 'task.label_space'; continuing.")
    else:
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

    if checkpoint_model_space is None:
        print("[EvaluationRunner] Warning: checkpoint missing 'model.semantic_space'; continuing.")
    else:
        assert (
            checkpoint_model_space
            == cfg["model"]["semantic_space"]
        ), (
            "Checkpoint model semantic-space mismatch:\n"
            f"Expected: "
            f"{cfg['model']['semantic_space']}\n"
            f"Found: {checkpoint_model_space}"
        )

    cfg = dict(cfg)

    staging_dir = None
    if use_staging:
        staging_dir = tempfile.mkdtemp(prefix="eval_staging_")
        print(f"[EvaluationRunner] Using staging directory: {staging_dir}")
        cfg["experiment"] = {"save_dir": staging_dir}
    else:
        cfg["experiment"] = {"save_dir": exp_dir}

    evaluator = None
    results = {}
    try:
        evaluator = ModelEvaluator(
            model=model,
            model_name=model_name,
            n_classes=n_classes,
            device=str(next(model.parameters()).device),
            cfg=cfg,
        )

        print("[EvaluationRunner] Running prediction pass...")
        results = evaluator.evaluate_all(loaders)
        results["task"] = task_cfg["name"]

        eval_base = staging_dir or exp_dir
        eval_path = os.path.join(eval_base, f"{stage}_eval_results.json")
        print("[EvaluationRunner] Exporting evaluation JSON...")
        evaluator.save(eval_path)

        if save_outputs:
            print("[EvaluationRunner] Exporting prediction artifacts...")
            for split_name in ["test"]:
                if split_name in evaluator.artifacts:
                    _save_outputs(eval_base, split_name, evaluator.artifacts[split_name])

            for split_name, _ in (loaders.get("ood", {}) or {}).items():
                if split_name in evaluator.artifacts:
                    _save_outputs(eval_base, split_name, evaluator.artifacts[split_name])

            for split_name in ["train", "val"]:
                loader = loaders.get(split_name)
                if loader is None:
                    continue
                artifact = evaluator.collect_artifact(loader, split_name)
                _save_outputs(eval_base, split_name, artifact)

        if staging_dir is not None:
            print("[EvaluationRunner] Copying staged outputs...")
            os.makedirs(exp_dir, exist_ok=True)
            _copy_tree_contents(staging_dir, exp_dir)

        from src.utils.logging import print_output_paths
        print_output_paths({"Evaluation Results JSON": os.path.join(exp_dir, f"{stage}_eval_results.json")})
        print("[EvaluationRunner] Evaluation complete.")
        return results
    finally:
        print("[EvaluationRunner] Cleaning up workers/resources...")
        try:
            del loaders
        except Exception:
            pass
        try:
            del registry
        except Exception:
            pass
        try:
            del evaluator
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        try:
            plt.close("all")
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if staging_dir and os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    save_outputs = not args.no_save_outputs

    if args.exp_dir and not args.compare:
        evaluate_one(args.exp_dir, args.seed, save_outputs=save_outputs, include_predictions=False, use_staging=True)
    elif args.compare:
        all_results = [
            evaluate_one(
                exp_dir,
                args.seed,
                save_outputs=save_outputs,
                include_predictions=True,
                use_staging=True,
            )
            for exp_dir in args.compare
        ]
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
