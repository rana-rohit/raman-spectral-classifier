"""
scripts/train.py

Main training entry point.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml

from src.data.augmentation import AugmentationPipeline
from src.data.dataloader import build_all_loaders
from src.data.preprocessing import SpectralPreprocessor
from src.data.registry import DataRegistry
from src.evaluation.evaluator import ModelEvaluator
from src.models.registry import get_model, model_summary
from src.training.finetuner import finetune
from src.training.trainer import build_trainer
from src.utils.checkpoint import load_best_model
from src.utils.config import load_config, save_config
from src.utils.seed import set_seed
from src.utils.class_subset import filter_and_remap_classes

def parse_args():
    p = argparse.ArgumentParser(description="Train a spectral classifier")
    p.add_argument("--model", required=True, choices=["cnn", "resnet1d", "transformer", "hybrid"])
    p.add_argument("--exp-name", default=None)
    p.add_argument("--exp-dir", default="experiments")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--override", nargs="*", default=[])
    return p.parse_args()


def apply_overrides(cfg: dict, overrides: list) -> dict:
    for item in overrides:
        key, val = item.split("=", 1)
        keys = key.split(".")
        cursor = cfg
        for part in keys[:-1]:
            cursor = cursor.setdefault(part, {})
        try:
            val = yaml.safe_load(val)
        except Exception:
            pass
        cursor[keys[-1]] = val
    return cfg


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(
        "configs/data/splits.yaml",
        "configs/data/preprocessing.yaml",
        "configs/data/augmentation.yaml",
        "configs/training/base.yaml",
        f"configs/model/{args.model}.yaml",
    )
    cfg = apply_overrides(dict(cfg), args.override)

    exp_name = args.exp_name or f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join(args.exp_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    save_config(cfg, os.path.join(exp_dir, "config.yaml"))
    print(f"\n  Experiment: {exp_name}")
    print(f"  Directory:  {exp_dir}")

    print("\n[1/4] Loading data...")
    registry = DataRegistry(data_root="data/raw", cfg=cfg)
    registry.load_all()

    X_ref, y_ref = registry.get_arrays("reference")

    shared_classes = cfg["dataset"]["shared_classes"]

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
        "consistency": cfg.get("training", {}).get("consistency", {}),
    }

    loaders = build_all_loaders(
        registry,
        preprocessor,
        augmentation,
        loader_cfg,
        shared_classes=shared_classes,
    )

    print(f"  Train: {len(loaders['train'].dataset):,} samples")
    print(f"  Val:   {len(loaders['val'].dataset):,} samples")

    print("\n[2/4] Building model...")
    model = get_model(args.model, cfg)
    model_summary(model) 
    
    print("\n[3/4] Training...")
    trainer = build_trainer(
        model=model,
        loaders=loaders,
        cfg=cfg,
        exp_dir=exp_dir,
        n_classes=cfg["dataset"]["n_classes_clinical"],
    )
    trainer.fit()
    load_best_model(exp_dir, model)

    print("\n[4/4] Final evaluation (pre-finetune, best checkpoint)...")
    evaluator = ModelEvaluator(
        model=model,
        model_name=args.model,
        n_classes=cfg["dataset"]["n_classes_clinical"],
        device=str(next(model.parameters()).device),
        cfg=cfg,
    )
    evaluator.evaluate_all(loaders)
    evaluator.save(os.path.join(exp_dir, "pretrain_results.json"))

    print(f"\n  Pretrain results saved in {exp_dir}/")
    print("\n[Finetune Phase] Adapting model to new domain...")

    finetune_dir = os.path.join(exp_dir, "finetune")
    finetune(
        model=model,
        pretrained_exp_dir=exp_dir,
        loaders=loaders,
        cfg=cfg,
        exp_dir=finetune_dir,
        freeze_epochs=3,
        n_classes=cfg["dataset"]["n_classes_clinical"],
    )

    print(f"\n  Done. Training artifacts: {exp_dir}/")
    print(f"  Fine-tune artifacts:     {finetune_dir}/")


if __name__ == "__main__":
    main()
