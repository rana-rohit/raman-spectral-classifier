"""
scripts/train.py

Main training entry point. Trains a single model end-to-end.

Usage:
    python scripts/train.py --model cnn
    python scripts/train.py --model transformer
    python scripts/train.py --model hybrid --exp-name hybrid_handoff2
    python scripts/train.py --model hybrid --override model.handoff_blocks=1

The experiment directory is auto-named with a timestamp if not specified.
All configs, metrics, and checkpoints are saved there.
"""

import argparse
from multiprocessing import dummy
import os
import sys
import time
from xml.parsers.expat import model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml

from src.utils.seed import set_seed
from src.utils.config import load_config, save_config
from src.data.registry import DataRegistry
from src.data.preprocessing import SpectralPreprocessor
from src.data.augmentation import AugmentationPipeline
from src.data.dataloader import build_all_loaders
from src.models.registry import get_model, model_summary
from src.training.trainer import build_trainer
from src.evaluation.evaluator import ModelEvaluator
from src.training.finetuner import finetune

def parse_args():
    p = argparse.ArgumentParser(description="Train a spectral classifier")
    p.add_argument("--model",    required=True,
                   choices=["cnn", "resnet1d", "transformer", "hybrid"],
                   help="Model architecture to train")
    p.add_argument("--exp-name", default=None,
                   help="Experiment name. Auto-generated if not set.")
    p.add_argument("--exp-dir",  default="experiments",
                   help="Root directory for all experiments")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--override", nargs="*", default=[],
                   help="Config overrides: key=value (e.g. model.dropout=0.5)")
    return p.parse_args()


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """Apply key=value overrides to a nested config dict."""
    for item in overrides:
        key, val = item.split("=", 1)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Attempt type coercion
        try:
            val = yaml.safe_load(val)
        except Exception:
            pass
        d[keys[-1]] = val
    return cfg


def main():
    args = parse_args()
    set_seed(args.seed)

    # ---- Load and merge configs ----
    cfg = load_config(
        "configs/data/splits.yaml",
        "configs/data/preprocessing.yaml",
        "configs/data/augmentation.yaml",
        "configs/training/base.yaml",
        f"configs/model/{args.model}.yaml",
    )
    cfg = apply_overrides(dict(cfg), args.override)

    # ---- Experiment directory ----
    exp_name = args.exp_name or f"{args.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir  = os.path.join(args.exp_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    save_config(cfg, os.path.join(exp_dir, "config.yaml"))
    print(f"\n  Experiment: {exp_name}")
    print(f"  Directory:  {exp_dir}")

    # ---- Data ----
    print("\n[1/4] Loading data...")
    registry = DataRegistry(data_root="data/raw", cfg=cfg)
    registry.load_all()

    X_ref, y_ref = registry.get_arrays("reference")
    preprocessor = SpectralPreprocessor.from_config(cfg["preprocessing"])
    X_ref_clean  = preprocessor.fit_transform(X_ref)   # fit AND transform in one call

    augmentation = AugmentationPipeline.from_config(cfg["augmentation"])
    if len(augmentation.steps) == 0 or augmentation.p == 0:
        augmentation = None

    loader_cfg = {
        "batch_size":  cfg.get("training", {}).get("batch_size", 256),
        "num_workers": cfg.get("training", {}).get("num_workers", 4),
        "validation":  cfg["validation"],
    }
    loaders = build_all_loaders(registry, preprocessor, augmentation, loader_cfg)
    print(f"  Train: {len(loaders['train'].dataset):,} samples")
    print(f"  Val:   {len(loaders['val'].dataset):,} samples")

    # ---- Model ----
    print("\n[2/4] Building model...")
    model = get_model(args.model, cfg)
    model_summary(model)
    
    # Sanity check: forward pass produces finite outputs
    model.eval()
    dummy = torch.zeros(2, 1, 1000)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 30), f"Expected (2, 30), got {out.shape}"
    assert not torch.isnan(out).any(), "Model produces NaN on zero input"
    assert not torch.isinf(out).any(), "Model produces Inf on zero input"
    print(f"  Model sanity check passed. Output range: [{out.min():.3f}, {out.max():.3f}]")
    model.train()
    
    # ---- Train ----
    print("\n[3/4] Training...")
    trainer = build_trainer(
        model=model,
        loaders=loaders,
        cfg=cfg,
        exp_dir=exp_dir,
        n_classes=cfg["dataset"]["n_classes_full"],
    )
    trainer.fit()

    # ---- Final evaluation ----
    print("\n[4/4] Final evaluation (all splits)...")
    evaluator = ModelEvaluator(
        model=model,
        model_name=args.model,
        n_classes=cfg["dataset"]["n_classes_full"],
    )
    results = evaluator.evaluate_all(loaders)
    evaluator.save(os.path.join(exp_dir, "results.json"))

    print(f"\n  Done. Results in {exp_dir}/")

    print("\n[Finetune Phase] Adapting model to new domain...")

    finetune(
        model=model,
        pretrained_exp_dir=exp_dir,   
        loaders=loaders,
        cfg=cfg,
        exp_dir=exp_dir,
        freeze_epochs=3,             
        n_classes=cfg["dataset"]["n_classes_full"],
    )

if __name__ == "__main__":
    main()