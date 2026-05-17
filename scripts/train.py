"""
scripts/train.py

Main training entry point.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

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
from src.utils.checkpoint import (
    load_best_model,
    load_backbone_weights,
    resolve_best_checkpoint_path,
)
from src.utils.config import load_config, save_config
from src.utils.seed import set_seed
from metadata.ontology import (
    CLINICAL_LABELS,
    INVERSE_COMPACT_LABEL_MAP,
    ONTOLOGY_VERSION,
)
from src.xai.saliency import compute_saliency
from scripts.plot_saliency import plot_saliency
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Train a spectral classifier")
    p.add_argument("--model", required=True, choices=["cnn", "resnet1d", "tcn", "transformer"])
    p.add_argument("--stage", required=True, choices=["s1_isolate","s2_treatment","s3_transfer"])
    p.add_argument("--exp-name", default=None)
    p.add_argument("--exp-dir", default="experiments")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--run-xai", action="store_true", help="Run saliency analysis after training")
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
        f"configs/stages/{args.stage}.yaml",
        f"configs/model/{args.model}.yaml",
    )
    cfg = apply_overrides(dict(cfg), args.override)

    # TASK CONFIGURATION
    task_cfg = cfg["task"]

    task_name = task_cfg["name"]
    label_space = task_cfg["label_space"]

    stage = task_cfg["stage"]

    clinical_sparse_global_ids = task_cfg.get(
        "clinical_sparse_global_ids",
        [],
    )

    if stage == "pretrain_30class":
        assert (
            label_space == "isolate_space"
        ), (
            "pretrain_30class must operate "
            "on isolate_space"
        )

        assert (
            cfg["model"]["semantic_space"]
            == "isolate_space"
        ), (
            "Stage-1 model must operate "
            "in isolate_space"
        )

        clinical_sparse_ids = []
        n_classes = cfg["dataset"]["n_classes_full"]

    elif stage == "pretrain_treatment_8class":
        assert (
            label_space == "global_treatment_space"
        ), (
            "pretrain_treatment_8class must operate "
            "on global_treatment_space"
        )
        
        assert (
            cfg["model"]["semantic_space"]
            == "global_treatment_space"
        ), (
            "Treatment pretraining model must operate "
            "in global_treatment_space"
        )

        clinical_sparse_ids = []
        n_classes = 8

    elif stage == "transfer_5class":

        clinical_sparse_ids = clinical_sparse_global_ids
        n_classes = len(clinical_sparse_ids)

    else:
        raise ValueError(f"Unknown training stage: {stage}")

    cfg["model"]["n_classes"] = n_classes
    
    # --------------------------------------------------------
    # Semantic-space integrity checks
    # --------------------------------------------------------
    if stage == "transfer_5class":
        assert (
            label_space == "sparse_global_treatment_space"
        ), (
            "transfer_5class must operate "
            "on sparse_global_treatment_space"
        )

        assert (
            cfg["model"]["semantic_space"]
            == "compact_transfer_space"
        ), (
            "Transfer model must operate "
            "in compact_transfer_space"
        )

    from src.utils.logging import print_stage_header, print_label_space_info
    print_stage_header(stage, task_name)
    print_label_space_info(label_space, clinical_sparse_ids)

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

    # Always fit preprocessor on the FULL reference set.
    # The pretrained backbone learned under this normalization.
    # Filtering before fitting creates a statistical mismatch.
    preprocessor = SpectralPreprocessor.from_config(cfg["preprocessing"])
    preprocessor.fit(X_ref)

    augmentation = AugmentationPipeline.from_config(cfg["augmentation"])
    if len(augmentation.steps) == 0 or augmentation.p == 0:
        augmentation = None
    
    cfg["batch_size"] = (
        cfg.get("training", {})
        .get("batch_size", 256)
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

    from src.utils.logging import print_split_provenance
    print_split_provenance(loaders, cfg, context="training")

    print("\n[2/4] Building model...")
    model = get_model(args.model, cfg)
    
    if stage in {
        "pretrain_treatment_8class",
        "transfer_5class",
    }:
    
        pretrained_dir = task_cfg.get("pretrained_exp_dir")

        if pretrained_dir is None:
            raise ValueError(
                f"{stage} requires pretrained_exp_dir"
            )

        print("\nLoading pretrained backbone...")
        pretrained_ckpt = resolve_best_checkpoint_path(pretrained_dir)

        checkpoint = load_backbone_weights(
            pretrained_ckpt,
            model,
        )

        checkpoint_cfg = checkpoint.get("config", {})

        checkpoint_stage = (
            checkpoint_cfg
            .get("task", {})
            .get("stage", None)
        )

        if stage == "pretrain_treatment_8class":
            assert checkpoint_stage == "pretrain_30class", (
                "Stage-2 must load a Stage-1 isolate checkpoint"
            )

        elif stage == "transfer_5class":
            assert checkpoint_stage == "pretrain_treatment_8class", (
                "Stage-3 must load a Stage-2 treatment checkpoint"
            )

        from src.utils.logging import print_checkpoint_info
        print_checkpoint_info(
            pretrained_ckpt,
            loaded=True,
            details={"epoch": checkpoint.get("epoch", "?")}
        )

    from src.utils.logging import print_model_summary
    print_model_summary(args.model, cfg["model"])

    print("\n[3/4] Training...")
    trainer = build_trainer(
        model=model,
        loaders=loaders,
        cfg=cfg,
        exp_dir=exp_dir,
        n_classes=n_classes,
    )
    trainer.fit()
    load_best_model(exp_dir, model)
    
    # --------------------------------------------------------
    # Stage 1:
    # isolate-space pretraining
    #
    # 30-class spectral representation learning.
    # --------------------------------------------------------
    if stage in {
        "pretrain_30class",
        "pretrain_treatment_8class",
    }:

        print(
            "\n[4/4] Final evaluation "
            "(pretraining checkpoint)..."
        )

    else:

        print(
            "\n[4/4] Final evaluation "
            "(transfer checkpoint)..."
        )

    evaluator = ModelEvaluator(
        model=model,
        model_name=args.model,
        n_classes=n_classes,
        device=str(next(model.parameters()).device),
        cfg=cfg,
    )
    evaluator.evaluate_all(loaders)

    if stage == "pretrain_30class":
        results_name = (
            "pretrain_30class_results.json"
        )

    elif stage == "pretrain_treatment_8class":
        results_name = (
            "pretrain_treatment_8class_results.json"
        )
    else:
        results_name = (
            "transfer_5class_results.json"
        )

    evaluator.save(os.path.join(exp_dir, results_name))

    print(f"\n  Results saved in {exp_dir}/")
    
    if stage == "transfer_5class":
        print("\n[Finetune Phase] Adapting model to new domain...")
        finetune_dir = os.path.join(exp_dir, "finetune")
        assert n_classes == 5, (
            "Finetuning must operate "
            "in compact transfer-space"
        )
        
        finetune(
            model=model,
            pretrained_exp_dir=exp_dir,
            loaders=loaders,
            cfg=cfg,
            exp_dir=finetune_dir,
            freeze_epochs=3,
            n_classes=n_classes,
        )

        print(f"\n  Fine-tune artifacts: {finetune_dir}/")

    print(f"\n  Done. Training artifacts: {exp_dir}/")
    
    # --------------------------------------------------------
    # XAI analysis operates on:
    # compact transfer-space predictions
    #
    # Predictions are restored back to:
    # sparse clinical ontology IDs
    #
    # before semantic visualization.
    # --------------------------------------------------------
    if args.run_xai and clinical_sparse_ids is not None:
        print("\n[XAI] Generating saliency maps...")

        xai_root = Path(exp_dir) / "xai"
        xai_root.mkdir(parents=True, exist_ok=True)

        loader = loaders["ood"]["2018clinical"]

        model.eval()
        device = next(model.parameters()).device

        class_counts = {
            remapped_label: 0
            for remapped_label in range(n_classes)
        }

        for x, y in loader:
            for i in range(x.shape[0]):

                label = int(y[i].item())

                if label not in class_counts or class_counts[label] >= 2:
                    continue

                xi = x[i:i+1].to(device)

                saliency = compute_saliency(model, xi)
                signal = xi[0].mean(dim=0).detach().cpu().numpy()
                
                assert label in INVERSE_COMPACT_LABEL_MAP, (
                    f"Unknown compact label: {label}"
                )
                
                original_label = INVERSE_COMPACT_LABEL_MAP[label]
                clinical_info = CLINICAL_LABELS[original_label]

                treatment = clinical_info["global_treatment"]

                safe_name = (
                    f"treatment_{original_label}_{treatment}"
                    .replace(" ", "_")
                    .replace("/", "_")
                )

                save_dir = xai_root / safe_name
                save_dir.mkdir(parents=True, exist_ok=True)

                save_path = save_dir / f"sample_{class_counts[label]}.png"

                plot_saliency(signal, saliency, save_path)

                class_counts[label] += 1

                if all(v >= 2 for v in class_counts.values()):
                    break

            if all(v >= 2 for v in class_counts.values()):
                break

        print(f"[XAI] Saved to {xai_root}")

if __name__ == "__main__":
    main()
