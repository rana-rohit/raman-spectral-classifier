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
from src.utils.split_modes import IID_REFERENCE, resolve_split_mode
from src.evaluation.evaluator import ModelEvaluator
from src.models.registry import get_model, model_summary
from src.training.finetuner import finetune
from src.training.trainer import build_trainer
from src.utils.checkpoint import (
    load_best_model,
    load_backbone_weights,
    resolve_best_checkpoint_path,
    load_encoder_only,
    resolve_pretrained_checkpoint,
)
from src.utils.config import load_config, save_config
from src.utils.seed import set_seed
from metadata.ontology import (
    CLINICAL_LABELS,
    INVERSE_COMPACT_LABEL_MAP,
    ONTOLOGY_VERSION,
)
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Train a spectral classifier")
    p.add_argument("--model", required=True, choices=["cnn", "resnet1d", "seresnet1d", "tcn", "transformer", "inception1d"])
    p.add_argument("--stage", required=True, choices=["s1_isolate","s2_treatment","s3_transfer"])
    p.add_argument("--exp-name", default=None)
    p.add_argument("--exp-dir", default="experiments")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--run-finetune", action="store_true", help="Run explicit post-training clinical finetuning")
    p.add_argument("--two-stage", action="store_true", help="Enable decoupled two-stage representation and linear classifier training")
    args, dotlist_overrides = p.parse_known_args()
    args.override = list(args.override) + dotlist_overrides
    return args


def apply_overrides(cfg: dict, overrides: list) -> dict:
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected dotted key=value, "
                "for example model.use_cbam=true"
            )
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


def canonicalize_runtime_config(cfg: dict, args) -> dict:
    """Resolve explicit feature flags into one saved, reproducible config."""
    cfg["seed"] = int(args.seed)
    train_cfg = cfg.setdefault("training", {})
    model_cfg = cfg.setdefault("model", {})
    supcon_cfg = train_cfg.setdefault("supcon", {})
    finetune_cfg = train_cfg.setdefault("finetune", {})

    split_mode = resolve_split_mode(cfg)
    cfg["split_mode"] = split_mode
    train_cfg["split_mode"] = split_mode

    if args.two_stage:
        train_cfg["two_stage"] = True
    else:
        train_cfg["two_stage"] = bool(train_cfg.get("two_stage", False))

    if args.run_finetune:
        finetune_cfg["enabled"] = True
    else:
        finetune_cfg["enabled"] = bool(finetune_cfg.get("enabled", False))

    legacy_contrastive = bool(model_cfg.get("contrastive", False))
    if legacy_contrastive and not supcon_cfg.get("enabled", False):
        print(
            "[Config] Deprecated model.contrastive=True detected. "
            "Canonicalizing to training.supcon.enabled=True."
        )
        supcon_cfg["enabled"] = True

    supcon_enabled = bool(supcon_cfg.get("enabled", False))
    model_cfg["contrastive"] = supcon_enabled
    if supcon_enabled:
        projection_dim = int(
            supcon_cfg.get(
                "projection_dim",
                model_cfg.get("projection_dim", 128),
            )
        )
        supcon_cfg["projection_dim"] = projection_dim
        model_cfg["projection_dim"] = projection_dim

    if train_cfg["two_stage"] and not supcon_enabled:
        raise ValueError(
            "training.two_stage=True requires training.supcon.enabled=True. "
            "Enable SupCon explicitly via training.supcon.enabled=true."
        )

    train_cfg["use_dann"] = bool(
        train_cfg.get("use_dann", False)
        or train_cfg.get("dann", {}).get("enabled", False)
    )
    train_cfg["use_coral"] = bool(
        train_cfg.get("use_coral", False)
        or train_cfg.get("coral", {}).get("enabled", False)
    )
    train_cfg["freeze_backbone"] = bool(train_cfg.get("freeze_backbone", False))
    return cfg


def apply_freeze_backbone_policy(model, cfg: dict) -> None:
    if not cfg.get("training", {}).get("freeze_backbone", False):
        return

    found_classifier = False
    for name, param in model.named_parameters():
        is_classifier = "classifier" in name and "domain_classifier" not in name
        param.requires_grad = is_classifier
        found_classifier = found_classifier or is_classifier

    if not found_classifier:
        raise ValueError("training.freeze_backbone=True requires a classifier module.")

    print("\n[Training] Explicit freeze_backbone=True: training classifier parameters only.")


def print_phase_status(
    phase: int,
    losses_enabled: list[str],
    frozen_modules: list[str],  
    trainable_modules: list[str],
    loaded_checkpoint: str | None,
    frozen_params: int,
    trainable_params: int,
    ) -> None:
    border = "=" * 60
    print(f"\n{border}")
    if phase == 1:
        print("PHASE 1: Pure SupCon Representation Learning")
    else:
        print("PHASE 2: Linear Evaluation / Frozen Encoder Classification")
    print("=" * 60)
    print(f"Active Phase:      Phase {phase}")
    print(f"Losses Enabled:    {', '.join(losses_enabled)}")
    if loaded_checkpoint:
        print(f"Loaded Checkpoint: {loaded_checkpoint}")
    else:
        print(f"Loaded Checkpoint: None (Initial)")
    print(f"Frozen Modules:    {', '.join(frozen_modules) if frozen_modules else 'None'}")
    print(f"Trainable Modules: {', '.join(trainable_modules) if trainable_modules else 'None'}")
    print(f"Frozen Params:     {frozen_params:,}")
    print(f"Trainable Params:  {trainable_params:,}")
    print(f"{border}\n")


def main():
    args = parse_args()
    set_seed(args.seed)
    model_config_name = "seresnet" if args.model == "seresnet1d" else args.model

    cfg = load_config(
        "configs/data/splits.yaml",
        "configs/data/preprocessing.yaml",
        "configs/data/augmentation.yaml",
        "configs/training/base.yaml",
        f"configs/stages/{args.stage}.yaml",
        f"configs/model/{model_config_name}.yaml",
    )
    cfg = apply_overrides(dict(cfg), args.override)
    cfg = canonicalize_runtime_config(cfg, args)

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
    cfg["runtime"] = {
        "n_classes": n_classes,
        "stage": stage,
        "resolved_by": "scripts/train.py",
    }
    
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

    split_mode = resolve_split_mode(cfg)
    split_suffix = "iid" if split_mode == IID_REFERENCE else "holdout"
    stage_suffix = {
        "pretrain_30class": "s1",
        "pretrain_treatment_8class": "s2",
        "transfer_5class": "s3",
    }.get(stage, stage)
    exp_name = args.exp_name or (
        f"{args.model}_{stage_suffix}_{split_suffix}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    exp_dir = os.path.join(args.exp_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    save_config(cfg, os.path.join(exp_dir, "config.yaml"))
    print(f"\n  Experiment: {exp_name}")
    print(f"  Directory:  {exp_dir}")

    print("\n[1/4] Loading data...")
    registry = DataRegistry(data_root="data/raw", cfg=cfg)
    if split_mode == IID_REFERENCE:
        registry.load("reference")
    else:
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
        ckpt_path, source_type, exp_name = resolve_pretrained_checkpoint(cfg, task_cfg, stage)

        print("\nLoading pretrained backbone...")
        try:
            checkpoint = load_backbone_weights(
                ckpt_path,
                model,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {ckpt_path}. Error: {e}")

        checkpoint_cfg = checkpoint.get("config", {})
        checkpoint_stage = (
            checkpoint_cfg
            .get("task", {})
            .get("stage", None)
        )

        if stage == "pretrain_treatment_8class":
            assert checkpoint_stage == "pretrain_30class", (
                f"Stage-2 must load a Stage-1 isolate checkpoint. Got: {checkpoint_stage}"
            )

        elif stage == "transfer_5class":
            assert checkpoint_stage == "pretrain_treatment_8class", (
                f"Stage-3 must load a Stage-2 treatment checkpoint. Got: {checkpoint_stage}"
            )

        from src.utils.logging import print_checkpoint_info
        print_checkpoint_info(
            ckpt_path,
            loaded=True,
            details={"epoch": checkpoint.get("epoch", "?")}
        )

    from src.utils.logging import print_model_summary
    from src.utils.logging import print_feature_summary
    from src.utils.logging import print_clinical_adaptation_config
    print_model_summary(args.model, cfg["model"])
    print_feature_summary(cfg)
    print_clinical_adaptation_config(cfg)

    print("\n[3/4] Training...")
    contrastive_enabled = cfg.get("training", {}).get("supcon", {}).get("enabled", False)
    two_stage_enabled = cfg.get("training", {}).get("two_stage", False)
    if contrastive_enabled and two_stage_enabled:
        import copy
        import shutil
        
        # ----------------------------------------------------
        # PHASE 1: Pure Supervised Contrastive Pretraining
        # ----------------------------------------------------
        # Disable classifier head influence
        classifier_module = None
        if hasattr(model, "classifier"):
            classifier_module = model.classifier
        elif hasattr(model, "backbone") and hasattr(model.backbone, "classifier"):
            classifier_module = model.backbone.classifier
            
        for p in model.parameters():
            p.requires_grad = True
            
        frozen_modules = []
        if classifier_module is not None:
            for p in classifier_module.parameters():
                p.requires_grad = False
            frozen_modules.append("classifier")
            
        if hasattr(model, "domain_classifier") and model.domain_classifier is not None:
            for p in model.domain_classifier.parameters():
                p.requires_grad = False
            frozen_modules.append("domain_classifier")
            
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        trainable_modules = ["backbone", "projection_head"]
        
        # Prepare Phase 1 config: pure supervised contrastive loss, classification loss disabled
        p1_cfg = copy.deepcopy(cfg)
        p1_cfg["training"]["supcon"]["enabled"] = True
        p1_cfg["training"]["supcon"]["weight"] = 1.0
        p1_cfg["training"]["supcon"]["classification_weight"] = 0.0
        p1_cfg["model"]["contrastive"] = True
        p1_cfg["training"]["monitor_metric"] = "loss"
        
        print_phase_status(
            phase=1,
            losses_enabled=["Supervised Contrastive Loss (SupCon)"],
            frozen_modules=frozen_modules,
            trainable_modules=trainable_modules,
            loaded_checkpoint=None,
            frozen_params=frozen_params,
            trainable_params=trainable_params
        )
        
        # Separate Phase 1 experiment directory so checkpoints don't collide
        p1_exp_dir = os.path.join(exp_dir, "phase1")
        trainer_p1 = build_trainer(
            model=model,
            loaders=loaders,
            cfg=p1_cfg,
            exp_dir=p1_exp_dir,
            n_classes=n_classes,
        )
        trainer_p1.is_pure_supcon = True
        trainer_p1.fit()
        
        # At the end of Phase 1, copy the best representation checkpoint to distinct names
        best_p1_ckpt_path = resolve_best_checkpoint_path(p1_exp_dir)
        os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        best_rep_path = os.path.join(exp_dir, "checkpoints", "best_representation_model.pt")
        best_supcon_path = os.path.join(exp_dir, "checkpoints", "best_supcon_encoder.pt")
        
        shutil.copy2(best_p1_ckpt_path, best_rep_path)
        shutil.copy2(best_p1_ckpt_path, best_supcon_path)
        
        # Also copy to root exp_dir
        shutil.copy2(best_p1_ckpt_path, os.path.join(exp_dir, "best_representation_model.pt"))
        shutil.copy2(best_p1_ckpt_path, os.path.join(exp_dir, "best_supcon_encoder.pt"))
        
        print(f"\nSaved representation checkpoints:")
        print(f"  - {best_rep_path}")
        print(f"  - {best_supcon_path}")
        
        # ----------------------------------------------------
        # PHASE 2: Linear Evaluation / Classifier Training
        # ----------------------------------------------------
        # Re-initialize model to ensure independent classifier initialization
        model = get_model(args.model, cfg)
        
        # Load the pretrained representation weights (encoder + projection head)
        load_encoder_only(best_rep_path, model)
        
        # Freeze encoder weights explicitly: requires_grad = False
        classifier_module = None
        if hasattr(model, "classifier"):
            classifier_module = model.classifier
        elif hasattr(model, "backbone") and hasattr(model.backbone, "classifier"):
            classifier_module = model.backbone.classifier
            
        for p in model.parameters():
            p.requires_grad = False
            
        trainable_modules = []
        if classifier_module is not None:
            for p in classifier_module.parameters():
                p.requires_grad = True
            trainable_modules.append("classifier")
            
        # Bypass or disable the projection head
        model.bypass_projection = True
        
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        frozen_modules = ["backbone", "projection_head"]
        if hasattr(model, "domain_classifier") and model.domain_classifier is not None:
            frozen_modules.append("domain_classifier")
            
        # Prepare Phase 2 config: classification only, contrastive bypassed
        p2_cfg = copy.deepcopy(cfg)
        p2_cfg["training"]["supcon"]["enabled"] = True
        p2_cfg["training"]["supcon"]["weight"] = 0.0
        p2_cfg["training"]["supcon"]["classification_weight"] = 1.0
        p2_cfg["model"]["contrastive"] = True
        
        print_phase_status(
            phase=2,
            losses_enabled=["Cross Entropy Loss (Classification)"],
            frozen_modules=frozen_modules,
            trainable_modules=trainable_modules,
            loaded_checkpoint=best_rep_path,
            frozen_params=frozen_params,
            trainable_params=trainable_params
        )
        
        # Build Phase 2 trainer and train ONLY the classifier
        trainer_p2 = build_trainer(
            model=model,
            loaders=loaders,
            cfg=p2_cfg,
            exp_dir=exp_dir,
            n_classes=n_classes,
        )
        trainer_p2.fit()
        
        # Load the best classifier checkpoint from Phase 2 for downstream evaluation
        load_best_model(exp_dir, model)
        
    else:
        # Standard joint or non-contrastive training pipeline
        apply_freeze_backbone_policy(model, cfg)
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
    
    finetune_cfg = cfg.get("training", {}).get("finetune", {})
    if stage == "transfer_5class" and finetune_cfg.get("enabled", False):
        print("\n[Finetune Phase] Explicit finetune.enabled=True: adapting model to new domain...")
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
            freeze_epochs=int(finetune_cfg.get("freeze_epochs", 0)),
            n_classes=n_classes,
        )

        print(f"\n  Fine-tune artifacts: {finetune_dir}/")

    print(f"\n  Done. Training artifacts: {exp_dir}/")


if __name__ == "__main__":
    main()
