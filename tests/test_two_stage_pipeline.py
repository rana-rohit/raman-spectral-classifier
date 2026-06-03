import os
import shutil
import uuid
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.registry import get_model
from src.training.trainer import build_trainer
from src.utils.checkpoint import load_encoder_only, resolve_best_checkpoint_path

def _tiny_loaders() -> dict:
    x = torch.randn(12, 1, 32)
    y = torch.tensor([0, 1, 2] * 4, dtype=torch.long)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    return {
        "train": loader,
        "val": loader,
        "finetune": loader,
        "ood": {},
    }

def _workspace_tmp_dir() -> Path:
    path = Path("experiments") / "_tmp_tests" / str(uuid.uuid4())
    path.mkdir(parents=True, exist_ok=True)
    return path

def test_two_stage_contrastive_pipeline():
    tmp_path = _workspace_tmp_dir()
    
    cfg = {
        "model": {
            "name": "cnn",
            "signal_length": 32,
            "n_classes": 3,
            "channels": [8, 16, 32, 64],
            "kernel_sizes": [3, 3, 3, 3],
            "dropout": 0.1,
            "contrastive": True,
            "projection_dim": 64,
            "in_channels": 1,
        },
        "augmentation": {
            "enabled": True,
            "apply_probability": 1.0,
            "steps": {
                "gaussian_noise": {
                    "enabled": True,
                    "max_std": 0.1,
                }
            }
        },
        "task": {
            "stage": "pretrain_30class",
            "name": "isolate_pretraining",
            "label_space": "isolate_space",
        },
        "training": {
            "max_epochs": 1,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "loss": "cross_entropy",
            "loss_kwargs": {},
            "scheduler": "cosine",
            "scheduler_cfg": {"T_max": 1, "eta_min": 1e-6},
            "early_stopping_patience": 2,
            "monitor_metric": "loss",
            "contrastive_weight": 1.0,
            "classification_weight": 0.0,
            "temperature": 0.07,
        },
    }
    
    # --------------------------------------------------------
    # 1. PHASE 1: Pure Supervised Contrastive Pretraining
    # --------------------------------------------------------
    model = get_model("cnn", cfg)
    assert model.contrastive is True
    assert hasattr(model, "projection_head")
    
    # Freeze the classifier head
    classifier_module = model.classifier
    for p in classifier_module.parameters():
        p.requires_grad = False
        
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert frozen_params > 0
    assert trainable_params > 0
    
    # Save original classifier weights to verify they don't change
    orig_classifier_weights = [p.clone().detach() for p in classifier_module.parameters()]
    
    # Run Phase 1 training
    p1_exp_dir = tmp_path / "phase1"
    trainer_p1 = build_trainer(
        model=model,
        loaders=_tiny_loaders(),
        cfg=cfg,
        exp_dir=str(p1_exp_dir),
        n_classes=3,
    )
    trainer_p1.is_pure_supcon = True
    
    metrics = trainer_p1.fit()
    assert "train_metrics" in metrics
    
    # Check that classifier weights did not change
    for p, orig_p in zip(classifier_module.parameters(), orig_classifier_weights):
        assert torch.allclose(p, orig_p)
        
    best_p1_ckpt_path = resolve_best_checkpoint_path(str(p1_exp_dir))
    assert Path(best_p1_ckpt_path).exists()
    
    # Copy checkpoint file to distinct names (mock train.py behavior)
    best_rep_path = tmp_path / "checkpoints" / "best_representation_model.pt"
    best_supcon_path = tmp_path / "checkpoints" / "best_supcon_encoder.pt"
    best_rep_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_p1_ckpt_path, best_rep_path)
    shutil.copy2(best_p1_ckpt_path, best_supcon_path)
    
    # --------------------------------------------------------
    # 2. PHASE 2: Linear Evaluation / Classifier Training
    # --------------------------------------------------------
    # Re-initialize model to ensure independent classifier initialization
    model_p2 = get_model("cnn", cfg)
    
    # Save the fresh randomly initialized classifier weights to verify they change in Phase 2
    fresh_classifier_weights = [p.clone().detach() for p in model_p2.classifier.parameters()]
    
    # Load encoder weights only
    load_encoder_only(str(best_rep_path), model_p2)
    
    # Verify encoder weights match the trained model_p1
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model_p2.named_parameters()):
        if "classifier" not in n1:
            assert torch.allclose(p1, p2)
            
    # Freeze encoder
    for p in model_p2.parameters():
        p.requires_grad = False
    for p in model_p2.classifier.parameters():
        p.requires_grad = True
        
    # Enable bypass of projection head
    model_p2.bypass_projection = True
    
    # Verify projection head is bypassed
    x = torch.randn(2, 1, 32)
    outputs = model_p2(x)
    assert "projection_features" not in outputs
    
    # Phase 2 config
    cfg_p2 = cfg.copy()
    cfg_p2["training"]["contrastive_weight"] = 0.0
    cfg_p2["training"]["classification_weight"] = 1.0
    cfg_p2["training"]["monitor_metric"] = "accuracy"
    
    trainer_p2 = build_trainer(
        model=model_p2,
        loaders=_tiny_loaders(),
        cfg=cfg_p2,
        exp_dir=str(tmp_path),
        n_classes=3,
    )
    
    assert trainer_p2.contrastive_learning_enabled is False
    
    metrics_p2 = trainer_p2.fit()
    assert "train_metrics" in metrics_p2
    
    # Check that classifier weights HAVE updated/changed from their fresh random values
    updated = False
    for p, fresh_p in zip(model_p2.classifier.parameters(), fresh_classifier_weights):
        if not torch.allclose(p, fresh_p):
            updated = True
            break
    assert updated is True
    
    # Clean up
    shutil.rmtree(tmp_path, ignore_errors=True)
