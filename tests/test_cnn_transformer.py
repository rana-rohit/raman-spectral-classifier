import torch
import pytest
from src.models.registry import get_model
from src.models.cnn_transformer import CNNTransformer
from src.interpretability.gradcam1d import GradCAM1D

def test_cnn_transformer_direct_instantiation():
    model = CNNTransformer(
        signal_length=1000,
        n_classes=5,
        in_channels=1,
        channels=[32, 64, 128, 256],
        cnn_kernel_size=5,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        dropout=0.1,
        attn_dropout=0.1,
    )
    
    assert model.embedding_dim == 256
    assert model.d_model == 256
    assert hasattr(model, "cnn_stem")
    assert hasattr(model, "classifier")
    assert hasattr(model, "domain_classifier")
    
    # Check parameters count function
    n_params = model.n_parameters()
    assert isinstance(n_params, int)
    assert n_params > 0

def test_cnn_transformer_forward_and_shapes():
    model = CNNTransformer(
        signal_length=1000,
        n_classes=5,
        in_channels=1,
        channels=[32, 64, 128, 256],
        d_model=256,
        n_heads=8,
        n_layers=4,
    )
    model.eval()
    
    x = torch.randn(2, 1, 1000)
    
    # Test standard forward
    outputs = model(x)
    assert isinstance(outputs, dict)
    assert "main_logits" in outputs
    assert "features" in outputs
    assert "aux_logits" in outputs
    assert outputs["main_logits"].shape == (2, 5)
    assert outputs["features"].shape == (2, 256)
    
    # Test forward_features and forward_logits
    feats = model.forward_features(x)
    assert feats.shape == (2, 256)
    logits = model.forward_logits(feats)
    assert logits.shape == (2, 5)
    
    # Test forward with return_attn=True
    logits_attn, attn_maps = model(x, return_attn=True)
    assert logits_attn.shape == (2, 5)
    assert isinstance(attn_maps, list)
    assert len(attn_maps) == 4  # n_layers = 4
    # Each attention map should have shape (B, seq_len, seq_len)
    # L' = 1000 // 16 = 62. +1 for CLS = 63.
    for attn in attn_maps:
        assert attn.shape == (2, 63, 63)

def test_cnn_transformer_xai_methods():
    model = CNNTransformer(
        signal_length=1000,
        n_classes=5,
        in_channels=1,
        channels=[32, 64, 128, 256],
        d_model=256,
        n_heads=8,
        n_layers=4,
    )
    model.eval()
    
    x = torch.randn(2, 1, 1000)
    
    # 1. Feature maps (for Grad-CAM)
    feat_maps = model.get_feature_maps(x)
    assert feat_maps.shape == (2, 256, 62)
    
    # 2. CNN features
    cnn_feats = model.get_cnn_features(x)
    assert cnn_feats.shape == (2, 256, 62)
    
    # 3. Attention maps
    attn_maps = model.get_attention_maps(x)
    assert len(attn_maps) == 4
    for attn in attn_maps:
        assert attn.shape == (2, 63, 63)

def test_cnn_transformer_registry():
    cfg = {
        "model": {
            "name": "cnn_transformer",
            "signal_length": 1000,
            "n_classes": 5,
            "in_channels": 1,
            "channels": [32, 64, 128, 256],
            "cnn_kernel_size": 5,
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 4,
            "d_ff": 512,
            "dropout": 0.1,
            "attn_dropout": 0.1,
        }
    }
    
    model = get_model("cnn_transformer", cfg)
    assert isinstance(model, CNNTransformer)
    
    x = torch.randn(2, 1, 1000)
    outputs = model(x)
    assert outputs["main_logits"].shape == (2, 5)

def test_cnn_transformer_gradcam_compatibility():
    model = CNNTransformer(
        signal_length=1000,
        n_classes=5,
        in_channels=1,
        channels=[32, 64, 128, 256],
        d_model=256,
        n_heads=8,
        n_layers=4,
    )
    
    gcam = GradCAM1D(model)
    x = torch.randn(1, 1, 1000)
    
    # Compute Grad-CAM
    cam = gcam.compute(x, target_class=2, signal_length=1000)
    assert cam.shape == (1000,)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0

def test_cnn_transformer_two_stage_pipeline():
    import shutil
    import uuid
    from pathlib import Path
    from torch.utils.data import DataLoader, TensorDataset
    from src.training.trainer import build_trainer
    from src.utils.checkpoint import load_encoder_only, resolve_best_checkpoint_path

    # Define a tiny loaders helper
    x_data = torch.randn(12, 1, 32)
    y_data = torch.tensor([0, 1, 2] * 4, dtype=torch.long)
    loader = DataLoader(TensorDataset(x_data, y_data), batch_size=4, shuffle=False)
    loaders = {"train": loader, "val": loader, "finetune": loader, "ood": {}}

    path = Path("experiments") / "_tmp_tests" / str(uuid.uuid4())
    path.mkdir(parents=True, exist_ok=True)

    cfg = {
        "model": {
            "name": "cnn_transformer",
            "signal_length": 32,
            "n_classes": 3,
            "channels": [8, 16, 32, 64],
            "cnn_kernel_size": 3,
            "d_model": 64,
            "n_heads": 8,
            "n_layers": 2,
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

    # Phase 1: Pretraining
    model = get_model("cnn_transformer", cfg)
    assert model.contrastive is True
    assert hasattr(model, "projection_head")

    # Freeze classifier
    for p in model.classifier.parameters():
        p.requires_grad = False

    orig_classifier_weights = [p.clone().detach() for p in model.classifier.parameters()]

    trainer_p1 = build_trainer(
        model=model,
        loaders=loaders,
        cfg=cfg,
        exp_dir=str(path / "phase1"),
        n_classes=3,
    )
    trainer_p1.is_pure_supcon = True
    metrics = trainer_p1.fit()
    assert "train_metrics" in metrics

    # Check classifier weights didn't change
    for p, orig_p in zip(model.classifier.parameters(), orig_classifier_weights):
        assert torch.allclose(p, orig_p)

    best_p1_ckpt_path = resolve_best_checkpoint_path(str(path / "phase1"))
    assert Path(best_p1_ckpt_path).exists()

    best_rep_path = path / "checkpoints" / "best_representation_model.pt"
    best_rep_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_p1_ckpt_path, best_rep_path)

    # Phase 2: Fine-tuning
    model_p2 = get_model("cnn_transformer", cfg)
    fresh_classifier_weights = [p.clone().detach() for p in model_p2.classifier.parameters()]
    load_encoder_only(str(best_rep_path), model_p2)

    # Verify encoder weights match
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model_p2.named_parameters()):
        if "classifier" not in n1:
            assert torch.allclose(p1, p2)

    # Freeze backbone
    for p in model_p2.parameters():
        p.requires_grad = False
    for p in model_p2.classifier.parameters():
        p.requires_grad = True

    model_p2.bypass_projection = True
    cfg_p2 = cfg.copy()
    cfg_p2["training"]["contrastive_weight"] = 0.0
    cfg_p2["training"]["classification_weight"] = 1.0
    cfg_p2["training"]["monitor_metric"] = "accuracy"

    trainer_p2 = build_trainer(
        model=model_p2,
        loaders=loaders,
        cfg=cfg_p2,
        exp_dir=str(path),
        n_classes=3,
    )
    trainer_p2.fit()

    # Check that classifier weights HAVE updated
    updated = False
    for p, fresh_p in zip(model_p2.classifier.parameters(), fresh_classifier_weights):
        if not torch.allclose(p, fresh_p):
            updated = True
            break
    assert updated is True

    # Clean up
    shutil.rmtree(path, ignore_errors=True)

def test_supcon_wrapper_signature_and_behaviors():
    import tempfile
    from pathlib import Path
    from src.utils.checkpoint import save_checkpoint, load_best_model
    import inspect


    cfg = {
        "model": {
            "name": "cnn_transformer",
            "signal_length": 1000,
            "n_classes": 5,
            "in_channels": 1,
            "channels": [32, 64, 128, 256],
            "d_model": 256,
            "n_heads": 8,
            "n_layers": 4,
            "dropout": 0.1,
            "contrastive": True,
            "projection_dim": 128,
        },
        "training": {
            "supcon": {
                "enabled": True,
                "projection_dim": 128,
            }
        }
    }

    # Test for both cnn_transformer and existing transformer
    for model_name in ["cnn_transformer", "transformer"]:
        model_cfg = cfg.copy()
        model_cfg["model"] = cfg["model"].copy()
        model_cfg["model"]["name"] = model_name
        if model_name == "transformer":
            model_cfg["model"].pop("channels", None)
            model_cfg["model"].pop("cnn_kernel_size", None)
        
        # Get wrapped model
        model = get_model(model_name, model_cfg)

        assert model.contrastive is True
        assert hasattr(model, "projection_head")
        
        # 1. Verify original forward signature is preserved
        sig = inspect.signature(model.forward)
        assert "return_attn" in sig.parameters
        assert sig.parameters["return_attn"].default is False
        
        x = torch.randn(2, 1, 1000)
        
        # 2. Test model(x)
        out = model(x)
        assert isinstance(out, dict)
        assert "projection_features" in out
        assert out["projection_features"].shape == (2, 128)
        norm = torch.linalg.norm(out["projection_features"], dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
        
        # 3. Test model(x, return_attn=True)
        logits, attn_maps = model(x, return_attn=True)
        assert isinstance(logits, torch.Tensor)
        assert isinstance(attn_maps, list)
        
        # 4. Test get_attention_maps()
        attn_maps2 = model.get_attention_maps(x)
        assert isinstance(attn_maps2, list)
        
        # 5. Test saliency compatibility (forward_features + forward_logits)
        feats = model.forward_features(x)
        assert feats.shape == (2, 256)
        logits = model.forward_logits(feats)
        assert logits.shape == (2, 5)
        
        # 6. Test checkpoint save/load
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "checkpoint.pt"
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            save_checkpoint(
                path=str(tmp_path),
                model=model,
                optimizer=optimizer,
                epoch=1,
                metrics={"accuracy": 0.8},
                config=model_cfg,
                is_best=True
            )
            
            reloaded_model = get_model(model_name, model_cfg)
            load_best_model(str(Path(tmpdir)), reloaded_model)
            
            # Verify weights loaded match
            for p1, p2 in zip(model.parameters(), reloaded_model.parameters()):
                assert torch.allclose(p1, p2)

