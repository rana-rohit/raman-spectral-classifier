import shutil
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.interpretability.gradcam1d import GradCAM1D
from src.models.modules.cbam1d import CBAM1D, SpatialAttention1D
from src.models.registry import get_model
from src.models.resnet1d import ResNet1D
from src.utils.checkpoint import (
    load_backbone_weights,
    load_checkpoint,
    load_encoder_only,
    save_checkpoint,
)


def _workspace_tmp_dir() -> Path:
    path = Path("experiments") / "_tmp_tests" / str(uuid.uuid4())
    path.mkdir(parents=True, exist_ok=True)
    return path


def _small_resnet_cfg(use_cbam: bool = False, contrastive: bool = False) -> dict:
    return {
        "model": {
            "name": "resnet1d",
            "signal_length": 64,
            "n_classes": 3,
            "channels": [8, 16, 32, 64],
            "n_blocks": [1, 1, 1, 1],
            "stem_kernel": 7,
            "dropout": 0.1,
            "in_channels": 1,
            "use_se": False,
            "se_reduction": 4,
            "use_cbam": use_cbam,
            "cbam_reduction": 4,
            "cbam_kernel_size": 7,
            "contrastive": contrastive,
            "projection_dim": 16,
        },
        "task": {
            "stage": "pretrain_30class",
            "label_space": "isolate_space",
        },
    }


def test_cbam1d_preserves_spectral_tensor_shape():
    module = CBAM1D(channels=8, reduction=4, kernel_size=7)
    x = torch.randn(2, 8, 31)
    y = module(x)

    assert y.shape == x.shape
    assert isinstance(module.spatial_attention, SpatialAttention1D)
    assert isinstance(module.spatial_attention.conv, nn.Conv1d)
    assert module.spatial_attention.conv.kernel_size == (7,)


def test_resnet1d_default_has_no_cbam_state_and_preserves_contracts():
    model = ResNet1D(
        signal_length=64,
        n_classes=3,
        channels=[8, 16, 32, 64],
        n_blocks=[1, 1, 1, 1],
        in_channels=1,
    )
    x = torch.randn(2, 1, 64)
    outputs = model(x)

    assert not any(".cbam." in key for key in model.state_dict())
    assert outputs["main_logits"].shape == (2, 3)
    assert outputs["features"].shape == (2, 64)
    assert model.get_feature_maps(x).shape == (2, 64, 8)


def test_resnet1d_cbam_preserves_forward_and_feature_map_shapes():
    model = ResNet1D(
        signal_length=64,
        n_classes=3,
        channels=[8, 16, 32, 64],
        n_blocks=[1, 1, 1, 1],
        in_channels=1,
        use_cbam=True,
        cbam_reduction=4,
        cbam_kernel_size=7,
    )
    x = torch.randn(2, 1, 64)
    outputs = model(x)

    assert any(".cbam." in key for key in model.state_dict())
    assert outputs["main_logits"].shape == (2, 3)
    assert outputs["features"].shape == (2, 64)
    assert model.get_feature_maps(x).shape == (2, 64, 8)


def test_registry_cbam_resnet_supcon_projection_shape():
    model = get_model("resnet1d", _small_resnet_cfg(use_cbam=True, contrastive=True))
    outputs = model(torch.randn(4, 1, 64))

    assert model.contrastive is True
    assert outputs["main_logits"].shape == (4, 3)
    assert outputs["features"].shape == (4, 64)
    assert outputs["projection_features"].shape == (4, 16)
    norms = torch.linalg.norm(outputs["projection_features"], dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_gradcam_smoke_with_cbam_resnet():
    model = get_model("resnet1d", _small_resnet_cfg(use_cbam=True))
    cam = GradCAM1D(model).compute(torch.randn(1, 1, 64), signal_length=64)

    assert cam.shape == (64,)
    assert np.all(np.isfinite(cam))


def test_old_resnet_checkpoint_loads_into_cbam_resnet_with_partial_warning():
    tmp_path = _workspace_tmp_dir()
    try:
        old_model = get_model("resnet1d", _small_resnet_cfg(use_cbam=False))
        optimizer = torch.optim.Adam(old_model.parameters(), lr=1e-3)
        checkpoint_path = tmp_path / "old_resnet.pt"
        save_checkpoint(
            path=str(checkpoint_path),
            model=old_model,
            optimizer=optimizer,
            epoch=1,
            metrics={"accuracy": 0.5},
            config={"training": {"monitor_metric": "accuracy"}},
        )

        cbam_model = get_model("resnet1d", _small_resnet_cfg(use_cbam=True))
        checkpoint = load_checkpoint(str(checkpoint_path), cbam_model)

        assert checkpoint["epoch"] == 1
        assert any(".cbam." in key for key in cbam_model.state_dict())
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_transfer_checkpoint_loaders_preserve_cbam_model_shape():
    tmp_path = _workspace_tmp_dir()
    try:
        old_model = get_model("resnet1d", _small_resnet_cfg(use_cbam=False))
        optimizer = torch.optim.Adam(old_model.parameters(), lr=1e-3)
        checkpoint_path = tmp_path / "old_stage_checkpoint.pt"
        save_checkpoint(
            path=str(checkpoint_path),
            model=old_model,
            optimizer=optimizer,
            epoch=1,
            metrics={"accuracy": 0.5},
            config={"training": {"monitor_metric": "accuracy"}},
        )

        stage2_model = get_model("resnet1d", _small_resnet_cfg(use_cbam=True))
        stage3_model = get_model("resnet1d", _small_resnet_cfg(use_cbam=True))

        load_backbone_weights(str(checkpoint_path), stage2_model)
        load_encoder_only(str(checkpoint_path), stage3_model)

        assert stage2_model(torch.randn(2, 1, 64))["features"].shape == (2, 64)
        assert stage3_model(torch.randn(2, 1, 64))["features"].shape == (2, 64)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
