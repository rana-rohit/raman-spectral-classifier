from pathlib import Path
import uuid

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.clinical_utils import clinical_subset_eval
from src.models.multitask import MultiHeadSpectralModel
from src.models.registry import get_model
from src.training.regularizers import L2SPRegularizer
from src.training.trainer import build_trainer
from src.utils.checkpoint import load_best_model, save_checkpoint
from src.utils.class_subset import prepare_subset_eval_logits
from src.data.dataset import SpectralDataset
from src.data.preprocessing import SpectralPreprocessor


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x).squeeze(-1)
        return self.classifier(x)


def _tiny_loaders() -> dict:
    x = torch.randn(12, 1, 16)
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


def test_load_best_model_finds_checkpoint_in_checkpoints_dir():
    tmp_path = _workspace_tmp_dir()
    model = nn.Linear(4, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint_path = tmp_path / "checkpoints" / "epoch_001.pt"

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        epoch=1,
        metrics={"accuracy": 1.0},
        config={"training": {"max_epochs": 1}},
        is_best=True,
    )

    reloaded = nn.Linear(4, 3)
    checkpoint = load_best_model(str(tmp_path), reloaded)
    assert checkpoint["epoch"] == 1


def test_clinical_subset_eval_restricts_logits_without_dropping_samples():
    logits = torch.tensor(
        [
            [5.0, 1.0, 4.0, 3.0, 0.0, 2.0, 1.5],
            [1.0, 6.0, 0.5, 4.0, 2.0, 3.0, 5.0],
        ]
    )
    targets = torch.tensor([2, 6], dtype=torch.long)

    subset_logits, mapped_targets = clinical_subset_eval(
        logits,
        targets,
        valid_classes=[0, 2, 3, 5, 6],
    )

    assert subset_logits.shape == (2, 5)
    assert mapped_targets.tolist() == [1, 4]


def test_prepare_subset_eval_logits_blends_auxiliary_head_for_clinical_eval():
    main_logits = torch.tensor(
        [
            [3.0, 0.2, 2.0, 1.0, 0.1, 0.5, 0.3],
            [0.5, 2.5, 0.7, 1.8, 0.2, 0.4, 1.6],
        ]
    )
    aux_logits = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )
    targets = torch.tensor([2, 6], dtype=torch.long)

    subset_logits, mapped_targets = prepare_subset_eval_logits(
        main_logits=main_logits,
        targets=targets,
        class_ids=[0, 2, 3, 5, 6],
        aux_logits=aux_logits,
        aux_blend=0.25,
    )

    expected_main = torch.tensor(
        [
            [3.0, 2.0, 1.0, 0.5, 0.3],
            [0.5, 0.7, 1.8, 0.4, 1.6],
        ]
    )
    expected = 0.75 * expected_main + 0.25 * aux_logits

    assert torch.allclose(subset_logits, expected)
    assert mapped_targets.tolist() == [1, 4]


def test_trainer_fit_does_not_run_implicit_finetune():
    tmp_path = _workspace_tmp_dir()
    model = TinyClassifier()
    cfg = {
        "model": {"name": "tiny"},
        "training": {
            "max_epochs": 1,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "loss": "cross_entropy",
            "loss_kwargs": {},
            "scheduler": "cosine",
            "scheduler_cfg": {"T_max": 1, "eta_min": 1e-6},
            "early_stopping_patience": 2,
            "monitor_metric": "accuracy",
        },
    }

    trainer = build_trainer(
        model=model,
        loaders=_tiny_loaders(),
        cfg=cfg,
        exp_dir=str(tmp_path),
        n_classes=3,
    )
    trainer.fit()

    assert model.backbone[0].weight.requires_grad is True


def test_l2sp_regularizer_penalizes_drift_and_respects_exclusions():
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
    reference_state = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }

    with torch.no_grad():
        model[0].weight.add_(1.0)
        model[2].bias.add_(2.0)

    regularizer = L2SPRegularizer(
        reference_state=reference_state,
        lambda_=0.5,
        exclude_patterns=["2.bias"],
    )
    penalty = regularizer(model)
    expected = 0.5 * torch.mean((model[0].weight - reference_state["0.weight"]) ** 2)

    assert torch.isclose(penalty, expected)


def test_spectral_dataset_returns_two_augmented_views_when_enabled():
    X = np.arange(24, dtype=np.float32).reshape(3, 8)
    y = np.array([0, 1, 2], dtype=np.int64)

    preprocessor = SpectralPreprocessor([]).fit(X)
    dataset = SpectralDataset(
        X,
        y,
        augmentation=None,
        training=True,
        n_views=2,
        preprocessor=preprocessor,
        expected_n_classes=3,
    )
    sample = dataset[1]

    assert set(sample.keys()) == {"x1", "x2", "y"}
    assert sample["x1"].shape == (2, 8)
    assert sample["x2"].shape == (2, 8)
    assert sample["y"].item() == 1


def test_registry_wraps_model_with_auxiliary_shared_head():
    cfg = {
        "model": {
            "name": "cnn",
            "signal_length": 32,
            "n_classes": 5,
            "channels": [8, 16, 32, 64],
            "kernel_sizes": [3, 3, 3, 3],
            "dropout": 0.1,
        },
        "dataset": {"shared_classes": [0, 2, 3, 5, 6]},
        "task": {"mode": "shared_clinical_5"},
        "multitask": {
            "auxiliary_shared_head": {
                "enabled": True,
                "classes": [0, 2, 3, 5, 6],
                "dropout": 0.0,
            }
        },
    }

    model = get_model("cnn", cfg)
    outputs = model(torch.randn(2, 1, 32))

    assert isinstance(model, MultiHeadSpectralModel)
    assert outputs["main_logits"].shape == (2, 5)
    assert outputs["aux_logits"].shape == (2, 5)
