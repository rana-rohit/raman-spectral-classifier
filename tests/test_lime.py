"""
tests/test_lime.py

Unit tests for the project-wide LIME explainability framework.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.preprocessing import SpectralPreprocessor
from src.xai.predict_wrapper import build_predict_fn, SpectralPredictWrapper
from src.xai.lime_explainer import SpectralLimeExplainer
from src.xai.xai_visualization import plot_lime_explanation, plot_lime_comparison


class DummyModel(nn.Module):
    """
    A simple dummy 1D CNN model to test the prediction wrapper and LIME.
    """
    def __init__(self, input_len: int = 32, n_classes: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv1d(2, 4, kernel_size=3, padding=1)
        self.fc = nn.Linear(4 * input_len, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (B, C, L) where C = 2 (normal + derivative)
        features = self.conv(x)
        features = features.view(features.size(0), -1)
        logits = self.fc(features)
        return logits


@pytest.fixture
def dummy_preprocessor() -> SpectralPreprocessor:
    cfg_preprocess = {
        "pipeline": ["per_sample_mean_subtraction"],
        "per_sample_mean_subtraction": {"enabled": True}
    }
    preprocessor = SpectralPreprocessor.from_config(cfg_preprocess)
    X_train = np.random.rand(10, 32).astype(np.float32)
    preprocessor.fit(X_train)
    return preprocessor


def test_predict_wrapper(dummy_preprocessor: SpectralPreprocessor):
    model = DummyModel(input_len=32, n_classes=3)
    wrapper = build_predict_fn(model, dummy_preprocessor, device="cpu")

    assert isinstance(wrapper, SpectralPredictWrapper)

    # Test mapping raw input spectra (N, L) -> class probabilities (N, n_classes)
    X_raw = np.random.rand(5, 32).astype(np.float32)
    probs = wrapper(X_raw)

    assert probs.shape == (5, 3)
    # Probabilities should sum to 1
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    # Check that it handles a single 1D spectrum correctly
    single_prob = wrapper(X_raw[0])
    assert single_prob.shape == (1, 3)


def test_lime_explainer_and_viz(dummy_preprocessor: SpectralPreprocessor):
    model = DummyModel(input_len=32, n_classes=3)
    wrapper = build_predict_fn(model, dummy_preprocessor, device="cpu")

    X_background = np.random.rand(50, 32).astype(np.float32)
    wavenumbers = np.arange(32) + 500

    explainer = SpectralLimeExplainer(
        predict_fn=wrapper,
        training_data=X_background,
        wavenumbers=wavenumbers,
        class_names=["A", "B", "C"],
        n_features=5,
        n_samples=50,  # Low number of samples for fast testing
        random_state=42,
    )

    query_spectrum = np.random.rand(32).astype(np.float32)
    
    # 1. Explain specific class
    explanation = explainer.explain_sample(query_spectrum, label=1)

    assert explanation.spectrum.shape == (32,)
    assert explanation.importance.shape == (32,)
    assert explanation.predicted_class in [0, 1, 2]
    assert explanation.explained_class == 1
    assert len(explanation.probabilities) == 3
    assert len(explanation.feature_weights) <= 5
    assert explanation.predicted_label in ["A", "B", "C"]
    assert explanation.explained_label == "B"
    assert 0.0 <= explanation.confidence <= 1.0
    
    # Check top features properties
    top_feats = explanation.top_features(n=3)
    assert len(top_feats) <= 3
    assert explanation.positive_importance.shape == (32,)
    assert explanation.negative_importance.shape == (32,)

    # 2. Explain batch of spectra
    spectra_batch = np.random.rand(3, 32).astype(np.float32)
    batch_exps = explainer.explain_batch(spectra_batch, labels=np.array([0, 1, 2]))
    assert len(batch_exps) == 3
    assert batch_exps[0].explained_class == 0

    # 3. Test visualization utilities save output files properly
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Test individual plot
        save_path = tmp_path / "lime_explanation.png"
        plot_lime_explanation(
            explanation=explanation,
            save_path=save_path,
            stage_label="Test Stage",
            split_label="test_split",
        )
        assert save_path.exists()
        assert save_path.stat().st_size > 0

        # Test comparison plot
        comp_path = tmp_path / "lime_comparison.png"
        plot_lime_comparison(
            explanations=[explanation, explanation],
            save_path=comp_path,
            title="LIME Comparison Unit Test",
        )
        assert comp_path.exists()
        assert comp_path.stat().st_size > 0


def test_lime_explain_cli():
    from unittest.mock import patch, MagicMock

    # Generate mock config
    mock_cfg = {
        "preprocessing": {
            "pipeline": ["per_sample_mean_subtraction"],
            "per_sample_mean_subtraction": {"enabled": True}
        },
        "task": {
            "stage": "pretrain_30class",
            "name": "isolate_pretraining",
            "label_space": "isolate_space",
        },
        "model": {
            "name": "cnn",
            "signal_length": 32,
        },
        "splits": {
            "test": {"label_space": "isolate_space"},
        }
    }

    # Mock DataRegistry
    mock_registry = MagicMock()
    mock_registry.get_arrays.side_effect = lambda split, allow_holdout=False: (
        np.random.rand(10, 32).astype(np.float32),
        np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
    )

    # Mock Model
    mock_model = DummyModel(input_len=32, n_classes=3)

    # Use temporary directory for cli outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        test_args = [
            "lime_explain.py",
            "--exp-dir", tmpdir,
            "--split", "test",
            "--per-class", "2",
            "--n-samples", "10",
            "--n-background", "5",
        ]

        # Patch dependencies in scripts.lime_explain
        with patch("sys.argv", test_args), \
             patch("scripts.lime_explain.load_config", return_value=mock_cfg), \
             patch("scripts.lime_explain.DataRegistry", return_value=mock_registry), \
             patch("scripts.lime_explain.get_model", return_value=mock_model), \
             patch("scripts.lime_explain.load_best_model", return_value={"config": mock_cfg}), \
             patch("scripts.lime_explain.plot_lime_explanation") as mock_plot_exp, \
             patch("scripts.lime_explain.plot_lime_comparison") as mock_plot_comp:

            from scripts.lime_explain import main as cli_main
            cli_main()

            # Verify that individual explanation plots were generated
            assert mock_plot_exp.call_count > 0
            # Verify that comparison plots were generated
            assert mock_plot_comp.call_count > 0
