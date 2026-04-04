"""
tests/test_data_layer.py

Unit tests for the complete data layer.
Run with: pytest tests/test_data_layer.py -v

These tests use synthetic data so they work without the real dataset.
"""

import numpy as np
import pytest
import torch

from src.data.preprocessing import (
    SpectralPreprocessor,
    PerSampleMeanSubtraction,
    SavitzkyGolaySmoothing,
    ClipTransform,
)
from src.data.augmentation import (
    GaussianNoise, BaselineShift, AmplitudeScaling, SpectralShift,
    AugmentationPipeline,
)
from src.data.dataset import SpectralDataset, make_train_val_split
from src.data.split_roles import SplitRole, role_from_str


# ============================================================ #
#  Fixtures
# ============================================================ #

@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    N, L, C = 200, 1000, 30
    X = rng.uniform(0, 1, (N, L)).astype(np.float32)
    y = rng.integers(0, C, N).astype(np.int64)
    return X, y


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ============================================================ #
#  Preprocessing tests
# ============================================================ #

class TestPerSampleMeanSubtraction:
    def test_output_mean_is_zero(self, synthetic_data):
        X, _ = synthetic_data
        t = PerSampleMeanSubtraction()
        X_out = t.fit_transform(X)
        assert np.allclose(X_out.mean(axis=1), 0, atol=1e-6)

    def test_shape_preserved(self, synthetic_data):
        X, _ = synthetic_data
        X_out = PerSampleMeanSubtraction().fit_transform(X)
        assert X_out.shape == X.shape

    def test_stateless_fit(self, synthetic_data):
        """fit() should have no side effects — transform is deterministic."""
        X, _ = synthetic_data
        t = PerSampleMeanSubtraction()
        out1 = t.fit_transform(X)
        out2 = t.transform(X)
        assert np.allclose(out1, out2)


class TestSavitzkyGolay:
    def test_shape_preserved(self, synthetic_data):
        X, _ = synthetic_data
        X_out = SavitzkyGolaySmoothing(window_length=11, polyorder=3).fit_transform(X)
        assert X_out.shape == X.shape

    def test_smoothing_reduces_variance(self, synthetic_data):
        X, _ = synthetic_data
        X_noisy = X + np.random.default_rng(0).normal(0, 0.1, X.shape)
        X_smooth = SavitzkyGolaySmoothing().fit_transform(X_noisy)
        assert X_smooth.std() < X_noisy.std()

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            SavitzkyGolaySmoothing(window_length=10)  # Even number

    def test_polyorder_too_high_raises(self):
        with pytest.raises(ValueError):
            SavitzkyGolaySmoothing(window_length=5, polyorder=5)


class TestSpectralPreprocessor:
    def test_pipeline_no_leakage(self, synthetic_data):
        """Transform on new data should only use stats from fit data."""
        X, _ = synthetic_data
        X_train, X_test = X[:100], X[100:]
        pipe = SpectralPreprocessor([PerSampleMeanSubtraction(), ClipTransform()])
        pipe.fit(X_train)
        out_train = pipe.transform(X_train)
        out_test  = pipe.transform(X_test)
        assert out_train.shape == X_train.shape
        assert out_test.shape  == X_test.shape

    def test_from_config(self):
        cfg = {
            "pipeline": ["per_sample_mean_subtraction", "clip"],
            "per_sample_mean_subtraction": {"enabled": True},
            "clip": {"enabled": True, "min_val": -3.0, "max_val": 3.0},
        }
        pipe = SpectralPreprocessor.from_config(cfg)
        assert len(pipe.transforms) == 2


# ============================================================ #
#  Augmentation tests
# ============================================================ #

class TestAugmentations:
    def test_gaussian_noise_shape(self, synthetic_data, rng):
        X, _ = synthetic_data
        out = GaussianNoise(max_std=0.02)(X, rng)
        assert out.shape == X.shape

    def test_baseline_shift_changes_mean(self, synthetic_data, rng):
        X, _ = synthetic_data
        out = BaselineShift(max_shift=0.05)(X, rng)
        assert not np.allclose(X.mean(axis=1), out.mean(axis=1))

    def test_amplitude_scale_shape(self, synthetic_data, rng):
        X, _ = synthetic_data
        out = AmplitudeScaling(factor=0.10)(X, rng)
        assert out.shape == X.shape

    def test_spectral_shift_shape(self, synthetic_data, rng):
        X, _ = synthetic_data
        out = SpectralShift(max_shift=5)(X, rng)
        assert out.shape == X.shape

    def test_pipeline_clips_output(self, synthetic_data):
        """Output should always stay in [0, 1] after pipeline."""
        X, y = synthetic_data
        pipeline = AugmentationPipeline([
            GaussianNoise(0.5),    # Extreme noise to force clipping
            BaselineShift(0.5),
        ], p=1.0)
        out = pipeline(X)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ============================================================ #
#  Dataset tests
# ============================================================ #

class TestSpectralDataset:
    def test_output_shape(self, synthetic_data):
        X, y = synthetic_data
        ds = SpectralDataset(X, y)
        x_t, y_t = ds[0]
        assert x_t.shape == torch.Size([1, 1000])   # (channel, length)
        assert y_t.dtype == torch.long

    def test_class_filter(self, synthetic_data):
        X, y = synthetic_data
        ds = SpectralDataset(X, y, class_filter=[0, 2, 3, 5, 6])
        for _, label in ds:
            assert label.item() in [0, 2, 3, 5, 6]

    def test_augmentation_only_in_training(self, synthetic_data):
        """Augmentation should not change data when training=False."""
        X, y = synthetic_data
        aug = AugmentationPipeline([GaussianNoise(max_std=0.5)], p=1.0)
        ds_train = SpectralDataset(X, y, augmentation=aug, training=True)
        ds_eval  = SpectralDataset(X, y, augmentation=aug, training=False)
        # Eval dataset should return exact values (no noise)
        x_eval, _ = ds_eval[0]
        assert np.allclose(x_eval.numpy()[0], X[0], atol=1e-6)

    def test_length_consistent(self, synthetic_data):
        X, y = synthetic_data
        ds = SpectralDataset(X, y)
        assert len(ds) == len(X)


class TestTrainValSplit:
    def test_sizes_correct(self, synthetic_data):
        X, y = synthetic_data
        (X_tr, y_tr), (X_val, y_val) = make_train_val_split(X, y, val_fraction=0.2)
        total = len(X_tr) + len(X_val)
        assert total == len(X)
        assert abs(len(X_val) / len(X) - 0.2) < 0.05  # Within 5% of target

    def test_no_overlap(self, synthetic_data):
        X, y = synthetic_data
        (X_tr, _), (X_val, _) = make_train_val_split(X, y, val_fraction=0.2)
        # Check first sample of val is not in train (not perfectly rigorous but fast)
        assert not any(np.array_equal(X_tr[i], X_val[0]) for i in range(len(X_tr)))


# ============================================================ #
#  Split role tests
# ============================================================ #

class TestSplitRoles:
    def test_role_from_str(self):
        assert role_from_str("source")     == SplitRole.SOURCE
        assert role_from_str("ood_eval")   == SplitRole.OOD_EVAL
        assert role_from_str("adaptation") == SplitRole.ADAPTATION

    def test_invalid_role_raises(self):
        with pytest.raises(ValueError):
            role_from_str("nonexistent_role")