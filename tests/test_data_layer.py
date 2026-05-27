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
from src.data.dataloader import build_all_loaders
from src.data.split_roles import SplitRole, role_from_str
from src.utils.class_subset import filter_and_remap_classes
from src.utils.split_modes import canonicalize_split_mode_config, resolve_split_mode


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
        assert out_train.shape == (len(X_train), 2, X_train.shape[1])
        assert out_test.shape  == (len(X_test), 2, X_test.shape[1])

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
        preprocessor = SpectralPreprocessor([]).fit(X)
        ds = SpectralDataset(X, y, preprocessor=preprocessor)
        x_t, y_t = ds[0]
        assert x_t.shape == torch.Size([2, 1000])   # (channels, length)
        assert y_t.dtype == torch.long

    def test_class_filter(self, synthetic_data):
        X, y = synthetic_data
        class_filter = [0, 2, 3, 5, 6]
        X, y = filter_and_remap_classes(X, y, class_filter)
        preprocessor = SpectralPreprocessor([]).fit(X)
        ds = SpectralDataset(
            X,
            y,
            preprocessor=preprocessor,
            expected_n_classes=len(class_filter),
            class_filter=class_filter,
        )
        for _, label in ds:
            assert label.item() in range(len(class_filter))

    def test_augmentation_only_in_training(self, synthetic_data):
        """Augmentation should not change data when training=False."""
        X, y = synthetic_data
        aug = AugmentationPipeline([GaussianNoise(max_std=0.5)], p=1.0)
        preprocessor = SpectralPreprocessor([]).fit(X)
        ds_train = SpectralDataset(X, y, augmentation=aug, training=True, preprocessor=preprocessor)
        ds_eval  = SpectralDataset(X, y, augmentation=aug, training=False, preprocessor=preprocessor)
        # Eval dataset should return exact values (no noise)
        x_eval, _ = ds_eval[0]
        expected = preprocessor.transform(X[:1])[0]
        assert np.allclose(x_eval.numpy(), expected, atol=1e-6)

    def test_length_consistent(self, synthetic_data):
        X, y = synthetic_data
        ds = SpectralDataset(X, y)
        assert len(ds) == len(X)

    def test_sparse_membership_validation_allows_semantic_ood_labels(self):
        X = np.random.default_rng(0).normal(size=(10, 32)).astype(np.float32)
        y = np.array([0, 2, 3, 5, 6] * 2, dtype=np.int64)
        ds = SpectralDataset(
            X,
            y,
            label_validation="membership",
            valid_label_ids=[0, 2, 3, 5, 6],
        )
        assert sorted(np.unique(ds.y).tolist()) == [0, 2, 3, 5, 6]


class _FakeRegistry:
    def __init__(self) -> None:
        rng = np.random.default_rng(1)
        self.calls = []
        self.arrays = {
            "reference": (
                rng.normal(size=(60, 32)).astype(np.float32),
                np.tile(np.arange(30, dtype=np.int64), 2),
            ),
            "test": (
                rng.normal(size=(60, 32)).astype(np.float32),
                np.tile(np.arange(30, dtype=np.int64), 2),
            ),
            "2018clinical": (
                rng.normal(size=(50, 32)).astype(np.float32),
                np.array([0, 2, 3, 5, 6] * 10, dtype=np.int64),
            ),
        }
        self.cfg = {
            "splits": {
                "reference": {"label_space": "isolate_space"},
                "test": {"label_space": "isolate_space"},
                "2018clinical": {"label_space": "sparse_global_treatment_space"},
            }
        }

    def get_arrays(self, split_name, allow_holdout=False):
        self.calls.append((split_name, allow_holdout))
        return self.arrays[split_name]

    def ood_split_names(self):
        return ["2018clinical"]


def test_stage1_loader_does_not_construct_clinical_ood_loaders():
    registry = _FakeRegistry()
    preprocessor = SpectralPreprocessor([]).fit(registry.arrays["reference"][0])
    cfg = {
        "validation": {
            "val_fraction": 0.2,
            "random_seed": 42,
            "clinical_val_fraction": 0.2,
            "clinical_eval_fraction": 0.2,
        },
        "task": {
            "stage": "pretrain_30class",
            "label_space": "isolate_space",
        },
        "training": {"batch_size": 8, "num_workers": 0},
    }

    loaders = build_all_loaders(
        registry,
        preprocessor,
        augmentation=None,
        cfg=cfg,
        clinical_sparse_ids=[],
        n_classes=30,
    )

    assert set(loaders) == {"train", "source_val", "val", "test", "ood"}
    assert loaders["ood"] == {}
    assert "clinical_train" not in loaders
    assert "clinical_val" not in loaders
    assert loaders["val"] is loaders["source_val"]


class _ReferenceOnlyRegistry:
    def __init__(self) -> None:
        rng = np.random.default_rng(2)
        self.calls = []
        self.arrays = {
            "reference": (
                rng.normal(size=(300, 32)).astype(np.float32),
                np.repeat(np.arange(30, dtype=np.int64), 10),
            ),
        }
        self.cfg = {
            "splits": {
                "reference": {"label_space": "isolate_space"},
                "test": {"label_space": "isolate_space"},
            }
        }

    def get_arrays(self, split_name, allow_holdout=False):
        self.calls.append((split_name, allow_holdout))
        if split_name == "test":
            raise AssertionError("iid_reference mode must not request the holdout test split")
        return self.arrays[split_name]

    def ood_split_names(self):
        return []


def test_iid_reference_mode_uses_reference_only_stratified_splits():
    registry = _ReferenceOnlyRegistry()
    preprocessor = SpectralPreprocessor([]).fit(registry.arrays["reference"][0])
    cfg = {
        "split_mode": "iid_reference",
        "validation": {
            "val_fraction": 0.2,
            "random_seed": 123,
            "iid_reference": {
                "train_fraction": 0.70,
                "val_fraction": 0.15,
                "test_fraction": 0.15,
                "random_seed": 123,
            },
        },
        "task": {
            "stage": "pretrain_30class",
            "label_space": "isolate_space",
        },
        "training": {
            "split_mode": "iid_reference",
            "batch_size": 16,
            "num_workers": 0,
        },
    }

    loaders = build_all_loaders(
        registry,
        preprocessor,
        augmentation=None,
        cfg=cfg,
        clinical_sparse_ids=[],
        n_classes=30,
    )

    assert set(loaders) == {"train", "source_val", "val", "test", "ood"}
    assert loaders["val"] is loaders["source_val"]
    assert len(loaders["train"].dataset) == 210
    assert len(loaders["source_val"].dataset) == 45
    assert len(loaders["test"].dataset) == 45
    assert all(call[0] != "test" for call in registry.calls)
    assert all(
        str(sample_id).startswith("reference:")
        for sample_id in loaders["test"].dataset.sample_ids
    )

    train_counts = loaders["train"].dataset.class_counts
    assert set(train_counts.values()) == {7}

    for split_name in ("source_val", "test"):
        counts = loaders[split_name].dataset.class_counts
        assert set(counts) == set(range(30))
        assert max(counts.values()) - min(counts.values()) <= 1

    registry_2 = _ReferenceOnlyRegistry()
    loaders_2 = build_all_loaders(
        registry_2,
        preprocessor,
        augmentation=None,
        cfg=cfg,
        clinical_sparse_ids=[],
        n_classes=30,
    )

    assert loaders["train"].dataset.sample_ids.tolist() == loaders_2["train"].dataset.sample_ids.tolist()
    assert loaders["source_val"].dataset.sample_ids.tolist() == loaders_2["source_val"].dataset.sample_ids.tolist()
    assert loaders["test"].dataset.sample_ids.tolist() == loaders_2["test"].dataset.sample_ids.tolist()


def test_validation_split_mode_override_beats_training_default():
    cfg = {
        "training": {"split_mode": "holdout"},
        "validation": {"split_mode": "iid_reference"},
    }

    assert resolve_split_mode(cfg) == "iid_reference"

    mode = canonicalize_split_mode_config(cfg)
    assert mode == "iid_reference"
    assert cfg["split_mode"] == "iid_reference"
    assert cfg["training"]["split_mode"] == "iid_reference"
    assert cfg["validation"]["split_mode"] == "iid_reference"


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
