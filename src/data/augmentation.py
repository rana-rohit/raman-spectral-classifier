"""
src/data/augmentation.py

Spectral data augmentations — applied ONLY during training, never at eval time.

All augmentations are designed to produce physically plausible signals:
- Noise levels are bounded to realistic SNR ranges
- Shifts are small enough not to move peaks past neighbours
- Scaling stays within a range seen in real acquisition variation

Each augmentation is independently togglable via config.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class GaussianNoise:
    """
    Add zero-mean Gaussian noise. Simulates detector noise.
    std is sampled uniformly from [0, max_std] per batch element.
    """

    def __init__(self, max_std: float = 0.02) -> None:
        self.max_std = max_std

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        std = rng.uniform(0, self.max_std, size=(len(X), 1))
        return X + rng.normal(0, 1, size=X.shape) * std


class BaselineShift:
    """
    Add a random constant offset to each signal.
    Simulates the residual baseline variation between domains
    that we observed in EDA (source mean ~0.43, clinical ~0.46).
    """

    def __init__(self, max_shift: float = 0.05) -> None:
        self.max_shift = max_shift

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        shift = rng.uniform(-self.max_shift, self.max_shift, size=(len(X), 1))
        return X + shift


class AmplitudeScaling:
    """
    Multiply each signal by a random scalar in [1-factor, 1+factor].
    Simulates gain variation between acquisition sessions.
    """

    def __init__(self, factor: float = 0.10) -> None:
        self.factor = factor

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        scale = rng.uniform(1 - self.factor, 1 + self.factor, size=(len(X), 1))
        return X * scale


class SpectralShift:
    """
    Shift signals by a random number of positions (circular).
    Simulates small wavelength calibration differences between instruments.
    Keep max_shift small — large shifts move peaks past neighbours.
    """

    def __init__(self, max_shift: int = 5) -> None:
        self.max_shift = max_shift

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        result = np.empty_like(X)
        for i in range(len(X)):
            shift = int(rng.integers(-self.max_shift, self.max_shift + 1))
            result[i] = np.roll(X[i], shift)
        return result


class Mixup:
    """
    Mixup augmentation adapted for spectral signals.
    Interpolates between two signals from the same class.
    alpha controls the Beta distribution shape (higher = closer to 0.5 mix).
    """

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha

    def __call__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ):
        """
        Returns mixed (X, y_a, y_b, lam) for soft-label loss computation.
        The trainer must use: loss = lam * loss(y_a) + (1-lam) * loss(y_b)
        """
        lam = rng.beta(self.alpha, self.alpha)
        idx = rng.permutation(len(X))
        X_mix = lam * X + (1 - lam) * X[idx]
        y_a, y_b = y, y[idx]
        return X_mix, y_a, y_b, lam


# ============================================================ #
#  Composed augmentation pipeline
# ============================================================ #

_AUG_REGISTRY = {
    "gaussian_noise":  GaussianNoise,
    "baseline_shift":  BaselineShift,
    "amplitude_scale": AmplitudeScaling,
    "spectral_shift":  SpectralShift,
}


class AugmentationPipeline:
    """
    Sequential augmentation pipeline.
    Each step is applied with a given probability.
    Only call during training — never at evaluation time.
    """

    def __init__(self, steps: list, p: float = 0.5, seed: int = 42) -> None:
        self.steps = steps
        self.p = p          # Probability of applying each step
        self._rng = np.random.default_rng(seed)

    @classmethod
    def from_config(cls, cfg: dict) -> "AugmentationPipeline":
        steps = []
        for name, step_cfg in cfg.get("steps", {}).items():
            if not step_cfg.get("enabled", True):
                continue
            aug_cls = _AUG_REGISTRY.get(name)
            if aug_cls is None:
                raise ValueError(f"Unknown augmentation '{name}'")
            params = {k: v for k, v in step_cfg.items() if k != "enabled"}
            steps.append(aug_cls(**params))
        p   = cfg.get("apply_probability", 0.5)
        seed = cfg.get("seed", 42)
        return cls(steps, p, seed)

    def __call__(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply augmentations. y is only needed if Mixup is in the pipeline."""
        for step in self.steps:
            if self._rng.random() < self.p:
                if isinstance(step, Mixup) and y is not None:
                    X, *_ = step(X, y, self._rng)
                else:
                    X = step(X, self._rng)
        return np.clip(X, 0.0, 1.0)  # Keep in valid range after augmentation