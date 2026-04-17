"""
src/data/augmentation.py

Spectral data augmentations applied only during training.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d


class GaussianNoise:
    def __init__(self, max_std: float = 0.02) -> None:
        self.max_std = max_std

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        std = rng.uniform(0, self.max_std, size=(len(X), 1))
        return X + rng.normal(0, 1, size=X.shape) * std


class BaselineShift:
    def __init__(self, max_shift: float = 0.05) -> None:
        self.max_shift = max_shift

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        shift = rng.uniform(-self.max_shift, self.max_shift, size=(len(X), 1))
        return X + shift


class AmplitudeScaling:
    def __init__(self, factor: float = 0.10) -> None:
        self.factor = factor

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        scale = rng.uniform(1 - self.factor, 1 + self.factor, size=(len(X), 1))
        return X * scale


class MultiplicativeIntensityScale:
    def __init__(self, factor: float = 0.20) -> None:
        self.factor = factor

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        scale = rng.uniform(1 - self.factor, 1 + self.factor, size=(len(X), 1))
        return X * scale


class SpectralShift:
    def __init__(self, max_shift: int = 5) -> None:
        self.max_shift = max_shift

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        result = np.empty_like(X)
        for i in range(len(X)):
            shift = int(rng.integers(-self.max_shift, self.max_shift + 1))
            result[i] = np.roll(X[i], shift)
        return result


class BaselineDrift:
    def __init__(self, max_strength: float = 0.10, n_control_points: int = 5) -> None:
        self.max_strength = max_strength
        self.n_control_points = max(2, int(n_control_points))

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        length = X.shape[1]
        control_x = np.linspace(0, length - 1, self.n_control_points)
        full_x = np.arange(length)

        result = np.empty_like(X)
        for i, spectrum in enumerate(X):
            control_y = rng.uniform(
                -self.max_strength,
                self.max_strength,
                size=self.n_control_points,
            )
            drift = np.interp(full_x, control_x, control_y)
            result[i] = spectrum + drift
        return result


class PeakBroadening:
    def __init__(self, max_sigma: float = 1.5) -> None:
        self.max_sigma = max_sigma

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        result = np.empty_like(X)
        for i, spectrum in enumerate(X):
            sigma = float(rng.uniform(0.0, self.max_sigma))
            result[i] = gaussian_filter1d(spectrum, sigma=sigma, mode="nearest")
        return result


class NonlinearSpectralWarp:
    def __init__(self, max_shift: float = 4.0, n_control_points: int = 6) -> None:
        self.max_shift = max_shift
        self.n_control_points = max(2, int(n_control_points))

    def __call__(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        length = X.shape[1]
        control_x = np.linspace(0, length - 1, self.n_control_points)
        full_x = np.arange(length)

        result = np.empty_like(X)
        for i, spectrum in enumerate(X):
            offsets = rng.uniform(-self.max_shift, self.max_shift, size=self.n_control_points)
            warped = control_x + offsets
            warped[0] = 0.0
            warped[-1] = float(length - 1)
            warped = np.maximum.accumulate(warped)
            warped[-1] = float(length - 1)

            source_positions = np.interp(full_x, control_x, warped)
            result[i] = np.interp(
                full_x,
                source_positions,
                spectrum,
                left=spectrum[0],
                right=spectrum[-1],
            )
        return result


class Mixup:
    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha

    def __call__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rng: np.random.Generator,
    ):
        lam = rng.beta(self.alpha, self.alpha)
        idx = rng.permutation(len(X))
        X_mix = lam * X + (1 - lam) * X[idx]
        y_a, y_b = y, y[idx]
        return X_mix, y_a, y_b, lam


_AUG_REGISTRY = {
    "gaussian_noise": GaussianNoise,
    "baseline_shift": BaselineShift,
    "amplitude_scale": AmplitudeScaling,
    "multiplicative_intensity": MultiplicativeIntensityScale,
    "spectral_shift": SpectralShift,
    "baseline_drift": BaselineDrift,
    "peak_broadening": PeakBroadening,
    "nonlinear_warp": NonlinearSpectralWarp,
}


class AugmentationPipeline:
    def __init__(
        self,
        steps: list,
        p: float = 0.5,
        seed: int = 42,
        clip_min: float | None = 0.0,
        clip_max: float | None = 1.0,
    ) -> None:
        self.steps = steps
        self.p = p
        self._rng = np.random.default_rng(seed)
        self.clip_min = clip_min
        self.clip_max = clip_max

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

        return cls(
            steps=steps,
            p=cfg.get("apply_probability", 0.5),
            seed=cfg.get("seed", 42),
            clip_min=cfg.get("clip_min", 0.0),
            clip_max=cfg.get("clip_max", 1.0),
        )

    def __call__(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        for step in self.steps:
            if self._rng.random() < self.p:
                if isinstance(step, Mixup) and y is not None:
                    X, *_ = step(X, y, self._rng)
                else:
                    X = step(X, self._rng)

        if self.clip_min is not None and self.clip_max is not None:
            return np.clip(X, self.clip_min, self.clip_max)
        return X
