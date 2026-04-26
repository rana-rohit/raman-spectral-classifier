"""
src/data/preprocessing.py

Composable preprocessing pipeline for 1D spectral signals.

Critical design rule:
    preprocessor.fit() is called ONLY on the reference (train) split.
    preprocessor.transform() is then applied identically to all other splits.
    This prevents any form of data leakage.

Each transform follows the sklearn convention:
    fit(X)         -> self
    transform(X)   -> X_transformed
    fit_transform  -> shorthand

Usage:
    pipe = SpectralPreprocessor.from_config(cfg["preprocessing"])
    pipe.fit(X_train)
    X_train_clean = pipe.transform(X_train)
    X_clinical    = pipe.transform(X_clinical)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from typing import List, Optional


# ============================================================ #
#  Individual transforms
# ============================================================ #

class PerSampleMeanSubtraction:
    """
    Subtract each signal's own mean.
    Removes baseline offset — the primary domain-shift mechanism we
    observed between source (mean ~0.43) and clinical (mean ~0.46) splits.
    Applied per-sample so no training statistics are needed.
    """

    def fit(self, X: np.ndarray) -> "PerSampleMeanSubtraction":
        return self  # Stateless — no fitting needed

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X - X.mean(axis=1, keepdims=True)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

def per_sample_normalize(X):
    """
    Normalize each spectrum independently.
    
    Args:
        X: numpy array of shape (N, L)
    
    Returns:
        normalized X
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std

def compute_first_derivative(X):
    """
    Compute first derivative using numpy gradient
    """
    return np.gradient(X, axis=1)

class SNVNormalization: 
    """
    Standard Normal Variate: per-sample centering + scaling.
    Removes multiplicative scatter AND additive baseline offset
    simultaneously — significantly more robust than mean subtraction
    alone for cross-session/cross-instrument generalization.
    """

    def fit(self, X: np.ndarray) -> "SNVNormalization":
        return self  # Stateless

    def transform(self, X: np.ndarray) -> np.ndarray:
        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)
        stds[stds < 1e-8] = 1.0
        return (X - means) / stds

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class FirstDerivative:
    """
    Savitzky-Golay first derivative — inherently invariant to additive
    baseline offsets and reduces sensitivity to slow-varying instrument
    artifacts.  Standard practice in chemometrics / spectroscopy.
    """

    def __init__(self, window_length: int = 11, polyorder: int = 3) -> None:
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if polyorder >= window_length:
            raise ValueError("polyorder must be < window_length")
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, X: np.ndarray) -> "FirstDerivative":
        return self  # Stateless

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(
            lambda sig: savgol_filter(
                sig, self.window_length, self.polyorder, deriv=1
            ),
            axis=1, arr=X,
        )

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class SavitzkyGolaySmoothing:
    """
    Smooth signals using a Savitzky-Golay filter.
    Reduces high-frequency noise while preserving peak shapes
    better than a simple moving average.
    window_length must be odd and > polyorder.
    """

    def __init__(self, window_length: int = 11, polyorder: int = 3) -> None:
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if polyorder >= window_length:
            raise ValueError("polyorder must be < window_length")
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, X: np.ndarray) -> "SavitzkyGolaySmoothing":
        return self  # Stateless

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(
            lambda sig: savgol_filter(sig, self.window_length, self.polyorder),
            axis=1, arr=X
        )

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class GlobalStandardisation:
    """
    Z-score standardisation using statistics fit on training data only.
    Optional — only needed if model training shows instability.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "GlobalStandardisation":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0  # Avoid division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("GlobalStandardisation must be fit before transform.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class ClipTransform:
    """Clip extreme values after all other transforms."""

    def __init__(self, min_val: float = -3.0, max_val: float = 3.0) -> None:
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X: np.ndarray) -> "ClipTransform":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.min_val, self.max_val)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ============================================================ #
#  Pipeline
# ============================================================ #

# Registry of available transforms
_TRANSFORM_REGISTRY = {
    "per_sample_mean_subtraction": PerSampleMeanSubtraction,
    "snv_normalization":           SNVNormalization,
    "first_derivative":            FirstDerivative,
    "savitzky_golay":              SavitzkyGolaySmoothing,
    "global_standardisation":      GlobalStandardisation,
    "clip":                        ClipTransform,
}


class SpectralPreprocessor:
    """
    Ordered pipeline of spectral transforms.
    Only stateful transforms (GlobalStandardisation) need fit().
    """

    def __init__(self, transforms: List) -> None:
        self.transforms = transforms
        self._is_fit = False

    @classmethod
    def from_config(cls, cfg: dict) -> "SpectralPreprocessor":
        pipeline_names = cfg["pipeline"]
        transforms = []
        for name in pipeline_names:
            if not cfg.get(name, {}).get("enabled", True):
                continue
            transform_cls = _TRANSFORM_REGISTRY.get(name)
            if transform_cls is None:
                raise ValueError(f"Unknown transform '{name}'. "
                                 f"Available: {list(_TRANSFORM_REGISTRY)}")
            step_cfg = cfg.get(name, {})
            # Pass config params, ignoring 'enabled' key
            params = {k: v for k, v in step_cfg.items() if k != "enabled"}
            transforms.append(transform_cls(**params))

        return cls(transforms)

    def fit(self, X: np.ndarray) -> "SpectralPreprocessor":
        """Fit all stateful transforms on training data ONLY."""
        for t in self.transforms:
            t.fit(X)
            # Apply transform so subsequent steps see the same distribution
            X = t.transform(X)
        self._is_fit = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            X = t.transform(X)

        # Step 1: normalize
        X_norm = per_sample_normalize(X)

        # Step 2: derivative
        X_deriv = compute_first_derivative(X_norm)

        # Step 3: stack channels
        X = np.stack([X_norm, X_deriv], axis=1)  # (N, 2, L)

        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            t.fit(X)
            X = t.transform(X)

        self._is_fit = True

        X_norm = per_sample_normalize(X)
        X_deriv = compute_first_derivative(X_norm)

        X = np.stack([X_norm, X_deriv], axis=1)

        return X

    def __repr__(self) -> str:
        steps = " -> ".join(type(t).__name__ for t in self.transforms)
        return f"SpectralPreprocessor([{steps}])"
    
class BaselineCorrection:
    """Asymmetric least squares baseline subtraction."""
    def __init__(self, lam: float = 1e4, p: float = 0.01, n_iter: int = 10):
        self.lam = lam
        self.p = p
        self.n_iter = n_iter

    def fit(self, X): return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        out = np.empty_like(X)
        L = X.shape[1]
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L-2, L)).toarray()
        D = self.lam * D.T @ D
        for i, sig in enumerate(X):
            w = np.ones(L)
            for _ in range(self.n_iter):
                W = sparse.diags(w)
                Z = spsolve(W + D, w * sig)
                w = self.p * (sig > Z) + (1 - self.p) * (sig <= Z)
            out[i] = sig - Z
        return out

    def fit_transform(self, X): return self.fit(X).transform(X)