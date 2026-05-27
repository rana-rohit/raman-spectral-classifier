"""
src/xai/lime_explainer.py

LIME (Local Interpretable Model-agnostic Explanations) for 1D Raman spectra.

Wraps lime.lime_tabular.LimeTabularExplainer with spectral-domain semantics:
    - Feature names map to wavenumber positions
    - Perturbation operates on preprocessed spectral intensities
    - Explanations are returned as per-wavenumber importance arrays

Works identically for:
    - Stage 1 (pretrain_30class)     — isolate-space labels
    - Stage 2 (pretrain_treatment_8class)  — treatment-space labels
    - Stage 3 (transfer_5class)      — compact clinical transfer labels

IMPORTANT:
    This module does NOT modify any existing model, trainer, dataloader,
    or evaluation logic. It is a pure read-only wrapper around inference.

Usage:
    from src.xai.lime_explainer import SpectralLimeExplainer

    explainer = SpectralLimeExplainer(
        predict_fn=predict_wrapper,
        training_data=X_background,
        wavenumbers=wavenumbers,
        class_names=["Meropenem", "TZP", ...],
    )

    explanation = explainer.explain_sample(spectrum, label=2)
"""

from __future__ import annotations

import warnings
from typing import Callable, List, Optional

import numpy as np

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    raise ImportError(
        "LIME is required for this module. Install it with:\n"
        "  pip install lime"
    )


class SpectralLimeExplainer:
    """
    LIME explainer specialized for 1D spectral signals.

    Parameters
    ----------
    predict_fn : callable
        A function that accepts a numpy array of shape (N, L) and returns
        a probability array of shape (N, n_classes). This is the
        prediction wrapper that handles preprocessing and model inference.

    training_data : np.ndarray
        Background dataset of shape (N_background, L) used by LIME to
        learn the local data distribution for perturbation. Typically
        the RAW (unpreprocessed) reference spectra.

    wavenumbers : np.ndarray or None
        Wavenumber axis of shape (L,). If provided, feature names in
        explanations will be mapped to wavenumber positions (e.g.,
        "1003 cm⁻¹"). If None, integer indices are used.

    class_names : list[str] or None
        Human-readable class labels. Stage-aware: may be isolate names,
        treatment names, or compact clinical labels depending on context.

    n_features : int
        Number of top features LIME should report per explanation.

    n_samples : int
        Number of perturbation samples LIME generates around each query
        point. Higher = more stable but slower.

    random_state : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        training_data: np.ndarray,
        wavenumbers: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        n_features: int = 20,
        n_samples: int = 2000,
        random_state: int = 42,
    ) -> None:
        self.predict_fn = predict_fn
        self.n_features = n_features
        self.n_samples = n_samples
        self.random_state = random_state
        self.wavenumbers = wavenumbers
        self.class_names = class_names

        signal_length = training_data.shape[1]

        # Build feature names from wavenumbers or indices
        if wavenumbers is not None:
            if len(wavenumbers) != signal_length:
                raise ValueError(
                    f"wavenumbers length ({len(wavenumbers)}) must match "
                    f"signal length ({signal_length})"
                )
            self.feature_names = [
                f"{wn:.0f} cm⁻¹" for wn in wavenumbers
            ]
        else:
            self.feature_names = [
                f"idx_{i}" for i in range(signal_length)
            ]

        # Initialize the LIME tabular explainer with background data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._explainer = LimeTabularExplainer(
                training_data=training_data,
                feature_names=self.feature_names,
                class_names=class_names,
                mode="classification",
                random_state=random_state,
                discretize_continuous=False,
            )

    def explain_sample(
        self,
        spectrum: np.ndarray,
        label: Optional[int] = None,
        n_features: Optional[int] = None,
        n_samples: Optional[int] = None,
    ) -> "SpectralLimeExplanation":
        """
        Generate a LIME explanation for a single spectrum.

        Parameters
        ----------
        spectrum : np.ndarray
            Raw (unpreprocessed) 1D spectrum of shape (L,).

        label : int or None
            Class to explain. If None, explains the predicted class.

        n_features : int or None
            Override default n_features for this explanation.

        n_samples : int or None
            Override default n_samples for this explanation.

        Returns
        -------
        SpectralLimeExplanation
            Structured result containing importance weights, predicted
            class, probabilities, and metadata for visualization.
        """
        spectrum = np.asarray(spectrum, dtype=np.float64)
        if spectrum.ndim != 1:
            raise ValueError(
                f"Expected 1D spectrum, got shape {spectrum.shape}"
            )

        n_feat = n_features or self.n_features
        n_samp = n_samples or self.n_samples

        # Get model prediction for the original sample
        probs = self.predict_fn(spectrum[np.newaxis, :])[0]
        predicted_class = int(np.argmax(probs))

        explain_class = label if label is not None else predicted_class

        # Generate LIME explanation
        lime_exp = self._explainer.explain_instance(
            spectrum,
            self.predict_fn,
            labels=(explain_class,),
            num_features=n_feat,
            num_samples=n_samp,
        )

        # Extract per-feature importance weights
        feature_weights = lime_exp.as_list(label=explain_class)

        # Build full-length importance array
        signal_length = len(spectrum)
        importance = np.zeros(signal_length, dtype=np.float64)

        for feature_name, weight in feature_weights:
            idx = self._resolve_feature_index(feature_name)
            if idx is not None and 0 <= idx < signal_length:
                importance[idx] = weight

        return SpectralLimeExplanation(
            spectrum=spectrum,
            importance=importance,
            predicted_class=predicted_class,
            explained_class=explain_class,
            probabilities=probs,
            feature_weights=feature_weights,
            class_names=self.class_names,
            wavenumbers=self.wavenumbers,
            lime_explanation=lime_exp,
        )

    def explain_batch(
        self,
        spectra: np.ndarray,
        labels: Optional[np.ndarray] = None,
        n_features: Optional[int] = None,
        n_samples: Optional[int] = None,
    ) -> List["SpectralLimeExplanation"]:
        """
        Generate LIME explanations for a batch of spectra.

        Parameters
        ----------
        spectra : np.ndarray
            Array of shape (N, L) containing raw spectra.

        labels : np.ndarray or None
            Array of shape (N,) with classes to explain per sample.

        Returns
        -------
        list[SpectralLimeExplanation]
        """
        results = []
        for i in range(len(spectra)):
            lbl = int(labels[i]) if labels is not None else None
            exp = self.explain_sample(
                spectra[i],
                label=lbl,
                n_features=n_features,
                n_samples=n_samples,
            )
            results.append(exp)
        return results

    def _resolve_feature_index(self, feature_name: str) -> Optional[int]:
        """Map a LIME feature name back to an integer index."""
        try:
            return self.feature_names.index(feature_name)
        except ValueError:
            # Fallback: try parsing "idx_N" format
            if feature_name.startswith("idx_"):
                try:
                    return int(feature_name.split("_")[1])
                except (ValueError, IndexError):
                    pass
            return None


class SpectralLimeExplanation:
    """
    Structured container for a single LIME explanation result.

    Attributes
    ----------
    spectrum : np.ndarray
        Original raw spectrum (L,).
    importance : np.ndarray
        Per-wavenumber importance weights (L,). Positive = supports
        predicted class, negative = supports other classes.
    predicted_class : int
        Model's argmax prediction.
    explained_class : int
        The class being explained (may differ from predicted).
    probabilities : np.ndarray
        Full probability vector (n_classes,).
    feature_weights : list[tuple]
        LIME's (feature_name, weight) pairs for top features.
    class_names : list[str] or None
        Human-readable class names.
    wavenumbers : np.ndarray or None
        Wavenumber axis.
    lime_explanation : object
        The raw lime Explanation object for advanced usage.
    """

    def __init__(
        self,
        spectrum: np.ndarray,
        importance: np.ndarray,
        predicted_class: int,
        explained_class: int,
        probabilities: np.ndarray,
        feature_weights: list,
        class_names: Optional[List[str]],
        wavenumbers: Optional[np.ndarray],
        lime_explanation,
    ) -> None:
        self.spectrum = spectrum
        self.importance = importance
        self.predicted_class = predicted_class
        self.explained_class = explained_class
        self.probabilities = probabilities
        self.feature_weights = feature_weights
        self.class_names = class_names
        self.wavenumbers = wavenumbers
        self.lime_explanation = lime_explanation

    @property
    def predicted_label(self) -> str:
        if self.class_names and self.predicted_class < len(self.class_names):
            return self.class_names[self.predicted_class]
        return f"Class {self.predicted_class}"

    @property
    def explained_label(self) -> str:
        if self.class_names and self.explained_class < len(self.class_names):
            return self.class_names[self.explained_class]
        return f"Class {self.explained_class}"

    @property
    def confidence(self) -> float:
        return float(self.probabilities[self.predicted_class])

    @property
    def positive_importance(self) -> np.ndarray:
        return np.maximum(self.importance, 0.0)

    @property
    def negative_importance(self) -> np.ndarray:
        return np.minimum(self.importance, 0.0)

    def top_features(self, n: int = 10) -> list:
        """Return top N features sorted by absolute importance."""
        sorted_pairs = sorted(
            self.feature_weights,
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return sorted_pairs[:n]
