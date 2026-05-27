"""
src/xai/predict_wrapper.py

Stage-aware prediction wrapper for XAI methods (LIME, SHAP, etc.).

Converts raw numpy spectra into class probabilities by:
    1. Applying the fitted SpectralPreprocessor
    2. Converting to a PyTorch tensor
    3. Running model inference
    4. Applying softmax to logits

This wrapper exists so that XAI tools (which expect a simple
numpy-in → numpy-out function) can interface with the full
preprocessing + model pipeline without modifying any existing code.

IMPORTANT:
    This module does NOT alter model behavior, training, or
    evaluation logic. It is a read-only inference adapter.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from src.data.preprocessing import SpectralPreprocessor


class SpectralPredictWrapper:
    """
    Wraps a trained PyTorch model + preprocessor into a single
    callable that maps raw spectra → class probabilities.

    Parameters
    ----------
    model : nn.Module
        Trained model in eval mode.

    preprocessor : SpectralPreprocessor
        Fitted preprocessor (must have been fit on training data).

    device : str or torch.device
        Device for inference.

    batch_size : int
        Internal batch size for processing large arrays efficiently.
        LIME may pass thousands of perturbed samples at once.
    """

    def __init__(
        self,
        model: nn.Module,
        preprocessor: SpectralPreprocessor,
        device: str | torch.device = "cpu",
        batch_size: int = 256,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for raw spectra.

        Parameters
        ----------
        X : np.ndarray
            Raw (unpreprocessed) spectra of shape (N, L).

        Returns
        -------
        np.ndarray
            Class probabilities of shape (N, n_classes).

        Flow
        ----
        X (N, L) raw
          → preprocessor.transform → (N, C, L) where C ∈ {1, 2}
            → model.forward → logits (N, n_classes)
              → softmax → probabilities (N, n_classes)
        """
        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X[np.newaxis, :]

        # Apply the fitted preprocessing pipeline
        # This produces (N, C, L) with C channels (raw + derivative)
        X_processed = self.preprocessor.transform(X)

        # Run inference in batches to handle LIME's large perturbation sets
        all_probs = []
        n_samples = len(X_processed)

        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch = torch.from_numpy(
                X_processed[start:end].astype(np.float32)
            ).to(self.device)

            outputs = self.model(batch)

            # Handle both dict and tensor outputs
            if isinstance(outputs, dict):
                logits = outputs["main_logits"]
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)


def build_predict_fn(
    model: nn.Module,
    preprocessor: SpectralPreprocessor,
    device: str = "cpu",
    batch_size: int = 256,
) -> SpectralPredictWrapper:
    """
    Convenience factory for creating a prediction wrapper.

    Returns a callable: (N, L) raw numpy → (N, n_classes) probabilities.
    """
    return SpectralPredictWrapper(
        model=model,
        preprocessor=preprocessor,
        device=device,
        batch_size=batch_size,
    )
