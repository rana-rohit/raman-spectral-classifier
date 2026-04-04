"""
src/interpretability/gradcam1d.py

1D Grad-CAM (Gradient-weighted Class Activation Mapping) for spectral signals.

Grad-CAM highlights which spectral positions most influenced a given
class prediction. For spectral data this directly answers:
  "Which wavelength/frequency ranges does the model use to distinguish class X?"

Algorithm:
  1. Forward pass → logits
  2. Backprop gradient of class score w.r.t. final conv feature maps
  3. Global-average the gradients to get per-channel importance weights
  4. Weighted sum of feature maps → raw CAM
  5. ReLU + normalise to [0, 1]
  6. Upsample to original signal length for overlay

Works with:
  - CNN1D      (uses model.get_feature_maps())
  - ResNet1D   (uses model.get_feature_maps())
  - HybridCNNTransformer (uses model.get_cnn_features())

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks"
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM1D:
    """
    Computes Grad-CAM saliency maps for 1D spectral signals.

    Usage:
        gcam = GradCAM1D(model)
        cam = gcam.compute(x, target_class=5)   # cam shape: (L,)
        # Overlay cam on the original signal for visualisation
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks: list = []

    def _register_hooks(self, feature_extractor: nn.Module) -> None:
        """Register forward and backward hooks on the feature extractor."""
        self._remove_hooks()

        def save_activation(module, inp, out):
            self._activations = out.detach()

        def save_gradient(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hooks.append(
            feature_extractor.register_forward_hook(save_activation)
        )
        self._hooks.append(
            feature_extractor.register_backward_hook(save_gradient)  # type: ignore
        )

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def compute(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        signal_length: int = 1000,
    ) -> np.ndarray:
        """
        Compute 1D Grad-CAM for a single input signal.

        Args:
            x:             Input tensor, shape (1, 1, L) — single sample
            target_class:  Class to explain. If None, uses the predicted class.
            signal_length: Length to upsample CAM to (default: original signal L)

        Returns:
            cam: np.ndarray of shape (L,), values in [0, 1]
                 Higher values = more important for the predicted/target class
        """
        assert x.shape[0] == 1, "GradCAM1D expects a single sample (batch size 1)"

        # Get the right feature extractor for this model type
        feature_extractor = self._get_feature_extractor()
        self._register_hooks(feature_extractor)

        self.model.eval()
        x = x.requires_grad_(True)

        # Forward pass
        logits = self.model(x)

        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        # Backward pass for target class only
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Grad-CAM computation
        gradients  = self._gradients   # (1, C, T)
        activations = self._activations  # (1, C, T)

        if gradients is None or activations is None:
            raise RuntimeError(
                "Grad-CAM hooks did not fire. "
                "Ensure the model calls get_feature_maps() in its forward pass."
            )

        # Global average pool gradients → importance weights per channel
        weights = gradients.mean(dim=-1, keepdim=True)  # (1, C, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1)          # (1, T)
        cam = F.relu(cam)                                 # Keep only positive influence

        # Normalise to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        # Upsample to original signal length
        cam = F.interpolate(
            cam.unsqueeze(0),                              # (1, 1, T)
            size=signal_length,
            mode="linear",
            align_corners=False,
        ).squeeze().detach().numpy()                       # (L,)

        self._remove_hooks()
        return cam

    def compute_batch(
        self,
        X: torch.Tensor,
        target_classes: Optional[list] = None,
    ) -> np.ndarray:
        """
        Compute Grad-CAM for a batch of signals.
        Returns array of shape (N, L).
        """
        cams = []
        for i in range(len(X)):
            tc = target_classes[i] if target_classes is not None else None
            cam = self.compute(X[i:i+1], target_class=tc, signal_length=X.shape[-1])
            cams.append(cam)
        return np.stack(cams, axis=0)

    def _get_feature_extractor(self) -> nn.Module:
        """
        Returns the module that produces the final conv feature maps.
        This is the target of both hooks.
        """
        model = self.model

        # CNN1D / ResNet1D — use the features/stage4 submodule
        if hasattr(model, "features"):
            return model.features
        if hasattr(model, "stage4"):
            return model.stage4
        # Hybrid — CNN stem is the feature extractor
        if hasattr(model, "cnn_stem"):
            return model.cnn_stem

        raise AttributeError(
            "Cannot determine feature extractor for Grad-CAM. "
            "Model must have 'features', 'stage4', or 'cnn_stem' attribute."
        )