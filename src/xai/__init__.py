"""
src/xai/__init__.py

Project-wide explainability (XAI) interface.
Exposes saliency methods, prediction wrappers, LIME explainers, and visualizations.
"""

from src.xai.saliency import compute_saliency, compute_smoothgrad
from src.xai.predict_wrapper import SpectralPredictWrapper, build_predict_fn
from src.xai.xai_visualization import plot_lime_explanation, plot_lime_comparison

try:
    from src.xai.lime_explainer import SpectralLimeExplainer, SpectralLimeExplanation
except ImportError:
    SpectralLimeExplainer = None
    SpectralLimeExplanation = None

__all__ = [
    "compute_saliency",
    "compute_smoothgrad",
    "SpectralPredictWrapper",
    "build_predict_fn",
    "SpectralLimeExplainer",
    "SpectralLimeExplanation",
    "plot_lime_explanation",
    "plot_lime_comparison",
]
