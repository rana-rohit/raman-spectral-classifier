"""
Compatibility shim for `src.xai.lime_visualization`.

This file previously contained the full plotting implementation. The code
has been moved to `src.xai.xai_visualization` to better reflect its
responsibility for rendering multiple explainability outputs (LIME,
gradient saliency, etc.).

To preserve backwards compatibility, we re-export the main helpers and
emit a deprecation warning when this module is imported.
"""

from __future__ import annotations

import warnings

from src.xai.xai_visualization import (
    plot_lime_explanation,
    plot_lime_comparison,
)

warnings.warn(
    "src.xai.lime_visualization is deprecated; use src.xai.xai_visualization instead",
    DeprecationWarning,
)

__all__ = ["plot_lime_explanation", "plot_lime_comparison"]
