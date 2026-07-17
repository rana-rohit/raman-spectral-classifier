"""
src/xai/xai_visualization.py

General-purpose explainability visualization utilities.

This module is a renamed, backwards-compatible successor to
`src/xai/lime_visualization.py`. It renders publication-quality figures for
both LIME explanations and gradient-based saliency maps.

Note: keep plotting-only behavior here; computation of importance maps
remains in `src/xai/saliency.py` and LIME construction in
`src/xai/lime_explainer.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - optional visualization dependency
    savgol_filter = None


# ------------------------------------------------------------------ #
#  Color scheme — research-quality, colorblind-friendly
# ------------------------------------------------------------------ #

_POSITIVE_COLOR = "#22C55E"  # Premium green — supports prediction
_POSITIVE_FILL_COLOR = "#86EFAC"  # Soft support fill
_POSITIVE_GLOW_COLOR = "#DCFCE7"  # Atmospheric support glow
_NEGATIVE_COLOR = "#F44336"  # Red — opposes prediction
_SPECTRUM_COLOR = "#212121"  # Near-black for spectrum line
_BACKGROUND_COLOR = "#FAFAFA"  # Light grey background
_GRID_COLOR = "#E0E0E0"  # Subtle grid


def _display_spectrum(signal: np.ndarray) -> np.ndarray:
    if savgol_filter is None:
        return signal

    if signal.size < 9:
        return signal

    window_length = 11 if signal.size >= 11 else 9
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < 5:
        return signal

    polyorder = 3 if window_length >= 11 else 2
    if signal.size <= polyorder:
        return signal

    return savgol_filter(signal, window_length, polyorder)


def plot_lime_explanation(
    explanation,
    save_path: str | Path,
    title: Optional[str] = None,
    stage_label: Optional[str] = None,
    split_label: Optional[str] = None,
    figsize: tuple = (14, 8),
    dpi: int = 300,
    show_top_features: int = 15,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    spectrum = explanation.spectrum
    importance = explanation.importance
    wavenumbers = explanation.wavenumbers
    x_axis = wavenumbers if wavenumbers is not None else np.arange(len(spectrum))
    x_label = "Wavenumber (cm⁻¹)" if wavenumbers is not None else "Spectral Index"

    if wavenumbers is not None:
        order = np.argsort(x_axis)
        x_axis = x_axis[order]
        spectrum = spectrum[order]
        importance = importance[order]

    if title is None:
        parts = [f"LIME Explanation — {explanation.explained_label}"]
        if stage_label:
            parts.append(f"[{stage_label}]")
        if split_label:
            parts.append(f"({split_label})")
        title = " ".join(parts)

    fig = plt.figure(figsize=figsize, facecolor="white")
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[3,2],
        width_ratios=[3,1],
        hspace=0.35,
        wspace=0.25,
    )

    ax_spectrum = fig.add_subplot(gs[0, :])
    ax_spectrum.set_facecolor(_BACKGROUND_COLOR)

    display_spectrum = _display_spectrum(spectrum)

    ax_spectrum.plot(
        x_axis,
        display_spectrum,
        color=_SPECTRUM_COLOR,
        linewidth=1.2,
        alpha=0.9,
        label="Raman Spectrum",
        zorder=3,
    )

    positive = np.maximum(importance, 0)
    negative = np.minimum(importance, 0)
    max_importance = float(np.abs(importance).max() or 1.0)

    if np.any(positive > 0):
        pos_indices = np.flatnonzero(positive > 0)
        for idx in pos_indices:
            x_center = float(x_axis[idx])
            strength = float(positive[idx] / max_importance)
            band_width = 18.0 + 17.0 * strength
            glow_alpha = 0.16 + 0.10 * strength
            fill_alpha = 0.08 + 0.06 * strength
            left = x_center - band_width / 2.0
            right = x_center + band_width / 2.0
            ax_spectrum.axvspan(
                left,
                right,
                facecolor=_POSITIVE_GLOW_COLOR,
                alpha=glow_alpha,
                edgecolor="none",
                zorder=1,
            )
            ax_spectrum.axvspan(
                left,
                right,
                facecolor=_POSITIVE_FILL_COLOR,
                alpha=fill_alpha,
                edgecolor="none",
                zorder=1.1,
            )

    if np.any(negative < 0):
        neg_indices = np.flatnonzero(negative < 0)
        for idx in neg_indices:
            x_center = float(x_axis[idx])
            strength = float(np.abs(negative[idx]) / max_importance)
            band_width = 18.0 + 17.0 * strength
            band_alpha = 0.08 + 0.08 * strength
            left = x_center - band_width / 2.0
            right = x_center + band_width / 2.0
            ax_spectrum.axvspan(
                left,
                right,
                facecolor=_NEGATIVE_COLOR,
                alpha=band_alpha,
                edgecolor="none",
                zorder=1,
            )


    ax_spectrum.set_xlabel(x_label, fontsize=11)
    ax_spectrum.set_ylabel("Intensity (a.u.)", fontsize=11)
    ax_spectrum.set_title(title, fontsize=13, fontweight="bold", pad=12)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=_SPECTRUM_COLOR,
            lw=1.2,
            label="Raman Spectrum",
        ),
        Patch(
            facecolor=_POSITIVE_FILL_COLOR,
            edgecolor=_POSITIVE_COLOR,
            linewidth=1,
            label="Supports prediction",
        ),
        Patch(
            facecolor=_NEGATIVE_COLOR,
            edgecolor=_NEGATIVE_COLOR,
            alpha=0.35,
            linewidth=1,
            label="Opposes prediction",
        ),
    ]

    ax_spectrum.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=9,
        framealpha=0.95,
    )

    ax_spectrum.grid(True, alpha=0.3, color=_GRID_COLOR)

    conf_text = (
        f"Predicted: {explanation.predicted_label}\n"
        f"Confidence: {explanation.confidence:.1%}"
    )
    ax_spectrum.text(
        0.02,
        0.95,
        conf_text,
        transform=ax_spectrum.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#BDBDBD"
        ),
    )

    ax_heatmap = fig.add_subplot(gs[1, 0])
    ax_heatmap.set_facecolor(_BACKGROUND_COLOR)

    cmap = LinearSegmentedColormap.from_list(
        "lime_diverging",
        [_NEGATIVE_COLOR, "#FFFFFF", _POSITIVE_COLOR],
    )

    imp_2d = np.tile(importance, (30, 1))
    vmax = np.abs(importance).max() or 1.0

    ax_heatmap.imshow(
        imp_2d,
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        extent=[x_axis[0], x_axis[-1], 0, 1],
        interpolation="bicubic",
    )
    cbar = fig.colorbar(
        ax_heatmap.images[0],
        ax=ax_heatmap,
        fraction=0.025,
        pad=0.02,
    )

    cbar.set_label("Feature Contribution", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax_heatmap.set_xlabel(x_label, fontsize=11)
    ax_heatmap.set_yticks([])
    ax_heatmap.set_title("Spectral Importance Map", fontsize=11, fontweight="bold")

    ax_bars = fig.add_subplot(gs[1, 1])

    top_features = explanation.top_features(n=show_top_features)
    if top_features:
        names = [f[0] for f in reversed(top_features)]
        weights = [f[1] for f in reversed(top_features)]
        colors = [_POSITIVE_COLOR if w >= 0 else _NEGATIVE_COLOR for w in weights]

        ax_bars.barh(range(len(names)), weights, color=colors, alpha=0.9, height=0.65)
        ax_bars.set_yticks(range(len(names)))
        ax_bars.set_yticklabels(names, fontsize=8)
        ax_bars.set_xlabel("Feature Contribution", fontsize=9)
        ax_bars.set_title("Top Features", fontsize=10, fontweight="bold")
        ax_bars.axvline(x=0, color="#757575", linewidth=0.5, linestyle="--")
        ax_bars.grid(True, axis="x", alpha=0.2, linestyle=":")

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def plot_lime_comparison(
    explanations: list,
    save_path: str | Path,
    title: str = "LIME Explanation Comparison",
    figsize_per_row: tuple = (14, 3),
    dpi: int = 300,
) -> None:
    n = len(explanations)
    if n == 0:
        return

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(figsize_per_row[0], figsize_per_row[1] * n),
        facecolor="white",
    )
    if n == 1:
        axes = [axes]

    for i, (ax, exp) in enumerate(zip(axes, explanations)):
        wavenumbers = exp.wavenumbers
        x_axis = (
            wavenumbers if wavenumbers is not None else np.arange(len(exp.spectrum))
        )
        display_spectrum = _display_spectrum(exp.spectrum)

        if wavenumbers is not None:
            order = np.argsort(x_axis)
            x_axis = x_axis[order]
            display_spectrum = display_spectrum[order]
            importance = exp.importance[order]
        else:
            importance = exp.importance

        ax.set_facecolor(_BACKGROUND_COLOR)
        ax.plot(
            x_axis,
            display_spectrum,
            color=_SPECTRUM_COLOR,
            linewidth=1.0,
            alpha=0.8,
        )

        pos = np.maximum(importance, 0)
        neg = np.abs(np.minimum(importance, 0))
        max_importance = float(np.abs(importance).max() or 1.0)
        if np.any(pos > 0):
            pos_indices = np.flatnonzero(pos > 0)
            for idx in pos_indices:
                x_center = float(x_axis[idx])
                strength = float(pos[idx] / max_importance)
                band_width = 18.0 + 17.0 * strength
                glow_alpha = 0.14 + 0.10 * strength
                fill_alpha = 0.07 + 0.05 * strength
                left = x_center - band_width / 2.0
                right = x_center + band_width / 2.0
                ax.axvspan(
                    left,
                    right,
                    facecolor=_POSITIVE_GLOW_COLOR,
                    alpha=glow_alpha,
                    edgecolor="none",
                    zorder=1,
                )
                ax.axvspan(
                    left,
                    right,
                    facecolor=_POSITIVE_FILL_COLOR,
                    alpha=fill_alpha,
                    edgecolor="none",
                    zorder=1.1,
                )

        if np.any(neg > 0):
            neg_indices = np.flatnonzero(neg > 0)
            for idx in neg_indices:
                x_center = float(x_axis[idx])
                strength = float(neg[idx] / max_importance)
                band_width = 18.0 + 17.0 * strength
                band_alpha = 0.07 + 0.07 * strength
                left = x_center - band_width / 2.0
                right = x_center + band_width / 2.0
                ax.axvspan(
                    left,
                    right,
                    facecolor=_NEGATIVE_COLOR,
                    alpha=band_alpha,
                    edgecolor="none",
                    zorder=1,
                )

        label = f"{exp.explained_label} — " f"P={exp.confidence:.1%}"
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.2)

        if i == n - 1:
            x_label = "Wavenumber (cm⁻¹)" if wavenumbers is not None else "Index"
            ax.set_xlabel(x_label, fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
