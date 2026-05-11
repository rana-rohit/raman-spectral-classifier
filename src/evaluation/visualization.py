from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


# --------------------------------------------------------
# Generate publication-style confusion matrix heatmaps.
#
# Supports:
# - spectrum-level confusion
# - isolate-level confusion
# - patient-level confusion
#
# Saves:
# - raw-count matrix
# - normalized matrix
# --------------------------------------------------------


def save_confusion_matrix_figure(
    targets,
    predictions,
    class_labels,
    save_path,
    title,
    normalize=False,
):
    """
    Save confusion matrix heatmap.

    Parameters
    ----------
    targets : array-like
    predictions : array-like
    class_labels : list[str]
    save_path : str | Path
    title : str
    normalize : bool
    """

    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    cm = confusion_matrix(
        targets,
        predictions,
        labels=np.unique(
            np.concatenate([targets, predictions])
        )
    )

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    fig, ax = plt.subplots(
        figsize=(10, 9)
    )
    fig.patch.set_facecolor("white")

    im = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="YlGnBu",
    )
    
    ax.set_xticks(
        np.arange(-0.5, len(class_labels), 1),
        minor=True,
    )

    ax.set_yticks(
        np.arange(-0.5, len(class_labels), 1),
        minor=True,
    )

    ax.grid(
        which="minor",
        color="black",
        linestyle="-",
        linewidth=0.5,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    cbar = plt.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.ax.tick_params(labelsize=11)

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))

    ax.set_xticklabels(
        class_labels,
        rotation=90,
        fontsize=12,
    )

    ax.set_yticklabels(
        class_labels,
        fontsize=12,
    )

    ax.set_xlabel(
        "Predicted Class",
        fontsize=16,
        weight="bold",
    )

    ax.set_ylabel(
        "True Class",
        fontsize=16,
        weight="bold",
    )

    ax.set_title(
        title,
        fontsize=18,
        weight="bold",
        pad=20,
    )

    threshold = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            value = (
                f"{cm[i, j]:.2f}"
                if normalize
                else str(int(cm[i, j]))
            )

            ax.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        save_path,
        dpi=500,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Saved figure: {save_path}")
    plt.close(fig)