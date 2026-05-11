from __future__ import annotations

from pathlib import Path

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
        labels=np.arange(len(class_labels)),
    )

    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cm, interpolation="nearest")

    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))

    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

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
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)