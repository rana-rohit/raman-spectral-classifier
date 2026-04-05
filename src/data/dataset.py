"""
src/data/dataset.py

PyTorch Dataset for 1D spectral signals.

Handles:
- Conversion to tensors with correct shape (N, 1, L) for Conv1d
- Optional training-time augmentation (disabled at eval automatically)
- Optional class filtering (for OOD evaluation on shared classes only)
- Stratified train/val splitting from the reference split
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit


class SpectralDataset(Dataset):
    """
    Dataset for 1D spectral signals.

    X shape on disk: (N, L)  — N signals of length L
    X shape returned: (1, L) — single-channel 1D signal for Conv1d compatibility
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augmentation=None,
        training: bool = False,
        class_filter: Optional[List[int]] = None,
    ) -> None:
        """
        Args:
            X:            Signal array, shape (N, L)
            y:            Label array, shape (N,)
            augmentation: AugmentationPipeline or None
            training:     If True, augmentation is applied; if False, it is skipped
            class_filter: If set, only samples from these classes are included.
                          Use for OOD evaluation on the 5 shared classes.
        """
        if class_filter is not None:
            mask = np.isin(y, class_filter)
            X, y = X[mask], y[mask]

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augmentation = augmentation
        self.training = training

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].copy()   # Copy so augmentation doesn't mutate the array

        if self.training and self.augmentation is not None:
            # Augmentation expects batched input (N, L), so expand/squeeze
            x = self.augmentation(x[None])[0]

        # Numpy augmentation ops can upcast to float64; keep model inputs float32.
        x = x.astype(np.float32, copy=False)

        # Add channel dimension: (L,) -> (1, L)
        x_tensor = torch.from_numpy(x).unsqueeze(0)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)
        return x_tensor, y_tensor

    @property
    def n_classes(self) -> int:
        return int(self.y.max()) + 1

    @property
    def signal_length(self) -> int:
        return self.X.shape[1]

    @property
    def class_counts(self) -> dict:
        unique, counts = np.unique(self.y, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


def make_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.20,
    random_seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified train/val split from the reference (source) split.
    Ensures equal class representation in both halves.

    Returns: (X_train, y_train), (X_val, y_val)
    """
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=random_seed,
    )
    train_idx, val_idx = next(sss.split(X, y))
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])