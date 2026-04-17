"""
src/data/dataset.py

PyTorch Dataset for 1D spectral signals.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset


class SpectralDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augmentation=None,
        training: bool = False,
        class_filter: Optional[List[int]] = None,
        n_views: int = 1,
    ) -> None:
        if class_filter is not None:
            mask = np.isin(y, class_filter)
            X, y = X[mask], y[mask]

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augmentation = augmentation
        self.training = training
        self.class_filter = list(class_filter) if class_filter is not None else None
        self.n_views = int(n_views)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()
        y_tensor = torch.tensor(self.y[idx], dtype=torch.long)

        if self.n_views == 2 and self.training:
            x1 = self._transform_sample(x.copy())
            x2 = self._transform_sample(x.copy())
            return {
                "x1": torch.from_numpy(x1).unsqueeze(0),
                "x2": torch.from_numpy(x2).unsqueeze(0),
                "y": y_tensor,
            }

        x = self._transform_sample(x)
        return torch.from_numpy(x).unsqueeze(0), y_tensor

    def _transform_sample(self, x: np.ndarray) -> np.ndarray:
        if self.training and self.augmentation is not None and len(self.augmentation.steps) > 0:
            x = self.augmentation(x[None])[0]
        return x.astype(np.float32, copy=False)

    @property
    def n_classes(self) -> int:
        return len(np.unique(self.y))

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
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=random_seed,
    )
    train_idx, val_idx = next(sss.split(X, y))
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx])
