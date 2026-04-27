"""
src/data/dataset.py

PyTorch Dataset for 1D spectral signals.
Supports optional multi-channel input (raw + first derivative).
"""

from __future__ import annotations

from typing import Mapping, Tuple

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
        n_views: int = 1,
        preprocessor=None,
        expected_n_classes: int | None = None,
        class_filter: list[int] | None = None,
        class_map: Mapping[int, int] | None = None,
        inverse_class_map: Mapping[int, int] | None = None,
        sample_ids=None,
    ) -> None:
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        if self.X.ndim != 2:
            raise ValueError(f"Expected X shape (N, L), got {self.X.shape}")

        if len(self.X) != len(self.y):
            raise ValueError(
                f"X and y size mismatch: {self.X.shape[0]} vs {len(self.y)}"
            )

        self.expected_n_classes = expected_n_classes
        self.class_filter = list(class_filter) if class_filter is not None else None
        self.class_map = dict(class_map or {})
        self.inverse_class_map = dict(inverse_class_map or {})
        if sample_ids is None:
            self.sample_ids = np.arange(len(self.y))
        else:
            self.sample_ids = np.asarray(sample_ids)
            if len(self.sample_ids) != len(self.y):
                raise ValueError(
                    f"sample_ids and y size mismatch: {len(self.sample_ids)} vs {len(self.y)}"
                )
        self._validate_labels()

        self.augmentation = augmentation
        self.training = training
        self.n_views = int(n_views) 
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()
        y_tensor = torch.as_tensor(self.y[idx], dtype=torch.long)

        if self.n_views == 2 and self.training:
            x1 = self._transform_sample(x.copy())
            x2 = self._transform_sample(x.copy())
            x1_tensor = self._to_multichannel(x1, idx)
            x2_tensor = self._to_multichannel(x2, idx)
            return {
                "x1": x1_tensor,
                "x2": x2_tensor,
                "y": y_tensor,
            }

        x = self._transform_sample(x)
        x_tensor = self._to_multichannel(x, idx)
        return x_tensor, y_tensor
    """
    Returns:
    - Dict with x1, x2, y if n_views == 2
    - Tuple (x, y) otherwise
    """

    def _to_multichannel(self, x: np.ndarray, idx: int) -> torch.Tensor:
        """
        x should be (C, L) after preprocessing (C=2 if derivative enabled)
        """
        return torch.from_numpy(x.astype(np.float32))

    def _transform_sample(self, x: np.ndarray) -> np.ndarray:
        # x is RAW (1000,)

        if self.training and self.augmentation is not None:
            if hasattr(self.augmentation, "steps") and len(self.augmentation.steps) > 0:
                x = self.augmentation(x[None])[0]

        if self.preprocessor is not None:
            x = self.preprocessor.transform(x[None])[0]
        else:
            raise ValueError("Preprocessor is required for dataset")
        
        if x.ndim != 2:
            raise ValueError(f"Expected (C, L) after preprocessing, got {x.shape}")

        if x.shape[0] not in (1, 2):
            raise ValueError(f"Unexpected channel count: {x.shape}")

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

    def _validate_labels(self) -> None:
        if len(self.y) == 0:
            raise ValueError("Dataset cannot be empty")

        unique = np.unique(self.y)
        expected_unique = np.arange(len(unique))
        if not np.array_equal(unique, expected_unique):
            raise ValueError(
                f"Labels must be contiguous [0..N-1], got {unique}"
            )

        if self.expected_n_classes is not None:
            if self.y.min() < 0 or self.y.max() >= self.expected_n_classes:
                raise ValueError(
                    f"Labels must be in [0, {self.expected_n_classes - 1}], "
                    f"got [{self.y.min()}, {self.y.max()}]"
                )


def make_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.20,
    random_seed: int = 42,
) -> Tuple:
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=random_seed,
    )
    train_idx, val_idx = next(sss.split(X, y))

    return (
        (X[train_idx], y[train_idx]),
        (X[val_idx], y[val_idx]),
    )
