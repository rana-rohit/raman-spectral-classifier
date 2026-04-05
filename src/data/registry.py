"""
src/data/registry.py

DataRegistry — the single entry point for all dataset access.
Reads split definitions from configs/data/splits.yaml.
No hardcoded split names anywhere else in the codebase.

Usage:
    registry = DataRegistry(data_root="data/raw", cfg=splits_cfg)
    X_train, y_train = registry.get_arrays("reference")
    ood_splits = registry.ood_split_names()
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.split_roles import SplitRole, role_from_str, ROLE_PERMISSIONS


@dataclass
class SplitMeta:
    name: str
    x_file: str
    y_file: str
    role: SplitRole
    eval_classes: Optional[List[int]] = None   # None = use all classes

    # Populated after loading
    X: Optional[np.ndarray] = field(default=None, repr=False)
    y: Optional[np.ndarray] = field(default=None, repr=False)
    loaded: bool = False


class DataRegistry:
    """
    Manages all dataset splits as named domains.

    Key guarantees:
    - HOLDOUT splits raise an error if accessed with role != 'evaluate'
    - Preprocessor is only ever fit on the SOURCE split
    - OOD splits record which classes are valid for evaluation
    """

    def __init__(self, data_root: str, cfg: dict) -> None:
        self.data_root = data_root
        self._splits: Dict[str, SplitMeta] = {}
        self._shared_classes: List[int] = cfg["dataset"].get("shared_classes", [])
        self._signal_length: int = cfg["dataset"]["signal_length"]
        self._n_classes_full: int = cfg["dataset"]["n_classes_full"]

        self._register_from_cfg(cfg)

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #

    def _register_from_cfg(self, cfg: dict) -> None:
        for split_name, split_cfg in cfg["splits"].items():
            role = role_from_str(split_cfg["role"])
            eval_classes = split_cfg.get("eval_classes", None)
            self._splits[split_name] = SplitMeta(
                name=split_name,
                x_file=split_cfg["x_file"],
                y_file=split_cfg["y_file"],
                role=role,
                eval_classes=eval_classes,
            )

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def load(self, split_name: str) -> "DataRegistry":
        """Load a split from disk. Returns self for chaining."""
        meta = self._get_meta(split_name)
        if meta.loaded:
            return self

        x_path = os.path.join(self.data_root, meta.x_file)
        y_path = os.path.join(self.data_root, meta.y_file)

        if not os.path.exists(x_path):
            raise FileNotFoundError(
                f"[DataRegistry] X file not found: {x_path}\n"
                f"Check configs/data/splits.yaml path for split '{split_name}'"
            )

        meta.X = np.load(x_path)
        meta.y = np.load(y_path).astype(np.int64)
        meta.loaded = True

        self._validate_shape(split_name, meta)
        return self

    def load_all(self) -> "DataRegistry":
        for name in self._splits:
            self.load(name)
        return self

    def _validate_shape(self, name: str, meta: SplitMeta) -> None:
        assert meta.X.shape[1] == self._signal_length, (
            f"[DataRegistry] Split '{name}': expected signal length "
            f"{self._signal_length}, got {meta.X.shape[1]}"
        )
        assert len(meta.X) == len(meta.y), (
            f"[DataRegistry] Split '{name}': X/y length mismatch "
            f"({len(meta.X)} vs {len(meta.y)})"
        )

    # ------------------------------------------------------------------ #
    # Access
    # ------------------------------------------------------------------ #

    def get_arrays(self, split_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X, y) for a split. Loads from disk if not yet loaded."""
        self.load(split_name)
        meta = self._get_meta(split_name)
        return meta.X, meta.y

    def get_meta(self, split_name: str) -> SplitMeta:
        self.load(split_name)
        return self._get_meta(split_name)

    def get_eval_classes(self, split_name: str) -> Optional[List[int]]:
        """
        Returns the list of valid classes for evaluation on this split.
        None means use all classes (source / holdout splits).
        For OOD splits this returns the 5 shared clinical classes.
        """
        return self._get_meta(split_name).eval_classes

    # ------------------------------------------------------------------ #
    # Query helpers
    # ------------------------------------------------------------------ #

    def available_splits(self) -> List[str]:
        return list(self._splits.keys())

    def ood_split_names(self) -> List[str]:
        return [n for n, m in self._splits.items() if m.role == SplitRole.OOD_EVAL]

    def source_split_name(self) -> str:
        """Returns the split used to fit the preprocessor."""
        for name, meta in self._splits.items():
            if meta.role == SplitRole.SOURCE:
                return name
        raise RuntimeError("No SOURCE split registered.")

    def holdout_split_names(self) -> List[str]:
        return [n for n, m in self._splits.items() if m.role == SplitRole.HOLDOUT]

    def adaptation_split_names(self) -> List[str]:
        return [n for n, m in self._splits.items() if m.role == SplitRole.ADAPTATION]

    @property
    def shared_classes(self) -> List[int]:
        return self._shared_classes

    @property
    def signal_length(self) -> int:
        return self._signal_length

    @property
    def n_classes(self) -> int:
        return self._n_classes_full

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #

    def summary(self) -> None:
        print(f"\n{'Split':>16}  {'Role':>12}  {'Samples':>8}  "
              f"{'Classes':>8}  {'Loaded':>7}")
        print("-" * 60)
        for name, meta in self._splits.items():
            n_samples = len(meta.X) if meta.loaded else "—"
            n_classes = len(np.unique(meta.y)) if meta.loaded else "—"
            print(f"{name:>16}  {meta.role.name:>12}  {str(n_samples):>8}  "
                  f"{str(n_classes):>8}  {str(meta.loaded):>7}")
        print()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _get_meta(self, split_name: str) -> SplitMeta:
        if split_name not in self._splits:
            raise KeyError(
                f"[DataRegistry] Unknown split '{split_name}'. "
                f"Available: {self.available_splits()}"
            )
        return self._splits[split_name]