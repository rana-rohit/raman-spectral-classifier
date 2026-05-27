"""
src/utils/config.py

Lightweight config system — loads YAML files, supports dot-access,
and merges multiple configs without requiring Hydra or OmegaConf.

Usage:
    cfg = load_config("configs/data/splits.yaml",
                      "configs/data/preprocessing.yaml",
                      "configs/model/cnn.yaml",
                      "configs/training/base.yaml")
    print(cfg.model.n_classes)
"""

from pathlib import Path
from typing import Iterable, Union

import yaml


class Config(dict):
    """
    Dict subclass with recursive dot-access.
    cfg["model"]["n_classes"] == cfg.model.n_classes
    """
    def __getattr__(self, key):
        try:
            val = self[key]
            return Config(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __repr__(self):
        import json
        return json.dumps(dict(self), indent=2, default=str)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflict."""
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(*paths: Union[str, Path]) -> Config:
    """
    Load and merge one or more YAML config files.
    Later files override earlier ones on key conflict.
    """
    merged = {}
    for path in paths:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, data)
    return Config(merged)


def apply_overrides(cfg: dict, overrides: Iterable[str] | None) -> dict:
    """
    Apply dotted key=value config overrides in-place.

    Values are parsed with yaml.safe_load so booleans, numbers, lists,
    and quoted strings behave consistently across entry points.
    """
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid override '{item}'. Expected dotted key=value, "
                "for example training.batch_size=64"
            )
        key, val = item.split("=", 1)
        keys = key.split(".")
        cursor = cfg
        for part in keys[:-1]:
            cursor = cursor.setdefault(part, {})
        try:
            val = yaml.safe_load(val)
        except Exception:
            pass
        cursor[keys[-1]] = val
    return cfg


def save_config(cfg: dict, path: Union[str, Path]) -> None:
    """Save a config dict to YAML — used to snapshot experiment configs."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False, sort_keys=False)
