"""
src/utils/checkpoint.py

Save and load model checkpoints with full experiment metadata.
Every checkpoint stores: model weights, optimizer state, epoch,
best metric, config snapshot, and training history.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    config: Dict,
    is_best: bool = False,
) -> None:
    """
    Save a full training checkpoint.
    If is_best=True, also copies to <dir>/best.pt
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch":          epoch,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics":        metrics,
        "config":         config,
    }
    torch.save(checkpoint, path)

    if is_best:
        best_path = str(Path(path).parent / "best.pt")
        torch.save(checkpoint, best_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a checkpoint into model (and optionally optimizer).
    Returns the full checkpoint dict so caller can access metrics/config.
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def load_best_model(
    experiment_dir: str,
    model: torch.nn.Module,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Convenience: load best.pt from an experiment directory."""
    best_path = os.path.join(experiment_dir, "best.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"No best.pt found in {experiment_dir}. "
            "Has training completed?"
        )
    return load_checkpoint(best_path, model, device=device)