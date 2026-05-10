"""
src/utils/checkpoint.py

Save and load model checkpoints with full experiment metadata.
Every checkpoint stores: model weights, optimizer state, epoch,
best metric, config snapshot, and training history.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch

def load_backbone_weights(
    path: str,
    model: torch.nn.Module,
    device: str = "cpu",
) -> dict:
    """
    Load pretrained weights EXCEPT classifier head.
    Used for transfer learning when output dimensions differ.
    """

    path_obj = Path(path)

    if path_obj.is_file():
        checkpoint_path = str(path_obj)
    else:
        checkpoint_path = resolve_best_checkpoint_path(path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    pretrained_state = checkpoint["model_state"]
    current_state = model.state_dict()

    filtered_state = {}

    for key, value in pretrained_state.items():

        # Skip classifier layers
        if key.startswith("classifier"):
            continue

        # Skip incompatible shapes
        if key not in current_state:
            continue

        if current_state[key].shape != value.shape:
            continue

        filtered_state[key] = value

    current_state.update(filtered_state)

    model.load_state_dict(current_state)

    return checkpoint

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
    If is_best=True, also copies to <dir>/best_model.pt
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics":         metrics,
        "best_metric_name": config.get("training", {}).get("monitor_metric", "f1_macro"),
        "config":          config,
        "n_parameters": sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        ),
        "stage": config.get("task", {}).get("stage"),
        "label_space": config.get(
            "task",
            {},
        ).get(
            "label_space"
        ),
        "ontology_version": config.get(
            "ontology_version",
            "unknown",
        ),
        "semantic_space": config.get(
            "model",
            {},
        ).get(
            "semantic_space",
            None,
        ),
    }
    torch.save(checkpoint, path)

    if is_best:
        best_path = str(Path(path).parent / "best_model.pt")
        torch.save(checkpoint, best_path)


def resolve_best_checkpoint_path(experiment_dir: str) -> str:
    """
    Resolve the canonical best checkpoint for an experiment.

    New runs save the best checkpoint under ``checkpoints/best_model.pt``.
    Older runs may have used ``best_model.pt`` at the experiment root, so we
    support both locations for backward compatibility.
    """
    exp_path = Path(experiment_dir)
    candidates = [
        exp_path / "checkpoints" / "best_model.pt",
        exp_path / "checkpoints" / "best.pt",
        exp_path / "best_model.pt",
        exp_path / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        f"No best checkpoint found in {experiment_dir}. "
        f"Looked in: {[str(path) for path in candidates]}"
    )


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
    
    if "semantic_space" in checkpoint:
        checkpoint_space = checkpoint["semantic_space"]
        model_space = getattr(
            model,
            "semantic_space",
            None,
        )

        if (
            model_space is not None
            and checkpoint_space is not None
            and model_space != checkpoint_space
        ):
            raise ValueError(
                "Checkpoint semantic space mismatch: "
                f"{checkpoint_space} vs {model_space}"
            )
            
    checkpoint_state = checkpoint["model_state"]
    model_state = model.state_dict()
    
    # Assert embedding dimension and n_classes match precisely if strict
    for key, param in model_state.items():
        if key in checkpoint_state:
            assert checkpoint_state[key].shape == param.shape, (
                f"Shape mismatch for {key}: "
                f"checkpoint={checkpoint_state[key].shape}, "
                f"model={param.shape}"
            )
    
    if "n_classes" in checkpoint:
        assert checkpoint["n_classes"] == model.classifier[-1].out_features, (
            f"Checkpoint n_classes={checkpoint['n_classes']} "
            f"!= model n_classes={model.classifier[-1].out_features}"
        )
    model.load_state_dict(checkpoint_state)
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def load_best_model(
    experiment_dir: str,
    model: torch.nn.Module,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Convenience: load the best checkpoint from an experiment directory."""
    best_path = resolve_best_checkpoint_path(experiment_dir)
    return load_checkpoint(best_path, model, device=device)
