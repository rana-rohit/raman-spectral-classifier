"""
src/utils/checkpoint.py

Save and load model checkpoints with full experiment metadata.
Every checkpoint stores: model weights, optimizer state, epoch,
best metric, config snapshot, and training history.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _is_cbam_key(key: str) -> bool:
    """Return True for parameters introduced by optional CBAM blocks."""
    return ".cbam." in key


def _warn_cbam_partial_load(missing_keys: list[str], unexpected_keys: list[str]) -> None:
    print("\n[Checkpoint] WARNING: Partial CBAM checkpoint load.")
    if missing_keys:
        print(
            "[Checkpoint] Missing CBAM weights will remain randomly initialized: "
            f"{len(missing_keys)} keys"
        )
    if unexpected_keys:
        print(
            "[Checkpoint] Checkpoint contains CBAM weights not present in model: "
            f"{len(unexpected_keys)} keys"
        )
    print()


def resolve_pretrained_checkpoint(cfg: Dict, task_cfg: Dict, stage: str) -> tuple:
    """
    Centralized logic to resolve the pretrained checkpoint based on priority:
    1. pretrained_checkpoint
    2. pretrained_experiment
    3. legacy fallback (pretrained_exp_dir)
    
    Returns:
        tuple containing (checkpoint_path, source_type, experiment_name)
    """
    train_cfg = cfg.get("training", {})
    explicit_ckpt = train_cfg.get("pretrained_checkpoint")
    explicit_exp = train_cfg.get("pretrained_experiment")
    legacy_dir = task_cfg.get("pretrained_exp_dir")
    
    source_type = ""
    exp_name = "N/A"
    ckpt_path = ""
    
    if explicit_ckpt:
        source_type = "explicit_checkpoint"
        ckpt_path = str(explicit_ckpt)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Requested pretrained_checkpoint='{ckpt_path}' but file does not exist or is not a file."
            )
    elif explicit_exp:
        source_type = "experiment_name"
        exp_name = str(explicit_exp)
        # resolve the experiment directory, usually under experiments/ unless it's an absolute path
        exp_dir = exp_name if os.path.isabs(exp_name) else os.path.join("experiments", exp_name)
        if not os.path.isdir(exp_dir):
            raise FileNotFoundError(
                f"Requested pretrained_experiment='{exp_name}' but directory '{exp_dir}' was not found."
            )
        try:
            ckpt_path = resolve_best_checkpoint_path(exp_dir)
        except Exception as e:
            raise FileNotFoundError(
                f"Requested pretrained_experiment='{exp_name}' but no valid checkpoint was found in '{exp_dir}'. Error: {e}"
            )
    elif legacy_dir:
        raise ValueError(
            "Implicit checkpoint loading via task.pretrained_exp_dir is disabled. "
            "Move this value to training.pretrained_experiment or provide "
            "training.pretrained_checkpoint explicitly."
        )
    else:
        raise ValueError(
            f"{stage} requires a pretrained checkpoint. Please specify "
            f"'training.pretrained_checkpoint' or 'training.pretrained_experiment' in config."
        )

    print("\n============================================================")
    print("PRETRAINED CHECKPOINT SOURCE")
    print("============================")
    print(f"Source Type:   {source_type}")
    print(f"Experiment:    {exp_name}")
    print(f"Checkpoint:    {ckpt_path}")
    print(f"Stage Source:  {stage}")
    print("======================")

    return ckpt_path, source_type, exp_name

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
        weights_only=False,
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

    missing_cbam_keys = [
        key for key in current_state
        if _is_cbam_key(key) and key not in filtered_state
    ]
    if missing_cbam_keys:
        _warn_cbam_partial_load(missing_cbam_keys, [])

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
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
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
    missing_keys = [
        key for key in model_state
        if key not in checkpoint_state
    ]
    unexpected_keys = [
        key for key in checkpoint_state
        if key not in model_state
    ]
    has_key_mismatch = bool(missing_keys or unexpected_keys)

    if has_key_mismatch and all(
        _is_cbam_key(key) for key in missing_keys + unexpected_keys
    ):
        _warn_cbam_partial_load(missing_keys, unexpected_keys)
        model.load_state_dict(checkpoint_state, strict=False)
        if optimizer is not None and "optimizer_state" in checkpoint:
            print(
                "[Checkpoint] WARNING: Skipping optimizer state because "
                "model parameters changed for optional CBAM."
            )
    else:
        try:
            model.load_state_dict(checkpoint_state)
        except RuntimeError as e:
            print("\n[Checkpoint] WARNING: strict load failed, attempting non-strict load.\n", e)
            model.load_state_dict(checkpoint_state, strict=False)
            if optimizer is not None and "optimizer_state" in checkpoint:
                print("[Checkpoint] WARNING: Loaded model with non-strict state; optimizer state may be incompatible.")
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


def load_encoder_only(
    path: str,
    model: torch.nn.Module,
    device: str = "cpu",
) -> dict:
    """
    Load pretrained representation learning weights except classifier head.
    Ensures classifier initializes independently.
    """
    path_obj = Path(path)
    checkpoint_path = str(path_obj) if path_obj.is_file() else resolve_best_checkpoint_path(path)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    pretrained_state = checkpoint["model_state"]
    current_state = model.state_dict()
    
    loaded_keys = []
    skipped_keys = []
    shape_mismatch_keys = []
    
    filtered_state = {}
    for key, value in pretrained_state.items():
        # Skip classifier layers to ensure independent initialization
        if key.startswith("classifier") or "classifier" in key:
            skipped_keys.append(key)
            continue
            
        if key not in current_state:
            skipped_keys.append(key)
            continue
            
        if current_state[key].shape != value.shape:
            shape_mismatch_keys.append((key, value.shape, current_state[key].shape))
            continue
            
        filtered_state[key] = value
        loaded_keys.append(key)
        
    current_state.update(filtered_state)
    model.load_state_dict(current_state)

    missing_cbam_keys = [
        key for key in current_state
        if _is_cbam_key(key) and key not in filtered_state
    ]
    if missing_cbam_keys:
        _warn_cbam_partial_load(missing_cbam_keys, [])
    
    print(f"\n[Checkpoint] Loaded encoder weights from: {checkpoint_path}")
    print(f"[Checkpoint] Loaded keys count:   {len(loaded_keys)}")
    print(f"[Checkpoint] Skipped keys count:  {len(skipped_keys)} (including classifier head)")
    if shape_mismatch_keys:
        print(f"[Checkpoint] Shape mismatch keys: {shape_mismatch_keys}")
    print()
    
    return checkpoint
