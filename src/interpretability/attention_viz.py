"""
src/interpretability/attention_viz.py

Attention map extraction and visualisation for Transformer and Hybrid models.

Attention maps answer:
  "Which spectral patches does the model attend to when classifying sample X?"
  "Do different classes produce different attention patterns?"
  "Do clinical samples show different attention than source samples?"

Functions:
  extract_attention()   — raw attention weights for one sample
  mean_attention_map()  — average across heads and/or layers
  cls_to_patch_attention() — CLS token's attention to each patch (classification relevance)
  rollout()             — attention rollout for multi-layer attribution
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def extract_attention(
    model: nn.Module,
    x: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Extract raw attention weights from all layers.

    Args:
        model: SpectralTransformer or HybridCNNTransformer
        x:     Input tensor, shape (1, 1, L)

    Returns:
        attn_maps: List of (1, n_heads, seq_len, seq_len) tensors,
                   one per Transformer layer.
    """
    assert hasattr(model, "get_attention_maps"), (
        "Model must implement get_attention_maps(). "
        "Available for SpectralTransformer and HybridCNNTransformer."
    )
    model.eval()
    with torch.no_grad():
        attn_maps = model.get_attention_maps(x)
    return attn_maps


def cls_to_patch_attention(
    attn_maps: List[torch.Tensor],
    layer: int = -1,
    head: str = "mean",
) -> np.ndarray:
    """
    Extract the CLS token's attention to each patch (position 0 → positions 1:).

    This is the most directly interpretable attention signal:
    high attention from CLS to patch i means patch i strongly contributed
    to the classification decision.

    Args:
        attn_maps: Output of extract_attention()
        layer:     Which layer to use (-1 = last layer)
        head:      "mean" (average across heads) or int (specific head index)

    Returns:
        attn_to_patches: np.ndarray of shape (n_patches,), values sum to ~1
    """
    attn = attn_maps[layer]   # (1, n_heads, seq_len, seq_len)
    attn = attn.squeeze(0)    # (n_heads, seq_len, seq_len)

    # CLS is token 0; its attention to all other tokens is row 0
    cls_attn = attn[:, 0, 1:]   # (n_heads, n_patches)

    if head == "mean":
        cls_attn = cls_attn.mean(dim=0)   # (n_patches,)
    else:
        cls_attn = cls_attn[int(head)]

    return cls_attn.numpy()


def attention_rollout(
    attn_maps: List[torch.Tensor],
    discard_ratio: float = 0.9,
) -> np.ndarray:
    """
    Attention Rollout — propagates attention through all layers to compute
    total information flow from each patch to the CLS token.

    Unlike single-layer attention, rollout accounts for residual connections
    and multiple attention layers (Abnar & Zuidema, 2020).

    Args:
        attn_maps:     Output of extract_attention()
        discard_ratio: Fraction of lowest attentions to zero out per layer
                       (removes noise from near-zero attention values)

    Returns:
        rollout: np.ndarray of shape (n_patches,) — cumulative attribution
    """
    # Average over heads, add residual (identity matrix)
    result = None
    seq_len = attn_maps[0].shape[-1]
    eye = torch.eye(seq_len)

    for attn in attn_maps:
        attn = attn.squeeze(0)           # (n_heads, seq, seq)
        attn = attn.mean(dim=0)          # (seq, seq)

        # Discard low-attention values (noise reduction)
        flat = attn.flatten()
        threshold = flat.quantile(discard_ratio)
        attn = torch.where(attn > threshold, attn, torch.zeros_like(attn))

        # Add residual connection and re-normalise rows
        attn = attn + eye
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        result = attn if result is None else torch.matmul(attn, result)

    # CLS token (row 0) attention to patches (cols 1:)
    cls_rollout = result[0, 1:].numpy()

    # Normalise
    if cls_rollout.max() > 1e-8:
        cls_rollout = cls_rollout / cls_rollout.max()

    return cls_rollout


def patch_attention_to_signal(
    patch_attn: np.ndarray,
    signal_length: int,
    patch_size: int,
) -> np.ndarray:
    """
    Upsample patch-level attention to the original signal resolution.
    Each patch's attention value is repeated patch_size times.

    Args:
        patch_attn:    (n_patches,) array of attention weights
        signal_length: Original signal length (e.g. 1000)
        patch_size:    Number of signal positions per patch (e.g. 20)

    Returns:
        signal_attn: (signal_length,) — attention at every signal position
    """
    n_patches = len(patch_attn)
    # Each patch covers exactly patch_size positions
    signal_attn = np.repeat(patch_attn, patch_size)

    # Handle any length mismatch from rounding
    if len(signal_attn) < signal_length:
        signal_attn = np.pad(signal_attn, (0, signal_length - len(signal_attn)))
    elif len(signal_attn) > signal_length:
        signal_attn = signal_attn[:signal_length]

    return signal_attn


def per_class_attention(
    model: nn.Module,
    dataset,
    class_ids: List[int],
    n_samples: int = 20,
    patch_size: int = 20,
    signal_length: int = 1000,
    device: str = "cpu",
) -> dict:
    """
    Compute mean CLS→patch attention for each class.
    Useful for comparing which spectral regions each class attends to.

    Returns:
        Dict[class_id -> np.ndarray of shape (signal_length,)]
    """
    model.eval()
    model.to(device)

    class_attentions = {cls: [] for cls in class_ids}

    for i in range(len(dataset)):
        x, y = dataset[i]
        y = int(y)
        if y not in class_ids:
            continue
        if len(class_attentions[y]) >= n_samples:
            continue

        x_t = x.unsqueeze(0).to(device)
        attn_maps = extract_attention(model, x_t)
        patch_attn = cls_to_patch_attention(attn_maps, layer=-1, head="mean")
        signal_attn = patch_attention_to_signal(patch_attn, signal_length, patch_size)
        class_attentions[y].append(signal_attn)

        if all(len(v) >= n_samples for v in class_attentions.values()):
            break

    return {
        cls: np.mean(atts, axis=0) if atts else np.zeros(signal_length)
        for cls, atts in class_attentions.items()
    }