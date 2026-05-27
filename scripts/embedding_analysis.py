# ============================================================
# scripts/embedding_analysis.py
# ============================================================
#
# PURPOSE:
# --------
# Extract latent embeddings from a trained model and
# visualize feature geometry using PCA and UMAP.
#
# ============================================================

import os
import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
import umap
import torch
from torch.utils.data import DataLoader

# ============================================================
# PROJECT IMPORTS
# ============================================================
from src.models.registry import get_model
from src.data.registry import DataRegistry
from src.data.preprocessing import SpectralPreprocessor
from src.data.augmentation import AugmentationPipeline
from src.data.dataloader import build_all_loaders
from src.utils.split_modes import IID_REFERENCE, resolve_split_mode
from src.utils.config import load_config
from src.utils.checkpoint import load_best_model
from sklearn.preprocessing import StandardScaler
from metadata.ontology import ISOLATE_TO_TREATMENT

# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to analyze (e.g. test, 2018clinical)")
    parser.add_argument("--use-projection", action="store_true", help="Extract and analyze projection head embeddings instead of backbone features")
    return parser.parse_args()


# ============================================================
# LOAD EXPERIMENT CONFIG
# ============================================================

def load_experiment_config(experiment_dir):
    config_path = os.path.join(experiment_dir, "config.yaml")
    cfg = load_config(config_path)
    return cfg


# ============================================================
# LOAD MODEL CHECKPOINT
# ============================================================

def load_model(cfg, experiment_dir, device):
    model_name = cfg["model"]["name"]
    model = get_model(model_name, cfg)
    checkpoint = load_best_model(experiment_dir, model, device=str(device))
    model.to(device)
    model.eval()
    return model


# ============================================================
# EXTRACT EMBEDDINGS
# ============================================================

@torch.no_grad()
def extract_embeddings(model, dataloader, device, use_projection: bool = False):
    all_embeddings = []
    all_labels = []
    all_predictions = []
    all_correctness = []

    for x, y in tqdm(dataloader):
        inputs = x.to(device)
        labels = y.to(device)

        # Extract latent features
        if use_projection and hasattr(model, "projection_head") and model.projection_head is not None:
            out = model(inputs)
            embeddings = out["projection_features"]
        else:
            embeddings = model.forward_features(inputs)

        # Get logits for prediction analysis
        logits = model.forward_logits(model.forward_features(inputs) if use_projection else embeddings)
        predictions = torch.argmax(logits, dim=1)
        correctness = (predictions == labels)

        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_correctness.append(correctness.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_correctness = np.concatenate(all_correctness, axis=0)

    return all_embeddings, all_labels, all_predictions, all_correctness


# ============================================================
# VISUALIZATION UTILS
# ============================================================

def plot_pca(embeddings, labels, save_path, title="PCA Embedding Visualization"):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    safe_title = title.lower().replace(" ", "_").replace("-", "_")
    np.save(
        os.path.join(
            os.path.dirname(save_path),
            f"{safe_title}_coords.npy"
        ),
        reduced
    )

    print(
        "Explained variance ratio:",
        pca.explained_variance_ratio_
    )
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, s=8, alpha=0.7, cmap="gist_ncar")
    plt.colorbar(scatter, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("PC-1")
    plt.ylabel("PC-2")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Saving PCA plot to: {save_path}")

    plt.savefig(save_path, bbox_inches="tight")

    print("PCA plot saved.")
    plt.close()


def plot_umap(
    embeddings,
    labels,
    save_path,
    output_dir,
    metadata,
    title="UMAP Embedding Visualization"
):
    reducer = umap.UMAP(n_components=2, random_state=42)
    # NOTE:
    # UMAP is computed independently for each plot.
    # Coordinates across different plots are therefore
    # NOT directly comparable.
    #
    # Future improvement:
    # compute UMAP once and reuse coordinates.
    reduced = reducer.fit_transform(embeddings)
    safe_title = title.lower().replace(" ", "_").replace("-", "_")
    np.save(
        os.path.join(output_dir, f"{safe_title}_coords.npy"),
        reduced
    )

    with open(
        os.path.join(output_dir, "metadata.json"),
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, s=8, alpha=0.7, cmap="gist_ncar")
    plt.colorbar(scatter, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Saving UMAP plot to: {save_path}")
    plt.savefig(save_path, bbox_inches="tight")
    print("UMAP plot saved.")
    plt.close()


def plot_correctness_umap(embeddings, correctness, save_path):
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced[correctness, 0], reduced[correctness, 1], s=8, alpha=0.6, label="Correct")
    plt.scatter(reduced[~correctness, 0], reduced[~correctness, 1], s=8, alpha=0.6, label="Incorrect")
    plt.legend()
    plt.title("Correct vs Incorrect Predictions")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Saving correctness plot to: {save_path}")
    plt.savefig(save_path, bbox_inches="tight")
    print("Correctness plot saved.")
    plt.close()


def save_raw_outputs(output_dir, embeddings, labels, predictions, correctness):
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    np.save(os.path.join(output_dir, "predictions.npy"), predictions)
    np.save(os.path.join(output_dir, "correctness.npy"), correctness)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nUsing device: {device}")

    cfg = load_experiment_config(args.experiment_dir)
    
    # --------------------------------------------------------
    # Determine task semantics
    # --------------------------------------------------------
    task_cfg = cfg.get("task", {})
    stage = task_cfg.get("stage")
    
    clinical_sparse_ids = task_cfg.get("clinical_sparse_global_ids", [])
    if stage == "pretrain_30class":
        n_classes = 30
        clinical_sparse_ids = []
    elif stage == "pretrain_treatment_8class":
        n_classes = 8
        clinical_sparse_ids = []
    elif stage == "transfer_5class":
        n_classes = len(clinical_sparse_ids)
    else:
        n_classes = cfg.get("model", {}).get("n_classes", 30)

    cfg["model"]["n_classes"] = n_classes

    # --------------------------------------------------------
    # Setup data
    # --------------------------------------------------------
    registry = DataRegistry(data_root="data/raw", cfg=cfg)
    split_mode = resolve_split_mode(cfg)
    if split_mode == IID_REFERENCE:
        registry.load("reference")
    else:
        registry.load_all()

    X_ref, _ = registry.get_arrays("reference")
    preprocessor = SpectralPreprocessor.from_config(cfg.get("preprocessing", {}))
    preprocessor.fit(X_ref)

    augmentation = None # No augmentation for evaluation

    # Set loader config for evaluation
    cfg["batch_size"] = 256
    cfg["num_workers"] = 0
    cfg["consistency"] = {}

    loaders = build_all_loaders(
        registry,
        preprocessor,
        augmentation,
        cfg,
        clinical_sparse_ids=clinical_sparse_ids,
        n_classes=n_classes,
    )

    if args.split in loaders:
        dataloader = loaders[args.split]
    elif args.split in loaders.get("ood", {}):
        dataloader = loaders["ood"][args.split]
    else:
        raise ValueError(f"Unknown split: {args.split}. Available loaders: {list(loaders.keys())} + {list(loaders.get('ood', {}).keys())}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = load_model(cfg, args.experiment_dir, device)

    # --------------------------------------------------------
    # Extract embeddings
    # --------------------------------------------------------
    print("\nExtracting embeddings...")
    embeddings, labels, predictions, correctness = extract_embeddings(model, dataloader, device, use_projection=args.use_projection)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
    print("\nNormalizing embeddings...")
    embeddings = StandardScaler().fit_transform(embeddings)
    
    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------
    output_dir = os.path.join(args.experiment_dir, "embedding_analysis", args.split)
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------
    # Save & Plot
    # --------------------------------------------------------
    save_raw_outputs(output_dir, embeddings, labels, predictions, correctness)

    print("\nGenerating PCA plot...")
    plot_pca(embeddings, labels, os.path.join(output_dir, "pca_true_labels.png"), title=f"PCA - True Labels ({args.split})")
    metadata = {
        "stage": stage,
        "split": args.split,
        "n_classes": n_classes,
        "embedding_dim": int(embeddings.shape[1]),
        "num_samples": int(embeddings.shape[0]),
    }

    # Prediction-label UMAP:
    #
    # Shows classifier partition geometry.
    #
    # Important:
    # classifier decision regions may differ
    # substantially from true manifold structure.
    print("\nGenerating prediction-label UMAP plot...")
    plot_umap(
        embeddings,
        predictions,
        os.path.join(output_dir, "umap_predicted_labels.png"),
        output_dir,
        metadata,
        title=f"UMAP - Predicted Labels ({args.split})"
    )
    
    # --------------------------------------------------------
    # SEMANTIC TREATMENT-SPACE VISUALIZATION
    # --------------------------------------------------------
    # Isolate-space (30 classes) vs Treatment-space (8 classes/5 classes)
    #
    # Why this visualization is important:
    # By coloring the exact same embedding geometry with higher-order
    # treatment semantics, we can visually verify whether treatment 
    # abstractions emerge naturally during isolate pretraining (Stage 1),
    # or confirm their structure during transfer learning (Stage 2/3).
    #
    # Note: The embeddings themselves are completely unchanged.
    # Only the semantic interpretation layer (the coloring) differs.
    
    print("\nGenerating treatment-label UMAP plot...")
    if n_classes == 30:
        treatment_labels = np.array([
            ISOLATE_TO_TREATMENT[int(lbl)]
            for lbl in labels
        ])
    else:
        # For Stage 2 (8 classes) or Stage 3 (5 classes), labels are already
        # treatment-space semantic groups.
        treatment_labels = labels

    plot_umap(
        embeddings,
        treatment_labels,
        os.path.join(output_dir, "umap_treatment_labels.png"),
        output_dir,
        metadata,
        title=f"UMAP - Treatment Labels ({args.split})"
    )
    
    print("\nGenerating true-label UMAP plot...")
    plot_umap(
        embeddings,
        labels,
        os.path.join(output_dir, "umap_true_labels.png"),
        output_dir,
        metadata,
        title=f"UMAP - True Labels ({args.split})"
    )
    
    # Correctness topology:
    #
    # Boundary-localized errors suggest
    # meaningful manifold learning.
    #
    # Distributed errors may indicate
    # unstable representation geometry.
    print("\nGenerating correctness plot...")
    plot_correctness_umap(
        embeddings,
        correctness,
        os.path.join(output_dir, "umap_correctness.png")
    )

    print(f"\nDone. Saved outputs to:\n{output_dir}")

if __name__ == "__main__":
    main()
