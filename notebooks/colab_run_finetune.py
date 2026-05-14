# =============================================================================
# Google Colab: Raman Spectral Classifier — Finetune Pipeline (Fixed)
# =============================================================================
#
# This script runs the complete pretrain → finetune → evaluate pipeline
# on Google Colab with GPU support. It includes sanity checks at every stage.
#
# Usage: Copy this entire file into a single Colab cell and run it.
#        Make sure GPU runtime is enabled (Runtime → Change runtime type → GPU).
# =============================================================================

# %% [1/7] Clone Repository and Install Dependencies
# ===================================================

import subprocess, sys, os

# Clone the repo (skip if already present)
REPO_DIR = "/content/raman-spectral-classifier"
if not os.path.exists(REPO_DIR):
    subprocess.run(
        ["git", "clone", "https://github.com/rana-rohit/raman-spectral-classifier.git", REPO_DIR],
        check=True,
    )
    print("✅ Repository cloned")
else:
    # Pull latest changes
    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)
    print("✅ Repository updated")

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

# Install dependencies (Colab already has torch, numpy, scipy, sklearn)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "pyyaml", "matplotlib", "seaborn", "pandas"],
    check=True,
)
print("✅ Dependencies installed")

# %% [2/7] Verify GPU and Imports
# =================================

import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ PyTorch {torch.__version__} | Device: {device}")
if device != "cuda":
    print("⚠️  WARNING: No GPU detected. Training will be very slow.")
    print("   Go to Runtime → Change runtime type → GPU")

from src.data.augmentation import AugmentationPipeline
from src.data.dataloader import build_all_loaders
from src.data.preprocessing import SpectralPreprocessor
from src.data.registry import DataRegistry
from src.evaluation.evaluator import ModelEvaluator
from src.models.registry import get_model, model_summary
from src.training.finetuner import finetune
from src.training.trainer import build_trainer
from src.utils.checkpoint import load_best_model
from src.utils.config import load_config, save_config
from src.utils.seed import set_seed

print("✅ All modules imported successfully")

# %% [3/7] Load Configuration and Data
# ======================================

SEED = 42
MODEL_NAME = "cnn"  # Options: cnn, resnet1d, transformer

set_seed(SEED)

cfg = load_config(
    "configs/data/splits.yaml",
    "configs/data/preprocessing.yaml",
    "configs/data/augmentation.yaml",
    "configs/training/base.yaml",
    f"configs/model/{MODEL_NAME}.yaml",
)
cfg = dict(cfg)

print(f"\n📋 Config loaded for model: {MODEL_NAME}")
print(f"   Classes: {cfg['dataset']['n_classes_full']}")
print(f"   Signal length: {cfg['dataset']['signal_length']}")

# Load all data splits
registry = DataRegistry(data_root="data/raw", cfg=cfg)
registry.load_all()
registry.summary()

# %% [4/7] Sanity Check: Verify Label Integrity
# ===============================================

print("\n" + "=" * 60)
print("  LABEL INTEGRITY CHECK")
print("=" * 60)

checks_passed = True

for split_name in registry.available_splits():
    meta = registry.get_meta(split_name)
    unique_labels = np.unique(meta.y)
    label_counts = {int(l): int(c) for l, c in zip(*np.unique(meta.y, return_counts=True))}

    status = "✅" if len(unique_labels) > 1 else "❌"
    if len(unique_labels) <= 1:
        checks_passed = False

    print(f"\n  {status} {split_name}:")
    print(f"     Samples: {len(meta.y):,}")
    print(f"     Unique labels: {len(unique_labels)}")
    print(f"     Label range: [{unique_labels.min()}, {unique_labels.max()}]")
    if len(unique_labels) <= 5:
        print(f"     Distribution: {label_counts}")

if not checks_passed:
    raise RuntimeError("❌ Label integrity check FAILED. Fix data before proceeding.")

print(f"\n{'✅ All label checks passed':=^60}")

# %% [5/7] Build Loaders, Model, and Pretrain
# =============================================

# Preprocessing
X_ref, _ = registry.get_arrays("reference")
preprocessor = SpectralPreprocessor.from_config(cfg["preprocessing"])
preprocessor.fit(X_ref)

# Augmentation
augmentation = AugmentationPipeline.from_config(cfg["augmentation"])
if len(augmentation.steps) == 0 or augmentation.p == 0:
    augmentation = None

# DataLoaders
loader_cfg = {
    "batch_size": cfg.get("training", {}).get("batch_size", 128),
    "num_workers": cfg.get("training", {}).get("num_workers", 2),
    "validation": cfg["validation"],
    "seed": SEED,
    "consistency": cfg.get("training", {}).get("consistency", {}),
}
loaders = build_all_loaders(registry, preprocessor, augmentation, loader_cfg)

print(f"\n📊 DataLoader summary:")
print(f"   Train:    {len(loaders['train'].dataset):>7,} samples")
print(f"   Val:      {len(loaders['val'].dataset):>7,} samples")
print(f"   Test:     {len(loaders['test'].dataset):>7,} samples")
print(f"   Finetune: {len(loaders['finetune'].dataset):>7,} samples")

if "clinical" in loaders:
    clin_ds = loaders["clinical"].dataset
    clin_unique = np.unique(clin_ds.y)
    print(f"   Clinical: {len(clin_ds):>7,} samples ({len(clin_unique)} classes)")
    if len(clin_unique) <= 1:
        print("   ❌ WARNING: Clinical labels are all the same!")
    else:
        print(f"   ✅ Clinical labels verified: {clin_unique.tolist()}")

for ood_name, ood_loader in loaders.get("ood", {}).items():
    print(f"   OOD ({ood_name}): {len(ood_loader.dataset):>7,} samples")

# Verify finetune loader labels (the critical check)
ft_labels = loaders["finetune"].dataset.y
ft_unique = np.unique(ft_labels)
print(f"\n🔍 Finetune label check: {len(ft_unique)} unique classes")
assert len(ft_unique) > 1, "FATAL: Finetune labels have only 1 class!"
print(f"   ✅ Finetune labels OK: classes {ft_unique[:10]}...")

# Build model
model = get_model(MODEL_NAME, cfg)
model_summary(model)

# Pretrain
import time

exp_name = f"{MODEL_NAME}_{time.strftime('%Y%m%d_%H%M%S')}"
exp_dir = os.path.join("experiments", exp_name)
os.makedirs(exp_dir, exist_ok=True)
save_config(cfg, os.path.join(exp_dir, "config.yaml"))

print(f"\n🏋️ Starting pretraining → {exp_dir}")

trainer = build_trainer(
    model=model,
    loaders=loaders,
    cfg=cfg,
    exp_dir=exp_dir,
    n_classes=cfg["dataset"]["n_classes_full"],
)
trainer.fit()
load_best_model(exp_dir, model)

print("✅ Pretraining complete")

# %% [6/7] Pre-Finetune Evaluation + Finetune
# =============================================

print("\n📊 Pre-finetune evaluation...")
evaluator = ModelEvaluator(
    model=model,
    model_name=MODEL_NAME,
    n_classes=cfg["dataset"]["n_classes_full"],
    device=str(next(model.parameters()).device),
    cfg=cfg,
)
evaluator.evaluate_all(loaders)
evaluator.save(os.path.join(exp_dir, "pretrain_results.json"))

# Finetune
print("\n🔧 Starting finetuning...")
finetune_dir = os.path.join(exp_dir, "finetune")
ft_results = finetune(
    model=model,
    pretrained_exp_dir=exp_dir,
    loaders=loaders,
    cfg=cfg,
    exp_dir=finetune_dir,
    freeze_epochs=3,
    n_classes=cfg["dataset"]["n_classes_full"],
)

print("✅ Finetuning complete")

# %% [7/7] Post-Finetune Evaluation and Results
# ================================================

print("\n" + "=" * 60)
print("  FINAL RESULTS (POST-FINETUNE)")
print("=" * 60)

# Extract metrics
val_f1 = ft_results.get("val", {}).get("f1_macro", float("nan"))
val_acc = ft_results.get("val", {}).get("accuracy", float("nan"))
test_f1 = ft_results.get("test", {}).get("f1_macro", float("nan"))
test_acc = ft_results.get("test", {}).get("accuracy", float("nan"))

print(f"\n  Val  — Accuracy: {val_acc:.4f}  |  F1 (macro): {val_f1:.4f}")
print(f"  Test — Accuracy: {test_acc:.4f}  |  F1 (macro): {test_f1:.4f}")

for ood_name, ood_metrics in ft_results.get("ood", {}).items():
    ood_f1 = ood_metrics.get("f1_macro", float("nan"))
    ood_acc = ood_metrics.get("accuracy", float("nan"))
    print(f"  {ood_name} — Accuracy: {ood_acc:.4f}  |  F1 (macro): {ood_f1:.4f}")

# Validation checklist
print("\n" + "-" * 60)
print("  HEALTH CHECK")
print("-" * 60)

if val_f1 > 0.50:
    print("  ✅ Val F1 > 0.50 — model is learning properly")
else:
    print("  ❌ Val F1 <= 0.50 — potential issue (check training logs)")

if test_f1 > 0.40:
    print("  ✅ Test F1 > 0.40 — generalizing to held-out data")
else:
    print("  ❌ Test F1 <= 0.40 — generalization may still need work")

if abs(val_f1 - test_f1) < 0.20:
    print("  ✅ Val-Test gap < 0.20 — no severe overfitting")
else:
    print("  ⚠️  Val-Test gap >= 0.20 — possible overfitting")

print(f"\n  📁 All artifacts saved to: {exp_dir}/")
print(f"  📁 Finetune artifacts:     {finetune_dir}/")
print("=" * 60)
