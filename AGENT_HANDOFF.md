# AGENT HANDOFF — Raman Spectral Classifier
> Read this entire file before doing anything. It contains the full project context,
> every design decision made, every file already built, and exactly what remains to do.
> Do not re-design anything already listed as DONE. Pick up from WHERE WE LEFT OFF.

---

## 1. YOUR ROLE

You are a technical co-researcher and ML system architect on this project.
You know this codebase in full — you helped design and build every file in it.
Think like a researcher, not just a coder. Justify decisions. Do not re-design
things that are already done. Ask before changing any core architecture.

---

## 2. PROJECT SUMMARY

**Task:** Research-grade deep learning classification of 1D Raman spectral signals.
**Goal:** Compare CNN, ResNet1D, Transformer, and Hybrid CNN-Transformer architectures
with systematic cross-domain generalisation evaluation on clinical data.
**Intended output:** Results strong enough for a research publication / internship presentation.

**Project root:** `raman-spectral-classifier/`
**Language:** Python 3.10, PyTorch 2.0+
**Environment:** conda env named `spectral-clf` (see `environment.yml`)

---

## 3. DATASET — CONFIRMED FACTS

All discovered via EDA. Do not assume anything different.

| Split | Samples | Classes | Samples/class | Role |
|---|---|---|---|---|
| reference | 60,000 | 30 (classes 0–29) | 2,000 | Primary training |
| finetune | 3,000 | 30 (classes 0–29) | 100 | Domain adaptation |
| test | 3,000 | 30 (classes 0–29) | 100 | Held-out evaluation |
| 2018clinical | 10,000 | 5 (classes 0,2,3,5,6) | 2,000 | OOD generalisation |
| 2019clinical | 2,500 | 5 (classes 0,2,3,5,6) | 500 | OOD generalisation |

- Signal length: **1000 points**, fixed across every split — no padding needed
- All data already normalised to **[0.0, 1.0]**
- All splits **perfectly balanced** within their class sets
- Shared classes present in ALL splits: **[0, 2, 3, 5, 6]**
- OOD evaluation ALWAYS filters to shared classes only — never score all 30
- Validation is carved from `reference` at runtime: 80/20 stratified split
  → 48,000 train samples, 12,000 val samples, 400 per class each
- File naming convention: `data/raw/X_reference.npy`, `data/raw/y_reference.npy`, etc.

---

## 4. EDA FINDINGS — KEY NUMBERS

These numbers directly drive preprocessing and augmentation choices.

### 4a. Mean absolute difference (reference mean vs clinical mean per class)
```
Class 0:  0.073 vs 2018clinical,  0.073 vs 2019clinical   ← largest shift
Class 2:  0.060 vs 2018clinical,  0.058 vs 2019clinical
Class 3:  0.060 vs 2018clinical,  0.051 vs 2019clinical
Class 5:  0.053 vs 2018clinical,  0.068 vs 2019clinical
Class 6:  0.029 vs 2018clinical,  0.027 vs 2019clinical   ← smallest shift
```

### 4b. Within-class shape variance (mean point-wise std)
```
split          class0   class2   class3   class5   class6
2018clinical   0.117    0.139    0.130    0.113    0.110
2019clinical   0.103    0.125    0.121    0.123    0.108
reference      0.089    0.109    0.106    0.098    0.111
test           0.083    0.102    0.082    0.065    0.115
```
Clinical variance is 25–30% higher than reference for classes 2 and 3.
Class 6 stays close to reference. Classes 0 and 5 moderate increase.

### 4c. PCA centroid distances (reference centroid vs clinical centroid)
```
Class 0:  15.89 vs 2018clinical,  10.08 vs 2019clinical   ← large drift
Class 2:  11.56 vs 2018clinical,   7.88 vs 2019clinical
Class 3:  14.81 vs 2018clinical,   6.99 vs 2019clinical
Class 5:  12.99 vs 2018clinical,  18.04 vs 2019clinical   ← largest 2019 drift
Class 6:   1.74 vs 2018clinical,   3.13 vs 2019clinical   ← safe, transfers easily
```

### 4d. Interpretation (filled by researcher after running EDA)
- Peak locations broadly similar across splits; main shift is **baseline/intensity offset**
- Baseline correction and robust intensity normalisation are highest priority
- Peak-shift augmentation is lower priority than baseline/amplitude handling
- Domain shift is **class-specific** — class 6 transfers easily, classes 0/2/3/5 need adaptation
- PCA shows splits visibly separated → **fine-tuning is essential, not optional**
- 2018clinical and 2019clinical behave differently — always report them as separate columns
- Class 5 drifts most in 2019clinical (distance 18.0) — expect it to dominate error analysis

### 4e. Config changes required before training (EDA-motivated)
These are NOT yet applied. Apply them before running any training:

1. In `configs/data/augmentation.yaml`:
   - Change `baseline_shift.max_shift` from `0.05` → **`0.08`**
     (default 0.05 doesn't cover the 0.073 observed shift for class 0)
   - Change `gaussian_noise.max_std` from `0.02` → **`0.03`**
     (clinical variance is 25–30% higher than reference for classes 2 and 3)

2. Create `configs/model/transformer_focal.yaml`:
   - Copy `transformer.yaml`, change `loss: label_smoothing` → `loss: focal`
   - Motivated by large centroid distances for classes 0, 3, 5 (all >10)
   - Run as additional variant alongside standard transformer

---

## 5. COMPLETE REPOSITORY STRUCTURE

Everything listed below is ALREADY BUILT and should NOT be recreated from scratch.
Read the actual files — do not rewrite them unless fixing a specific bug.

```
raman-spectral-classifier/
│
├── configs/
│   ├── data/
│   │   ├── splits.yaml          ✅ DONE — all split definitions, roles, shared classes
│   │   ├── preprocessing.yaml   ✅ DONE — per-sample mean sub, SavGol, clip pipeline
│   │   └── augmentation.yaml    ✅ DONE — gaussian noise, baseline shift, amplitude, spectral shift
│   ├── model/
│   │   ├── cnn.yaml             ✅ DONE — channels [32,64,128,256], kernels [7,15,15,31]
│   │   ├── resnet1d.yaml        ✅ DONE — channels [64,128,256,512], n_blocks [2,2,2,2]
│   │   ├── transformer.yaml     ✅ DONE — patch_size=20, d_model=256, 6 layers, 8 heads
│   │   └── hybrid.yaml          ✅ DONE — handoff_blocks=2, d_model=256, 4 layers
│   └── training/
│       └── base.yaml            ✅ DONE — AdamW lr=0.001, cosine schedule, patience=10
│
├── src/
│   ├── __init__.py              ✅ DONE
│   ├── data/
│   │   ├── __init__.py          ✅ DONE
│   │   ├── split_roles.py       ✅ DONE — SplitRole enum (SOURCE, ADAPTATION, HOLDOUT, OOD_EVAL, VALIDATION)
│   │   ├── registry.py          ✅ DONE — DataRegistry: loads splits, enforces roles, returns arrays
│   │   ├── preprocessing.py     ✅ DONE — SpectralPreprocessor: fit on train only, composable steps
│   │   ├── augmentation.py      ✅ DONE — AugmentationPipeline: 4 physically-valid augmentations
│   │   ├── dataset.py           ✅ DONE — SpectralDataset (PyTorch), class_filter, make_train_val_split
│   │   └── dataloader.py        ✅ DONE — build_all_loaders(): returns dict with train/val/test/finetune/ood
│   ├── models/
│   │   ├── __init__.py          ✅ DONE
│   │   ├── cnn.py               ✅ DONE — CNN1D: 4 ConvBlocks, GlobalAveragePool, Grad-CAM compatible
│   │   ├── resnet1d.py          ✅ DONE — ResNet1D: 4 stages of ResidualBlock1D, shortcut connections
│   │   ├── transformer.py       ✅ DONE — SpectralTransformer: PatchEmbedding, CLS token, pre-norm layers
│   │   ├── hybrid.py            ✅ DONE — HybridCNNTransformer: CNNStem (1/2/3 blocks) + Transformer
│   │   └── registry.py          ✅ DONE — get_model(name, cfg): factory for all architectures
│   ├── training/
│   │   ├── __init__.py          ✅ DONE
│   │   ├── losses.py            ✅ DONE — CrossEntropy, LabelSmoothingCrossEntropy, FocalLoss, get_loss()
│   │   ├── scheduler.py         ✅ DONE — WarmupCosineScheduler, get_scheduler()
│   │   ├── trainer.py           ✅ DONE — Trainer class: fit(), evaluate(), evaluate_ood(), build_trainer()
│   │   └── finetuner.py         ✅ DONE — finetune(): pretrain→adapt, few-shot subsampling, freeze stem
│   ├── evaluation/
│   │   ├── __init__.py          ✅ DONE
│   │   ├── metrics.py           ✅ DONE — accuracy, macro F1, per-class F1, ROC-AUC, MCC, confusion matrix
│   │   └── evaluator.py         ✅ DONE — ModelEvaluator, compare_models(), mcnemar_test()
│   ├── interpretability/
│   │   ├── __init__.py          ✅ DONE
│   │   ├── gradcam1d.py         ✅ DONE — GradCAM1D: hooks on features/stage4/cnn_stem, upsample to 1000
│   │   └── attention_viz.py     ✅ DONE — extract_attention, cls_to_patch_attention, attention_rollout
│   └── utils/
│       ├── __init__.py          ✅ DONE
│       ├── seed.py              ✅ DONE — set_seed(42): seeds random, numpy, torch, cudnn deterministic
│       ├── config.py            ✅ DONE — load_config(*paths): YAML merge with dot-access Config class
│       ├── checkpoint.py        ✅ DONE — save_checkpoint(), load_checkpoint(), load_best_model()
│       └── logging.py           ✅ DONE — ExperimentLogger: console + metrics.json + summary.json
│
├── scripts/
│   ├── setup_data.py            ✅ DONE — verify pipeline, smoke test all loaders before training
│   ├── train.py                 ✅ DONE — entry point: --model cnn|resnet1d|transformer|hybrid
│   ├── evaluate.py              ✅ DONE — single model eval or --compare multiple models
│   └── run_ablation.py          ✅ DONE — hybrid_handoff, patch_size, augmentation, few_shot
│
├── notebooks/
│   ├── 02_domain_analysis_complete.ipynb  ✅ DONE — EDA gaps 1/2/3, interpretation table filled
│   ├── 04_interpretability.ipynb          ✅ DONE — Grad-CAM + attention map visualisations
│   └── 05_results_analysis.ipynb          ✅ DONE — paper figures: heatmap, transfer gap, ablations, McNemar
│
├── tests/
│   └── test_data_layer.py       ✅ DONE — 18 unit tests for preprocessing, augmentation, dataset, roles
│
├── pyproject.toml               ✅ DONE — pip install -e . makes src importable everywhere
├── setup.py                     ✅ DONE — fallback for older setuptools
├── conftest.py                  ✅ DONE — pytest path fix
├── requirements.txt             ✅ DONE
├── environment.yml              ✅ DONE — conda env: spectral-clf
├── README.md                    ✅ DONE
└── IMPLEMENTATION_PLAN.md       ✅ DONE
```

---

## 6. KEY DESIGN DECISIONS — DO NOT CHANGE WITHOUT DISCUSSION

These are non-negotiable invariants built into the system:

1. **Preprocessor fit on `reference` only.** `SpectralPreprocessor.fit()` is called
   once on the reference training split. All other splits use `.transform()` only.
   Changing this introduces data leakage.

2. **`test` split touched once per model.** Used only at final evaluation.
   Never during development, hyperparameter tuning, or model selection.

3. **OOD splits are evaluation-only.** Never train on 2018clinical or 2019clinical.
   Always filter to shared classes [0,2,3,5,6] when evaluating them.

4. **`set_seed(42)` called before everything.** Every training script calls this
   before data loading and model instantiation. Reproducibility is non-negotiable.

5. **Class filtering for OOD evaluation.** The `SpectralDataset` `class_filter`
   parameter and `DataRegistry.get_eval_classes()` handle this automatically.
   The 30-class model is evaluated on the 5 clinical classes only — never retrained.

6. **Two clinical cohorts always reported separately.** Never average 2018clinical
   and 2019clinical into one number — they behave differently (class 5 drifts
   more in 2019clinical than 2018clinical).

7. **All LR values as floats in YAML** (e.g. `0.001` not `1e-3`). PyYAML parses
   scientific notation as strings on some versions. Always use decimal notation.

---

## 7. MODEL ARCHITECTURES — EXACT SPECS

### CNN1D (`src/models/cnn.py`)
```
Input (B, 1, 1000)
→ ConvBlock(1→32,   k=7)  + MaxPool(2)  → (B, 32,  500)
→ ConvBlock(32→64,  k=15) + MaxPool(2)  → (B, 64,  250)
→ ConvBlock(64→128, k=15)               → (B, 128, 250)
→ ConvBlock(128→256,k=31) + MaxPool(2)  → (B, 256, 125)
→ GlobalAveragePool                     → (B, 256)
→ Dropout(0.3) → Linear(256, 30)       → (B, 30)
```
- ConvBlock = Conv1d + BatchNorm1d + ReLU, padding="same"
- GAP (not Flatten) — required for Grad-CAM to work correctly
- `get_feature_maps(x)` exposes final conv output for Grad-CAM

### ResNet1D (`src/models/resnet1d.py`)
```
Input (B, 1, 1000)
→ Stem: Conv(1→64, k=15) + BN + ReLU + MaxPool(2)  → (B, 64, 500)
→ Stage1: 2× ResBlock(64→64,   stride=1)            → (B, 64,  500)
→ Stage2: 2× ResBlock(64→128,  stride=2)            → (B, 128, 250)
→ Stage3: 2× ResBlock(128→256, stride=2)            → (B, 256, 125)
→ Stage4: 2× ResBlock(256→512, stride=2)            → (B, 512,  63)
→ GlobalAveragePool → Dropout(0.3) → Linear(512,30) → (B, 30)
```
- ResidualBlock1D: Conv(k=3)+BN+ReLU → Conv(k=3)+BN → +shortcut → ReLU
- Shortcut is 1×1 Conv when channels or stride change, else Identity
- `get_feature_maps(x)` returns stage4 output for Grad-CAM

### SpectralTransformer (`src/models/transformer.py`)
```
Input (B, 1, 1000)
→ PatchEmbedding(patch=20): Conv1d(1→256, k=20, stride=20) → (B, 50, 256)
→ LayerNorm                                                  → (B, 50, 256)
→ Prepend CLS token                                          → (B, 51, 256)
→ Sinusoidal positional encoding + Dropout(0.1)
→ 6× TransformerEncoderLayer(d=256, heads=8, d_ff=512)
   [pre-norm: LN → MHA → +x → LN → FFN → +x]
→ LayerNorm on CLS token                                     → (B, 256)
→ Dropout(0.1) → Linear(256, 30)                            → (B, 30)
```
- Uses label smoothing loss (0.1) and warmup cosine LR (warmup 10 epochs)
- `get_attention_maps(x)` returns list of (B, 8, 51, 51) per layer
- Attention weights stored in each layer for interpretability

### HybridCNNTransformer (`src/models/hybrid.py`)
```
Input (B, 1, 1000)
→ CNNStem(n_blocks=2):
    ConvBlock(1→64,   k=7)  + MaxPool(2)  → (B, 64,  500)
    ConvBlock(64→128, k=15) + MaxPool(2)  → (B, 128, 250)
→ Transpose: (B, 250, 128)
→ Linear(128→256) + LayerNorm            → (B, 250, 256)
→ Prepend CLS token                      → (B, 251, 256)
→ Positional encoding
→ 4× TransformerEncoderLayer(d=256, heads=8, d_ff=512)
→ LayerNorm on CLS → Dropout(0.1) → Linear(256,30) → (B, 30)
```
- `handoff_blocks` ∈ {1,2,3} controls CNN depth before Transformer
- n_blocks=1 → 500 tokens, n_blocks=2 → 250 tokens, n_blocks=3 → 125 tokens
- `get_cnn_features(x)` for Grad-CAM on CNN stem
- `get_attention_maps(x)` for attention visualisation

---

## 8. TRAINING INFRASTRUCTURE

### Trainer (`src/training/trainer.py`)
- `build_trainer(model, loaders, cfg, exp_dir, n_classes)` — main factory
- AdamW optimiser, gradient clipping at 1.0 (essential for Transformer)
- Early stopping on val accuracy (patience=10)
- Saves checkpoint every epoch + `best.pt` when val accuracy improves
- Device auto-detection: CUDA → MPS → CPU
- `trainer.fit()` → returns best metrics dict
- `trainer.evaluate(loader, split_name)` → runs on any loader
- `trainer.evaluate_ood()` → runs all OOD splits in `loaders["ood"]`

### Losses (`src/training/losses.py`)
- `get_loss(name, **kwargs)`: `"cross_entropy"`, `"label_smoothing"`, `"focal"`
- LabelSmoothingCrossEntropy: smoothing=0.1 for Transformer/Hybrid
- FocalLoss: gamma=2.0, for hard cross-domain examples (classes 0,3,5)

### Schedulers (`src/training/scheduler.py`)
- CNN/ResNet: cosine annealing (T_max=100, eta_min=1e-6)
- Transformer/Hybrid: WarmupCosineScheduler (warmup=10 epochs, then cosine)
- `get_scheduler(name, optimizer, cfg)`: factory

### Finetuner (`src/training/finetuner.py`)
- `finetune(model, pretrained_exp_dir, loaders, cfg, exp_dir, n_shots_per_class, freeze_epochs)`
- Loads best.pt from pretrained experiment
- Optional few-shot subsampling for Experiment 4
- Optional frozen stem phase before full fine-tuning

---

## 9. EVALUATION

### Metrics (`src/evaluation/metrics.py`)
All implemented in pure numpy — no sklearn dependency:
- Accuracy, Macro F1, Per-class F1
- ROC-AUC (one-vs-rest, trapezoidal rule — uses `np.trapezoid` for numpy 2.x compat)
- MCC (Matthews Correlation Coefficient)
- Confusion matrix

### Evaluator (`src/evaluation/evaluator.py`)
- `ModelEvaluator.evaluate_all(loaders)` → full results matrix
- `transfer_gap = test_accuracy - ood_accuracy` computed automatically
- `ModelEvaluator.mcnemar_test(preds_a, preds_b, targets)` → pairwise significance
- `compare_models(results_list, split_names)` → formatted comparison table

---

## 10. INTERPRETABILITY

### Grad-CAM (`src/interpretability/gradcam1d.py`)
- `GradCAM1D(model).compute(x, target_class, signal_length)` → (1000,) saliency array
- Hooks on `model.features` (CNN), `model.stage4` (ResNet), `model.cnn_stem` (Hybrid)
- ReLU + normalise → upsample to original signal length via linear interpolation

### Attention (`src/interpretability/attention_viz.py`)
- `extract_attention(model, x)` → list of (B, heads, seq, seq) per layer
- `cls_to_patch_attention(attn_maps, layer=-1)` → (50,) patch weights
- `attention_rollout(attn_maps)` → multi-layer attribution (Abnar & Zuidema 2020)
- `patch_attention_to_signal(patch_attn, signal_length=1000, patch_size=20)` → (1000,)
- `per_class_attention(model, dataset, class_ids)` → dict of mean signal attentions

---

## 11. KNOWN BUGS FIXED

1. **`src/data/__init__.py` was empty** → caused `ModuleNotFoundError: No module named 'src.data.split_roles'`
   Fix: all `__init__.py` files in src tree now contain `# package`

2. **YAML scientific notation parsed as string** → `lr: 1e-3` came through as the string `"1e-3"` not float
   Fix: all LR values in YAML use decimal notation (`0.001`, `0.0001`, etc.)

3. **`ClipTransform` config key mismatch** → YAML had `min:` but constructor expected `min_val:`
   Fix: `configs/data/preprocessing.yaml` uses `min_val:` and `max_val:`

4. **`np.trapz` removed in numpy 2.x** → `AttributeError: module numpy has no attribute trapz`
   Fix: `src/evaluation/metrics.py` uses `getattr(np, "trapezoid", None) or np.trapz`

5. **`pyproject.toml` bad build backend** → `setuptools.backends.legacy:build` is wrong
   Fix: corrected to `setuptools.build_meta`

6. **Scripts use `os.path.dirname(__file__)` which can be relative on Windows**
   Fix: `scripts/setup_data.py` uses `Path(__file__).resolve().parent.parent`

---

## 12. FIRST-TIME SETUP (run once)

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate spectral-clf

# 2. Install package so src.* is importable from anywhere
pip install -e .

# 3. Verify data pipeline
python scripts/setup_data.py
# Expected: all 5 splits load, batch shape (256, 1, 1000), OOD classes [0,2,3,5,6]
```

---

## 13. WHAT REMAINS — ORDERED EXECUTION PLAN

### Step 1 — Apply EDA-motivated config changes (5 minutes)
- [ ] Edit `configs/data/augmentation.yaml`: `baseline_shift.max_shift` → `0.08`
- [ ] Edit `configs/data/augmentation.yaml`: `gaussian_noise.max_std` → `0.03`
- [ ] Create `configs/model/transformer_focal.yaml` (copy transformer.yaml, set `loss: focal`)

### Step 2 — Run setup verification
- [ ] `python scripts/setup_data.py` — must pass with no errors

### Step 3 — Train all models (run sequentially or on separate GPUs)
- [ ] `python scripts/train.py --model cnn --exp-name cnn_v1`
- [ ] `python scripts/train.py --model resnet1d --exp-name resnet1d_v1`
- [ ] `python scripts/train.py --model transformer --exp-name transformer_v1`
- [ ] `python scripts/train.py --model transformer --exp-name transformer_focal --override training.loss=focal`
- [ ] `python scripts/train.py --model hybrid --exp-name hybrid_v1`

### Step 4 — Ablation studies
- [ ] `python scripts/run_ablation.py --ablation hybrid_handoff`
  (trains hybrid with handoff_blocks=1,2,3 — core research contribution)
- [ ] `python scripts/run_ablation.py --ablation patch_size`
  (patch sizes 10,20,25,50 for Transformer)
- [ ] `python scripts/run_ablation.py --ablation augmentation`
  (disable one augmentation at a time — validates EDA findings)

### Step 5 — Fine-tuning and few-shot (Experiments 3 and 4)
- [ ] `python scripts/train.py --model hybrid --exp-name hybrid_finetuned --override training.lr=0.0001 training.max_epochs=50`
- [ ] `python scripts/run_ablation.py --ablation few_shot --pretrained experiments/hybrid_v1`
  (10, 25, 50, 100 shots/class — produces the learning curve figure)

### Step 6 — Final evaluation
- [ ] `python scripts/evaluate.py --compare experiments/cnn_v1 experiments/resnet1d_v1 experiments/transformer_v1 experiments/transformer_focal experiments/hybrid_v1 experiments/hybrid_finetuned`
  (generates full comparison table + McNemar significance tests)

### Step 7 — Interpretability
- [ ] Run `notebooks/04_interpretability.ipynb` — update `EXPERIMENT_DIRS` dict with actual exp names
- [ ] Key question: does Grad-CAM highlight different regions for class 5 (high drift) vs class 6 (low drift)?
  If yes, that is a direct finding linking EDA domain shift to model behaviour.

### Step 8 — Paper figures
- [ ] Run `notebooks/05_results_analysis.ipynb` — update `EXP_NAMES` dict with actual exp names
- [ ] Generates: results heatmap, transfer gap chart, ablation tables, few-shot curve, per-class F1

---

## 14. EXPECTED RESULTS (rough targets based on dataset structure)

- CNN val accuracy: >85% (sanity check — if below this, debug data pipeline)
- ResNet1D val accuracy: 87–92% (should outperform plain CNN)
- Transformer val accuracy: similar to ResNet1D, slower to converge
- Hybrid val accuracy: should match or exceed best single architecture
- Transfer gap (test → clinical): expect 10–20% drop for classes 0,2,3,5; <5% for class 6
- After fine-tuning: expect clinical accuracy to recover 5–15 percentage points
- Few-shot curve: class 6 should plateau early (~25 shots); class 5 needs more samples

---

## 15. PAPER RESEARCH CONTRIBUTIONS (what makes this publishable)

1. Systematic comparison of CNN, ResNet1D, Transformer, Hybrid on 1D Raman spectral data
2. Characterisation of class-specific domain shift between source and clinical cohorts
3. Ablation of optimal CNN-Transformer handoff point (handoff_blocks ∈ {1,2,3})
4. Few-shot adaptation efficiency: learning curve with varying shots/class
5. Interpretability: Grad-CAM/attention attribution correlated with spectral domain shift
6. Separate reporting per clinical cohort (2018 vs 2019 behave differently for class 5)

---

## 16. INVARIANTS — NEVER CHANGE THESE

| Rule | Reason |
|---|---|
| Preprocessor fit on `reference` only | Prevents data leakage into val/test/clinical |
| `test` split used once per model | Prevents test set overfitting |
| OOD splits never used for training | Maintains honest generalisation measurement |
| `set_seed(42)` before all else | Reproducibility |
| OOD eval always filters to classes [0,2,3,5,6] | 25 absent classes would inflate accuracy |
| 2018clinical and 2019clinical always separate columns | They behave differently — class 5 especially |
| All YAML LR values as decimals not scientific notation | PyYAML string parsing bug |
