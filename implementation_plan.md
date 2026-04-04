# Implementation Plan — Spectral Signal Classifier
## Research-Grade Deep Learning System

> Last updated after EDA completion. All decisions are grounded in observed data properties.

---

## Confirmed Dataset Facts (from EDA)

| Property | Value |
|---|---|
| Signal length | 1,000 (fixed across all splits) |
| Source classes | 30 (classes 0–29, perfectly balanced) |
| Clinical classes | 5 (classes 0, 2, 3, 5, 6 only) |
| Reference samples | 60,000 (2,000/class) |
| Finetune samples | 3,000 (100/class, all 30 classes) |
| Test samples | 3,000 (100/class, all 30 classes) |
| 2018clinical samples | 10,000 (2,000/class, 5 classes) |
| 2019clinical samples | 2,500 (500/class, 5 classes) |
| Value range | [0.0, 1.0] — already normalised |
| Mean offset | Source ~0.43, Clinical ~0.46 |
| Validation split | 20% of reference → 12,000 samples |

---

## Experimental Protocol (Locked)

```
Exp 1 — In-domain classification
  Train: reference (80%)   Val: reference (20%)   Test: test split
  Models: CNN, ResNet1D, Transformer, Hybrid

Exp 2 — Zero-shot cross-domain generalisation
  Same models as Exp 1, evaluated directly on 2018clinical + 2019clinical
  Classes filtered to shared 5: {0, 2, 3, 5, 6}
  Key metric: transfer_gap = test_acc - clinical_acc

Exp 3 — Fine-tune adaptation
  Pretrained models from Exp 1, fine-tuned on finetune split
  Re-evaluated on test + both clinical splits

Exp 4 — Few-shot learning curve (if time permits)
  Subsample finetune: 10, 25, 50, 100 samples/class
  Shows adaptation efficiency vs data budget
```

---

## Phase Roadmap

### PHASE 0 — Environment & Repository ✅ DONE
- [x] Directory structure created
- [x] configs/data/ — splits, preprocessing, augmentation
- [x] src/data/ — registry, preprocessing, augmentation, dataset, dataloader
- [x] src/utils/seed.py
- [x] tests/test_data_layer.py
- [x] scripts/setup_data.py

**Checkpoint:** Run `pytest tests/test_data_layer.py -v` → all green.
Run `python scripts/setup_data.py` → all splits load, batches correct shape.

---

### PHASE 1 — Complete EDA ← CURRENT PHASE
**Notebook:** `notebooks/02_domain_analysis_complete.ipynb`

Tasks:
- [ ] Run Gap 1: Cross-split mean spectra per shared class
- [ ] Run Gap 1b: Mean absolute difference table
- [ ] Run Gap 2: Within-class variance by split
- [ ] Run Gap 3: PCA on shared classes (coloured by class AND by split)
- [ ] Run Gap 3b: Centroid distance table
- [ ] Fill in interpretation table from notebook

**Decision gate after Phase 1:**
Use centroid distances from Gap 3b to calibrate augmentation strength.
If centroid distance > 5.0: increase baseline_shift max to 0.08.
If class clusters are split-separated in PCA: fine-tuning is critical (prioritise Exp 3).

---

### PHASE 2 — CNN Baseline
**Files to create:**
- `src/models/cnn.py`
- `src/models/registry.py`
- `configs/model/cnn.yaml`
- `src/training/trainer.py`
- `src/training/losses.py`
- `scripts/train.py`

**Architecture:**
```
Input (B, 1, 1000)
→ Conv1d(1, 32, kernel=7)  + BN + ReLU
→ Conv1d(32, 64, kernel=7) + BN + ReLU + MaxPool(2)
→ Conv1d(64, 128, kernel=15) + BN + ReLU + MaxPool(2)
→ Conv1d(128, 256, kernel=15) + BN + ReLU + MaxPool(2)
→ GlobalAveragePool → (B, 256)
→ Dropout(0.3)
→ Linear(256, 30)
```

Receptive field at output: 7+7+15+15 = 44 positions minimum.
Use three kernel sizes: 7 (narrow peaks), 15 (medium features), 31 (broad envelopes).

**Training config:**
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Scheduler: CosineAnnealingLR, T_max=50 epochs
- Early stopping: patience=10 on val loss
- Batch size: 256
- Max epochs: 100

**Checkpoint:** CNN achieves >85% val accuracy on 30-class problem.
Record: val_acc, val_f1_macro, training time per epoch.

---

### PHASE 3 — ResNet1D (Strong Baseline Extension)
**Files to create:**
- `src/models/resnet1d.py`
- `configs/model/resnet1d.yaml`

**Architecture:**
Residual blocks with 1D convolutions. Skip connections allow deeper networks
without degradation. Start with depth-18 equivalent (8 residual blocks).

```
ResidualBlock1D:
  Conv1d(C, C, 3) + BN + ReLU
  Conv1d(C, C, 3) + BN
  + identity (or 1x1 conv if channels change)
  → ReLU
```

**Why:** ResNet1D is the standard strong baseline for time-series/spectral
classification in the literature. If the Transformer doesn't beat ResNet1D,
that is a publishable finding.

**Checkpoint:** Compare val accuracy with plain CNN.
ResNet1D should outperform CNN by at least 1-2% due to depth.

---

### PHASE 4 — Transformer
**Files to create:**
- `src/models/transformer.py`
- `configs/model/transformer.yaml`

**Architecture:**
```
Input (B, 1, 1000)
→ PatchEmbedding: split into patches of size P=20 → (B, 50, d_model)
→ CLS token prepended → (B, 51, d_model)
→ Positional encoding (learned) → (B, 51, d_model)
→ TransformerEncoder × N_layers
   [MultiHeadAttention(n_heads=8) + FFN + LayerNorm + Dropout]
→ CLS token output → (B, d_model)
→ Linear(d_model, 30)
```

Key hyperparameters to sweep (via config):
- d_model: [128, 256]
- n_layers: [4, 6]
- n_heads: [4, 8]
- patch_size: [10, 20, 25]

**Training differences from CNN:**
- Use warmup: linear increase for first 10 epochs, then cosine decay
- Lower lr: 5e-4 (Transformers are sensitive to lr)
- Higher dropout: 0.1 in attention, 0.2 in FFN
- Label smoothing: 0.1

**Checkpoint:** Transformer reaches comparable val accuracy to CNN/ResNet1D.
Extract and visualise attention maps for a representative sample from each class.

---

### PHASE 5 — Hybrid CNN-Transformer
**Files to create:**
- `src/models/hybrid.py`
- `configs/model/hybrid.yaml`

**Architecture:**
```
Input (B, 1, 1000)
→ CNN Stem (local feature extraction):
   Conv1d(1, 64, 7) + BN + ReLU
   Conv1d(64, 128, 7) + BN + ReLU + MaxPool(2) → (B, 128, 500)
   Conv1d(128, 256, 15) + BN + ReLU + MaxPool(2) → (B, 256, 250)
→ Linear projection to d_model → (B, 250, d_model)
→ CLS token + Positional encoding
→ TransformerEncoder × N_layers
→ CLS token → Linear(d_model, 30)
```

**Ablation experiments (critical for research contribution):**
1. How many CNN layers before Transformer? (1, 2, 3 conv blocks)
2. What d_model at the handoff? (64, 128, 256)
3. CNN stem with residual connections vs plain CNN stem

Each ablation is a separate config. Run all → produces a table for the paper.

**Checkpoint:** Hybrid outperforms both standalone CNN and Transformer.
If it doesn't, the ablation itself is the finding.

---

### PHASE 6 — Evaluation Suite
**Files to create:**
- `src/evaluation/metrics.py`
- `src/evaluation/evaluator.py`
- `src/evaluation/transfer_gap.py`
- `scripts/evaluate.py`

**Metrics computed for every model × split combination:**
- Accuracy (overall)
- Macro F1
- Per-class F1 (for clinical 5-class subset)
- Confusion matrix
- ROC-AUC (one-vs-rest)
- Matthews Correlation Coefficient

**Transfer gap analysis:**
```python
transfer_gap(model, ood_split) = accuracy(model, test) - accuracy(model, ood_split)
```
Produces a ranked table: which architecture generalises best to clinical data?

**Statistical testing:**
McNemar's test for each pair of models on the test split.
Report: test statistic, p-value, whether improvement is significant.

---

### PHASE 7 — Fine-Tuning Protocol
**Files to create:**
- `src/training/finetuner.py`
- `scripts/finetune.py`
- `configs/training/finetune.yaml`

**Protocol:**
1. Load pretrained checkpoint (Exp 1 best model per architecture)
2. Optionally freeze CNN stem weights for first N epochs (config-controlled)
3. Fine-tune with low lr (1e-4) on finetune split (100 samples/class)
4. Evaluate on test + both clinical splits

**Few-shot ablation (Exp 4):**
```python
for n_shots in [10, 25, 50, 100]:
    subsample finetune split to n_shots per class
    finetune → evaluate
    record clinical accuracy
```
Produces learning curve: x=n_shots, y=clinical_accuracy, one line per architecture.
This is a clean, compelling paper figure.

---

### PHASE 8 — Interpretability
**Files to create:**
- `src/interpretability/gradcam1d.py`
- `src/interpretability/attention_viz.py`
- `notebooks/04_interpretability.ipynb`

**Grad-CAM for CNN/Hybrid:**
Compute class-activation maps over the 1,000-position signal.
Visualise: which spectral regions drive each class prediction?
Overlay on mean class spectrum for maximum clarity.

**Attention maps for Transformer/Hybrid:**
Extract attention weights from each head in each layer.
Average across heads → produces 51×51 attention matrix.
Visualise: which spectral patches attend to which other patches?

**Validation step (research quality):**
If domain knowledge about the spectral data is available:
- Do highlighted regions correspond to known discriminative features?
- Do attention patterns differ between correct and incorrect predictions?
- Do clinical misclassifications show attention on different regions than correct ones?

---

### PHASE 9 — Ablation Studies & Final Results
**Notebook:** `notebooks/05_results_analysis.ipynb`

**Ablation matrix to run:**
| Ablation | Variable | Values |
|---|---|---|
| Preprocessing | Which steps matter? | Remove each step in turn |
| Augmentation | Which augmentations help? | Disable each in turn |
| CNN depth | How many layers? | 2, 3, 4, 5 blocks |
| Patch size | Transformer tokenisation | 10, 20, 25, 50 |
| Hybrid handoff | CNN layers before Transformer | 1, 2, 3 |
| Training data | Learning curve | 10%, 25%, 50%, 100% of reference |

**Final results table format:**
```
Model          | Test Acc | Test F1 | 2018clin Acc | 2019clin Acc | Gap
CNN            |          |         |              |              |
ResNet1D       |          |         |              |              |
Transformer    |          |         |              |              |
Hybrid         |          |         |              |              |
Hybrid + FT    |          |         |              |              |
```

---

### PHASE 10 — Report & Writeup
**Document:** `report/spectral_classifier_report.md` (or LaTeX)

**Sections:**
1. Introduction — problem framing, clinical motivation
2. Related work — 1D CNN for spectral, Transformers for sequences, domain adaptation
3. Dataset — describe structure (anonymise if needed), EDA findings
4. Methods — preprocessing, architectures, training, evaluation protocol
5. Results — Exp 1 (in-domain), Exp 2 (zero-shot transfer), Exp 3 (fine-tuning)
6. Ablation studies — systematic component analysis
7. Interpretability — Grad-CAM and attention visualisations
8. Discussion — what works, what doesn't, and why
9. Conclusion — key findings, limitations, future work

---

## Implementation Order Summary

```
Week 1:  Phase 0 (done) + Phase 1 (EDA gaps)
Week 2:  Phase 2 (CNN) + Phase 3 (ResNet1D)
Week 3:  Phase 4 (Transformer)
Week 4:  Phase 5 (Hybrid) + ablations
Week 5:  Phase 6 (Eval) + Phase 7 (Fine-tuning)
Week 6:  Phase 8 (Interpretability) + Phase 9 (Results)
Week 7+: Phase 10 (Report)
```

---

## Key Design Principles (Never Break These)

1. **Preprocessor is fit on reference only.** Always. No exceptions.
2. **Test split is touched once per model.** Never use it during development.
3. **Clinical splits are evaluation-only.** Never train on them.
4. **Every experiment is seeded.** `set_seed(42)` before everything.
5. **Every result is logged.** No manually recorded numbers.
6. **Every config is versioned.** Results are reproducible from config alone.
7. **OOD evaluation uses class filtering.** Only shared 5 classes scored clinically.
