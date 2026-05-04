# v1.2 — Hybrid Multi-Scale ResNet1D

## 1. Objective

Evaluate whether combining:

* efficient early-stage spectral filtering
  with:
* dense late-stage feature interaction

can improve Raman spectral classification and OOD clinical generalization.

This experiment was designed to preserve the strong spectral-context modeling discovered in the previous multi-scale ResNet while reducing unnecessary parameter redundancy in earlier layers.

The primary goal was to improve:

* OOD robustness
* parameter efficiency
* spectral representation quality

without using explicit domain adaptation methods.

---

# 2. Experimental Design

## 2.1 Model Architecture

### Architecture

* Model: Hybrid Multi-Scale ResNet1D
* Input channels: 2

  * Raman signal
  * First derivative
* Output classes: 5

---

## 2.2 Residual Structure

### Channel Progression

```yaml
[32, 64, 128, 256]
```

### Residual Blocks Per Stage

```yaml
[2, 2, 2, 2]
```

### Dropout

```yaml
0.3
```

---

## 2.3 Multi-Scale Kernel Design

| Stage   | Kernel Size |
| ------- | ----------: |
| Stem    |           7 |
| Stage 1 |           7 |
| Stage 2 |           9 |
| Stage 3 |          13 |
| Stage 4 |          17 |

The progressively expanded receptive fields were preserved from the previous successful multi-scale architecture.

This allowed the network to simultaneously model:

* narrow Raman peaks
* medium-range spectral dependencies
* broad biochemical structures

---

## 2.4 Hybrid Convolution Strategy

This experiment introduced a hybrid convolution design.

| Stage   | Convolution Type     |
| ------- | -------------------- |
| Stage 1 | Depthwise separable  |
| Stage 2 | Depthwise separable  |
| Stage 3 | Standard convolution |
| Stage 4 | Standard convolution |

---

### Motivation

The architecture was based on the hypothesis that:

#### Early Layers

Primarily learn:

* local motifs
* low-level spectral filters
* positional features

These can be extracted efficiently using:

```text
depthwise separable convolutions
```

---

#### Late Layers

Require:

* dense channel interaction
* biochemical abstraction
* higher-order spectral relationships

These are better modeled using:

```text
standard convolutions
```

---

## 2.5 Normalization

No normalization layers were used:

* no BatchNorm
* no InstanceNorm

This decision was based on earlier experiments showing:

* normalization degraded OOD robustness
* normalization suppressed important spectral amplitude information

---

# 3. Training Configuration

## Optimizer

* AdamW
* Learning rate: 1e-4
* Weight decay: 5e-4

## Scheduler

* Warmup cosine decay

## Batch Size

* 128

## Early Stopping

* Enabled
* Based on clinical validation F1 macro

---

## 3.1 Supervision Strategy

### Source Supervision

Enabled.

### Target Supervised Learning

```yaml
enabled: true
weight: 1.0
```

### DANN

```yaml
enabled: false
```

### CORAL

```yaml
enabled: false
```

### Consistency Regularization

```yaml
enabled: false
```

This experiment intentionally excluded explicit domain adaptation methods to isolate architectural effects.

---

# 4. Dataset and Preprocessing

## 4.1 Dataset Setup

### Shared Clinical Classes

Original classes:

```text
[0, 2, 3, 5, 6]
```

Remapped classes:

```text
[0, 1, 2, 3, 4]
```

---

## 4.2 Dataset Splits

| Split               | Samples | Purpose             |
| ------------------- | ------: | ------------------- |
| Source Train        |    8000 | Source supervision  |
| Source Validation   |    2000 | Validation          |
| Clinical Train      |    7500 | Target supervision  |
| Clinical Validation |    2500 | Early stopping      |
| Clinical 2018       |    2000 | Clinical evaluation |
| Clinical 2019       |     500 | OOD evaluation      |

---

## 4.3 Preprocessing Pipeline

The preprocessing pipeline included:

1. Savitzky-Golay smoothing
2. SNV normalization
3. First derivative computation
4. Clipping to [-3, 3]

### Configuration

* Window length: 11
* Polynomial order: 3

The derivative signal was used as the second input channel.

---

# 5. Augmentation Configuration

Augmentations were applied with probability:

```yaml
0.25
```

---

## Enabled Augmentations

* Gaussian noise
* Amplitude scaling
* Baseline drift
* Peak broadening
* Spectral shift

---

## Final Tuned Parameters

```yaml
baseline_drift:
  max_strength: 0.07

peak_broadening:
  max_sigma: 0.14

spectral_shift:
  max_shift: 1.8
```

---

# 6. Results

## Evaluation Performance

| Dataset       | Accuracy | F1 Macro |    MCC |
| ------------- | -------: | -------: | -----: |
| Source Test   |    0.634 |    0.619 |  0.569 |
| Clinical 2018 |   0.9765 |   0.9765 | 0.9707 |
| Clinical 2019 |   0.9560 |   0.9558 | 0.9451 |

The hybrid architecture achieved the strongest overall performance observed in the project so far. 

---

# 7. Analysis

## 7.1 OOD Generalization

Clinical 2019 performance improved substantially:

```text
~0.93 → ~0.956
```

This represented the strongest OOD robustness achieved during the project.

Importantly:

* no adversarial alignment was used
* no explicit domain invariance loss was required

The robustness emerged directly from architectural design.

---

## 7.2 Hybrid Convolution Findings

The results strongly suggest:

> late-stage dense spectral interaction is critical for Raman domain robustness.

The earlier fully-depthwise architecture:

* reduced parameters successfully
* preserved some robustness

but suffered a major performance drop on OOD data. 

Restoring dense convolutions in deeper stages recovered and exceeded previous performance.

---

## 7.3 Spectral Context Modeling

The progressively expanded receptive fields continued to provide major benefits.

The architecture successfully modeled:

* local Raman peaks
* medium-range relationships
* broad biochemical context

This reinforced the earlier conclusion that:

```text
hierarchical spectral context is a primary driver of Raman robustness
```

---

## 7.4 Domain Adaptation Findings

The best-performing architecture in the project:

* used no DANN
* used no CORAL
* used no consistency learning

This strongly suggests that:

```text
feature extraction quality mattered more than explicit feature alignment
```

for this domain adaptation problem.

---

## 7.5 Source vs Clinical Performance

Source-domain performance remained significantly lower than clinical-domain performance.

Interpretation:

* target supervision strongly guided representation learning toward clinical distributions
* the shared clinical subset remained easier and more structured than the full source distribution

This pattern was consistently observed across experiments.

---

## 7.6 Training Stability

Training remained highly stable despite:

* removal of normalization layers
* large receptive fields
* hybrid convolution strategies

Observed behavior:

* smooth convergence
* no exploding activations
* stable early stopping behavior

The best checkpoint achieved:

```text
val_f1_macro = 0.9772
```

after 46 epochs. 

---

# 8. Comparison with Previous Architectures

| Model                                | 2018 F1 | 2019 F1 |
| ------------------------------------ | ------: | ------: |
| CNN Baseline                         |   ~0.91 |   ~0.85 |
| Clean ResNet1D                       |   ~0.95 |   ~0.85 |
| Multi-Scale ResNet1D                 |   ~0.97 |   ~0.93 |
| Fully Depthwise Multi-Scale ResNet1D |   ~0.85 |   ~0.80 |
| Hybrid Multi-Scale ResNet1D          |  0.9765 |  0.9558 |

---

# 9. Key Scientific Findings

## 1. Hierarchical Spectral Context Was Critical

The largest gains consistently came from:

* progressively expanded receptive fields
* hierarchical spectral modeling

This effect was substantially stronger than explicit domain alignment methods.

---

## 2. Late-Stage Dense Interaction Was Essential

Fully depthwise architectures lost significant OOD performance.

This demonstrated that:

* high-level biochemical abstraction requires dense channel interaction
* efficient early filtering alone is insufficient

---

## 3. Explicit Domain Alignment Was Unnecessary

The strongest model in the project used:

* no DANN
* no CORAL
* no consistency regularization

OOD robustness emerged naturally from:

* improved spectral representation learning
* spectroscopy-aware architectural inductive bias

---

## 4. Normalization-Free Training Improved Robustness

Removing normalization layers consistently improved:

* stability
* OOD robustness
* transfer performance

This suggests Raman spectral amplitudes contain meaningful discriminative information that may be distorted by normalization.

---

# 10. Conclusions

This experiment established the strongest and most robust architecture developed in the project so far.

Key findings:

* hybrid depthwise-standard architectures were highly effective
* late-stage dense feature interaction was essential
* hierarchical spectral context dramatically improved robustness
* explicit adversarial alignment was unnecessary

Most importantly:

> Raman domain robustness emerged primarily from spectroscopy-aware hierarchical representation learning rather than explicit domain adaptation.

---

# 11. Next Experiment

The next phase will investigate:

* channel attention mechanisms
* adaptive feature recalibration
* spectral importance weighting

This leads to:

```text
SE-Hybrid Multi-Scale ResNet1D
```

using:

* squeeze-and-excitation attention
* channel-wise feature recalibration
* adaptive spectral emphasis.
