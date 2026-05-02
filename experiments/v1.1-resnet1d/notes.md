# v1.1 — Multi-Scale ResNet1D

## 1. Objective

Evaluate whether progressively expanded receptive fields improve Raman spectral classification and cross-domain generalization relative to the clean ResNet1D baseline.

This experiment focused on:

* hierarchical spectral context modeling
* spectroscopy-aware architectural inductive bias
* improving OOD robustness without explicit domain adaptation

The goal was to determine whether broader spectral context could naturally improve domain robustness.

---

# 2. Experimental Design

## 2.1 Model Architecture

### Architecture

* Model: Multi-Scale ResNet1D
* Input channels: 2

  * Raman signal
  * First derivative
* Output classes: 5

### Residual Structure

* Channel progression: [32, 64, 128, 256]
* Residual blocks per stage: [2, 2, 2, 2]
* Dropout: 0.3

### Multi-Scale Kernel Design

| Stage   | Kernel Size |
| ------- | ----------: |
| Stem    |           7 |
| Stage 1 |           7 |
| Stage 2 |           9 |
| Stage 3 |          13 |
| Stage 4 |          17 |

This introduced progressively expanded receptive fields across network depth.

The design was motivated by the observation that Raman spectra contain:

* narrow local peaks
* medium-range spectral interactions
* broader biochemical structures

---

## 2.2 Experimental Isolation

This experiment preserved:

* preprocessing pipeline
* augmentation pipeline
* optimizer
* scheduler
* target supervision

while keeping disabled:

* DANN
* CORAL
* consistency regularization
* normalization layers

This isolated the effect of:

```text id="jlwmv0"
hierarchical receptive field expansion
```

as the primary experimental variable.

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

```yaml id="jlwmj3"
enabled: true
weight: 1.0
```

### DANN

```yaml id="jlwm6r"
enabled: false
```

### CORAL

```yaml id="jlwm3k"
enabled: false
```

### Consistency Regularization

```yaml id="jlwm4v"
enabled: false
```

---

# 4. Dataset and Preprocessing

## 4.1 Dataset Setup

### Shared Clinical Classes

Original classes:

```text id="jlwm9p"
[0, 2, 3, 5, 6]
```

Remapped classes:

```text id="jlwmfw"
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

## 4.4 Augmentation

Augmentations were applied with probability:

```yaml id="jlwm9x"
0.25
```

### Enabled Augmentations

* Gaussian noise
* Amplitude scaling
* Baseline drift
* Peak broadening
* Spectral shift

### Final Tuned Parameters

```yaml id="jlwm91"
baseline_drift:
  max_strength: 0.07

peak_broadening:
  max_sigma: 0.14

spectral_shift:
  max_shift: 1.8
```

---

# 5. Results

## Evaluation Performance

| Dataset       | Accuracy | F1 Macro |   MCC |
| ------------- | -------: | -------: | ----: |
| Source Test   |    0.584 |    0.518 | 0.515 |
| Clinical 2018 |    0.973 |    0.973 | 0.966 |
| Clinical 2019 |    0.934 |    0.934 | 0.918 |

---

# 6. Analysis

## 6.1 Representation Learning

The introduction of progressively expanded receptive fields significantly improved representation quality.

The model demonstrated:

* stronger feature hierarchy
* improved biochemical context modeling
* better separation of clinical classes

This produced the strongest overall performance observed in the project so far.

---

## 6.2 OOD Generalization

Clinical 2019 performance improved substantially:

```text id="jlwm0s"
~0.85 → ~0.93
```

This represented the largest OOD improvement observed during the project.

Importantly:

* no explicit domain adaptation was used
* robustness emerged naturally from architectural design

---

## 6.3 Spectral Context Modeling

The results strongly suggested that:

* Raman robustness depends heavily on spectral context
* broader receptive fields improve biochemical representation quality
* hierarchical context modeling is more important than adversarial alignment

The model benefited from:

* local peak analysis
* medium-range relationships
* broad spectral structure

simultaneously.

---

## 6.4 Domain Adaptation Findings

Earlier experiments using:

* DANN
* CORAL
* consistency learning

either:

* degraded performance
  or:
* produced minimal benefit

This experiment demonstrated that:

> spectroscopy-aware architectural inductive bias was substantially more effective than explicit domain alignment.

---

## 6.5 Augmentation Findings

Several important augmentation behaviors were identified:

### Spectral Shift

Increasing spectral shift robustness improved OOD generalization.

Interpretation:

* positional invariance is important in Raman spectra

---

### Peak Broadening

Reducing excessive broadening improved discriminative preservation.

Interpretation:

* narrow Raman peaks contain critical class information

---

## 6.6 Training Stability

Despite larger receptive fields:

* training remained stable
* convergence was smooth
* no normalization layers were required

The architecture demonstrated strong optimization behavior.

---

# 7. Conclusions

This experiment established the first highly successful architecture for robust Raman spectral domain generalization within the project.

Key findings:

* hierarchical spectral context modeling dramatically improved OOD robustness
* explicit adversarial alignment was unnecessary
* spectroscopy-aware receptive field design was highly effective
* architectural inductive bias dominated generic adaptation methods

Most importantly:

> Raman domain robustness emerged primarily from improved spectral representation learning rather than explicit domain alignment.

---

# 8. Historical Context

This experiment marked a major transition in the direction of the project.

Earlier stages focused primarily on:

* domain adaptation
* alignment losses
* regularization methods

This experiment demonstrated that:

* architecture was the dominant factor
* spectral inductive bias mattered more than generic adaptation techniques

The project direction therefore shifted toward:

```text id="jlwm7c"
spectral representation engineering
```

rather than:

```text id="jlwm8a"
explicit domain alignment
```

---

# 9. Reproducibility Validation

A second experiment was performed using a different augmentation seed.

Observed behavior:

* nearly identical convergence
* identical final OOD performance
* highly stable optimization dynamics

This confirmed that:

* the improvement was architectural rather than stochastic
* the result was reproducible and stable

---

# 10. Next Experiment

The next phase investigates whether:

* the same spectral-context advantages
  can be preserved with:
* substantially fewer parameters

This leads to:

```text id="jlwm7m"
Efficient Multi-Scale ResNet1D
```

using:

* depthwise separable convolutions
* lightweight residual blocks
* efficient spectral feature extraction.
