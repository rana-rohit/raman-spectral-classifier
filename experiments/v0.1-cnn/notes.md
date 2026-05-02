# v0.1 — CNN Baseline (Tuned Configuration)

## 1. Objective

Establish a stronger and more stable CNN baseline for Raman spectral bacterial classification using the finalized 5-class shared clinical setup.

This experiment focused on:

* improving domain robustness
* refining augmentation strength
* stabilizing target supervision
* evaluating clinical generalization under controlled preprocessing and augmentation settings

---

# 2. Experimental Design

## 2.1 Model Architecture

### Architecture

* Model: 1D CNN
* Input channels: 2

  * Raman signal
  * First derivative
* Output classes: 5

### Convolutional Structure

* Channel progression: [32, 64, 128, 256]
* Kernel sizes: [7, 15, 15, 31]
* Dropout: 0.5

The architecture was designed to capture:

* local spectral peaks
* medium-range spectral structure
* broader biochemical context

---

## 2.2 Controlled Components

This experiment retained:

* domain adaptation
* target supervision
* tuned augmentation

while removing:

* consistency regularization

Consistency learning had previously shown unstable behavior due to non-label-preserving spectral augmentations.

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
* Monitored metric: clinical validation F1 macro

---

## 3.1 Supervision Strategy

### Target Supervised Learning

```yaml id="3w3t0v"
enabled: true
weight: 1.0
```

### DANN

```yaml id="xrljyr"
enabled: true
weight: 0.2
```

### CORAL

```yaml id="n0h6q8"
enabled: true
weight: 0.05
```

### Consistency Regularization

```yaml id="6u6d0q"
enabled: false
```

---

# 4. Dataset and Preprocessing

## 4.1 Dataset Setup

### Shared Clinical Classes

Original classes:

```text id="6hsv4z"
[0, 2, 3, 5, 6]
```

Remapped classes:

```text id="wz8y8i"
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
4. Value clipping to [-3, 3]

### Configuration

* Window length: 11
* Polynomial order: 3

The derivative signal was used as the second input channel.

---

## 4.4 Augmentation

Augmentations were applied with probability:

```yaml id="b9k8r0"
0.25
```

### Enabled Augmentations

* Gaussian noise
* Amplitude scaling
* Baseline drift
* Peak broadening
* Spectral shift

### Tuned Parameters

* Baseline drift strength: 0.05
* Peak broadening sigma: 0.17
* Spectral shift magnitude: ~1.6

The augmentation pipeline was carefully tuned to:

* improve robustness
* preserve label semantics
* avoid excessive spectral distortion

---

# 5. Results

## Evaluation Performance

| Dataset       | Accuracy | F1 Macro |
| ------------- | -------: | -------: |
| Source Test   |    0.442 |    0.410 |
| Clinical 2018 |    0.908 |    0.908 |
| Clinical 2019 |    0.854 |    0.854 |

---

# 6. Analysis

## 6.1 Clinical Transfer Performance

The model achieved strong transfer from laboratory reference spectra to clinical datasets.

Despite relatively weak source-domain performance:

* clinical evaluation remained highly stable
* target supervision effectively guided adaptation toward deployment distributions

---

## 6.2 Domain Shift Behavior

Performance degradation between:

```text id="3a1jlwm"
2018 → 2019
```

remained moderate:

```text id="jlwmg2"
~0.91 → ~0.85
```

This indicated:

* partial robustness to OOD clinical distributions
* remaining unresolved domain shift

---

## 6.3 Source vs Clinical Performance

The source-domain metrics were significantly lower than clinical-domain metrics.

Interpretation:

* the optimization objective favored clinical representation quality
* the model prioritized deployment-domain performance over laboratory-domain accuracy

This behavior aligned with the project objective.

---

## 6.4 Augmentation Effects

Augmentation emerged as a major contributor to domain robustness.

Important findings:

* moderate augmentation improved OOD generalization
* excessive augmentation damaged class semantics
* spectral shift augmentation was especially beneficial

---

## 6.5 Consistency Learning

Consistency regularization was excluded because:

* spectral augmentations were not strictly label-preserving
* enforced invariances degraded discriminative structure

This established an important limitation for contrastive-style training in Raman spectral pipelines.

---

# 7. Conclusions

This experiment established the first strong and stable clinical baseline for the Raman classification project.

Key findings:

* target-supervised adaptation was highly effective
* augmentation quality strongly influenced OOD performance
* clinical-domain optimization naturally reduced source-domain accuracy
* consistency regularization was harmful in this setup

This CNN configuration became the primary baseline for all subsequent ResNet experiments.

---

# 8. Historical Context

At the time of this experiment:

* DANN and CORAL were still believed to contribute positively
* architecture was not yet identified as the dominant bottleneck

Later experiments demonstrated:

* domain alignment methods were often harmful
* spectral inductive bias and receptive field design were significantly more important

This experiment therefore represents the final optimized CNN baseline before the transition to residual architectures.

---

# 9. Next Experiment

The next phase investigated whether deeper residual architectures could improve:

* spectral representation learning
* clinical robustness
* OOD generalization

This led to the introduction of:

```text id="jlwm1u"
ResNet1D
```

as the next architectural baseline.
