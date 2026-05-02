# v1.0 — ResNet1D Baseline

## 1. Objective

Evaluate whether a residual 1D convolutional architecture improves Raman spectral classification and cross-domain generalization relative to the optimized CNN baseline.

This experiment isolates architectural capacity as the primary variable by removing:

* explicit domain adaptation
* normalization layers
* consistency regularization

The goal was to determine whether improved feature representation alone could improve OOD robustness.

---

# 2. Experimental Design

## 2.1 Model Architecture

### Architecture

* Model: ResNet1D
* Input channels: 2

  * Raman signal
  * First derivative
* Output classes: 5

### Residual Structure

* Channel progression: [32, 64, 128, 256]
* Residual blocks per stage: [2, 2, 2, 2]
* Stem kernel size: 7
* Dropout: 0.3

### Classification Head

* Global Average Pooling
* Linear classifier

The architecture was designed to improve:

* representation learning depth
* hierarchical feature extraction
* optimization stability through residual connections

---

## 2.2 Experimental Isolation

To isolate the effect of architecture alone, the following components were removed:

### Removed Components

* Batch Normalization
* Instance Normalization
* DANN
* CORAL
* Consistency regularization

### Preserved Components

* preprocessing pipeline
* augmentation strategy
* optimizer
* scheduler
* target-supervised learning

This ensured that any observed performance changes originated primarily from the residual architecture itself.

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

```yaml id="rlfym1"
enabled: true
weight: 1.0
```

### DANN

```yaml id="j1yk5f"
enabled: false
```

### CORAL

```yaml id="j3mjlwm"
enabled: false
```

### Consistency Regularization

```yaml id="5jlwm4"
enabled: false
```

---

# 4. Dataset and Preprocessing

## 4.1 Dataset Setup

### Shared Clinical Classes

Original classes:

```text id="jlwm8e"
[0, 2, 3, 5, 6]
```

Remapped classes:

```text id="jlwmrx"
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

```yaml id="jlwm9m"
0.25
```

### Enabled Augmentations

* Gaussian noise
* Amplitude scaling
* Baseline drift
* Peak broadening
* Spectral shift

The augmentation configuration was inherited from the optimized CNN baseline.

---

# 5. Results

## Evaluation Performance

| Dataset       | Accuracy | F1 Macro |   MCC |
| ------------- | -------: | -------: | ----: |
| Source Test   |    0.502 |    0.439 | 0.431 |
| Clinical 2018 |    0.952 |    0.952 | 0.940 |
| Clinical 2019 |    0.854 |    0.853 | 0.818 |

---

# 6. Analysis

## 6.1 Representation Learning

The ResNet architecture improved feature representation quality compared to the CNN baseline.

This was most visible on:

* Clinical 2018 evaluation
* source-domain metrics

Residual learning improved:

* optimization depth
* feature reuse
* hierarchical representation quality

---

## 6.2 OOD Generalization

Despite improved in-domain representation quality, no major improvement was observed on the:

```text id="jlwm7u"
2019 clinical dataset
```

Performance remained approximately:

```text id="jlwm1f"
~0.85 F1
```

This suggested:

* model capacity was not the primary bottleneck
* residual learning alone was insufficient for domain robustness

---

## 6.3 Domain Shift

A substantial performance gap persisted between:

* source and clinical domains
* 2018 and 2019 clinical datasets

Interpretation:

* domain shift remained the dominant challenge
* improved representation learning alone did not resolve acquisition variability

---

## 6.4 Training Stability

Training remained highly stable despite:

* removal of normalization layers
* deeper architecture

Observed behavior:

* stable convergence
* no exploding activations
* no optimization instability

This demonstrated that:

* Raman spectral inputs were relatively well-behaved
* normalization was not strictly necessary in this setup

---

## 6.5 Clinical Optimization Behavior

The model continued prioritizing:

* clinical-domain performance
  over:
* source-domain accuracy

This remained consistent with deployment objectives.

---

# 7. Conclusions

This experiment established the first clean residual baseline for the project.

Key findings:

* residual learning improved representation quality
* deeper architectures benefited clinical classification
* OOD robustness did not improve significantly
* domain shift remained unresolved

Most importantly:

> architecture depth alone was insufficient to solve domain generalization.

---

# 8. Historical Context

At the time of this experiment:

* domain adaptation was still considered potentially useful
* architecture improvements were being evaluated independently

This experiment provided an important control baseline by:

* removing adaptation methods
* isolating pure architectural effects

It established that:

* the CNN baseline was not fundamentally capacity-limited
* the remaining challenge involved spectral robustness rather than simple feature extraction depth

---

# 9. Next Experiment

The next phase investigated whether:

* broader receptive fields
* hierarchical spectral context modeling

could improve:

* OOD robustness
* biochemical context representation
* natural domain invariance

This led to the development of:

```text id="jlwmfj"
Multi-Scale ResNet1D
```

with progressively expanded convolution kernels across network stages.
