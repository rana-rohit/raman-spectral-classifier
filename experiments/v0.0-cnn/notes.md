# v0.0 — CNN Baseline

## 1. Objective

Establish the first stable baseline for Raman spectral bacterial classification using the corrected 5-class clinical pipeline.

This experiment evaluates whether a standard 1D CNN combined with domain adaptation can achieve reliable transfer from laboratory reference spectra to clinical datasets.

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

The architecture was designed to capture both:

* local spectral peaks
* broader biochemical structures

---

## 2.2 Domain Adaptation Setup

The model was trained using:

* source supervision
* target supervision
* adversarial domain adaptation
* feature alignment

### Enabled Components

* Target supervised learning
* DANN
* CORAL

This experiment represents the earliest domain-adaptive baseline within the corrected pipeline.

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
* Based on clinical validation F1 score

---

## 3.1 Supervision Strategy

### Target Supervised Learning

```yaml
enabled: true
weight: 1.0
```

### DANN

```yaml
enabled: true
weight: 0.2
```

### CORAL

```yaml
enabled: true
weight: 0.05
```

### Consistency Regularization

Not used in this experiment.

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

The following preprocessing pipeline was applied:

1. Savitzky-Golay smoothing
2. First derivative computation
3. Value clipping

### Configuration

* Savitzky-Golay window: 11
* Polynomial order: 3
* Clipping range: [-3, 3]

The derivative signal was used as a second input channel.

---

## 4.4 Augmentation

Training augmentations included:

* Gaussian noise
* Baseline drift
* Spectral shift
* Peak broadening

These augmentations were designed to simulate real-world spectral variability and improve robustness to acquisition differences.

---

# 5. Results

## Clinical Validation

| Metric   | Value |
| -------- | ----: |
| F1 Score | ~0.92 |
| Accuracy | ~0.92 |

---

## Evaluation Performance

| Dataset       | Accuracy | F1 Macro |
| ------------- | -------: | -------: |
| Clinical 2018 |    ~0.91 |    ~0.91 |
| Clinical 2019 |    ~0.85 |    ~0.85 |

---

## Source Validation

| Metric   |      Value |
| -------- | ---------: |
| F1 Macro | ~0.85–0.88 |

---

# 6. Analysis

## 6.1 Training Stability

Training remained stable throughout optimization:

* no divergence
* no oscillatory behavior
* consistent convergence

Domain adversarial loss remained numerically stable.

---

## 6.2 Clinical Generalization

The model achieved strong performance on:

* clinical validation
* clinical 2018 evaluation

This demonstrated successful transfer from laboratory reference spectra to clinical distributions.

---

## 6.3 OOD Performance

Performance decreased on the 2019 clinical dataset:

```text
~0.91 → ~0.85
```

This indicated:

* residual domain shift
* limited robustness to stronger acquisition variability

---

## 6.4 Source vs Clinical Behavior

Clinical performance exceeded source-domain performance.

Interpretation:

* the model adapted strongly toward clinical distributions
* target supervision significantly influenced representation learning

---

# 7. Conclusions

This experiment established the first stable and reproducible baseline for the corrected 5-class Raman classification pipeline.

Key findings:

* domain-adaptive training was feasible
* target supervision substantially improved clinical performance
* meaningful transfer to clinical datasets was achieved
* OOD robustness remained limited

This experiment became the foundation for all subsequent architecture and domain generalization studies.

---

# 8. Historical Context

At the time of this experiment:

* DANN and CORAL appeared beneficial
* domain adaptation was believed to be a primary driver of performance

Later experiments revealed:

* explicit adversarial alignment was often harmful
* architectural inductive bias played a larger role than domain alignment

This experiment therefore represents an important early baseline within the evolution of the project.

---

# 9. Next Experiment

The next experimental phase focused on:

* tuning domain adaptation strength
* evaluating augmentation sensitivity
* testing alternative architectures

Primary motivation:

* improve robustness on the 2019 clinical dataset
* reduce remaining domain shift effects
