# v1.4 — Batch Normalization Ablation on Hybrid Multi-Scale ResNet1D

## 1. Objective

The objective of this experiment was to evaluate the impact of Batch Normalization (BN) on model performance and domain generalization in Raman spectral classification.

This experiment builds directly on the best-performing model (v1.2), which utilized a hybrid multi-scale architecture without normalization. The goal was to determine whether introducing standard normalization techniques improves training stability or degrades spectral representation quality.

---

## 2. Experimental Setup

### 2.1 Baseline (v1.2)

* Hybrid Multi-Scale ResNet1D
* Kernel sizes: 7 → 9 → 13 → 17
* Hybrid convolution (depthwise early, standard later)
* No normalization layers
* Achieved best performance across all datasets

---

### 2.2 Modification

Batch Normalization (BatchNorm1d) was introduced at the following locations:

* After the first convolution in each residual block
* After the second convolution in each residual block
* In residual shortcut connections
* In the input stem layer

---

### 2.3 Controlled Variables

All other components were kept identical to v1.2:

* Model architecture and channel configuration
* Multi-scale kernel design
* Training configuration (optimizer, scheduler, batch size)
* Preprocessing pipeline and derivative input
* Data augmentation strategy
* Supervision setup (including target supervision)

This ensures that performance differences are solely due to the inclusion of Batch Normalization.

---

## 3. Results

### 3.1 Evaluation Performance

| Dataset       | Accuracy | F1 Macro |    MCC |
| ------------- | -------: | -------: | -----: |
| Source Test   |   0.4100 |   0.3668 | 0.3263 |
| Clinical 2018 |   0.8970 |   0.8977 | 0.8735 |
| Clinical 2019 |   0.8420 |   0.8406 | 0.8060 |

---

## 4. Comparison with Baseline (v1.2)

| Model                            | BatchNorm | Source F1 | 2018 F1 | 2019 F1 |
| -------------------------------- | --------: | --------: | ------: | ------: |
| v1.2 Hybrid Multi-Scale ResNet1D |        No |    0.6190 |  0.9765 |  0.9558 |
| v1.4 + Batch Normalization       |       Yes |    0.3668 |  0.8977 |  0.8406 |

---

## 5. Observations

### 5.1 Significant Performance Degradation

The inclusion of Batch Normalization resulted in a substantial decrease in performance across all evaluation datasets:

* Source-domain performance dropped sharply
* Clinical (in-domain) performance degraded
* Out-of-distribution (2019 clinical) performance decreased significantly

---

### 5.2 Representation Collapse

Class-wise evaluation revealed severe instability:

* At least one class exhibited an F1 score of 0.0
* Multiple classes showed near-random performance

This indicates failure to learn stable and discriminative representations.

---

### 5.3 Persistent Domain Bias

Despite the degradation, the model continued to perform better on clinical datasets than on the source dataset. This is consistent with the use of target-domain supervision during training, indicating a bias toward the clinical distribution.

---

## 6. Analysis

### 6.1 Impact on Spectral Information

Raman spectral data is highly dependent on:

* absolute intensity values
* peak amplitude relationships
* biochemical signal characteristics

Batch Normalization standardizes feature distributions by removing mean and scaling variance. This disrupts the inherent structure of spectral signals and removes meaningful amplitude information.

---

### 6.2 Effect on Feature Learning

The introduction of Batch Normalization negatively affects:

* amplitude-sensitive feature extraction
* peak-level discriminative patterns
* stability of learned representations

As a result, the model develops unstable decision boundaries and fails to generalize effectively.

---

### 6.3 Effect on Domain Generalization

Batch Normalization introduces dependence on batch-level statistics, which vary across domains. This leads to:

* reduced robustness to distribution shifts
* increased domain-specific bias
* degradation in out-of-distribution performance

---

## 7. Conclusion

Batch Normalization is detrimental to Raman spectral classification.

Key conclusions:

* normalization disrupts critical spectral information
* representation quality is significantly reduced
* both in-domain and out-of-domain performance degrade

---

## 8. Implications

* The normalization-free architecture (v1.2) remains the optimal design
* Preserving raw spectral characteristics is essential for learning
* Standard deep learning practices such as Batch Normalization may not be suitable for spectroscopy-based tasks

---

## 9. Final Takeaway

The experiment demonstrates that:

> Raman spectral classification requires preservation of raw signal structure, and normalization layers such as BatchNorm should be avoided.

---

## 10. Next Steps

* Proceed with Explainable AI (XAI) integration
* Analyze spectral regions contributing to predictions
* Validate interpretability of learned representations for clinical relevance

---
