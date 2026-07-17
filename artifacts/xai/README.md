# Explainable AI (XAI) Results

This directory contains representative **LIME (Local Interpretable Model-Agnostic Explanations)** visualizations generated for the proposed Raman spectroscopy-based bacterial classification framework.

The figures presented here are **curated examples** selected from the complete explainability analysis performed during experimentation. They demonstrate how the trained models identify discriminative Raman spectral regions throughout the transfer learning pipeline while keeping the repository concise.

---

## Directory Contents

| File | Description |
|------|-------------|
| `stage1_lime_example.png` | Representative LIME explanation for a correctly classified sample from **Stage 1 – Isolate Space (30 Classes)**. |
| `stage1_lime_misclassified.png` | Example illustrating model reasoning for a misclassified Stage 1 sample. |
| `stage2_lime_example.png` | Representative LIME explanation after semantic transfer in **Stage 2 – Treatment Space**. |
| `stage2_lime_misclassified.png` | Example highlighting feature attribution for an incorrect Stage 2 prediction. |
| `stage3_models_comparison.png` | Qualitative comparison of LIME explanations produced by multiple model architectures for the same Raman spectrum. |

---

## Explainability Pipeline

The repository employs **LIME (Local Interpretable Model-Agnostic Explanations)** to interpret individual predictions.

For each selected Raman spectrum:

1. Local perturbations are generated around the original spectrum.
2. A local surrogate model is fitted around the prediction.
3. Spectral regions contributing positively or negatively to the prediction are identified.
4. Important Raman bands are visualized directly on the spectrum together with their corresponding feature contributions.

Positive (green) regions support the predicted class, while negative (red) regions oppose the prediction.

---

## Repository Scope

To maintain a lightweight and navigable repository, only representative visualizations are included.

The complete collection of LIME explanations generated during experimentation is intentionally omitted, as many figures are highly similar and primarily differ in the analyzed spectrum.

Stage 3 of the framework focuses on **patient-level clinical transfer evaluation** using cross-validation. Consequently, representative Stage 3 LIME visualizations are not included as part of the primary experimental artifacts.

---

## Purpose

These visualizations complement the quantitative evaluation metrics by providing qualitative evidence that the trained models rely on meaningful Raman spectral features rather than arbitrary signal characteristics.

They are intended to assist in:

- Understanding individual model predictions.
- Inspecting correctly classified and misclassified samples.
- Comparing feature attribution before and after semantic transfer learning.
- Qualitatively comparing feature attribution across different deep learning architectures.