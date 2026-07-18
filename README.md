# Explainable AI Framework for Raman Spectroscopy-Based Antimicrobial Treatment Classification


## Research Motivation & Problem Statement
Raman spectroscopy offers rapid, label-free bacterial identification. However, analyzing raw Raman spectra is complex due to biological variance, instrument noise, and domain shifts between laboratory and clinical settings. This project provides a reproducible and interpretable deep learning pipeline. It addresses the domain shift between laboratory and clinical Raman spectra using a **Three-Stage Transfer Learning Framework**, ultimately mapping isolate-level reference spectra to clinical treatment categories.

## Pipeline Overview
The repository implements the exact methodology described in the final research paper:

1. **Robust Preprocessing Pipeline**:
   - Savitzky-Golay Smoothing
   - Standard Normal Variate (SNV) Normalization
   - First Derivative Computation
   - Clip Transform
   - Data Augmentation
2. **Three-Stage Transfer Learning Framework**:
   - **Stage 1**: 30 Isolate Classification (Pre-training on Reference Data)
   - **Stage 2**: 8 Treatment Groups (Semantic Alignment)
   - **Stage 3**: Clinical Transfer Learning (5 Treatment Classes)
3. **Robust Evaluation**:
   - Patient-Level Probabilistic Voting (Achieving 100% Patient-Level Accuracy)
4. **Explainable AI (XAI)**:
   - LIME-based Explainability
   - Consensus Raman Peak Analysis to extract biologically meaningful spectral features.

<p align="center">
  <img src="assets/images/framework_overview.png"
       alt="Framework Overview"
       width="650">
</p>

<p align="center">
<b>Figure 1.</b> Overall workflow of the proposed Three-Stage Transfer Learning Framework.
</p>

## Project Structure
```text
.
|-- artifacts/            # Publication checkpoints and released result figures
|-- assets/               # Documentation images
|-- configs/              # YAML configuration files
|   |-- data/             # Splits, preprocessing, and augmentation configs
|   |-- model/            # Architecture-specific hyperparameters
|   |-- stages/           # Stage 1/2/3 task definitions
|   `-- training/         # Shared optimizer, loss, scheduler, and evaluation defaults
|-- data/
|   `-- raw/              # Local .npy datasets; not redistributed
|-- docs/                 # Additional documentation
|-- metadata/             # Label ontology, isolates, treatments, and patient IDs
|-- notebooks/            # Canonical reproduction and analysis notebooks
|   `-- archive/          # Archived exploratory notebooks
|-- scripts/              # Executable entry points
|   |-- setup_data.py     # Data preparation and integrity checks
|   |-- train.py          # Stage 1/2/3 training entry point
|   |-- run_patient_cv.py # Stage 3 patient-aware CV orchestration
|   |-- evaluate.py       # Standalone evaluation
|   |-- analyze_experiment.py
|   |-- xai.py
|   `-- archive/          # Archived development scripts
|-- src/                  # Core package
|   |-- data/             # Dataset, dataloader, preprocessing, and split logic
|   |-- evaluation/       # Metrics, visualization, and clinical utilities
|   |-- models/           # CNN, ResNet1D, TCN, Transformer, Inception1D, hybrids
|   |-- training/         # Training loops, finetuning, losses, and schedulers
|   |-- utils/            # Config, checkpoints, logging, seeds, and split modes
|   `-- xai/              # LIME, saliency, and XAI orchestration
`-- tests/                # Unit and regression tests
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/rana-rohit/raman-spectral-classifier.git
cd raman-spectral-classifier
```
2. Create a virtual environment, activate it, and install the project dependencies:
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Dataset

This project is built on the publicly available Raman spectroscopy datasets provided through the **RamanSPy** project. The repository **does not redistribute** the original datasets. Please download them from the official source before running the training pipeline.

The Raman spectroscopy datasets are available through the
[RamanSPy documentation](https://ramanspy.readthedocs.io/en/latest/datasets.html).

The datasets originate from:

Ho, C. S., et al. (2019), *Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning.*

Due to size constraints, the raw `.npy` Raman spectral datasets are not hosted in this repository. Place them under `data/raw/` before running the pipeline. The Colab workflow in `notebooks/00_getting_started.ipynb` links this directory from Google Drive.

To prepare and validate your data:
```bash
python scripts/setup_data.py --stage s1_isolate --split-mode iid_reference
python scripts/setup_data.py --stage s2_treatment --split-mode iid_reference
python scripts/setup_data.py --stage s3_transfer --split-mode patient_cv
```

## Model Architectures

The framework supports six deep learning architectures:

- CNN
- CNN-Transformer
- TCN
- Inception1D
- ResNet1D
- Transformer

## Quick Start

After preparing the dataset, the recommended workflow is:

1. Set up the dataset using `scripts/setup_data.py`.
2. Train the three-stage transfer learning framework.
3. Evaluate the trained models.
4. Generate explainability results using LIME.
5. Produce publication-quality figures.

A complete step-by-step walkthrough, including Google Colab instructions and reproduction notebooks, is available in:

- `notebooks/00_getting_started.ipynb`
- `docs/`

## Results & Highlights

Representative results obtained with the best-performing TCN configuration:

- Best Performing Architecture: Temporal Convolutional Network (TCN)
- Stage 3 Clinical Classification Accuracy (Best TCN Configuration): 96.0%
- Patient-Level Classification Accuracy: 100%
- LIME-based Explainability
- Consensus Raman Peak Analysis 

## License
Released under the MIT License. See `LICENSE` for more information.
