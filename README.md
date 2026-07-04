# Explainable AI Framework for Raman Spectroscopy-Based Antimicrobial Treatment Classification

## Project Title
Development of an Explainable AI Framework for Raman Spectroscopy-Based Antimicrobial Treatment Classification Using Three-Stage Transfer Learning

## Research Motivation & Problem Statement
Raman spectroscopy offers rapid, label-free bacterial identification. However, analyzing raw Raman spectra is complex due to biological variance, instrument noise, and domain shifts between laboratory and clinical settings. This project provides a reproducible and interpretable deep learning pipeline. It solves the domain-shift problem using a **Three-Stage Transfer Learning** approach, ultimately mapping isolate-level reference spectra to clinical treatment categories.

## Pipeline Overview
The repository implements the exact methodology described in the final research paper:

1. **Robust Preprocessing Pipeline**:
   - Savitzky-Golay Smoothing
   - Standard Normal Variate (SNV) Normalization
   - First Derivative Computation
   - Clip Transform
   - Data Augmentation
2. **Three-Stage Transfer Learning**:
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
<b>Figure 1.</b> Overall workflow of the proposed three-stage transfer learning framework.
</p>

## Repository Structure
```text
raman-spectral-classifier/
├── configs/            # YAML configurations for stages and models
├── data/               # Raw and processed datasets (Not included in Git)
├── docs/               # Detailed documentation for components
├── scripts/            # Public-facing execution scripts
│   ├── archive/        # Archived exploratory scripts
├── src/                # Core library source code
└── README.md           # This file
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/rana-rohit/raman-spectral-classifier.git
cd raman-spectral-classifier
```
2. Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
Due to size constraints, the raw `.npy` and `.csv` datasets are not hosted in this repository. Ensure your raw datasets are placed at `data/raw/`. 

To prepare and validate your data:
```bash
python scripts/setup_data.py
```

## Official Workflow
The official execution order to reproduce the paper's results is:

1. **Data Setup:**
```bash
python scripts/setup_data.py
```
2. **Train the Model (Stage 1 to 2 with TCN):**
```bash
python scripts/train.py --model tcn
```
3. **Train the Model (Stage 3 with TCN):**
```bash
python scripts/run_patient_cv.py --model tcn
```
4. **Evaluate the Model:**
```bash
python scripts/analyze_experiments.py --exp-dir experiments/tcn_...
```
5. **Generate LIME Explanations:**
```bash
python scripts/xai.py --exp-dir experiments/tcn_...
```
6. **Compare Models & Consensus Peak Analysis:**
```bash
python scripts/compare_models_xai.py --results-root experiments/
```
7. **Generate Research Plots:**
```bash
python scripts/generate_research_plots.py --exp_dir experiments/tcn_...
```

## Results & Highlights
- **Best Model Architecture:** Temporal Convolutional Network (TCN)
- **Final Stage 3 Accuracy:** 96.0%
- **Final Patient-Level Accuracy:** 100%
- **Interpretability:** Successfully mapped predictive features back to known biological and chemical Raman peaks using Consensus Peak Analysis.

## License
Distributed under the MIT License. See `LICENSE` for more information.