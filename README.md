# Raman Spectral Classification System

This repository implements a modular, research-grade deep learning pipeline for 1D spectral classification with PyTorch. It is designed to support rapid experimentation, robust evaluation, and achieving state-of-the-art accuracy (~97%) on the Raman dataset.

## Current Architecture

The core pipeline has been restructured for maximum performance:

- **Fast Data Loading**: Direct `.npy` array loading eliminates the I/O bottleneck of per-file CSV reading.
- **Robust Preprocessing**: Built-in SNV, baseline correction (ALS), Savitzky-Golay smoothing, and standard scaling.
- **On-the-fly Augmentation**: Gaussian noise, intensity scaling, wavenumber shifting, and baseline drift injection to improve generalization.
- **Advanced Architectures**:
  - `resnet1d`: Deep residual network with Squeeze-and-Excitation (SE) attention (the top-performing model)
  - `cnn1d`: Enhanced baseline convolutional network with SE-attention
  - `hybrid`: Convolutional stem with SE + transformer encoder
  - `transformer1d`: Attention-based sequence model
  - `classical`: Highly-tuned LogReg + PCA and HistGradientBoosting + PCA baselines
- **Enhanced Training**: Label smoothing, Mixup augmentation, and OneCycleLR scheduling.
- **Unified Pipeline**: A single script to clean data, train all models, evaluate, and build a probability ensemble.
- **Target Accuracy**: ~97% F1 macro score on holdout sets.

## 💾 Dataset Handling & Google Colab

Due to file size limits, the dataset (large `.npy` files) is **not** included in this repository. 

**Expected Folder Structure:**
Before running any training scripts, ensure your dataset is placed at `data/raw/`.

### Running in Google Colab

If you want to train this model on Google Colab, follow these steps to securely mount your dataset from Google Drive:

**Step 1: Clone the Repository & Install Dependencies**
```python
# Run this in a Colab cell
!git clone https://github.com/rana-rohit/raman-spectral-classifier.git
%cd raman-spectral-classifier
!pip install -r requirements.txt
```

**Step 2: Mount Google Drive & Load Dataset**
Upload your your dataset to your personal Google Drive, then mount it in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')

# Create the expected directory structure in the repo
!mkdir -p data/raw

# Copy your data from Drive into the colab workspace for faster disk I/O during training
# (Change the 'path_to_your_data' to your actual Drive path)
!cp /content/drive/MyDrive/path_to_your_data/*.npy ./data/raw/
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the full unified pipeline (cleans data, trains all models, builds ensemble):
```bash
python experiments/run_full_pipeline.py
```
*Note: Use `--device cuda:0` to train on the GPU and `--epochs 40` to set max epochs.*

3. Analyze the results:
Open the `notebooks/03_full_pipeline_results.ipynb` notebook to visualize the model leaderboard, per-class F1 metrics, and confusion matrices for the test and clinical splits.

## Project Structure

```text
.
|-- data/
|   `-- raw/                     # .npy files go here
|-- experiments/
|   |-- configs/                 # YAML model configs
|   |-- run_full_pipeline.py     # Main unified entry point
|-- notebooks/
|   `-- 03_full_pipeline_results.ipynb
|-- outputs/
|   `-- pipeline_runs/           # Output directory for each run
|-- src/
|   `-- raman_classifier/
|       |-- classical/           # Classical ML trainers
|       |-- config.py            
|       |-- data/                # NumpyDataset and Augmentations
|       |-- evaluation/          # Metrics and confusion matrices
|       |-- models/              # Deep Architectures (ResNet1D, CNN, Hybrid)
|       `-- training/            # PyTorch Trainer with Mixup/OneCycleLR
`-- requirements.txt
```

## Running Individual Components

The new unified pipeline automatically configures and runs the full suite, but you can also run specific subsets:

```bash
# Run only classical models
python experiments/run_full_pipeline.py --skip-deep

# Run only specific deep models
python experiments/run_full_pipeline.py --skip-classical --models resnet1d cnn1d_se
```

All metrics, model checkpoints, probability CSVs, and confusion matrices are saved automatically to `outputs/pipeline_runs/run_YYYYMMDD_HHMMSS/`.
