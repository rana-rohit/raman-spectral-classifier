# Raman Spectral Classification System

This repository implements a modular, research-grade deep learning pipeline for 1D spectral classification with PyTorch. It is designed to support rapid experimentation, robust evaluation, and advanced domain adaptation to achieve high accuracy on both reference datasets and out-of-distribution (OOD) clinical Raman spectral data.

## Current Architecture

The core pipeline is heavily optimized for both speed and generalization:

- **Fast Data Loading**: Direct `.npy` array loading eliminates the I/O bottleneck of per-file CSV reading.
- **Robust Preprocessing**: Configurable SNV, baseline correction (ALS), Savitzky-Golay smoothing, and standard scaling.
- **On-the-fly Augmentation**: Gaussian noise, intensity scaling, wavenumber shifting, and baseline drift injection to improve generalization.
- **Advanced Deep Architectures**:
  - `resnet1d`: Deep residual network with Squeeze-and-Excitation (SE) attention (typically top-performing).
  - `cnn`: Enhanced baseline convolutional network with SE-attention.
  - `hybrid`: Convolutional stem with SE + transformer encoder.
  - `transformer`: Attention-based sequence model.
- **Domain Adaptation & Finetuning**: A dedicated finetuning pipeline incorporating Domain-Adversarial Neural Networks (DANN), CORAL loss, and consistency regularization to close the generalization gap between reference and clinical distributions.
- **YAML-driven Configuration**: Complete control over data splits, preprocessing, augmentations, and model architectures via modular YAML files.
- **FastAPI Inference**: Built-in production-ready REST API for model serving.

## Dataset Handling

Due to file size limits, the dataset (large `.npy` files) is **not** included in this repository. 

**Expected Folder Structure:**
Before running any training scripts, ensure your dataset is placed at `data/raw/`.

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Train a model:**
You can train a model by specifying the architecture (`cnn`, `resnet1d`, `hybrid`, `transformer`). The script automatically loads configs from the `configs/` directory, trains the model, evaluates it, and runs the finetuning/domain adaptation phase.

```bash
python scripts/train.py --model resnet1d
```
*Note: Use `--override training.batch_size=64` to easily override specific configuration parameters.*

3. **Evaluate a trained model:**
```bash
python scripts/evaluate.py --exp-dir experiments/resnet1d_YYYYMMDD_HHMMSS
```

4. **Compare multiple models:**
```bash
python scripts/evaluate.py --compare experiments/run1 experiments/run2 --split test
```

## Running the API

You can serve a trained model locally using the included FastAPI backend.

```bash
# Set the environment variable to your trained experiment directory
export RAMAN_EXPERIMENT_DIR=experiments/resnet1d_YYYYMMDD_HHMMSS

# Run the API
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
Then, visit `http://localhost:8000/docs` to test the `/predict` endpoint via the Swagger UI.

## Project Structure

```text
.
├── app/                  # FastAPI backend for inference
├── configs/              # YAML configuration files
│   ├── data/             # Splits, preprocessing, and augmentation configs
│   ├── model/            # Architecture-specific hyperparameters
│   └── training/         # Base training config, optimizer, losses, etc.
├── data/
│   └── raw/              # .npy files go here
├── experiments/          # Output directory for training logs and artifacts
├── notebooks/            # Jupyter notebooks for data exploration and analysis
├── scripts/              # Executable entry points
│   ├── train.py          # Main training and finetuning script
│   ├── evaluate.py       # Standalone evaluation script
│   ├── setup_data.py     # Data preparation and integrity checks
│   └── deep_analysis.py  # Model interpretation and metric analysis
├── src/                  # Core package
│   ├── data/             # NumpyDataset, Dataloaders, and Augmentations
│   ├── evaluation/       # Metrics, confusion matrices, McNemar's test
│   ├── interpretability/ # Grad-CAM, Integrated Gradients, etc.
│   ├── models/           # CNN, Hybrid, ResNet1D, Transformer
│   ├── training/         # Main Trainer, Finetuner, Losses, and Schedulers
│   └── utils/            # Helper functions for config parsing and I/O
└── tests/                # Unit tests
```

## Google Colab Execution

If you want to train this model on Google Colab, you can securely mount your dataset from Google Drive:

```python
# 1. Clone the repo and install dependencies
!git clone https://github.com/rana-rohit/raman-spectral-classifier.git
%cd raman-spectral-classifier
!pip install -r requirements.txt

# 2. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Create the expected directory structure and copy data
!mkdir -p data/raw
!cp /content/drive/MyDrive/path_to_your_data/*.npy ./data/raw/

# 4. Run training
!python scripts/train.py --model resnet1d
```
