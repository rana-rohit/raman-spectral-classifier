# Project Structure

```text
raman-spectral-classifier/
├── configs/            # YAML configurations for stages and models
├── data/               # Raw and processed datasets
├── docs/               # Component documentation
├── experiments/        # Generated model checkpoints, outputs, and metrics
├── notebooks/          # Clean tutorial and demonstration notebooks
├── scripts/            # CLI Entry points (train, evaluate, xai, setup_data, etc.)
│   └── archive/        # Obsolete scripts safely archived
├── src/                # Core library source code
│   ├── data/           # Dataset, preprocessing, and augmentations
│   ├── evaluation/     # Metrics, visualizations, and clinical utils
│   ├── interpretability/# Archived interpretability utilities
│   ├── models/         # Network architectures and registries
│   ├── training/       # Training loops, finetuning logic, schedulers
│   ├── utils/          # Config, logging, and seed utilities
│   └── xai/            # LIME explainer and visualization logic
└── tests/              # Unit and integration tests
```
