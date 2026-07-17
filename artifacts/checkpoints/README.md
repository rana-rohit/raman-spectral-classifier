# Model Checkpoints

This directory stores model checkpoints generated during training.

The repository preserves the expected checkpoint directory structure using `.gitkeep` files so that training scripts can write outputs to consistent locations.

Generated checkpoint files (for example, `*.pt`) are intentionally excluded from version control because they are large runtime artifacts and can be reproduced by running the training pipeline described in the repository documentation and notebooks.

## Directory Structure

```
artifacts/
└── checkpoints/
    ├── cnn/
    ├── cnn_transformer/
    ├── inception1d/
    ├── resnet1d/
    ├── tcn/
    └── transformer/
```

Each subdirectory stores the checkpoints produced for its corresponding model architecture.