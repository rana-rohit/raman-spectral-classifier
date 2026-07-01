# Dataset and Preprocessing

## Overview
The dataset consists of 1D Raman spectra mapping to various antimicrobial treatments. The data pipeline is designed to load raw `.npy` files efficiently and apply a robust preprocessing pipeline.

## Preprocessing Pipeline
To handle domain shift between reference and clinical data, the following transformations are applied sequentially:
1. **Savitzky-Golay Smoothing**: Reduces high-frequency noise while preserving peak shapes.
2. **Standard Normal Variate (SNV) Normalization**: Per-sample centering and scaling to remove multiplicative scatter and baseline offset.
3. **First Derivative**: Inherently invariant to additive baseline offsets.
4. **Clip Transform**: Clips extreme values to prevent instability.

## Usage
Data splits and preprocessing are configured in `configs/data/`. Run `python scripts/setup_data.py` to validate and prepare the datasets before training.
