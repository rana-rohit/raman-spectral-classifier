# Explainable AI (XAI)

## Methodology
To ensure transparency in clinical decision making, the framework uses **LIME (Local Interpretable Model-agnostic Explanations)** to explain individual spectral predictions.

## Consensus Raman Peak Analysis
Rather than interpreting isolated samples, we extract the top-K most important peaks across multiple LIME explanations and compute a **Consensus Peak Frequency**. This maps the deep learning network's attention directly back to known biological Raman peaks.

## Execution
```bash
python scripts/xai.py --exp-dir <stage3_fold_experiment_dir>
python scripts/compare_models_xai.py --results-root experiments/
```
