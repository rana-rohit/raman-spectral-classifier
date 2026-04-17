# Domain Robustness Implementation Log

Date: 2026-04-17

## Purpose

This document records the code changes made after the pipeline repair pass to improve domain robustness for Raman spectral classification while preserving the existing multi-model training structure.

The goal of this pass was to keep the pipeline architecture intact and add four modular capabilities:

- auxiliary shared-class supervision for the clinical subset
- L2-SP regularized finetuning
- clinical-style spectral augmentation
- multi-view consistency regularization

These changes were implemented so they work across all supported model families:

- CNN
- ResNet1D
- Transformer
- Hybrid

## Summary Of What Changed

The system now supports:

1. A shared auxiliary head over clinically relevant classes `{0, 2, 3, 5, 6}` while still training the main 30-class classifier.
2. Finetuning regularized against the pretrained solution instead of allowing uncontrolled drift.
3. Augmentations that better simulate clinical Raman distortions.
4. Two-view training with prediction consistency loss for acquisition-invariant learning.

## Component 1: Auxiliary Shared-Class Head

### Objective

Add clinically focused supervision without collapsing the main task to 5-class training.

### Changes Made

- Added `src/models/multitask.py` with `MultiHeadSpectralModel`.
- Added `src/utils/class_subset.py` with reusable subset utilities:
  - `subset_mask`
  - `remap_targets_to_subset`
  - `slice_logits_to_subset`
  - `prepare_subset_eval_logits`
- Updated all backbone models to expose:
  - `embedding_dim`
  - `forward_features(x)`
- Updated the model registry to wrap any backbone with the auxiliary head when enabled.
- Updated the trainer to compute:
  - main 30-class supervised loss
  - auxiliary 5-class supervised loss only for samples in the shared subset
- Updated the evaluator to blend sliced main logits with auxiliary logits during clinical evaluation.

### Files

- `src/models/cnn.py`
- `src/models/resnet1d.py`
- `src/models/transformer.py`
- `src/models/hybrid.py`
- `src/models/multitask.py`
- `src/models/registry.py`
- `src/utils/class_subset.py`
- `src/training/trainer.py`
- `src/evaluation/clinical_utils.py`
- `src/evaluation/evaluator.py`
- `configs/training/base.yaml`

### Result

- Main training remains 30-class.
- Shared classes receive additional targeted supervision.
- Clinical evaluation can use blended subset-aware logits instead of relying only on the 30-class head.

## Component 2: L2-SP Regularized Finetuning

### Objective

Reduce destructive representation drift during adaptation on the small finetune split.

### Changes Made

- Added `src/training/regularizers.py` with `L2SPRegularizer`.
- Captured a frozen copy of the pretrained parameter state immediately after loading the best pretrained checkpoint.
- Passed that reference state into both finetuning phases.
- Integrated the L2-SP penalty into the trainer loss path through config flags under `training.l2sp`.
- Kept the same reference anchor for both the frozen and full finetuning phases.

### Files

- `src/training/regularizers.py`
- `src/training/trainer.py`
- `src/training/finetuner.py`
- `configs/training/base.yaml`

### Result

- Finetuning now has an architecture-independent mechanism to stay near the pretrained solution.
- The regularization can be enabled or disabled per experiment through config only.

## Component 3: Clinical-Style Augmentation

### Objective

Expose the model to distortions that are more representative of clinical Raman acquisition conditions.

### Changes Made

- Reworked `src/data/augmentation.py` to support additional spectral perturbations:
  - `BaselineDrift`
  - `MultiplicativeIntensityScale`
  - `PeakBroadening`
  - `NonlinearSpectralWarp`
- Kept the existing augmentations:
  - `GaussianNoise`
  - `BaselineShift`
  - `AmplitudeScaling`
  - `SpectralShift`
- Added configurable clipping bounds to `AugmentationPipeline`.
- Updated the augmentation config with a clinical-style default recipe.

### Files

- `src/data/augmentation.py`
- `configs/data/augmentation.yaml`

### Result

- Training augmentation is now able to simulate low-frequency drift, gain variation, slight smoothing, and mild nonlinear acquisition warp instead of only simple noise and shifts.

## Component 4: Consistency Regularization

### Objective

Train the model to produce stable predictions for two independently augmented views of the same spectrum.

### Changes Made

- Extended `SpectralDataset` to optionally return two training views:
  - `{"x1", "x2", "y"}`
- Updated loader construction to enable two-view batches only for train and finetune when consistency is enabled.
- Added `consistency_loss` to `src/training/losses.py`.
- Updated `Trainer` batch parsing to support:
  - standard tuple batches
  - multi-view dict batches
- Added consistency loss computation for:
  - main logits
  - auxiliary logits on shared-class samples
- Added config control for whether supervised loss is applied on both views.

### Files

- `src/data/dataset.py`
- `src/data/dataloader.py`
- `src/training/losses.py`
- `src/training/trainer.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/run_ablation.py`
- `configs/training/base.yaml`

### Result

- Training can now enforce view-invariant predictions without changing evaluation behavior.
- Validation, test, and clinical inference remain single-view.

## Supporting Integration Changes

### Training Flow

- `build_trainer()` now receives the full config and optional pretrained reference state.
- Trainer normalizes both legacy tensor outputs and multitask dict outputs.
- Total training loss is now:

```text
main_loss
+ aux_loss_weight * auxiliary_loss
+ consistency_weight * consistency_loss
+ l2sp_penalty
```

### Finetuning Flow

- Finetuning snapshots pretrained weights once and reuses that anchor through all adaptation phases.
- Frozen and unfrozen phases both use the same shared training logic.
- Few-shot subsampling remains supported.

### Evaluation Flow

- Standard splits use the main 30-class logits.
- Clinical splits use subset-logit evaluation and optional blending with the auxiliary head.
- Batch parsing remains compatible with both standard and multi-view dataset outputs.

## Configuration Added

### `configs/training/base.yaml`

```yaml
training:
  l2sp:
    enabled: false
    lambda: 0.001
    exclude_patterns: []

  consistency:
    enabled: false
    views: 2
    supervised_on_both_views: true
    loss_type: mse_probs
    temperature: 1.0
    loss_weight: 0.1

multitask:
  auxiliary_shared_head:
    enabled: false
    classes: [0, 2, 3, 5, 6]
    dropout: 0.0
    loss_weight: 0.3
    clinical_blend: 0.5
```

### `configs/data/augmentation.yaml`

The augmentation recipe now includes:

- `multiplicative_intensity`
- `baseline_drift`
- `peak_broadening`
- `nonlinear_warp`

and configurable clip bounds:

- `clip_min`
- `clip_max`

## Files Added

- `src/models/multitask.py`
- `src/training/regularizers.py`
- `src/utils/class_subset.py`
- `DOMAIN_ROBUSTNESS_CHANGES.md`

## Tests Added Or Expanded

Updated `tests/test_pipeline_regressions.py` with coverage for:

- checkpoint resolution
- subset-logit evaluation
- auxiliary-logit blending
- hidden finetune prevention
- L2-SP penalty behavior
- two-view dataset output shape
- multitask model wrapping

## Validation

Validation command run:

```powershell
python -m pytest tests -q --basetemp experiments\.pytest_tmp
```

Result:

- `29 passed`

Note:
Pytest emitted one cache warning because `.pytest_cache` is not writable in the current environment. Test execution still completed successfully.

## Operational Impact

The pipeline now supports moving from a source-only classifier to a domain-robust training recipe without redesigning the overall system:

1. train the main model with optional auxiliary shared-class supervision
2. expose training to clinically realistic spectral perturbations
3. optionally enforce multi-view consistency
4. finetune with L2-SP anchoring to the pretrained solution
5. evaluate clinical splits using subset-aware logits, with optional auxiliary blending

## Remaining Work

This pass implemented the required capabilities, but it did not run full dataset-scale retraining in this session.

Still required:

- enable the new config flags in experiment configs
- run clean reference-to-finetune experiments for CNN, ResNet1D, Transformer, and Hybrid
- compare clinical F1 with and without auxiliary head, consistency, and L2-SP
- tune only the new feature weights if needed

## Bottom Line

The system is no longer only a repaired lab-data pipeline. It now includes the code paths needed for subset-aware supervision, regularized adaptation, clinically motivated augmentation, and view-invariant training while preserving compatibility with all four model families.
