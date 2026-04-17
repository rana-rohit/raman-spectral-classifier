# Multi-Model Pipeline Repair Log

Date: 2026-04-17

## Purpose

This document records the system-level changes made during the audit and repair pass for the Raman spectral classification pipeline.

The focus of this pass was not model-specific tuning. The changes were made to fix shared failures affecting all architectures:

- CNN
- ResNet1D
- Transformer
- Hybrid

## Root Problem

The main issue was a broken shared pipeline, not a weak individual model.

The system had several coupled failures:

- training and finetuning phases were mixed together
- the model was being mutated before final evaluation
- checkpoint save/load paths did not match
- clinical evaluation used inconsistent 30-class vs 5-class logic
- invalid clinical predictions were being dropped instead of counted as errors
- reproducibility was weak because loader seeding and augmentation state were not isolated
- ablation scripts reused stale loaders, so some experiments were not actually changing what they claimed to change

## Change Summary

### 1. Separated training from finetuning

Problem:
The shared trainer was secretly running a finetune phase inside `Trainer.fit()`. Then `scripts/train.py` called finetuning again. This caused double adaptation, hidden state changes, and misleading evaluation.

Changes made:

- removed implicit finetuning from `src/training/trainer.py`
- made `Trainer.fit()` responsible only for source training
- updated `scripts/train.py` so pretraining evaluation happens after reloading the best pretrained checkpoint
- moved finetuning into an explicit follow-up stage with its own output directory

Files:

- `src/training/trainer.py`
- `scripts/train.py`

Expected effect:

- no duplicate finetune execution
- clean separation between source training and adaptation
- test metrics now reflect the best pretrained model instead of a mutated model

### 2. Rebuilt finetuning as an explicit 2-phase process

Problem:
Freeze/unfreeze logic and optimizer handling were incorrect. The old implementation did not clearly separate the frozen-head phase from the full-model phase, and optimizer state was not reset in a clean architecture-independent way.

Changes made:

- rewrote `src/training/finetuner.py`
- created an explicit frozen-backbone phase when `freeze_epochs > 0`
- reloaded the best checkpoint between finetune phases
- rebuilt the trainer, optimizer, and scheduler for each phase
- kept validation, test, and clinical evaluation fixed while only replacing the training loader with the finetune split
- saved finetune outputs in phase-specific directories and a summary file

Files:

- `src/training/finetuner.py`

Expected effect:

- prevents catastrophic updates from stale optimizer state
- makes finetuning behavior architecture-independent
- produces cleaner and reproducible adaptation runs

### 3. Fixed checkpoint save/load mismatch

Problem:
Training saved best checkpoints under `checkpoints/best.pt`, but loading code looked for `best.pt` at the experiment root. This caused `FileNotFoundError` and made some evaluations load the wrong state or fail entirely.

Changes made:

- added canonical checkpoint resolution in `src/utils/checkpoint.py`
- support both `checkpoints/best.pt` and legacy root-level `best.pt`
- updated training, finetuning, evaluation, and ablation flows to reload the best checkpoint before reporting metrics

Files:

- `src/utils/checkpoint.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/run_ablation.py`
- `src/training/finetuner.py`

Expected effect:

- no more checkpoint path mismatch
- evaluation consistently uses the best saved model

### 4. Corrected clinical label-space evaluation

Problem:
The model predicts in 30-class space, but clinical evaluation is valid only for the shared subset `{0, 2, 3, 5, 6}`. The previous logic mapped predictions after argmax and dropped invalid predictions, which made the evaluation logically incorrect and fragile.

Changes made:

- replaced the old clinical mapping logic with subset-logit evaluation
- for clinical splits, the evaluator now slices logits to the valid class subset before scoring
- targets are remapped to contiguous 5-class indices
- invalid predictions are no longer discarded; they become wrong choices inside the valid subset evaluation
- evaluator now uses `loader.dataset.class_filter` instead of split-name string matching

Files:

- `src/evaluation/clinical_utils.py`
- `src/evaluation/evaluator.py`
- `src/data/dataset.py`

Expected effect:

- removes the 30-class vs 5-class evaluation mismatch
- prevents `KeyError` from unsupported clinical predictions
- makes clinical evaluation architecture-independent and consistent

### 5. Stabilized metric computation and model selection

Problem:
The monitoring logic was too weak for this setup. Validation accuracy alone was being used to decide best checkpoints, and training metrics did not compute F1 directly. Metric code also had edge cases around empty inputs and class coverage.

Changes made:

- added configurable `monitor_metric` to training config
- defaulted model selection to `f1_macro`
- trainer now computes train metrics from accumulated logits and targets
- added safe handling for empty metric inputs
- updated MCC calculation to consider both true and predicted label coverage
- added empty confusion-matrix handling

Files:

- `configs/training/base.yaml`
- `src/training/trainer.py`
- `src/evaluation/metrics.py`

Expected effect:

- better alignment between checkpoint selection and class-balanced performance
- more stable logging and fewer NaN-style metric failures

### 6. Improved reproducibility and loader behavior

Problem:
DataLoader workers and augmentation RNG state were not isolated cleanly. This weakened reproducibility and made loader behavior phase-dependent.

Changes made:

- added explicit seed support to loader construction
- added worker seeding for NumPy and Python random
- cloned augmentation pipelines per training loader so train and finetune do not share mutable RNG state
- preserved class-filter metadata in datasets
- changed dataset `n_classes` logic to use unique labels instead of `max + 1`

Files:

- `src/data/dataloader.py`
- `src/data/dataset.py`

Expected effect:

- more reproducible training and finetuning
- cleaner separation between training phases
- safer evaluation on filtered label subsets

### 7. Fixed augmentation output behavior

Problem:
The existing tests assumed augmentation outputs stay within the normalized range `[0, 1]`, but the pipeline did not clip them back after applying noise and shifts.

Changes made:

- clipped augmentation pipeline output to `[0, 1]`

Files:

- `src/data/augmentation.py`

Expected effect:

- keeps augmented inputs within expected bounds
- restores consistency with the existing data-layer assumptions and tests

### 8. Corrected evaluation and ablation scripts

Problem:
Evaluation and ablation code reused shared loaders built from a default config, which could make experiments inconsistent with the model config being evaluated. Some ablations reused stale augmentation settings.

Changes made:

- evaluation now rebuilds loaders from the experiment config being evaluated
- ablations now rebuild loaders per configuration instead of reusing stale ones
- ablation runs reload best checkpoints before reporting results
- training now saves pretraining evaluation separately from finetuning outputs

Files:

- `scripts/evaluate.py`
- `scripts/run_ablation.py`
- `scripts/train.py`

Expected effect:

- each experiment is evaluated under its own configuration
- augmentation ablations now actually change augmentation
- better comparability across models and experiment variants

### 9. Added regression coverage for the repaired system

Changes made:

- added tests for checkpoint resolution
- added tests for clinical subset evaluation
- added tests to ensure the shared trainer does not perform hidden finetuning

Files:

- `tests/test_pipeline_regressions.py`

Expected effect:

- protects the shared pipeline from regressing in future edits

## Files Changed

- `configs/training/base.yaml`
- `scripts/evaluate.py`
- `scripts/run_ablation.py`
- `scripts/train.py`
- `src/data/augmentation.py`
- `src/data/dataloader.py`
- `src/data/dataset.py`
- `src/evaluation/clinical_utils.py`
- `src/evaluation/evaluator.py`
- `src/evaluation/metrics.py`
- `src/training/finetuner.py`
- `src/training/trainer.py`
- `src/utils/checkpoint.py`
- `tests/test_pipeline_regressions.py`

## Validation

Validation command run:

```powershell
python -m pytest tests -q --basetemp experiments\.pytest_tmp
```

Result:

- `25 passed`

Note:
Pytest emitted one cache warning because `.pytest_cache` is not writable in the current environment. This did not affect test execution.

## Known Remaining Work

This pass repaired the shared pipeline, but it did not rerun full end-to-end training on the real dataset in this session.

Still required:

- retrain each architecture with the repaired pipeline
- compare pretrain vs finetune behavior on clean runs
- measure actual test and clinical F1 after repair
- tune adaptation strategy if domain gap remains large after the pipeline bugs are removed

## Bottom Line

The key change was turning a fragile, state-leaking, architecture-coupled workflow into an explicit staged pipeline:

1. train on source
2. reload best source checkpoint
3. evaluate source performance cleanly
4. finetune in explicit phases
5. evaluate test and clinical splits with correct label-space logic

This is the foundation needed before any model-level comparison or domain adaptation tuning can be trusted.
