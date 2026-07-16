# Evaluation and Patient Voting

## Standalone Evaluation
Models can be evaluated on static test sets using `scripts/evaluate.py`. This script outputs JSON files containing key performance metrics.

## Patient-Level Probabilistic Voting
Since multiple spectra are collected per patient, relying on a single spectrum is error-prone. The evaluation pipeline implements **Patient-Level Probabilistic Voting**, which aggregates predictions across all spectra belonging to a single patient to determine the final prediction.

This approach achieved a **100% Patient-Level Accuracy** in Stage 3.

## Execution
```bash
python scripts/run_patient_cv.py \
  --model tcn \
  --stage s3_transfer \
  --exp-name tcn_s3_transfer_ts_iid_patient_cv \
  --override training.pretrained_checkpoint=experiments/tcn_s2_treatment_iid/checkpoints/best_model.pt

python scripts/aggregate_folds.py --run-dir experiments/tcn_s3_transfer_ts_iid_patient_cv
python scripts/analyze_experiment.py --exp_dir experiments/tcn_s3_transfer_ts_iid_patient_cv
```
