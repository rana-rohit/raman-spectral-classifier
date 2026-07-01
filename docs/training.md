# Training and Transfer Learning

## Three-Stage Architecture
The project utilizes a Three-Stage Transfer Learning architecture to handle complex clinical mapping:

1. **Stage 1 (Isolate Classification)**: Pre-trains on reference datasets to classify 30 distinct bacterial isolates.
2. **Stage 2 (Semantic Alignment)**: Maps the 30 isolate classes into 8 broader treatment groups.
3. **Stage 3 (Clinical Transfer Learning)**: Fine-tunes the network to classify 5 specific treatment classes on the target clinical dataset.

## Execution
Training is managed via `scripts/train.py`.
```bash
python scripts/train.py --model tcn
```
Model hyperparameters are managed in `configs/model/`. Stage hyperparameters are in `configs/stages/`.
