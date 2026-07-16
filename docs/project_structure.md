# Project Structure

```text
raman-spectral-classifier/
|-- artifacts/          # Publication checkpoints and released result figures
|-- assets/             # Documentation images
|-- configs/            # YAML configurations for data, models, stages, and training
|-- data/               # Local raw datasets; not redistributed
|-- docs/               # Component documentation
|-- metadata/           # Label ontology, isolates, treatments, and patient IDs
|-- notebooks/          # Canonical reproduction and analysis notebooks
|   `-- archive/        # Archived exploratory notebooks
|-- scripts/            # CLI entry points
|   `-- archive/        # Archived development scripts
|-- src/                # Core library source code
|   |-- data/           # Dataset, preprocessing, split, and dataloader logic
|   |-- evaluation/     # Metrics, visualizations, and clinical utilities
|   |-- interpretability/ # Archived interpretability utilities
|   |-- models/         # Network architectures and registries
|   |-- training/       # Training loops, finetuning logic, and schedulers
|   |-- utils/          # Config, checkpoint, logging, and seed utilities
|   `-- xai/            # LIME, saliency, and XAI visualization logic
`-- tests/              # Unit and regression tests
```
