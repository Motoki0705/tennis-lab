# ball_detection

`src/ball_detection` implements a staged workflow for tennis ball detection:

1. supervised pretraining on labeled data
2. pseudo-label generation on unlabeled data via ensemble + refinement
3. self-training on mixed labeled/pseudo data

## Key structure

- `data/io`: dataset layout, annotation ingestion/merge, and writing policies
- `pseudo/components`: clip sampling, trajectory refinement, event tagging, quality checks
- `pseudo/orchestrator.py`: end-to-end pseudo-label generation workflow
- `training`: Lightning training modules and runner
- `scripts`: Hydra entrypoints
