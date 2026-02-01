# Run Configuration Schema

This document defines the common run configuration schema used across all training tasks in Tennis-Lab.

## Overview

All training tasks use a standardized set of configuration keys under `config.run.*` to control training execution, resumption, and debugging options. These keys are referenced by the `BaseTrainingRunner` class.

## Common Keys

All `configs/run/*.yaml` files for training tasks should include the following keys:

### `output_dir`
- **Type**: `string`
- **Default**: Task-specific (e.g., `outputs/blcs/single`, `outputs/plcs/frame`)
- **Description**: Directory where training outputs (checkpoints, logs, config) will be saved
- **Usage**: Referenced in `BaseTrainingRunner.prepare_output_dir()`

### `seed`
- **Type**: `integer` or `null`
- **Default**: `42`
- **Description**: Random seed for reproducibility. If set, PyTorch Lightning's `seed_everything()` is called
- **Usage**: Referenced in `BaseTrainingRunner.seed_everything()`

### `gpus`
- **Type**: `integer`
- **Default**: `1`
- **Description**: Number of GPUs to use for training. If `0` or CUDA is unavailable, falls back to CPU
- **Usage**: Referenced in `BaseTrainingRunner.select_devices()`

### `resume`
- **Type**: `string` or `null`
- **Default**: `null`
- **Description**: Path to checkpoint file to resume training from. If `null`, training starts from scratch
- **Usage**: Referenced in `BaseTrainingRunner.resolve_resume()`

### `fast_dev_run`
- **Type**: `boolean`
- **Default**: `false`
- **Description**: If `true`, runs a quick sanity check (1 batch train/val). Skips test phase
- **Usage**: Referenced in `BaseTrainingRunner.build_trainer()` and `BaseTrainingRunner.skip_test()`

### `dry_run`
- **Type**: `boolean`
- **Default**: `false`
- **Description**: If `true`, loads one batch, prints shapes, and runs 1 epoch on CPU without full training
- **Usage**: Referenced in `BaseTrainingRunner.is_dry_run()` and `BaseTrainingRunner.run_dry_run()`

## Standard Key Order

For consistency, use the following key order in all run configuration files:

```yaml
output_dir: <path>
seed: <integer or null>
gpus: <integer>
resume: <path or null>
fast_dev_run: <boolean>
dry_run: <boolean>
```

## Common CLI Options

With Hydra, these keys can be overridden from the command line:

```bash
# Override output directory
uv run python -m src.blcs.scripts.train run.output_dir=custom/path

# Resume from checkpoint
uv run python -m src.plcs.scripts.train run.resume=outputs/plcs/checkpoints/last.ckpt

# Quick sanity check (1 batch)
uv run python -m src.wasb.scripts.train run.fast_dev_run=true

# Dry run (load data and check shapes only)
uv run python -m src.court_detection.scripts.train run.dry_run=true

# Use CPU only
uv run python -m src.blcs.scripts.train run.gpus=0

# Set custom seed
uv run python -m src.plcs.scripts.train run.seed=123
```

## Implementation Reference

The common run configuration schema is implemented in `src/base/training/runner.py`:

- **Line 70**: `prepare_output_dir()` uses `config.run.output_dir`
- **Line 73-76**: `resolve_resume()` uses `config.run.resume`
- **Line 138-140**: `seed_everything()` uses `config.run.seed`
- **Line 143**: `is_dry_run()` uses `config.run.dry_run`
- **Line 146**: `skip_test()` uses `config.run.fast_dev_run`
- **Line 197**: `build_trainer()` uses `config.run.fast_dev_run`
- **Line 205**: `select_devices()` uses `config.run.gpus`

## Tasks Using This Schema

The following training tasks use the standardized run configuration schema:

- `src/blcs/configs/run/train.yaml`
- `src/blcs/configs/run/train_multiview.yaml`
- `src/plcs/configs/run/train.yaml`
- `src/plcs/configs/run/train_multiview.yaml`
- `src/plcs/configs/run/train_sequence.yaml`
- `src/wasb/configs/run/ball_detection.yaml`
- `src/court_detection/configs/run/default.yaml`
- `src/evnet_detection/configs/run/train.yaml`
- `src/trajectory_completion/configs/run/train.yaml`

## Non-Training Configs

Some run configurations are for non-training tasks (e.g., dataset generation, visualization, analysis) and may have different schemas tailored to their specific needs:

- `src/plcs/configs/run/generate_dataset.yaml`
- `src/plcs/configs/run/analyze_dataset_distribution.yaml`
- `src/blcs/configs/run/generate_dataset.yaml`
- `src/blcs/configs/run/visualize.yaml`

These files are not required to follow the training run schema.
