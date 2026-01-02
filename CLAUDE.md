# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tennis-Lab is a research project for 3D reconstruction of tennis scenes from monocular video. It estimates 3D trajectories of players and balls from single-camera tennis match footage.

### Pipeline Architecture

The system consists of four main modules that work together:

1. **GVHMR** (third_party/): Estimates 3D player meshes (SMPL parameters) and 2D skeletons (ViTPose) from video
2. **WASB** (src/wasb/): Detects 2D ball positions in image space ("Where's the Ball?")
3. **PLCS** (src/plcs/): Converts 2D human skeletons + court keypoints → 3D player position/orientation on court
4. **BLCS** (src/blcs/): Converts 2D ball positions + court keypoints → 3D ball trajectory on court

All modules use a common **court coordinate system** with 20 keypoints (CourtKP20) and COCO-17 format for human poses.

## Core Commands

### Development Setup

```bash
# Install dependencies
uv sync

# Install dev dependencies
uv sync --dev
```

### Testing

```bash
# Run all tests (excluding local_data marker)
uv run pytest -m "not local_data"

# Run specific test file
uv run pytest tests/unit/test_example.py

# Run specific test case
uv run pytest tests/unit/test_example.py::test_case

# Run with coverage
uv run pytest --cov=src --cov-report=html --cov-report=term-missing

# Using the test subagent (recommended)
uv run python -m src.agents.scripts.test
uv run python -m src.agents.scripts.test 'task.test_cmd=uv run pytest tests/...'
```

### Linting and Type Checking

```bash
# Run ruff (auto-fix)
uv run ruff check --fix src tests

# Run mypy
uv run mypy src tests

# Using pre-commit subagent (recommended, runs both)
uv run python -m src.agents.scripts.pre_commit
```

### Training

Each module has its own training script using Hydra configuration:

```bash
# PLCS training
uv run python -m src.plcs.scripts.train
uv run python -m src.plcs.scripts.train run.gpus=1 training.max_epochs=100

# BLCS training
uv run python -m src.blcs.scripts.train

# WASB training (ball detection)
uv run python -m src.wasb.scripts.train.ball_detection
```

### Dataset Generation

PLCS and BLCS use synthetic datasets generated via simulation:

```bash
# Generate PLCS dataset
uv run python -m src.plcs.scripts.generate_dataset
uv run python -m src.plcs.scripts.generate_dataset simulation.num_scenes=1000

# Generate BLCS dataset
uv run python -m src.blcs.scripts.generate_dataset
```

### Visualization

```bash
# Visualize PLCS predictions
uv run python -m src.plcs.scripts.visualize \
    visualization.mode=predict \
    visualization.checkpoint=outputs/plcs/checkpoints/last.ckpt

# Visualize BLCS predictions
uv run python -m src.blcs.scripts.visualize \
    visualization.mode=predict \
    visualization.checkpoint=outputs/blcs/checkpoints/last.ckpt

# WASB ball detection visualization
uv run python -m src.wasb.scripts.visualize.ball_video_ensemble \
    video_path=inputs/demo/match.mp4 \
    output_path=outputs/ball_detection.mp4
```

## Architecture and Design Patterns

### Module Organization

```
src/
├── base/           # Shared abstractions (BasePredictor)
├── blcs/           # Ball Localization in Court System
├── plcs/           # Player Localization in Court System
├── wasb/           # Ball detection ("Where's the Ball")
└── utils/          # Shared utilities
    ├── geometry/   # Court geometry, coordinate systems
    ├── projection/ # Camera projection utilities
    └── rendering/  # Visualization renderers
```

**CRITICAL**: Never duplicate code between modules. Common functionality MUST be placed in:
- `src/base/` for shared abstractions (e.g., BasePredictor)
- `src/utils/` for shared utilities (geometry, rendering, projection)

### Coordinate Systems and Normalization

All modules use a unified tennis court coordinate system:

- **Origin**: Court center at ground level
- **X-axis**: Court width (-5.485 to +5.485 meters, doubles width)
- **Y-axis**: Court length (-11.885 to +11.885 meters)
- **Z-axis**: Height above ground (0 to ~3 meters)

**Normalization convention** (used across all tasks):
```python
x_norm = X / HALF_DOUBLES_WIDTH  # 5.485m
y_norm = Y / HALF_LENGTH          # 11.885m
z_norm = Z / NET_HEIGHT_POST      # 1.07m
```

See `src/utils/geometry/constants.py` for complete definitions.

### PyTorch Lightning Structure

All training modules follow a consistent Lightning pattern:

1. **DataModule**: `src/{task}/data/datamodule.py` - Data loading and batching
2. **Dataset**: `src/{task}/data/dataset.py` - Dataset implementation
3. **LightningModule**: `src/{task}/training/lightning_module.py` - Training loop, losses, metrics
4. **Model**: `src/{task}/models/` - Neural network architectures
5. **Predictor**: `src/{task}/inference/predictor.py` - Inherits from BasePredictor for inference

### Configuration with Hydra

All scripts under `src/{task}/scripts/` MUST use Hydra for configuration management. **Never use argparse**.

Configuration structure:
```
src/{task}/configs/
├── {script_name}.yaml      # Entry point config
├── run/                    # Run settings (gpus, epochs, etc.)
├── model/                  # Model architecture configs
├── training/               # Training hyperparameters
└── paths/                  # Data paths
```

### BasePredictor Pattern

All inference predictors must inherit from `src/base/api/predictor.BasePredictor` and implement:

```python
@classmethod
def load_from_checkpoint(cls, checkpoint_path, device="cpu", **kwargs) -> Self:
    """Load model from checkpoint."""

def predict(self, *args, **kwargs) -> dict[str, Any]:
    """Run inference and return results."""
```

## Development Workflow

### Mandatory Workflow (AI Agents MUST follow)

1. **NEVER work directly on main/master/develop**:
   ```bash
   git checkout -b feature/<task>-<short-desc>
   ```

2. **After ANY code changes**:
   ```bash
   # Run linting/type checking
   bash agents_workspace/sub_agents/pre_commit_subagent.sh

   # Run ONLY affected tests (do NOT run all tests)
   # Identify test files related to your changes and specify them
   bash agents_workspace/sub_agents/test_subagent.sh --test-cmd 'uv run --no-sync pytest -q -n auto tests/unit/test_affected.py'
   ```

3. **Check documentation consistency**:
   - Update `src/{task}/README.md` if your changes affect the documented behavior
   - Keep READMEs in sync with implementation (new features, API changes, config changes)

4. **Exception handling**: If tools fail due to environment issues (e.g., permission errors), apply workarounds and document in final report.

### UV Cache Workaround

If you encounter `Permission denied` errors with `uv`, use one of these:

```bash
# Option 1: Specify cache directory
uv --cache-dir agents_workspace/tmp_cache/uv_cache run pytest

# Option 2: Disable cache (slower but reliable)
uv --no-cache run pytest
```

The subagent scripts handle this automatically.

## Code Quality Standards

### Hydra Configuration Requirements

All executable scripts in `src/{task}/scripts/` must:
- Use Hydra for configuration (no argparse)
- Include module-level docstring with:
  - Brief description
  - Example commands
  - Config entry point path

Example:
```python
"""Train a PLCS model with Hydra-managed configuration.

Example commands:
    `uv run python -m src.plcs.scripts.train`
    `uv run python -m src.plcs.scripts.train run.gpus=1`

Config entry point: `src/plcs/configs/train.yaml`
"""
```

### Type Checking

This project enforces strict type checking:
- All functions must have type hints (`disallow_untyped_defs = true`)
- mypy runs in CI on changed files (PRs) or full codebase (main branch)
- Third-party library types are ignored where necessary (see pyproject.toml)

### Testing

- Use pytest with markers: `unit`, `integration`, `e2e`, `slow`, `local_data`
- CI skips tests marked `local_data` (for large datasets/models)
- Coverage reports generated in `htmlcov/`
- All tests in `tests/` directory

### Pre-commit Hooks

Pre-commit runs:
1. ruff (linting + auto-fix)
2. mypy (type checking)

Only on files matching `^(src|tests)/` pattern.

## CI/CD

GitHub Actions runs on:
- Push to `main` or `feature/*` branches
- Pull requests to `main`

**Lint job**: Runs ruff + mypy
- PRs: Only on changed Python files
- Main push: Full codebase

**Test job**: Runs pytest with coverage
- Excludes `local_data` marker
- Uploads coverage to Codecov
- Python 3.11

## Key Files and Locations

- `pyproject.toml`: Dependencies, ruff/mypy/pytest config
- `uv.lock`: Lockfile for reproducible builds
- `.pre-commit-config.yaml`: Pre-commit hook definitions
- `AGENTS.md`: Detailed workflow rules (Japanese, for AI agents)
- `src/base/api/predictor.py`: BasePredictor abstract class
- `src/utils/geometry/constants.py`: CourtKP20, COCO-17, SMPL-H definitions
- `src/utils/geometry/court.py`: Court dimensions and 3D keypoints
- `court.py`: Standalone court geometry helpers

## Important Notes

1. **No code duplication**: Always check `src/base/` and `src/utils/` before implementing shared functionality
2. **Coordinate system**: Use the normalized court coordinate convention consistently
3. **Configuration**: All scripts use Hydra, never argparse
4. **Testing**: Always run pre-commit and tests before committing
5. **Branching**: Never commit directly to main
