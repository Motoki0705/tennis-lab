"""
Train the SLCS multimodal temporal fusion model with Hydra-managed configuration.

Usage:
    python -m src.tasks.slcs.scripts.train
    python -m src.tasks.slcs.scripts.train run.gpus=0 training.trainer.max_epochs=1
    python -m src.tasks.slcs.scripts.train model=small data.batch_size=2
    python -m src.tasks.slcs.scripts.train court_coordinate_normalization=v2
    python -m src.tasks.slcs.scripts.train run.dry_run=true

Notes:
    - Configuration is loaded from `src/tasks/slcs/configs/train.yaml`.
    - The dataset must follow the issue #634 contract with completed
      tennis_scene annotations, precomputed DINOv3 tokens
      (`scripts/precompute_dino_tokens.py`) and a split file
      (`scripts/make_splits.py`).
    - The script uses Hydra for configuration loading.
    - The normalization selection is persisted in checkpoints; resume and
      init_weights reject metadata/config mismatches before loading state.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.slcs.training.runner import SLCSTrainingRunner
from src.utils.hydra import hydra_main


def run_training(config: DictConfig) -> None:
    """Execute SLCS training with the provided configuration."""
    runner = SLCSTrainingRunner()
    runner.run(config)


@hydra_main(
    config_path="../configs",
    config_name="train",
    version_base="1.3",
    validation_boundary="slcs.train",
)
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for SLCS training."""
    run_training(config)


if __name__ == "__main__":
    main()
