"""Evaluate a ground-UV KP14 alignment checkpoint on its fixed test split.

Usage:
    python -m src.tasks.court_alignment.scripts.evaluate \
        evaluation.checkpoint_path=/absolute/path/to/model.ckpt
    python -m src.tasks.court_alignment.scripts.evaluate --cfg job

Notes:
    - Hydra loads configuration from ``src/tasks/court_alignment/configs``.
    - Evaluation uses the standard test prediction and metric bundle.
    - ``evaluation.checkpoint_path`` is required for execution but not for config display.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.court_alignment import configuration as _configuration  # noqa: F401
from src.tasks.court_alignment.training.runner import CourtAlignmentTrainingRunner
from src.utils.hydra import hydra_main


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="evaluate",
    validation_boundary="court_alignment.evaluate",
)
def main(cfg: DictConfig) -> None:
    """Run checkpoint evaluation on the configured test split."""
    CourtAlignmentTrainingRunner().evaluate(cfg)


if __name__ == "__main__":
    main()
