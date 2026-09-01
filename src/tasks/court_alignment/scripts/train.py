"""Train a ground-UV KP14 multi-court alignment model.

Usage:
    python -m src.tasks.court_alignment.scripts.train
    python -m src.tasks.court_alignment.scripts.train data.sigma_px=0.75
    python -m src.tasks.court_alignment.scripts.train --config-name smoke

Notes:
    - Hydra loads configuration from ``src/tasks/court_alignment/configs``.
    - Local GPU jobs must be submitted through the shared training queue.
    - Test-after-fit writes the standard reproducible prediction bundle.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.court_alignment import configuration as _configuration  # noqa: F401
from src.tasks.court_alignment.training.runner import CourtAlignmentTrainingRunner
from src.utils.hydra import hydra_main


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="train",
    validation_boundary="court_alignment.train",
)
def main(cfg: DictConfig) -> None:
    """Run config-driven court-alignment training."""
    CourtAlignmentTrainingRunner().run(cfg)


if __name__ == "__main__":
    main()
