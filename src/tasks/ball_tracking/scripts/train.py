"""
Train the synthetic multi-ball track-query baseline.

Usage:
    .venv/bin/python -m src.tasks.ball_tracking.scripts.train model.role_rope_enabled=true

Notes:
    - Configuration is loaded from ``../configs/train.yaml`` with Hydra.
    - Test-split predictions are saved after a successful non-fast-dev run.
"""

from typing import Any

from src.tasks.ball_tracking.training import BallTrackingTrainingRunner
from src.utils.hydra import hydra_main


@hydra_main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: Any) -> None:
    BallTrackingTrainingRunner().run(config)


if __name__ == "__main__":
    main()
