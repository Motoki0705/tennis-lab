"""
Train the synthetic multi-person track-query baseline.

Usage:
    .venv/bin/python -m src.tasks.player_tracking.scripts.train model.role_rope_enabled=true

Notes:
    - Configuration is loaded from ``../configs/train.yaml`` with Hydra.
    - Track smoothness is disabled in the default loss configuration.
"""

from typing import Any

from src.tasks.player_tracking.training import PlayerTrackingTrainingRunner
from src.utils.hydra import hydra_main


@hydra_main(config_path="../configs", config_name="train", version_base="1.3")
def main(config: Any) -> None:
    PlayerTrackingTrainingRunner().run(config)


if __name__ == "__main__":
    main()
