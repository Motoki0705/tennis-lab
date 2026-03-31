"""Train the 3D event detection model with Hydra-managed configuration.

Usage:
    python -m src.tasks.event_detection.scripts.train_3d
    python -m src.tasks.event_detection.scripts.train_3d run.dry_run=true
    python -m src.tasks.event_detection.scripts.train_3d data.scene_dir=data/blcs run.dry_run=false

Notes:
    - Configuration is loaded from `src/tasks/event_detection/configs/train_3d.yaml`.
    - The script uses Hydra for configuration loading.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.tasks.event_detection.training.runner import EventDetectionTrainingRunner


@hydra.main(config_path="../configs", config_name="train_3d", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI entry point
    runner = EventDetectionTrainingRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()
