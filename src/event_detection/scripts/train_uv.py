"""Train a UV-based event detection model using Hydra-managed configuration.

Example commands:
    `uv run python -m src.event_detection.scripts.train_uv`
    `uv run python -m src.event_detection.scripts.train_uv run.dry_run=true`
    `uv run python -m src.event_detection.scripts.train_uv data.scene_dir=data/blcs run.dry_run=false`

Config entry point: `src/event_detection/configs/train_uv.yaml`
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.event_detection.training.runner import EventDetectionTrainingRunner


@hydra.main(config_path="../configs", config_name="train_uv", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI entry point
    runner = EventDetectionTrainingRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()
