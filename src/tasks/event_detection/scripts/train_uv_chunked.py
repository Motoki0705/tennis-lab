"""Train the UV event-detection model with background-generated chunk rotation.

Usage:
    python -m src.tasks.event_detection.scripts.train_uv_chunked
    python -m src.tasks.event_detection.scripts.train_uv_chunked data.chunk.scenes_per_chunk=500
    python -m src.tasks.event_detection.scripts.train_uv_chunked data.generator_device=cuda

Notes:
    - Configuration is loaded from `src/tasks/event_detection/configs/train_uv_chunked.yaml`.
    - The script uses Hydra for configuration loading.
    - Training chunks are generated from the default BLCS generator configuration.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.tasks.event_detection.training.runner import EventDetectionTrainingRunner


@hydra.main(config_path="../configs", config_name="train_uv_chunked", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for chunked UV event-detection training."""
    runner = EventDetectionTrainingRunner()
    runner.run(cfg)


if __name__ == "__main__":
    main()