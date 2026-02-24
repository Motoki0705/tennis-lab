"""Training script for WASB tennis models.

Run with Hydra-style overrides, for example:

```
uv run python -m src.tasks.wasb.scripts.train.ball_detection training.max_epochs=50 data.batch_size=32
```

Config entry point: `src/tasks/wasb/configs/train_ball_detection.yaml`

This script uses WASBTrainingRunner which extends BaseTrainingRunner with
WASB-specific behavior for datamodule selection, model construction,
curriculum learning callbacks, and dry-run visualizations.
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.tasks.wasb.training.runner import WASBTrainingRunner


def _setup_logging(config: DictConfig) -> None:
    """Initialize Python logging from the config.

    Expects a ``logging`` section in the root config with keys:

        level: str   (e.g. "INFO", "DEBUG", ...)
        fmt: str     (logging format string)
        datefmt: str (date format string)
    """
    log_cfg = getattr(config, "logging", None)
    if log_cfg is None:
        return

    level_name = str(getattr(log_cfg, "level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = getattr(log_cfg, "fmt", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    datefmt = getattr(log_cfg, "datefmt", "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


@hydra.main(config_path="../../configs", config_name="train_ball_detection", version_base="1.3")
def main(config: DictConfig) -> None:
    """Hydra entry point."""
    _setup_logging(config)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))

    data_name = str(config.data.get("name", "ball_detection")).lower()
    if data_name == "patch_embeddings":
        logging.getLogger(__name__).warning(
            "Using patch_embeddings data: ensure model/handlers accept frames shaped [B, T, N, C]."
        )

    runner = WASBTrainingRunner()
    runner.run(config)


if __name__ == "__main__":
    main()
