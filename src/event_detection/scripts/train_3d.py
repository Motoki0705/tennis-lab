"""**Deprecated** \u2013 use ``python -m src.event_detection.scripts.train --config-name train_3d``.

This wrapper is kept for backward compatibility and will be removed in a
future release.
"""

from __future__ import annotations

import warnings

import hydra
from omegaconf import DictConfig

from src.event_detection.scripts.train import run_training

warnings.warn(
    "train_3d.py is deprecated. "
    "Use `python -m src.event_detection.scripts.train --config-name train_3d` instead.",
    DeprecationWarning,
    stacklevel=1,
)


@hydra.main(config_path="../configs", config_name="train_3d", version_base="1.3")
def main(cfg: DictConfig) -> None:  # pragma: no cover - CLI entry point
    run_training(cfg)


if __name__ == "__main__":
    main()
