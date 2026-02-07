"""**Deprecated** – use ``python -m src.wasb.scripts.train``.

This wrapper is kept for backward compatibility and will be removed in a
future release.
"""

from __future__ import annotations

import warnings

import hydra
from omegaconf import DictConfig

warnings.warn(
    "src.wasb.scripts.train.ball_detection is deprecated. "
    "Use `python -m src.wasb.scripts.train` instead.",
    DeprecationWarning,
    stacklevel=1,
)


@hydra.main(config_path="../../configs", config_name="train_ball_detection", version_base="1.3")
def main(config: DictConfig) -> None:
    """Hydra entry point (deprecated)."""
    from src.wasb.training.runner import WASBTrainingRunner

    runner = WASBTrainingRunner()
    runner.run(config)


if __name__ == "__main__":
    main()
