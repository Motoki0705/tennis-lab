"""**Deprecated** – use ``python -m src.blcs.scripts.train --config-name train_multiview``.

This wrapper is kept for backward compatibility and will be removed in a
future release.
"""

from __future__ import annotations

import warnings

import hydra
from omegaconf import DictConfig

from src.blcs.scripts.train import run_training

warnings.warn(
    "train_multiview.py is deprecated. "
    "Use `python -m src.blcs.scripts.train --config-name train_multiview` instead.",
    DeprecationWarning,
    stacklevel=1,
)


@hydra.main(config_path="../configs", config_name="train_multiview", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry point
    """Hydra entry point for multi-view BLCS training."""
    run_training(config)


if __name__ == "__main__":
    main()
