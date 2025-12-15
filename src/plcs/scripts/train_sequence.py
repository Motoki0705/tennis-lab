"""Train the sequence PLCS model (Hydra-based).

Example commands:
    `uv run python -m src.plcs.scripts.train_sequence`

Config entry point: `src/plcs/configs/train_sequence.yaml`
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.plcs.scripts.train import run_training


@hydra.main(config_path="../configs", config_name="train_sequence", version_base="1.3")
def main(config: DictConfig) -> None:  # pragma: no cover - CLI entry
    """Hydra entry point for sequence-based PLCS training."""
    run_training(config)


if __name__ == "__main__":
    main()
