"""Sequence PLCS training entrypoint using Hydra."""

from __future__ import annotations

import hydra

from src.plcs.configs import PLCSSequenceConfig, register_configs
from src.plcs.scripts.train import run_training

register_configs()


@hydra.main(version_base=None, config_name="plcs_sequence")
def main(config: PLCSSequenceConfig) -> None:  # pragma: no cover - CLI entry
    """Hydra entry point for sequence-based PLCS training."""

    run_training(config)


if __name__ == "__main__":
    main()
