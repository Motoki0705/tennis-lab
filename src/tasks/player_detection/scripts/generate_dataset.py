"""Build the player-detection frame store from processed chat annotations.

Usage (from a worktree, point the roots at the main checkout)::

    .venv/bin/python -m src.tasks.player_detection.scripts.generate_dataset \
        paths.output_root=/abs/tennis-lab/outputs paths.data_root=/abs/tennis-lab/data

Only frames that list at least one player are stored. The dataset version
directory must not exist; the build publishes it atomically.
"""

from __future__ import annotations

import json

from omegaconf import DictConfig

from src.tasks.player_detection.configuration import (
    GenerateDatasetConfig,
    validate_generate_boundary,
)
from src.tasks.player_detection.generate_dataset.builder import build_dataset
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "player_detection.generate_dataset"
register_boundary_validator(_BOUNDARY, validate_generate_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="generate_dataset",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> None:
    destination = build_dataset(GenerateDatasetConfig.from_config(cfg))
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    print(json.dumps(metadata["counts"], indent=2))
    print(f"Published {destination}")


if __name__ == "__main__":
    main()
