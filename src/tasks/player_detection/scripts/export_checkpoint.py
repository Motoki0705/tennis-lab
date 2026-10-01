"""Export a fine-tuned Lightning checkpoint as a DinoPersonDetector checkpoint.

Usage::

    .venv/bin/python -m src.tasks.player_detection.scripts.export_checkpoint \
        export.lightning_checkpoint=player_detection/train/.../checkpoints/player-dino-epoch=07.ckpt \
        export.destination=player_detection/dino_swinl_4scale_player_ft_<dataset>_<run>.pth

``lightning_checkpoint`` is output-root-relative, ``destination`` is
checkpoint-root-relative (``ckpt/``); existing destinations are never replaced.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.player_detection.configuration import (
    ExportConfig,
    validate_export_boundary,
)
from src.tasks.player_detection.model_export import export_player_checkpoint
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "player_detection.export_checkpoint"
register_boundary_validator(_BOUNDARY, validate_export_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="export_checkpoint",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> None:
    print(f"Exported {export_player_checkpoint(ExportConfig.from_config(cfg))}")


if __name__ == "__main__":
    main()
