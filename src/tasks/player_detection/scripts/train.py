"""Fine-tune DINO 4-scale Swin-L as a tennis-player detector.

Usage (from a worktree, point roots at the main checkout)::

    .venv/bin/python -m src.tasks.player_detection.scripts.train \
        paths.data_root=/abs/tennis-lab/data paths.output_root=/abs/tennis-lab/outputs \
        paths.checkpoint_root=/abs/tennis-lab/ckpt paths.external_asset_root=/abs/tennis-lab/third_party

``run.dry_run=true`` only loads a train batch (DINO training requires CUDA).
Export the selected checkpoint with ``scripts.export_checkpoint``.
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.tasks.player_detection.configuration import validate_train_boundary
from src.tasks.player_detection.training.runner import PlayerDetectionTrainingRunner
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "player_detection.train"
register_boundary_validator(_BOUNDARY, validate_train_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="train",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> None:
    PlayerDetectionTrainingRunner().run(cfg)


if __name__ == "__main__":
    main()
