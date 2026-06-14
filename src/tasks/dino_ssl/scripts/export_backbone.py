"""Export a tennis-adapted DINOv3 backbone from an SSL training checkpoint.

Usage:
    python -m src.tasks.dino_ssl.scripts.export_backbone \
        checkpoint=outputs/dino_ssl/<run>/logs/version_0/checkpoints/last.ckpt \
        train_config=outputs/dino_ssl/<run>/config.yaml \
        output=outputs/dino_ssl/exported/dinov3_vitb16_tennis.pth

Notes:
    - Hydra loads configuration from ``src/tasks/dino_ssl/configs/export.yaml``.
    - ``checkpoint`` and ``train_config`` are mandatory; override them on the CLI.
    - The exported file merges the LoRA adapters into the base ViT and can be
      passed to downstream tasks via their DINOv3 ``checkpoint_path`` setting.
"""

from __future__ import annotations

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.tasks.dino_ssl.models.export import export_backbone


@hydra.main(version_base="1.3", config_path="../configs", config_name="export")
def main(cfg: DictConfig) -> None:
    export_backbone(
        checkpoint_path=to_absolute_path(str(cfg.checkpoint)),
        config_path=to_absolute_path(str(cfg.train_config)),
        output_path=to_absolute_path(str(cfg.output)),
    )


if __name__ == "__main__":
    main()
