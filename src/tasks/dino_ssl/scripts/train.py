"""Fine-tune DINOv3 on tennis imagery with LoRA self-distillation.

Usage:
    python -m src.tasks.dino_ssl.scripts.train
    python -m src.tasks.dino_ssl.scripts.train run.dry_run=true
    python -m src.tasks.dino_ssl.scripts.train training.trainer.max_epochs=1

Notes:
    - Hydra loads configuration from ``src/tasks/dino_ssl/configs/train.yaml``.
    - The dataset must be collected first via ``scripts.collect``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.tasks.dino_ssl.training.runner import DinoSSLTrainingRunner


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    DinoSSLTrainingRunner().run(cfg)


if __name__ == "__main__":
    main()
