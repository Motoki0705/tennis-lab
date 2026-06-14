"""Collect a web-derived tennis image dataset for DINOv3 SSL.

Usage:
    python -m src.tasks.dino_ssl.scripts.collect
    python -m src.tasks.dino_ssl.scripts.collect collector=tennis_sample

Notes:
    - Hydra loads configuration from ``src/tasks/dino_ssl/configs/collect.yaml``.
    - The script forwards the resolved config to the collection runner, which
      writes an image folder plus a ``meta.json`` manifest.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.tasks.dino_ssl.generate_dataset.runner import DinoSSLCollectionRunner


@hydra.main(version_base="1.3", config_path="../configs", config_name="collect")
def main(cfg: DictConfig) -> None:
    DinoSSLCollectionRunner().run(cfg)


if __name__ == "__main__":
    main()
