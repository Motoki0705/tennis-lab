"""Generate stratified human-readable GIF samples for BLCS datasets.

Usage:
    .venv/bin/python -m src.tasks.blcs.scripts.generate_dataset_samples
    .venv/bin/python -m src.tasks.blcs.scripts.generate_dataset_samples paths.project_root=/path/to/tennis-lab
    .venv/bin/python -m src.tasks.blcs.scripts.generate_dataset_samples samples.overwrite=true

Notes:
    - Hydra loads configuration from `src/tasks/blcs/configs/generate_dataset_samples.yaml`.
    - Each configured dataset receives `samples/*.gif` and `samples/manifest.json`.
    - Selection is deterministic and covers a 3x3 semantic/duration grid.
    - Camera visibility is deliberately varied; rendered timelines contain at most 120 frames by default.
"""

from __future__ import annotations

import sys

import matplotlib
from omegaconf import DictConfig

from src.tasks.base.generate_dataset.dataset_samples import DatasetSamplesConfig
from src.tasks.blcs.generate_dataset.samples import generate_blcs_dataset_samples
from src.utils.hydra import hydra_main


@hydra_main(
    config_path="../configs",
    config_name="generate_dataset_samples",
    version_base="1.3",
    validation_boundary="blcs.generate_dataset_samples",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - CLI entry point
    """Validate configuration and generate every configured BLCS sample set."""
    matplotlib.use("Agg")
    config = DatasetSamplesConfig.from_config(cfg, task="blcs")
    manifests = generate_blcs_dataset_samples(config)
    for manifest in manifests:
        print(f"Generated BLCS dataset samples: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
