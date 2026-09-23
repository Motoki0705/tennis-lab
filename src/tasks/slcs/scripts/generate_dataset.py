"""Prepare SLCS features and video splits from an existing scene dataset."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.tasks.slcs.configuration import SLCSGenerationConfig
from src.tasks.slcs.generate_dataset.preparation import prepare_dataset
from src.tasks.slcs.model_io.factory import create_slcs_frame_token_encoder
from src.utils.hydra import hydra_main

LOGGER = logging.getLogger(__name__)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="generate_dataset",
    validation_boundary="slcs.generate_dataset",
)
def main(config: DictConfig) -> int:
    """Validate scenes, prepare RGB features, and publish reproducible splits."""
    runtime = SLCSGenerationConfig.from_config(config)
    report = prepare_dataset(
        runtime,
        encoder_factory=lambda: create_slcs_frame_token_encoder(runtime.precompute),
    )
    LOGGER.info(
        "processed=%d reused=%d failed=%d split_ready=%s split_reused=%s",
        len(report.features.processed),
        len(report.features.skipped_existing),
        len(report.features.failed),
        report.split_ready,
        report.split_reused,
    )
    for clip_id, error in report.features.failed.items():
        LOGGER.error("%s: %s", clip_id, error)
    # Hydra does not forward the task's return value when invoked as a CLI.
    if not report.ok:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
