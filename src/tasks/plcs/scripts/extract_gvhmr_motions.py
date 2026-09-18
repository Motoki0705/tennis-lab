"""Extract a configured dataset's foreground motions using Hydra composition."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.tasks.plcs.motion.extraction_config import ExtractionConfig
from src.tasks.plcs.motion.gvhmr_extraction import GvhmrMotionExtractor
from src.tasks.plcs.motion.reproducibility import (
    content_digest,
    describe_run,
    publish_run,
    seed_motion,
)
from src.utils.hydra import hydra_main

LOGGER = logging.getLogger(__name__)


def run_extraction(config: DictConfig) -> dict[str, int]:
    """Reusable Python API using the same validated config as the CLI."""
    settings = ExtractionConfig.from_config(config)
    run = settings.resolved["run"]
    seed_motion(run["seed"], "model-initialization", run["deterministic"])
    description = describe_run(
        config=settings.resolved,
        model_runtime=settings.model_runtime,
        selection=settings.selection.to_dict(),
        dataset_root=settings.dataset_root,
    )
    publish_run(settings.output_root, description)
    extractor = GvhmrMotionExtractor(
        dataset_root=settings.dataset_root,
        output_root=settings.output_root,
        selection=settings.selection,
        selection_digest=content_digest(settings.selection.to_dict()),
        model_runtime=settings.model_runtime,
        max_frames=run["max_frames"],
        overwrite=run["overwrite"],
        write_preview=run["write_preview"],
        reproducibility_digest=description["sha256"],
        seed=run["seed"],
        deterministic=run["deterministic"],
    )
    return extractor.run(
        clip_ids=None if run["clip_ids"] is None else tuple(run["clip_ids"]),
        camera_ids=None if run["camera_ids"] is None else tuple(run["camera_ids"]),
        max_clips=run["max_clips"],
    )


@hydra_main(
    config_path="../configs",
    config_name="extract_gvhmr_motions",
    version_base="1.3",
    validation_boundary="plcs.extract_gvhmr_motions",
)
def main(config: DictConfig) -> None:
    LOGGER.info("GVHMR extraction complete: %s", run_extraction(config))


if __name__ == "__main__":
    main()
