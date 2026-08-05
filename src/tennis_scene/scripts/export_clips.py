"""
Headlessly append the clips of a clip studio project to a structured dataset
as synchronized per-camera videos plus clip and dataset manifests, matching
the tennis_scene pipeline contract.

Usage:
    python -m src.tennis_scene.scripts.export_clips project_path=tennis_scene/project.json

Notes:
    - The project and output directory are explicit role-relative paths.
    - Configuration is loaded from `src/tennis_scene/configs/export_clips.yaml`.
    - Exported videos are re-probed and validated against the plan; a
      contract violation aborts with an error instead of writing a bad clip.
    - Clips are namespaced as `clips/<recording_id>/<clip_name>` so later
      recording sessions can be appended without `clip_000` collisions.
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.tennis_scene.configuration import validate_export_clips_boundary
from src.utils.hydra import hydra_main, register_boundary_validator

LOGGER = logging.getLogger(__name__)
_BOUNDARY = "tennis_scene.export_clips"
register_boundary_validator(_BOUNDARY, validate_export_clips_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="export_clips",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:
    """Export clips from a clip studio project."""
    from src.tennis_scene.clip_studio.export import ExportSettings, export_clips
    from src.tennis_scene.clip_studio.project import ClipStudioProject
    from src.tennis_scene.configuration import parse_export_config

    runtime = parse_export_config(cfg)
    if not runtime.project_path.is_file():
        raise FileNotFoundError(f"project not found: {runtime.project_path}")
    settings = ExportSettings(
        output_dir=runtime.output_dir,
        fps=runtime.fps,
        width=runtime.width,
        height=runtime.height,
        crf=runtime.crf,
        overwrite=runtime.overwrite,
    )

    try:
        project = ClipStudioProject.load(runtime.project_path, runtime.resolver)
        results = export_clips(project, settings, clip_names=runtime.clip_names)
    except (KeyError, ValueError, RuntimeError) as error:
        LOGGER.error(str(error))
        return 1

    LOGGER.info("=" * 60)
    LOGGER.info(f"Exported {len(results)} clip(s):")
    for result in results:
        LOGGER.info(f"  {result.clip_dir}")
    dataset_fragment = runtime.output_dir.relative_to(runtime.roots.artifact_root)
    LOGGER.info(
        "Generate pseudo annotations: python -m "
        "src.tennis_scene.scripts.generate_dataset "
        f"dataset_directory={dataset_fragment}"
    )
    LOGGER.info("=" * 60)
    return 0


if __name__ == "__main__":
    main()
