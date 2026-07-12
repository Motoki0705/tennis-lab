"""
Headlessly append the clips of a clip studio project to a structured dataset
as synchronized per-camera videos plus clip and dataset manifests, matching
the tennis_scene pipeline contract.

Usage:
    python -m src.tennis_scene.scripts.export_clips match_id=match1
    python -m src.tennis_scene.scripts.export_clips match_id=match1 clip_names='[clip_000]' export.overwrite=true

Notes:
    - By default, match_id resolves the project and dataset below
      data/tennis_multivew/processed/<match_id>.
    - Configuration is loaded from `src/tennis_scene/configs/export_clips.yaml`.
    - Exported videos are re-probed and validated against the plan; a
      contract violation aborts with an error instead of writing a bad clip.
    - Clips are namespaced as `clips/<recording_id>/<clip_name>` so later
      recording sessions can be appended without `clip_000` collisions.
"""

from __future__ import annotations

import logging
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig

from src.utils.hydra import hydra_main

LOGGER = logging.getLogger(__name__)


def _resolve_project_path(cfg: DictConfig) -> Path:
    raw_project_path = cfg.get("project_path")
    if raw_project_path is not None:
        return Path(to_absolute_path(str(raw_project_path)))

    raw_match_id = cfg.get("match_id")
    if raw_match_id is None:
        raise ValueError("match_id or project_path is required")

    from src.tennis_scene.clip_studio.paths import standard_clip_studio_paths

    data_root = Path(to_absolute_path(str(cfg.data_root)))
    return standard_clip_studio_paths(data_root, str(raw_match_id)).project_path


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="export_clips",
)
def main(cfg: DictConfig) -> int:
    """Export clips from a clip studio project."""
    from src.tennis_scene.clip_studio.export import ExportSettings, export_clips
    from src.tennis_scene.clip_studio.project import ClipStudioProject

    try:
        project_path = _resolve_project_path(cfg)
    except ValueError as error:
        LOGGER.error(str(error))
        return 1
    if not project_path.exists():
        LOGGER.error(f"project not found: {project_path}")
        return 1

    raw_clip_names = cfg.get("clip_names")
    clip_names = None
    if raw_clip_names is not None:
        if not isinstance(raw_clip_names, (list, tuple, ListConfig)):
            LOGGER.error("clip_names must be null or a list")
            return 1
        clip_names = [str(name) for name in raw_clip_names]

    raw_output_dir = cfg.export.get("output_dir")
    output_dir = (
        Path(to_absolute_path(str(raw_output_dir)))
        if raw_output_dir is not None
        else project_path.parent / "dataset"
    )
    settings = ExportSettings(
        output_dir=output_dir,
        fps=None if cfg.export.get("fps") is None else float(cfg.export.fps),
        width=None if cfg.export.get("width") is None else int(cfg.export.width),
        height=None if cfg.export.get("height") is None else int(cfg.export.height),
        crf=int(cfg.export.get("crf", 17)),
        overwrite=bool(cfg.export.get("overwrite", False)),
    )

    try:
        project = ClipStudioProject.load(project_path)
        results = export_clips(project, settings, clip_names=clip_names)
    except (KeyError, ValueError, RuntimeError) as error:
        LOGGER.error(str(error))
        return 1

    LOGGER.info("=" * 60)
    LOGGER.info(f"Exported {len(results)} clip(s):")
    for result in results:
        video_list = ",".join(str(path) for path in result.video_paths)
        LOGGER.info(f"  {result.clip_dir}")
        LOGGER.info(
            "    run: python -m src.tennis_scene.scripts.run_pipeline "
            f"video_paths='[{video_list}]'"
        )
    LOGGER.info("=" * 60)
    return 0


if __name__ == "__main__":
    main()
