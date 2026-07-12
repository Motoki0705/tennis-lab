"""
Launch the multi-camera clip studio GUI: synchronize long unsynchronized
match videos on a shared timeline, cut rally clips, and export them in the
format `run_pipeline.py` expects (equal fps / frame count / resolution).

Usage:
    python -m src.tennis_scene.scripts.clip_studio match_id=match1
    python -m src.tennis_scene.scripts.clip_studio project_path=/custom/project.json recording_id=match1 video_paths='[/data/cam0.mp4,/data/cam1.mp4]'

Notes:
    - With match_id, source videos default to data/tennis_multivew/raw/<match_id>/
      cam{0,1,2}.mp4 and processed data goes below processed/<match_id>.
    - Explicit project_path / recording_id / video_paths override the standard
      convention when creating a project outside that layout.
    - Configuration is loaded from `src/tennis_scene/configs/clip_studio.yaml`.
    - Key bindings are shown in-app with `h`; the project is autosaved on quit.
    - Export (`e`/`E` in the GUI, or the export_clips script) writes one
      directory per clip with per-camera mp4s plus a clip.json manifest.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig

from src.utils.hydra import hydra_main

if TYPE_CHECKING:
    from src.tennis_scene.clip_studio.export import ExportSettings
    from src.tennis_scene.clip_studio.paths import StandardClipStudioPaths
    from src.tennis_scene.clip_studio.project import ClipStudioProject

LOGGER = logging.getLogger(__name__)


def _standard_paths(cfg: DictConfig) -> StandardClipStudioPaths | None:
    from src.tennis_scene.clip_studio.paths import standard_clip_studio_paths

    raw_match_id = cfg.get("match_id")
    if raw_match_id is None:
        return None
    data_root = Path(to_absolute_path(str(cfg.data_root)))
    return standard_clip_studio_paths(data_root, str(raw_match_id))


def _resolve_project_path(
    cfg: DictConfig, standard_paths: StandardClipStudioPaths | None
) -> Path:
    raw_project_path = cfg.get("project_path")
    if raw_project_path is not None:
        return Path(to_absolute_path(str(raw_project_path)))
    if standard_paths is None:
        raise ValueError("match_id or project_path is required")
    return standard_paths.project_path


def _resolve_export_settings(cfg: DictConfig, project_path: Path) -> ExportSettings:
    from src.tennis_scene.clip_studio.export import ExportSettings

    raw_output_dir = cfg.export.get("output_dir")
    output_dir = (
        Path(to_absolute_path(str(raw_output_dir)))
        if raw_output_dir is not None
        else project_path.parent / "dataset"
    )
    return ExportSettings(
        output_dir=output_dir,
        fps=None if cfg.export.get("fps") is None else float(cfg.export.fps),
        width=None if cfg.export.get("width") is None else int(cfg.export.width),
        height=None if cfg.export.get("height") is None else int(cfg.export.height),
        crf=int(cfg.export.get("crf", 17)),
        overwrite=bool(cfg.export.get("overwrite", False)),
    )


def _bootstrap_project(
    cfg: DictConfig,
    project_path: Path,
    standard_paths: StandardClipStudioPaths | None,
) -> ClipStudioProject:
    from src.tennis_scene.clip_studio.project import ClipSource, ClipStudioProject

    raw_video_paths = cfg.get("video_paths")
    if project_path.exists():
        if raw_video_paths is not None:
            raise ValueError(
                f"project already exists at {project_path}; omit video_paths "
                "when reopening (delete the project file to start over)"
            )
        LOGGER.info(f"Loading project: {project_path}")
        return ClipStudioProject.load(project_path)

    if raw_video_paths is None and standard_paths is not None:
        video_paths = list(standard_paths.video_paths)
    elif not isinstance(raw_video_paths, (list, tuple, ListConfig)) or not raw_video_paths:
        raise ValueError(
            "match_id or a non-empty video_paths list is required when creating "
            "a new project"
        )
    else:
        video_paths = [Path(to_absolute_path(str(path))) for path in raw_video_paths]
    missing = [path for path in video_paths if not path.exists()]
    if missing:
        raise ValueError(f"video not found: {missing}")

    raw_camera_ids = cfg.get("camera_ids")
    if raw_camera_ids is None:
        camera_ids = [f"cam{index}" for index in range(len(video_paths))]
    else:
        camera_ids = [str(camera_id) for camera_id in raw_camera_ids]
        if len(camera_ids) != len(video_paths):
            raise ValueError("camera_ids length must match video_paths length")

    raw_recording_id = cfg.get("recording_id")
    if raw_recording_id is None:
        raw_recording_id = cfg.get("match_id")
    if raw_recording_id is None:
        raise ValueError(
            "match_id or recording_id is required when creating a new project"
        )

    project = ClipStudioProject(
        recording_id=str(raw_recording_id),
        sources=[
            ClipSource(path=path, camera_id=camera_id)
            for path, camera_id in zip(video_paths, camera_ids, strict=True)
        ]
    )
    project.save(project_path)
    LOGGER.info(f"Created new project: {project_path}")
    return project


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="clip_studio",
)
def main(cfg: DictConfig) -> int:
    """Launch the clip studio GUI."""
    from src.tennis_scene.clip_studio.app import ClipStudioApp, ClipStudioAppConfig

    try:
        standard_paths = _standard_paths(cfg)
        project_path = _resolve_project_path(cfg, standard_paths)
        project = _bootstrap_project(cfg, project_path, standard_paths)
        app_config = ClipStudioAppConfig(
            project_path=project_path,
            export=_resolve_export_settings(cfg, project_path),
            canvas_width=int(cfg.gui.canvas_width),
            tile_width=int(cfg.gui.tile_width),
            cache_frames=int(cfg.gui.cache_frames),
            seek_grab_threshold=int(cfg.gui.seek_grab_threshold),
            window_name=str(cfg.gui.window_name),
            audio_sample_rate=int(cfg.audio_sync.sample_rate),
            audio_envelope_rate=float(cfg.audio_sync.envelope_rate),
            audio_max_seconds=(
                None
                if cfg.audio_sync.get("max_seconds") is None
                else float(cfg.audio_sync.max_seconds)
            ),
            zoom_step=float(cfg.gui.zoom_step),
        )
        app = ClipStudioApp(app_config, project)
    except ValueError as error:
        LOGGER.error(str(error))
        return 1

    LOGGER.info(f"Cameras: {[source.camera_id for source in project.sources]}")
    LOGGER.info("Press h in the window for key bindings.")
    app.run()

    LOGGER.info(
        "Run the pipeline on an exported clip with:\n"
        "  python -m src.tennis_scene.scripts.run_pipeline "
        "video_paths='[<clip_dir>/cam0.mp4,<clip_dir>/cam1.mp4]'"
    )
    return 0


if __name__ == "__main__":
    main()
