"""
Launch the multi-camera clip studio GUI with one explicit project-file contract.

Usage:
    python -m src.tennis_scene.scripts.clip_studio project_path=tennis_scene/project.json

Notes:
    - All configured paths are role-relative and resolved by RuntimePathRoots.
    - Existing projects must omit video_paths and camera_ids; new projects require both.
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from src.tennis_scene.configuration import validate_clip_studio_boundary
from src.utils.hydra import hydra_main, register_boundary_validator

LOGGER = logging.getLogger(__name__)
_BOUNDARY = "tennis_scene.clip_studio"
register_boundary_validator(_BOUNDARY, validate_clip_studio_boundary)


@hydra_main(
    version_base="1.3",
    config_path="../configs",
    config_name="clip_studio",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:
    """Validate the full GUI boundary, then load or create one project."""
    from src.tennis_scene.clip_studio.app import ClipStudioApp, ClipStudioAppConfig
    from src.tennis_scene.clip_studio.export import ExportSettings
    from src.tennis_scene.clip_studio.project import ClipSource, ClipStudioProject
    from src.tennis_scene.configuration import parse_clip_studio_config

    runtime = parse_clip_studio_config(cfg)
    project_path = runtime.export.project_path
    if project_path.is_file():
        if runtime.video_paths is not None or runtime.camera_ids is not None:
            raise ValueError(
                "Existing projects forbid video_paths and camera_ids; remove both keys."
            )
        project = ClipStudioProject.load(project_path, runtime.export.resolver)
    else:
        if runtime.video_paths is None or runtime.camera_ids is None:
            raise ValueError("New projects require video_paths and camera_ids.")
        missing = [path for path in runtime.video_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"video not found: {missing[0]}")
        project = ClipStudioProject(
            recording_id=runtime.recording_id,
            sources=[
                ClipSource(path=path, camera_id=camera_id)
                for path, camera_id in zip(
                    runtime.video_paths, runtime.camera_ids, strict=True
                )
            ],
        )
        project.save(project_path, runtime.export.resolver)

    export = runtime.export
    export_settings = ExportSettings(
        output_dir=export.output_dir,
        fps=export.fps,
        width=export.width,
        height=export.height,
        crf=export.crf,
        overwrite=export.overwrite,
    )
    app = ClipStudioApp(
        ClipStudioAppConfig(
            project_path=project_path,
            resolver=runtime.export.resolver,
            export=export_settings,
            canvas_width=runtime.gui.canvas_width,
            tile_width=runtime.gui.tile_width,
            cache_frames=runtime.gui.cache_frames,
            seek_grab_threshold=runtime.gui.seek_grab_threshold,
            window_name=runtime.gui.window_name,
            audio_sample_rate=runtime.audio_sync.sample_rate,
            audio_envelope_rate=runtime.audio_sync.envelope_rate,
            audio_max_seconds=runtime.audio_sync.max_seconds,
            zoom_step=runtime.gui.zoom_step,
        ),
        project,
    )
    app.run()
    return 0


if __name__ == "__main__":
    main()
