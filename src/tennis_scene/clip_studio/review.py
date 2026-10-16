"""Read-only loading of an existing Clip Studio project.

The package ``__main__`` owns the command-line interface; see README.md.
"""

from __future__ import annotations

from src.tennis_scene.clip_studio.project import ClipStudioProject
from src.tennis_scene.configuration import ClipStudioRuntimeConfig


def load_review_project(runtime: ClipStudioRuntimeConfig) -> ClipStudioProject:
    """Require a saved entry and exact raw source identity; never initialize it."""
    project = ClipStudioProject.load(
        runtime.export.projects_path,
        runtime.export.resolver,
        dataset_id=runtime.dataset_id,
        video_id=runtime.video_id,
    )
    saved = tuple((source.path, source.camera_id) for source in project.sources)
    discovered = tuple(zip(runtime.video_paths, runtime.camera_ids, strict=True))
    if saved != discovered:
        raise ValueError("保存projectのcamera/pathがraw動画と一致しません。")
    return project
