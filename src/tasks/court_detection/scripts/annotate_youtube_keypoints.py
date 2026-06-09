"""Annotate YouTube court keypoints with per-keypoint visibility.

Usage:
    python -m src.tasks.court_detection.scripts.annotate_youtube_keypoints
    python -m src.tasks.court_detection.scripts.annotate_youtube_keypoints annotate.split=val
    python -m src.tasks.court_detection.scripts.annotate_youtube_keypoints annotate.image_id=yt_000001_f00001234

Notes:
    - Hydra loads configuration from `src/tasks/court_detection/configs/annotate_youtube_keypoints.yaml`.
    - The script edits `data/court/youtube/annotations/{split}.json` in place.
    - Visibility shortcuts are `v` visible, `o` occluded, `x` out-of-frame, and delete clear.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.tasks.court_detection.annotation import (
    AnnotationSessionConfig,
    run_annotation_session,
)

F = TypeVar("F", bound=Callable[..., Any])


def hydra_main(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper for ``hydra.main``."""
    return cast(Callable[[F], F], hydra.main(*args, **kwargs))


@hydra_main(
    config_path="../configs",
    config_name="annotate_youtube_keypoints",
    version_base="1.3",
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - interactive CLI
    """Hydra entry point."""
    ann_cfg = cfg.annotate
    return run_annotation_session(
        AnnotationSessionConfig(
            root=Path(to_absolute_path(str(ann_cfg.root))).resolve(),
            split=str(ann_cfg.split),
            source_file_pattern=str(ann_cfg.source_file_pattern),
            target_file_pattern=str(ann_cfg.target_file_pattern),
            target_file_name=None if ann_cfg.target_file_name is None else str(ann_cfg.target_file_name),
            manual_adjusted_field=str(ann_cfg.manual_adjusted_field),
            image_id=None if ann_cfg.image_id is None else str(ann_cfg.image_id),
            start_index=int(ann_cfg.start_index),
            skip_completed=bool(ann_cfg.skip_completed),
            window_name=str(ann_cfg.window_name),
            max_display_width=int(ann_cfg.max_display_width),
            max_display_height=int(ann_cfg.max_display_height),
            editable_indices=tuple(int(value) for value in ann_cfg.editable_indices),
            required_indices=tuple(int(value) for value in ann_cfg.required_indices),
            drag_radius_px=float(ann_cfg.drag_radius_px),
            drag_start_threshold_px=float(ann_cfg.drag_start_threshold_px),
            annotation_format=str(ann_cfg.annotation_format),
            image_path_key=str(ann_cfg.image_path_key),
            homography_auto_fill=bool(ann_cfg.homography_auto_fill),
            start_after_last_completed=bool(ann_cfg.start_after_last_completed),
            keypoint_format=str(ann_cfg.keypoint_format),
            include_source_types=tuple(str(value) for value in ann_cfg.include_source_types),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
