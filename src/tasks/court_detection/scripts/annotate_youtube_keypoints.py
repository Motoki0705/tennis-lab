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

from typing import cast

from omegaconf import DictConfig

from src.tasks.base.configuration import require_config_mapping, require_config_value
from src.tasks.court_detection.configuration import validate_paths_boundary
from src.tasks.court_detection.generate_dataset import (
    AnnotationSessionConfig,
    run_annotation_session,
)
from src.utils.configuration import PathRole
from src.utils.hydra import hydra_main, register_boundary_validator

_BOUNDARY = "court_detection.annotate_youtube_keypoints"


def _runtime(cfg: DictConfig) -> AnnotationSessionConfig:
    root, resolver = validate_paths_boundary(cfg, expected_sections={"annotate"})
    ann_cfg = require_config_mapping(root, "annotate", path="configuration")
    expected = {
        "root",
        "split",
        "source_file_pattern",
        "target_file_pattern",
        "target_file_name",
        "manual_adjusted_field",
        "annotation_format",
        "image_path_key",
        "homography_auto_fill",
        "keypoint_format",
        "include_source_types",
        "image_id",
        "start_index",
        "skip_completed",
        "start_after_last_completed",
        "window_name",
        "max_display_width",
        "max_display_height",
        "editable_indices",
        "required_indices",
        "drag_radius_px",
        "drag_start_threshold_px",
    }
    if set(ann_cfg) != expected:
        raise ValueError(f"annotate requires exactly {sorted(expected)}.")
    for key in (
        "root",
        "split",
        "source_file_pattern",
        "target_file_pattern",
        "manual_adjusted_field",
        "annotation_format",
        "image_path_key",
        "keypoint_format",
        "window_name",
    ):
        require_config_value(ann_cfg, key, str, path="annotate")
    for key in ("target_file_name", "image_id"):
        require_config_value(ann_cfg, key, (str, type(None)), path="annotate")
    for key in ("start_index", "max_display_width", "max_display_height"):
        require_config_value(ann_cfg, key, int, path="annotate")
    for key in ("skip_completed", "homography_auto_fill", "start_after_last_completed"):
        require_config_value(ann_cfg, key, bool, path="annotate")
    for key in ("editable_indices", "required_indices", "include_source_types"):
        require_config_value(ann_cfg, key, list, path="annotate")
    for key in ("drag_radius_px", "drag_start_threshold_px"):
        require_config_value(ann_cfg, key, (float, int), path="annotate")
    if any(
        type(value) is not int
        for key in ("editable_indices", "required_indices")
        for value in cast("list[object]", ann_cfg[key])
    ):
        raise TypeError("annotate editable/required indices must contain integers.")
    if any(
        type(value) is not str
        for value in cast("list[object]", ann_cfg["include_source_types"])
    ):
        raise TypeError("annotate.include_source_types must contain strings.")
    root_raw = cast("str", ann_cfg["root"])
    split = cast("str", ann_cfg["split"])
    if split not in {"train", "val"}:
        raise ValueError("annotate.split must be train or val.")
    annotation_format = cast("str", ann_cfg["annotation_format"])
    if annotation_format != "named_keypoints":
        raise ValueError("annotate.annotation_format must be 'named_keypoints'.")
    keypoint_format = cast("str", ann_cfg["keypoint_format"])
    if keypoint_format not in {"kp15", "kp20"}:
        raise ValueError("annotate.keypoint_format must be kp15 or kp20.")
    editable = tuple(cast("list[int]", ann_cfg["editable_indices"]))
    required = tuple(cast("list[int]", ann_cfg["required_indices"]))
    if (
        not editable
        or len(set(editable)) != len(editable)
        or any(index < 0 or index >= 20 for index in editable)
    ):
        raise ValueError("annotate.editable_indices must be unique indices in [0, 19].")
    if len(set(required)) != len(required) or not set(required).issubset(editable):
        raise ValueError("annotate.required_indices must be a unique editable subset.")
    start_index = cast("int", ann_cfg["start_index"])
    display_width = cast("int", ann_cfg["max_display_width"])
    display_height = cast("int", ann_cfg["max_display_height"])
    drag_radius = float(cast("float | int", ann_cfg["drag_radius_px"]))
    drag_threshold = float(cast("float | int", ann_cfg["drag_start_threshold_px"]))
    if start_index < 0 or display_width <= 0 or display_height <= 0:
        raise ValueError("annotate index/display dimensions are invalid.")
    if drag_radius <= 0 or drag_threshold <= 0:
        raise ValueError("annotate drag radii must be positive.")
    source_pattern = cast("str", ann_cfg["source_file_pattern"])
    target_pattern = cast("str", ann_cfg["target_file_pattern"])
    try:
        source_relative = source_pattern.format(split=split)
        target_relative = target_pattern.format(split=split)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(
            "annotate file patterns may only use the {split} field."
        ) from exc
    source_path = resolver.resolve(PathRole.DATA, root_raw, source_relative)
    target_path = resolver.resolve(PathRole.DATA, root_raw, target_relative)
    target_file_name = cast("str | None", ann_cfg["target_file_name"])
    if target_file_name is not None:
        if not target_file_name:
            raise ValueError("annotate.target_file_name must be null or non-empty.")
        target_path = resolver.resolve(PathRole.DATA, root_raw, target_file_name)
    return AnnotationSessionConfig(
        root=resolver.resolve(PathRole.DATA, root_raw),
        root_fragment=root_raw,
        resolver=resolver,
        source_path=source_path,
        target_path=target_path,
        split=split,
        manual_adjusted_field=str(ann_cfg["manual_adjusted_field"]),
        image_id=None if ann_cfg["image_id"] is None else str(ann_cfg["image_id"]),
        start_index=start_index,
        skip_completed=bool(ann_cfg["skip_completed"]),
        window_name=str(ann_cfg["window_name"]),
        max_display_width=display_width,
        max_display_height=display_height,
        editable_indices=editable,
        required_indices=required,
        drag_radius_px=drag_radius,
        drag_start_threshold_px=drag_threshold,
        annotation_format=annotation_format,
        image_path_key=str(ann_cfg["image_path_key"]),
        homography_auto_fill=bool(ann_cfg["homography_auto_fill"]),
        start_after_last_completed=bool(ann_cfg["start_after_last_completed"]),
        keypoint_format=keypoint_format,
        include_source_types=tuple(
            str(value)
            for value in cast("list[object]", ann_cfg["include_source_types"])
        ),
    )


def _validate_boundary(cfg: DictConfig) -> None:
    _runtime(cfg)


register_boundary_validator(_BOUNDARY, _validate_boundary)


@hydra_main(
    config_path="../configs",
    config_name="annotate_youtube_keypoints",
    version_base="1.3",
    validation_boundary=_BOUNDARY,
)
def main(cfg: DictConfig) -> int:  # pragma: no cover - interactive CLI
    """Hydra entry point."""
    status: int = run_annotation_session(_runtime(cfg))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
