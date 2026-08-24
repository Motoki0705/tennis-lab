"""OpenCV overlays for canonical Court, BLCS, and PLCS labels."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.components.labels import (
    SEMANTIC_CLASS_NAMES,
    SEMANTIC_CLASS_NAMES_V2,
)
from src.synthetic_data_generation.dataset.court.schema import CourtDatasetSchemaVersion
from src.synthetic_data_generation.dataset.runtime import LogicalRenderSample
from src.synthetic_data_generation.visualization.sources import (
    BLCSSourceFrame,
    CourtSourceFrame,
    PLCSSourceFrame,
)
from src.utils.schema.court import CAMERA_VIEW_HALF_TURN_INDEX, COURT_SKELETON
from src.utils.schema.player import COCO17_SKELETON

_COURT_CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (80, 210, 255),
    (255, 170, 70),
    (80, 255, 140),
    (255, 100, 210),
    (60, 220, 220),
    (220, 120, 80),
    (230, 230, 80),
)
_COURT_CLASS_COLORS_V2: tuple[tuple[int, int, int], ...] = (
    (80, 210, 255),
    (255, 170, 70),
    (80, 255, 140),
    (255, 100, 210),
    (60, 220, 220),
    (220, 120, 80),
    (230, 230, 80),
    (150, 100, 255),
    (100, 240, 200),
    (255, 130, 130),
    (130, 200, 255),
    (200, 255, 100),
    (230, 150, 255),
    (100, 180, 230),
)
_IDENTITY_COLORS: tuple[tuple[int, int, int], ...] = (
    (70, 220, 255),
    (255, 120, 80),
    (90, 240, 130),
    (230, 100, 230),
    (80, 180, 255),
    (255, 210, 80),
    (180, 120, 255),
    (120, 255, 230),
)
_VISIBLE_COURT_BGR = (80, 220, 220)
_INVISIBLE_BGR = (120, 120, 120)
_TEXT_BGR = (245, 245, 245)


BallHistory = dict[str, deque[tuple[int, int]]]


def new_ball_history(object_ids: Sequence[str], *, history_frames: int) -> BallHistory:
    """Create fixed-capacity trajectory histories in canonical object order."""
    capacity = max(1, history_frames)
    return {object_id: deque(maxlen=capacity) for object_id in object_ids}


def render_court_overlay(
    frame: CourtSourceFrame,
    *,
    trajectory_id: str,
) -> NDArray[np.uint8]:
    """Overlay seven-class physical keypoints and renderer visibility."""
    if frame.schema_version in (
        CourtDatasetSchemaVersion.V2,
        CourtDatasetSchemaVersion.V3,
    ):
        return _render_court_overlay_singleton(frame, trajectory_id=trajectory_id)
    if frame.schema_version is not CourtDatasetSchemaVersion.V1:
        raise TypeError("Court overlay requires an explicit supported schema version.")
    canvas = _rgb_float_to_bgr(frame.rgb)
    projection = _object(frame.projection, name="Court projection")
    courts = _array(projection.get("courts"), name="Court projected courts")
    visible_points = 0
    total_points = 0
    for court_index, value in enumerate(courts):
        court = _object(value, name="Court projection court")
        court_id = _text(court.get("court_instance_id"), name="court_instance_id")
        classes = _array(court.get("classes"), name="Court semantic classes")
        if len(classes) != len(SEMANTIC_CLASS_NAMES):
            raise ValueError("Court overlay requires exactly seven semantic classes.")
        label_anchor: tuple[int, int] | None = None
        for class_index, (raw_class, expected_name) in enumerate(
            zip(classes, SEMANTIC_CLASS_NAMES, strict=True)
        ):
            semantic_class = _object(raw_class, name="Court semantic class")
            if (
                semantic_class.get("class_id") != class_index
                or semantic_class.get("class_name") != expected_name
            ):
                raise ValueError("Court semantic class identity/order changed.")
            points = _array(semantic_class.get("points"), name="Court points")
            if len(points) != 2:
                raise ValueError("Court semantic class must contain two points.")
            parsed = tuple(_court_point(value) for value in points)
            total_points += len(parsed)
            visible_points += sum(point[2] for point in parsed)
            color = _COURT_CLASS_COLORS[class_index]
            if all(point[1] for point in parsed):
                cv2.line(
                    canvas,
                    parsed[0][0],
                    parsed[1][0],
                    color,
                    2,
                    cv2.LINE_AA,
                )
            for point, in_frame, renderer_visible in parsed:
                if not in_frame:
                    continue
                if label_anchor is None:
                    label_anchor = point
                if renderer_visible:
                    cv2.circle(canvas, point, 5, color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, point, 7, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, point, 6, _INVISIBLE_BGR, 2, cv2.LINE_AA)
                    cv2.line(
                        canvas,
                        (point[0] - 4, point[1] - 4),
                        (point[0] + 4, point[1] + 4),
                        _INVISIBLE_BGR,
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.line(
                        canvas,
                        (point[0] + 4, point[1] - 4),
                        (point[0] - 4, point[1] + 4),
                        _INVISIBLE_BGR,
                        1,
                        cv2.LINE_AA,
                    )
        if label_anchor is not None:
            _outlined_text(
                canvas,
                court_id,
                (label_anchor[0] + 8, label_anchor[1] - 8 - 16 * court_index),
                color=_TEXT_BGR,
                scale=0.48,
            )
    _header(
        canvas,
        (
            f"COURT trajectory={trajectory_id} view={frame.view_id} "
            f"frame={frame.trajectory_frame_index} sample={frame.sample_id}"
        ),
        second_line=(
            f"filled=renderer-visible  x=not-visible  count={visible_points}/{total_points}"
        ),
    )
    _court_class_legend(canvas)
    return cast(NDArray[np.uint8], cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def _render_court_overlay_singleton(
    frame: CourtSourceFrame,
    *,
    trajectory_id: str,
) -> NDArray[np.uint8]:
    """Overlay exact V2/V3 singleton classes and physical CourtKP lines."""
    version = frame.schema_version.value
    canvas = _rgb_float_to_bgr(frame.rgb)
    projection = _object(frame.projection, name="Court projection")
    courts = _array(projection.get("courts"), name="Court projected courts")
    visible_points = 0
    total_points = 0
    for court_index, value in enumerate(courts):
        court = _object(value, name="Court projection court")
        court_id = _text(court.get("court_instance_id"), name="court_instance_id")
        classes = _array(court.get("classes"), name="Court semantic classes")
        if len(classes) != len(SEMANTIC_CLASS_NAMES_V2):
            raise ValueError(
                f"Court {version} overlay requires fourteen semantic classes."
            )
        points_by_physical_index: dict[
            int, tuple[tuple[int, int], bool, bool, str]
        ] = {}
        physical_index_by_class: list[int] = []
        for class_index, (raw_class, expected_name) in enumerate(
            zip(classes, SEMANTIC_CLASS_NAMES_V2, strict=True)
        ):
            semantic_class = _object(raw_class, name="Court semantic class")
            if (
                semantic_class.get("class_id") != class_index
                or semantic_class.get("class_name") != expected_name
            ):
                raise ValueError(
                    f"Court {version} semantic class identity/order changed."
                )
            points = _array(semantic_class.get("points"), name="Court points")
            if len(points) != 1:
                raise ValueError(
                    f"Court {version} semantic class must contain one point."
                )
            point = _object(points[0], name=f"Court {version} point")
            physical_index = _nonnegative_integer(
                point.get("physical_index"), name="physical_index"
            )
            physical_index_by_class.append(physical_index)
            if physical_index >= 14 or physical_index in points_by_physical_index:
                raise ValueError(
                    f"Court {version} physical point inventory is invalid."
                )
            uv, in_frame, renderer_visible = _court_point(point)
            points_by_physical_index[physical_index] = (
                uv,
                in_frame,
                renderer_visible,
                expected_name,
            )
        if set(points_by_physical_index) != set(range(14)):
            raise ValueError(
                f"Court {version} physical point inventory must cover 0..13."
            )
        if frame.schema_version is CourtDatasetSchemaVersion.V3:
            physical_order = tuple(physical_index_by_class)
            if physical_order not in (tuple(range(14)), CAMERA_VIEW_HALF_TURN_INDEX):
                raise ValueError(
                    "Court v3 physical inventory must be identity or full half-turn."
                )
        for first, second in COURT_SKELETON:
            if first >= 14 or second >= 14:
                continue
            first_point = points_by_physical_index[first]
            second_point = points_by_physical_index[second]
            if first_point[1] and second_point[1]:
                cv2.line(
                    canvas,
                    first_point[0],
                    second_point[0],
                    _VISIBLE_COURT_BGR,
                    1,
                    cv2.LINE_AA,
                )
        label_anchor: tuple[int, int] | None = None
        for class_index, expected_name in enumerate(SEMANTIC_CLASS_NAMES_V2):
            physical_index = physical_index_by_class[class_index]
            parsed_point, in_frame, renderer_visible, _ = points_by_physical_index[
                physical_index
            ]
            total_points += 1
            visible_points += int(renderer_visible)
            if not in_frame:
                continue
            if label_anchor is None:
                label_anchor = parsed_point
            color = _COURT_CLASS_COLORS_V2[class_index]
            if renderer_visible:
                cv2.circle(canvas, parsed_point, 5, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, parsed_point, 7, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, parsed_point, 6, _INVISIBLE_BGR, 2, cv2.LINE_AA)
            _outlined_text(
                canvas,
                expected_name,
                (parsed_point[0] + 5, parsed_point[1] - 5),
                color=color,
                scale=0.28,
            )
        if label_anchor is not None:
            _outlined_text(
                canvas,
                court_id,
                (label_anchor[0] + 8, label_anchor[1] - 8 - 16 * court_index),
                color=_TEXT_BGR,
                scale=0.48,
            )
    _header(
        canvas,
        (
            f"COURT {version} trajectory={trajectory_id} view={frame.view_id} "
            f"frame={frame.trajectory_frame_index} sample={frame.sample_id}"
        ),
        second_line=(f"filled=renderer-visible  count={visible_points}/{total_points}"),
    )
    _court_class_legend(
        canvas,
        class_names=SEMANTIC_CLASS_NAMES_V2,
        colors=_COURT_CLASS_COLORS_V2,
    )
    return cast(NDArray[np.uint8], cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def render_blcs_overlay(
    frame: BLCSSourceFrame,
    *,
    logical_scene_id: str,
    camera_id: str,
    object_ids: Sequence[str],
    court_kp: NDArray[np.float32],
    court_vis: NDArray[np.bool_],
    history: BallHistory,
    history_frames: int,
) -> NDArray[np.uint8]:
    """Overlay ball identity, observations, presence, and short track history."""
    canvas = _rgb_float_to_bgr(frame.render.rgb)
    _draw_normalized_court(
        canvas,
        court_kp,
        court_vis,
        normalized=False,
    )
    metadata = _object(frame.metadata, name="BLCS metadata")
    objects = tuple(
        _object(value, name="BLCS object")
        for value in _array(metadata.get("objects"), name="BLCS objects")
    )
    if len(objects) != len(object_ids):
        raise ValueError("BLCS object labels differ from the canonical track axis.")
    arrays = _object(metadata.get("semantic_arrays"), name="BLCS semantic_arrays")
    ball_uv = _float_array(arrays.get("ball_uv"), shape=(len(object_ids), 2))
    present = _bool_array(arrays.get("present"), shape=(len(object_ids),))
    geometric = _bool_array(arrays.get("geometric_visible"), shape=(len(object_ids),))
    rendered = _bool_array(arrays.get("rendered_visible"), shape=(len(object_ids),))
    instance_ids = _positive_integer_array(
        arrays.get("instance_ids"), shape=(len(object_ids),)
    )
    if len(set(int(value) for value in instance_ids)) != len(instance_ids):
        raise ValueError("BLCS semantic instance IDs must be unique.")
    if np.any(rendered & ~present):
        raise ValueError("Absent BLCS objects cannot be renderer-visible.")
    for object_index, (object_id, raw_object) in enumerate(
        zip(object_ids, objects, strict=True)
    ):
        if raw_object.get("object_id") != object_id:
            raise ValueError("BLCS object identity/order changed during visualization.")
        if raw_object.get("present") is not bool(present[object_index]):
            raise ValueError("BLCS object presence differs from semantic arrays.")
        if raw_object.get("geometric_visible") is not bool(
            geometric[object_index]
        ) or raw_object.get("rendered_visible") is not bool(rendered[object_index]):
            raise ValueError("BLCS object visibility differs from semantic arrays.")
        if _positive_integer(
            raw_object.get("instance_id"), name="BLCS object instance_id"
        ) != int(instance_ids[object_index]):
            raise ValueError("BLCS object instance ID differs from semantic arrays.")
    rendered_counts = _renderer_instance_counts(frame.render)
    foreign_ids = set(rendered_counts) - set(int(value) for value in instance_ids)
    if foreign_ids:
        raise ValueError(
            "BLCS render contains undeclared foreground instance IDs: "
            f"{sorted(foreign_ids)}."
        )
    observed = np.asarray(
        [int(instance_id) in rendered_counts for instance_id in instance_ids],
        dtype=np.bool_,
    )
    if not np.array_equal(observed, rendered):
        raise ValueError(
            "BLCS rendered_visible claims disagree with streamed renderer instance IDs."
        )
    statuses: list[str] = []
    for object_index, (object_id, _) in enumerate(
        zip(object_ids, objects, strict=True)
    ):
        color = _IDENTITY_COLORS[object_index % len(_IDENTITY_COLORS)]
        coordinate = _pixel(ball_uv[object_index])
        inside = _inside(canvas, coordinate)
        renderer_observed = bool(
            present[object_index] and rendered[object_index] and inside
        )
        if history_frames > 0:
            if renderer_observed:
                history[object_id].append(coordinate)
            else:
                history[object_id].clear()
            _draw_history(canvas, history[object_id], color=color)
        if renderer_observed:
            cv2.circle(canvas, coordinate, 8, color, 2, cv2.LINE_AA)
            cv2.circle(canvas, coordinate, 3, color, -1, cv2.LINE_AA)
            _outlined_text(
                canvas,
                object_id,
                (coordinate[0] + 10, coordinate[1] - 8),
                color=color,
                scale=0.5,
            )
        elif present[object_index] and geometric[object_index] and inside:
            cv2.drawMarker(
                canvas,
                coordinate,
                color,
                cv2.MARKER_TILTED_CROSS,
                12,
                2,
                cv2.LINE_AA,
            )
        status = "absent"
        if present[object_index]:
            status = "observed" if rendered[object_index] else "present/occluded"
        statuses.append(f"{object_id}: {status}")
    _header(
        canvas,
        (
            f"BLCS scene={logical_scene_id} camera={camera_id} "
            f"source_frame={frame.source_frame_index} global={frame.global_frame_index}"
        ),
        second_line="circle=renderer observation  cross=geometric-only",
    )
    _status_panel(canvas, statuses)
    return cast(NDArray[np.uint8], cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def render_plcs_overlay(
    frame: PLCSSourceFrame,
    *,
    logical_scene_id: str,
    camera_id: str,
    object_ids: Sequence[str],
) -> NDArray[np.uint8]:
    """Overlay projected COCO17 skeletons with identity and presence state."""
    canvas = _rgb_float_to_bgr(frame.render.rgb)
    _draw_normalized_court(
        canvas,
        frame.court_kp,
        frame.court_vis,
        normalized=True,
    )
    objects = tuple(
        _object(value, name="PLCS object")
        for value in _array(frame.label.get("objects"), name="PLCS objects")
    )
    if len(objects) != len(object_ids) or frame.present.shape != (len(object_ids),):
        raise ValueError("PLCS object labels differ from the canonical track axis.")
    rendered_counts = _renderer_instance_counts(frame.render)
    declared_instance_ids: list[int] = []
    for object_index, (object_id, raw_object) in enumerate(
        zip(object_ids, objects, strict=True)
    ):
        if raw_object.get("object_id") != object_id:
            raise ValueError("PLCS object identity/order changed during visualization.")
        present = bool(frame.present[object_index])
        if raw_object.get("present") is not present:
            raise ValueError("PLCS object presence differs from supervision arrays.")
        instance_id = _positive_integer(
            raw_object.get("instance_id"), name="PLCS object instance_id"
        )
        if instance_id in declared_instance_ids:
            raise ValueError("PLCS object instance IDs must be unique.")
        declared_instance_ids.append(instance_id)
        visible_pixels = _nonnegative_integer(
            raw_object.get("visible_pixel_count"), name="visible_pixel_count"
        )
        actual_visible_pixels = rendered_counts.get(instance_id, 0)
        if visible_pixels != actual_visible_pixels:
            raise ValueError(
                "PLCS visible_pixel_count disagrees with streamed renderer "
                f"instance ID {instance_id}: claimed {visible_pixels}, "
                f"observed {actual_visible_pixels}."
            )
        if not present and visible_pixels != 0:
            raise ValueError("Absent PLCS objects cannot have renderer-visible pixels.")
        keypoints = frame.human_kp[object_index]
        visible = frame.human_vis[object_index]
        if keypoints.shape != (17, 2) or visible.shape != (17,):
            raise ValueError("PLCS COCO17 supervision shape changed.")
        if not present and np.any(visible):
            raise ValueError("Absent PLCS objects cannot have visible keypoints.")
    foreign_ids = set(rendered_counts) - set(declared_instance_ids)
    if foreign_ids:
        raise ValueError(
            "PLCS render contains undeclared foreground instance IDs: "
            f"{sorted(foreign_ids)}."
        )
    statuses: list[str] = []
    for object_index, (object_id, raw_object) in enumerate(
        zip(object_ids, objects, strict=True)
    ):
        present = bool(frame.present[object_index])
        visible_pixels = _nonnegative_integer(
            raw_object.get("visible_pixel_count"), name="visible_pixel_count"
        )
        color = _IDENTITY_COLORS[object_index % len(_IDENTITY_COLORS)]
        keypoints = frame.human_kp[object_index]
        visible = frame.human_vis[object_index]
        pixels = np.empty_like(keypoints)
        pixels[:, 0] = keypoints[:, 0] * canvas.shape[1]
        pixels[:, 1] = keypoints[:, 1] * canvas.shape[0]
        points = tuple(_pixel(value) for value in pixels)
        for first, second in COCO17_SKELETON:
            if visible[first] and visible[second]:
                cv2.line(
                    canvas,
                    points[first],
                    points[second],
                    color,
                    2,
                    cv2.LINE_AA,
                )
        visible_indices = np.flatnonzero(visible)
        for joint_index in visible_indices:
            cv2.circle(
                canvas,
                points[int(joint_index)],
                4,
                color,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                points[int(joint_index)],
                5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if len(visible_indices):
            anchor_values = pixels[visible]
            anchor = (
                int(round(float(np.mean(anchor_values[:, 0])))),
                int(round(float(np.min(anchor_values[:, 1])))) - 8,
            )
            _outlined_text(canvas, object_id, anchor, color=color, scale=0.52)
        status = "absent"
        if present:
            status = f"present pixels={visible_pixels} joints={len(visible_indices)}/17"
        statuses.append(f"{object_id}: {status}")
    _header(
        canvas,
        (f"PLCS scene={logical_scene_id} camera={camera_id} frame={frame.frame_index}"),
        second_line="COCO17 projections; panel reports physical/rendered presence",
    )
    _status_panel(canvas, statuses)
    return cast(NDArray[np.uint8], cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def _rgb_float_to_bgr(value: NDArray[np.float32]) -> NDArray[np.uint8]:
    if value.dtype != np.float32 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("Visualization source RGB must be float32 [H,W,3].")
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 1.0):
        raise ValueError("Visualization source RGB must be finite and in [0,1].")
    rgb = np.rint(value * 255.0).astype(np.uint8)
    return cast(NDArray[np.uint8], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _court_point(value: object) -> tuple[tuple[int, int], bool, bool]:
    point = _object(value, name="Court point")
    uv = _float_array(point.get("uv"), shape=(2,))
    in_frame = point.get("in_frame")
    renderer_visible = point.get("renderer_visible")
    if not isinstance(in_frame, bool) or not isinstance(renderer_visible, bool):
        raise TypeError("Accepted Court point visibility must be boolean.")
    return _pixel(uv), in_frame, renderer_visible


def _draw_normalized_court(
    canvas: NDArray[np.uint8],
    keypoints: NDArray[np.float32],
    visible: NDArray[np.bool_],
    *,
    normalized: bool,
) -> None:
    if keypoints.shape != (20, 2) or visible.shape != (20,):
        raise ValueError("CourtKP20 overlay requires shapes [20,2] and [20].")
    values = np.asarray(keypoints, dtype=np.float32).copy()
    if normalized:
        values[:, 0] *= canvas.shape[1]
        values[:, 1] *= canvas.shape[0]
    points = tuple(_pixel(value) for value in values)
    for first, second in COURT_SKELETON:
        if visible[first] and visible[second]:
            cv2.line(
                canvas,
                points[first],
                points[second],
                _VISIBLE_COURT_BGR,
                1,
                cv2.LINE_AA,
            )
    for index, is_visible in enumerate(visible):
        if is_visible:
            cv2.circle(
                canvas,
                points[index],
                2,
                _VISIBLE_COURT_BGR,
                -1,
                cv2.LINE_AA,
            )


def _draw_history(
    canvas: NDArray[np.uint8],
    points: deque[tuple[int, int]],
    *,
    color: tuple[int, int, int],
) -> None:
    values = tuple(points)
    for index in range(1, len(values)):
        cv2.line(
            canvas,
            values[index - 1],
            values[index],
            color,
            max(1, 3 - (len(values) - index) // 5),
            cv2.LINE_AA,
        )


def _header(canvas: NDArray[np.uint8], text: str, *, second_line: str) -> None:
    width = canvas.shape[1]
    cv2.rectangle(canvas, (0, 0), (width, 52), (20, 20, 20), -1)
    _outlined_text(canvas, text, (10, 20), color=_TEXT_BGR, scale=0.48)
    _outlined_text(canvas, second_line, (10, 42), color=(205, 205, 205), scale=0.42)


def _status_panel(canvas: NDArray[np.uint8], statuses: Sequence[str]) -> None:
    if not statuses:
        return
    line_height = 19
    panel_height = 9 + line_height * len(statuses)
    top = canvas.shape[0] - panel_height
    cv2.rectangle(
        canvas,
        (0, max(0, top)),
        (min(canvas.shape[1] - 1, 500), canvas.shape[0] - 1),
        (20, 20, 20),
        -1,
    )
    for index, status in enumerate(statuses):
        color = _IDENTITY_COLORS[index % len(_IDENTITY_COLORS)]
        y = max(15, top + 17 + line_height * index)
        cv2.circle(canvas, (10, y - 4), 4, color, -1, cv2.LINE_AA)
        _outlined_text(canvas, status, (21, y), color=_TEXT_BGR, scale=0.43)


def _court_class_legend(
    canvas: NDArray[np.uint8],
    *,
    class_names: Sequence[str] = SEMANTIC_CLASS_NAMES,
    colors: Sequence[tuple[int, int, int]] = _COURT_CLASS_COLORS,
) -> None:
    line_height = 18
    panel_width = 145
    left = max(0, canvas.shape[1] - panel_width)
    bottom = min(canvas.shape[0] - 1, 58 + line_height * len(class_names))
    cv2.rectangle(canvas, (left, 52), (canvas.shape[1] - 1, bottom), (20, 20, 20), -1)
    for index, (name, color) in enumerate(zip(class_names, colors, strict=True)):
        y = 68 + line_height * index
        cv2.circle(canvas, (left + 9, y - 4), 4, color, -1, cv2.LINE_AA)
        _outlined_text(canvas, name, (left + 18, y), color=_TEXT_BGR, scale=0.36)


def _outlined_text(
    canvas: NDArray[np.uint8],
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int],
    scale: float,
) -> None:
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


def _pixel(value: Sequence[float] | NDArray[np.floating]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("Overlay coordinate must contain exactly two values.")
    x = float(value[0])
    y = float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Overlay coordinate must be finite.")
    return int(round(x)), int(round(y))


def _inside(canvas: NDArray[np.uint8], point: tuple[int, int]) -> bool:
    return bool(0 <= point[0] < canvas.shape[1] and 0 <= point[1] < canvas.shape[0])


def _renderer_instance_counts(render: object) -> dict[int, int]:
    if not isinstance(render, LogicalRenderSample):
        raise TypeError("Visualization render must be a LogicalRenderSample.")
    instance_ids = render.instance_ids
    if (
        instance_ids.dtype != np.int32
        or instance_ids.shape != render.rgb.shape[:2]
        or np.any(instance_ids < 0)
    ):
        raise ValueError(
            "Streamed renderer instance IDs must be non-negative int32 [H,W]."
        )
    values, counts = np.unique(instance_ids[instance_ids > 0], return_counts=True)
    return {
        int(instance_id): int(count)
        for instance_id, count in zip(values, counts, strict=True)
    }


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    return cast(Mapping[str, object], value)


def _array(value: object, *, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an array.")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer.")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _float_array(value: object, *, shape: tuple[int, ...]) -> NDArray[np.float64]:
    result = np.asarray(value)
    if result.shape != shape or not np.issubdtype(result.dtype, np.number):
        raise ValueError(f"Numeric overlay array must have shape {shape}.")
    result = result.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("Numeric overlay array must be finite.")
    return result


def _bool_array(value: object, *, shape: tuple[int, ...]) -> NDArray[np.bool_]:
    result = np.asarray(value)
    if result.shape != shape or result.dtype != np.bool_:
        raise ValueError(f"Boolean overlay array must have shape {shape}.")
    return cast(NDArray[np.bool_], result)


def _positive_integer_array(
    value: object, *, shape: tuple[int, ...]
) -> NDArray[np.int64]:
    result = np.asarray(value)
    if (
        result.shape != shape
        or not np.issubdtype(result.dtype, np.integer)
        or np.any(result <= 0)
    ):
        raise ValueError(f"Positive integer overlay array must have shape {shape}.")
    return cast(NDArray[np.int64], result.astype(np.int64, copy=False))


__all__ = [
    "BallHistory",
    "new_ball_history",
    "render_blcs_overlay",
    "render_court_overlay",
    "render_plcs_overlay",
]
