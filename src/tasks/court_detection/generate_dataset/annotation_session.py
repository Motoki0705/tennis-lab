"""Shared OpenCV UI for CourtKP20 annotation and pseudo-label adjustment."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.utils.io import find_existing_file, save_json_atomic
from src.utils.schema.court import COURT_KP_NAMES, court_keypoints_3d

COURT_KP20_COUNT = 20
GROUND_KEYPOINT_COUNT = 14
NET_CENTER_IDX = 14
HOMOGRAPHY_GROUND_INDICES: tuple[int, ...] = tuple(range(NET_CENTER_IDX + 1))
HOMOGRAPHY_MIN_ANCHORS = 4
LEFT_ARROW_KEY = 81
UP_ARROW_KEY = 82
RIGHT_ARROW_KEY = 83
DOWN_ARROW_KEY = 84
NET_DRAW_PAIRS: tuple[tuple[int, int], ...] = (
    (15, 16),
    (17, 18),
    (16, 19),
    (19, 18),
)
GROUND_DRAW_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (2, 3),
    (0, 2),
    (1, 3),
    (4, 8),
    (6, 9),
    (8, 10),
    (9, 11),
    (10, 5),
    (11, 7),
    (8, 9),
    (10, 11),
    (12, 13),
)


@dataclass(frozen=True)
class AnnotationSessionConfig:
    """Runtime settings for a CourtKP20 annotation session."""

    root: Path
    split: str
    source_file_pattern: str
    target_file_pattern: str
    target_file_name: str | None
    manual_adjusted_field: str
    image_id: str | None
    start_index: int
    skip_completed: bool
    window_name: str
    max_display_width: int
    max_display_height: int
    editable_indices: tuple[int, ...]
    required_indices: tuple[int, ...]
    drag_radius_px: float
    drag_start_threshold_px: float
    annotation_format: str = "legacy_kps"
    image_path_key: str = "image_path"
    homography_auto_fill: bool = False
    start_after_last_completed: bool = False
    keypoint_format: str = "kp20"
    include_source_types: tuple[str, ...] = ()


@dataclass
class UiState:
    """Mutable state for the active OpenCV window."""

    editable_indices: tuple[int, ...]
    selected_keypoint: int
    display_scale: float = 1.0
    show_help: bool = True
    complete_button_rect: tuple[int, int, int, int] | None = None
    pending_action: str | None = None
    mouse_down_display_xy: tuple[int, int] | None = None
    dragging_idx: int | None = None
    drag_started: bool = False
    drag_radius_px: float = 18.0
    drag_start_threshold_px: float = 4.0
    input_visibility: int = 1
    image_width: int | None = None
    image_height: int | None = None
    auto_filled_indices: set[int] = field(default_factory=set)


@dataclass
class AnnotationDocument:
    """Parsed annotation document with optional top-level metadata."""

    metadata: dict[str, Any] | None
    items: list[dict[str, Any]]


def run_annotation_session(config: AnnotationSessionConfig) -> int:
    """Run an interactive CourtKP20 annotation session."""
    source_path = config.root / config.source_file_pattern.format(split=config.split)
    if config.target_file_name is not None:
        target_path = config.root / config.target_file_name
    else:
        target_path = config.root / config.target_file_pattern.format(split=config.split)
    if not source_path.exists():
        raise FileNotFoundError(f"Source annotation file not found: {source_path}")

    source_doc = read_annotation_document(source_path)
    target_doc = read_annotation_document(target_path)
    all_source_entries = source_doc.items
    source_entries = filter_source_entries(all_source_entries, config.include_source_types)
    target_entries = target_doc.items
    target_by_id = {str(entry["id"]): entry for entry in target_entries}
    if not source_entries:
        raise ValueError(
            f"No matching entries found in {source_path}; "
            f"include_source_types={list(config.include_source_types)}."
        )

    current_index = find_start_index(source_entries, target_by_id, config)
    ui = UiState(
        editable_indices=config.editable_indices,
        selected_keypoint=config.editable_indices[0],
        drag_radius_px=config.drag_radius_px,
        drag_start_threshold_px=config.drag_start_threshold_px,
    )

    images_dir = config.root / "images"
    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
    print(
        f"[court_kp20_annotation] split={config.split} source={source_path} "
        f"target={target_path} completed={completed_count(source_entries, target_by_id, config)}/{len(source_entries)}"
    )

    try:
        while True:
            source_entry = source_entries[current_index]
            image_id = str(source_entry["id"])
            image_path = find_image_for_entry(config.root, images_dir, source_entry, image_id, config)
            if image_path is None:
                print(f"  SKIP (missing image): {image_id}")
                current_index = advance_index(
                    source_entries,
                    target_by_id,
                    current_index,
                    1,
                    skip_completed=False,
                    config=config,
                )
                continue

            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"  SKIP (imread): {image_id}")
                current_index = advance_index(
                    source_entries,
                    target_by_id,
                    current_index,
                    1,
                    skip_completed=False,
                    config=config,
                )
                continue

            ui.image_height, ui.image_width = image_bgr.shape[:2]
            kps, visibility = display_keypoints(source_entry, target_by_id, config)
            ui.auto_filled_indices.clear()
            maybe_auto_fill_ground_keypoints(kps, visibility, ui, config)
            ui.selected_keypoint = first_missing_keypoint(kps, config.editable_indices, visibility)
            set_mouse_callback(config.window_name, kps, visibility, ui, config)

            while True:
                frame = render(
                    image_bgr,
                    kps,
                    image_id=image_id,
                    index=current_index,
                    total=len(source_entries),
                    completed_count=completed_count(source_entries, target_by_id, config),
                    ui=ui,
                    config=config,
                    visibility=visibility,
                )
                cv2.imshow(config.window_name, frame)
                key = cv2.waitKey(20) & 0xFF

                if ui.pending_action == "complete":
                    ui.pending_action = None
                    if save_completed_kp20(
                        source_entry,
                        kps,
                        visibility,
                        all_source_entries,
                        target_by_id,
                        target_path,
                        config,
                        target_doc.metadata or source_doc.metadata,
                    ):
                        current_index = advance_index(
                            source_entries,
                            target_by_id,
                            current_index,
                            1,
                            skip_completed=config.skip_completed,
                            config=config,
                        )
                        ui.selected_keypoint = config.editable_indices[0]
                        break
                    print(f"[court_kp20_annotation] Complete requires indices={list(config.required_indices)}.")

                if key == 255:
                    continue
                if key in _number_keys_for_editable_indices(config.editable_indices):
                    ui.selected_keypoint = config.editable_indices[key - ord("1")]
                elif key in (RIGHT_ARROW_KEY, DOWN_ARROW_KEY):
                    advance_selected_keypoint(ui)
                elif key in (LEFT_ARROW_KEY, UP_ARROW_KEY):
                    advance_selected_keypoint(ui, step=-1)
                elif key in (8, 127):
                    kps[ui.selected_keypoint] = [None, None]
                    visibility[ui.selected_keypoint] = 0
                    ui.auto_filled_indices.discard(ui.selected_keypoint)
                    maybe_auto_fill_ground_keypoints(
                        kps,
                        visibility,
                        ui,
                        config,
                        changed_idx=ui.selected_keypoint,
                    )
                elif key == ord("c"):
                    if save_completed_kp20(
                        source_entry,
                        kps,
                        visibility,
                        all_source_entries,
                        target_by_id,
                        target_path,
                        config,
                        target_doc.metadata or source_doc.metadata,
                    ):
                        current_index = advance_index(
                            source_entries,
                            target_by_id,
                            current_index,
                            1,
                            skip_completed=config.skip_completed,
                            config=config,
                        )
                        ui.selected_keypoint = config.editable_indices[0]
                        break
                    print(f"[court_kp20_annotation] Complete requires indices={list(config.required_indices)}.")
                elif key == ord("g"):
                    center = compute_net_center(to_numpy(kps))
                    if center is not None:
                        kps[NET_CENTER_IDX] = [float(center[0]), float(center[1])]
                        visibility[NET_CENTER_IDX] = 1
                elif key == ord("h"):
                    ui.show_help = not ui.show_help
                elif key == ord("r"):
                    kps, visibility = display_keypoints(source_entry, target_by_id, config)
                    ui.auto_filled_indices.clear()
                    maybe_auto_fill_ground_keypoints(kps, visibility, ui, config)
                    ui.selected_keypoint = first_missing_keypoint(kps, config.editable_indices, visibility)
                    set_mouse_callback(config.window_name, kps, visibility, ui, config)
                elif key == ord("v"):
                    ui.input_visibility = 1
                elif key == ord("o"):
                    ui.input_visibility = 1 if ui.input_visibility == 2 else 2
                elif key == ord("x"):
                    kps[ui.selected_keypoint] = [None, None]
                    visibility[ui.selected_keypoint] = 3
                    ui.auto_filled_indices.discard(ui.selected_keypoint)
                    maybe_auto_fill_ground_keypoints(
                        kps,
                        visibility,
                        ui,
                        config,
                        changed_idx=ui.selected_keypoint,
                    )
                    advance_selected_keypoint(ui)
                elif key == ord("w"):
                    save_completed_kp20(
                        source_entry,
                        kps,
                        visibility,
                        all_source_entries,
                        target_by_id,
                        target_path,
                        config,
                        target_doc.metadata or source_doc.metadata,
                    )
                elif key == ord("n"):
                    current_index = advance_index(
                        source_entries,
                        target_by_id,
                        current_index,
                        1,
                        skip_completed=config.skip_completed,
                        config=config,
                    )
                    ui.selected_keypoint = config.editable_indices[0]
                    break
                elif key == ord("p"):
                    current_index = advance_index(
                        source_entries,
                        target_by_id,
                        current_index,
                        -1,
                        skip_completed=False,
                        config=config,
                    )
                    ui.selected_keypoint = config.editable_indices[0]
                    break
                elif key == ord("q"):
                    print("[court_kp20_annotation] Quit without writing incomplete edits.")
                    return 0
    finally:
        cv2.destroyWindow(config.window_name)


def read_json(path: Path) -> list[dict[str, Any]]:
    """Read a JSON list, returning an empty list when the file does not exist."""
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def read_annotation_document(path: Path) -> AnnotationDocument:
    """Read either a legacy JSON list or a metadata-wrapped annotation document."""
    if not path.exists():
        return AnnotationDocument(metadata=None, items=[])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return AnnotationDocument(metadata=None, items=payload)
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        metadata = dict(payload)
        items = metadata.pop("items")
        return AnnotationDocument(metadata=metadata, items=items)
    raise ValueError(f"Unsupported annotation JSON shape: {path}")


def normalize_serialized_keypoints(raw_kps: Any) -> list[list[float | None]]:
    """Normalize a serialized keypoint list to CourtKP20 shape."""
    normalized: list[list[float | None]] = []
    for point in list(raw_kps)[:COURT_KP20_COUNT]:
        if point is None:
            normalized.append([None, None])
            continue
        point_values = list(point)
        if len(point_values) < 2:
            normalized.append([None, None])
            continue
        normalized.append([optional_float(point_values[0]), optional_float(point_values[1])])
    while len(normalized) < COURT_KP20_COUNT:
        normalized.append([None, None])
    return normalized


def normalize_named_keypoints(entry: dict[str, Any]) -> tuple[list[list[float | None]], list[int]]:
    """Normalize named keypoints to ordered coordinates and visibility labels."""
    kps = [[None, None] for _ in range(COURT_KP20_COUNT)]
    visibility = [0 for _ in range(COURT_KP20_COUNT)]
    raw_keypoints = entry.get("keypoints", [])
    if not isinstance(raw_keypoints, list):
        return kps, visibility

    for point in raw_keypoints:
        if not isinstance(point, dict):
            continue
        idx = keypoint_index(point)
        if idx is None:
            continue
        expected_name = COURT_KP_NAMES[idx]
        name = point.get("name")
        if name is not None and str(name) != expected_name:
            raise ValueError(
                f"Keypoint name mismatch for id={entry.get('id')!r}: "
                f"index={idx} expected={expected_name!r} got={name!r}"
            )
        kps[idx] = [optional_float(point.get("x")), optional_float(point.get("y"))]
        visibility[idx] = normalize_visibility(point.get("visibility"))
    return kps, visibility


def keypoint_index(point: dict[str, Any]) -> int | None:
    """Return the CourtKP20 index for a named keypoint object."""
    raw_index = point.get("index")
    if raw_index is not None:
        idx = int(raw_index)
        if 0 <= idx < COURT_KP20_COUNT:
            return idx
    raw_name = point.get("name")
    if raw_name is not None:
        try:
            return COURT_KP_NAMES.index(str(raw_name))
        except ValueError:
            return None
    return None


def normalize_visibility(value: Any) -> int:
    """Return a supported visibility value."""
    if value is None:
        return 0
    visibility = int(value)
    if visibility not in (0, 1, 2, 3):
        raise ValueError(f"Unsupported visibility={value!r}; expected one of 0, 1, 2, 3.")
    return visibility


def optional_float(value: Any) -> float | None:
    """Return a finite float or None."""
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        return None
    return result


def to_numpy(kps: list[list[float | None]]) -> np.ndarray:
    """Convert keypoints with None values to a float array with NaNs."""
    arr = np.full((len(kps), 2), np.nan, dtype=np.float32)
    for idx, point in enumerate(kps):
        if point[0] is not None and point[1] is not None:
            arr[idx] = (float(point[0]), float(point[1]))
    return arr


def source_keypoints(
    source_entry: dict[str, Any],
    config: AnnotationSessionConfig,
) -> tuple[list[list[float | None]], list[int]]:
    """Build a KP20 draft from a source entry."""
    if config.annotation_format == "named_keypoints":
        kps, visibility = normalize_named_keypoints(source_entry)
    else:
        kps = normalize_serialized_keypoints(source_entry.get("kps", []))
        visibility = [1 if point[0] is not None and point[1] is not None else 0 for point in kps]

    center = compute_net_center(to_numpy(kps))
    if center is not None:
        kps[NET_CENTER_IDX] = [float(center[0]), float(center[1])]
        visibility[NET_CENTER_IDX] = 1
    return kps, visibility


def display_keypoints(
    source_entry: dict[str, Any],
    target_by_id: dict[str, dict[str, Any]],
    config: AnnotationSessionConfig,
) -> tuple[list[list[float | None]], list[int]]:
    """Return target KP20 if available, otherwise a draft from the source entry."""
    image_id = str(source_entry["id"])
    target_entry = target_by_id.get(image_id)
    if config.annotation_format == "named_keypoints":
        if target_entry is not None and len(target_entry.get("keypoints", [])) == COURT_KP20_COUNT:
            return normalize_named_keypoints(target_entry)
        return source_keypoints(source_entry, config)

    if target_entry is not None and len(target_entry.get("kps", [])) == COURT_KP20_COUNT:
        kps = normalize_serialized_keypoints(target_entry["kps"])
        return kps, [1 if point[0] is not None and point[1] is not None else 0 for point in kps]
    return source_keypoints(source_entry, config)


def compute_net_center(kps: np.ndarray) -> np.ndarray | None:
    """Compute net center from the four doubles corners."""
    if kps.shape[0] < 4 or not np.isfinite(kps[:4]).all():
        return None
    intersection = line_intersection(kps[0], kps[3], kps[1], kps[2])
    if intersection is not None:
        return intersection.astype(np.float32)
    return np.mean(kps[:4], axis=0).astype(np.float32)


def maybe_auto_fill_ground_keypoints(
    kps: list[list[float | None]],
    visibility: list[int],
    ui: UiState,
    config: AnnotationSessionConfig,
    *,
    changed_idx: int | None = None,
) -> bool:
    """Fill missing ground-plane keypoints from any four or more ground anchors."""
    if not config.homography_auto_fill:
        return False
    if changed_idx is not None and changed_idx not in HOMOGRAPHY_GROUND_INDICES:
        return False

    anchor_indices = homography_anchor_indices(kps, visibility, ui)
    if len(anchor_indices) < HOMOGRAPHY_MIN_ANCHORS:
        clear_auto_filled_ground_keypoints(kps, visibility, ui)
        return False

    projected = project_ground_keypoints_from_anchors(kps, anchor_indices)
    if projected is None:
        clear_auto_filled_ground_keypoints(kps, visibility, ui)
        return False

    changed = False
    arr = to_numpy(kps)
    for idx in HOMOGRAPHY_GROUND_INDICES:
        if idx in anchor_indices:
            continue
        should_update = (
            idx in ui.auto_filled_indices
            or not is_finite_idx(arr, idx)
            or (idx < len(visibility) and visibility[idx] == 0)
        )
        if not should_update:
            continue
        projected_point = [float(projected[idx, 0]), float(projected[idx, 1])]
        if is_inside_image(projected_point, ui):
            kps[idx] = projected_point
            visibility[idx] = 1
        else:
            kps[idx] = [None, None]
            visibility[idx] = 3
        ui.auto_filled_indices.add(idx)
        changed = True
    return changed


def homography_anchor_indices(
    kps: list[list[float | None]],
    visibility: list[int],
    ui: UiState,
) -> tuple[int, ...]:
    """Return manually provided ground-plane indices usable as homography anchors."""
    arr = to_numpy(kps)
    anchors: list[int] = []
    for idx in HOMOGRAPHY_GROUND_INDICES:
        if idx in ui.auto_filled_indices:
            continue
        if idx >= len(visibility) or visibility[idx] not in (1, 2):
            continue
        if is_finite_idx(arr, idx):
            anchors.append(idx)
    return tuple(anchors)


def is_inside_image(point: list[float], ui: UiState) -> bool:
    """Return whether a projected point lands inside the current image frame."""
    if ui.image_width is None or ui.image_height is None:
        return True
    x, y = point
    return 0.0 <= x < float(ui.image_width) and 0.0 <= y < float(ui.image_height)


def clear_auto_filled_ground_keypoints(
    kps: list[list[float | None]],
    visibility: list[int],
    ui: UiState,
) -> None:
    """Clear homography-filled points when a defining corner is removed."""
    for idx in tuple(ui.auto_filled_indices):
        kps[idx] = [None, None]
        visibility[idx] = 0
    ui.auto_filled_indices.clear()


def project_ground_keypoints_from_anchors(
    kps: list[list[float | None]],
    anchor_indices: tuple[int, ...],
) -> np.ndarray | None:
    """Project CourtKP20 ground-plane keypoints from arbitrary ground anchors."""
    image_points = to_numpy(kps)
    if len(anchor_indices) < HOMOGRAPHY_MIN_ANCHORS:
        return None
    if not all(is_finite_idx(image_points, idx) for idx in anchor_indices):
        return None

    court_xy = court_keypoints_3d()[: NET_CENTER_IDX + 1].numpy()[:, :2].astype(np.float32)
    source_points = court_xy[list(anchor_indices)]
    target_points = image_points[list(anchor_indices)].astype(np.float32)
    homography, _status = cv2.findHomography(source_points, target_points, 0)
    if homography is None or not np.isfinite(homography).all():
        return None
    projected = cv2.perspectiveTransform(court_xy.reshape(1, -1, 2), homography).reshape(-1, 2)
    if not np.isfinite(projected).all():
        return None
    return projected


def line_intersection(
    a0: np.ndarray,
    a1: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
) -> np.ndarray | None:
    """Return the intersection of two 2-D lines."""
    da = a1 - a0
    db = b1 - b0
    cross = float(da[0] * db[1] - da[1] * db[0])
    if abs(cross) < 1e-6:
        return None
    delta = b0 - a0
    t = float(delta[0] * db[1] - delta[1] * db[0]) / cross
    return a0 + t * da


def is_finite_idx(kps: np.ndarray, idx: int) -> bool:
    """Return whether a keypoint index is finite."""
    return idx < kps.shape[0] and bool(np.isfinite(kps[idx]).all())


def is_complete(
    kps: list[list[float | None]],
    required_indices: tuple[int, ...],
    *,
    visibility: list[int] | None = None,
    annotation_format: str = "legacy_kps",
) -> bool:
    """Return whether all required indices are present."""
    if annotation_format == "named_keypoints":
        arr = to_numpy(kps)
        for idx in required_indices:
            if idx >= len(visibility or []):
                return False
            state = (visibility or [])[idx]
            if state in (1, 2):
                if not is_finite_idx(arr, idx):
                    return False
            elif state == 3:
                continue
            else:
                return False
        return True

    arr = to_numpy(kps)
    return all(is_finite_idx(arr, idx) for idx in required_indices)


def is_completed_id(
    image_id: str,
    target_by_id: dict[str, dict[str, Any]],
    config: AnnotationSessionConfig,
) -> bool:
    """Return whether an image already has a complete target entry."""
    target_entry = target_by_id.get(image_id)
    if target_entry is None:
        return False
    if config.annotation_format == "named_keypoints":
        kps, visibility = normalize_named_keypoints(target_entry)
        return is_complete(
            kps,
            config.required_indices,
            visibility=visibility,
            annotation_format=config.annotation_format,
        )
    kps = normalize_serialized_keypoints(target_entry.get("kps", []))
    return is_complete(kps, config.required_indices)


def first_missing_keypoint(
    kps: list[list[float | None]],
    editable_indices: tuple[int, ...],
    visibility: list[int] | None = None,
) -> int:
    """Return the first missing editable keypoint, or the first editable index."""
    arr = to_numpy(kps)
    for idx in editable_indices:
        if visibility is not None and idx < len(visibility) and visibility[idx] == 3:
            continue
        if not is_finite_idx(arr, idx):
            return idx
    return editable_indices[0]


def completed_count(
    source_entries: list[dict[str, Any]],
    target_by_id: dict[str, dict[str, Any]],
    config: AnnotationSessionConfig,
) -> int:
    """Count complete target entries among source entries."""
    return sum(is_completed_id(str(entry["id"]), target_by_id, config) for entry in source_entries)


def save_completed_kp20(
    source_entry: dict[str, Any],
    kps: list[list[float | None]],
    visibility: list[int],
    source_entries: list[dict[str, Any]],
    target_by_id: dict[str, dict[str, Any]],
    target_path: Path,
    config: AnnotationSessionConfig,
    document_metadata: dict[str, Any] | None,
) -> bool:
    """Save a complete KP20 entry and mark it as manually adjusted."""
    if not is_complete(
        kps,
        config.required_indices,
        visibility=visibility,
        annotation_format=config.annotation_format,
    ):
        return False
    image_id = str(source_entry["id"])
    existing_entry = dict(target_by_id.get(image_id, {}))
    if config.annotation_format == "named_keypoints":
        existing_entry.update(named_annotation_entry(source_entry, existing_entry, kps, visibility, config))
    else:
        existing_entry.update({
            "id": image_id,
            "metric": source_entry.get("metric"),
            "kps": [[float(point[0]), float(point[1])] for point in kps[:COURT_KP20_COUNT]],
            config.manual_adjusted_field: True,
        })
    target_by_id[image_id] = existing_entry
    write_annotation_entries_atomic(
        target_path,
        ordered_target_entries(source_entries, target_by_id),
        config,
        document_metadata,
    )
    print(f"[court_kp20_annotation] saved {config.keypoint_format}: {image_id}")
    return True


def named_annotation_entry(
    source_entry: dict[str, Any],
    existing_entry: dict[str, Any],
    kps: list[list[float | None]],
    visibility: list[int],
    config: AnnotationSessionConfig,
) -> dict[str, Any]:
    """Build an updated named-keypoint annotation entry."""
    output = dict(source_entry)
    output.update(existing_entry)
    output["id"] = str(source_entry["id"])
    for key in ("image_path", "width", "height", "split", "source"):
        if key in source_entry:
            output[key] = source_entry[key]
    output["annotation_status"] = "completed"
    output[config.manual_adjusted_field] = True
    output["keypoint_format"] = config.keypoint_format
    output["labeled_keypoint_indices"] = list(labeled_keypoint_indices(config.keypoint_format))
    output["is_yastrebksv_kp15"] = bool(source_entry.get("is_yastrebksv_kp15", False))
    source = dict(output.get("source", {}))
    for key in ("dataset", "keypoint_format", "labeled_keypoint_indices"):
        source.pop(key, None)
    if "type" not in source and source.get("source_url"):
        source["type"] = "youtube"
    output["source"] = source
    output["keypoints"] = [
        {
            "index": idx,
            "name": COURT_KP_NAMES[idx],
            "x": optional_point_value(kps[idx][0]),
            "y": optional_point_value(kps[idx][1]),
            "visibility": normalize_visibility(visibility[idx] if idx < len(visibility) else 0),
        }
        for idx in range(COURT_KP20_COUNT)
    ]
    annotation = dict(output.get("annotation", {}))
    annotation["updated_at"] = datetime.now(UTC).isoformat()
    if annotation.get("created_at") is None:
        annotation["created_at"] = annotation["updated_at"]
    output["annotation"] = annotation
    return output


def labeled_keypoint_indices(keypoint_format: str) -> tuple[int, ...]:
    """Return labeled keypoint indices for a dataset variant."""
    if keypoint_format == "kp15":
        return tuple(range(NET_CENTER_IDX + 1))
    if keypoint_format == "kp20":
        return tuple(range(COURT_KP20_COUNT))
    raise ValueError(f"Unsupported keypoint_format={keypoint_format!r}.")


def filter_source_entries(
    entries: list[dict[str, Any]],
    include_source_types: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Filter entries by provenance type when a filter is configured."""
    if not include_source_types:
        return entries
    allowed = set(include_source_types)
    return [
        entry
        for entry in entries
        if str(entry.get("source", {}).get("type", "")) in allowed
    ]


def optional_point_value(value: float | None) -> float | None:
    """Return a JSON-friendly keypoint coordinate."""
    if value is None:
        return None
    return float(value)


def write_annotation_entries_atomic(
    path: Path,
    entries: list[dict[str, Any]],
    config: AnnotationSessionConfig,
    metadata: dict[str, Any] | None,
) -> None:
    """Write entries in either legacy-list or named-document format."""
    if config.annotation_format != "named_keypoints":
        save_json_atomic(entries, path)
        return

    payload = dict(metadata or {})
    payload["items"] = entries
    save_json_atomic(payload, path)


def ordered_target_entries(
    source_entries: list[dict[str, Any]],
    target_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Order target entries by source order, appending unknown ids at the end."""
    ordered: list[dict[str, Any]] = []
    written: set[str] = set()
    for source_entry in source_entries:
        image_id = str(source_entry["id"])
        target_entry = target_by_id.get(image_id)
        if target_entry is not None:
            ordered.append(target_entry)
            written.add(image_id)
    for image_id in sorted(set(target_by_id) - written):
        ordered.append(target_by_id[image_id])
    return ordered


def find_image(images_dir: Path, image_id: str) -> Path | None:
    """Find an image file for a dataset id."""
    return find_existing_file(images_dir, image_id, (".png", ".jpg", ".jpeg"))


def find_image_for_entry(
    root: Path,
    images_dir: Path,
    entry: dict[str, Any],
    image_id: str,
    config: AnnotationSessionConfig,
) -> Path | None:
    """Find an image path using entry metadata before legacy image lookup."""
    image_path_value = entry.get(config.image_path_key)
    if image_path_value:
        image_path = Path(str(image_path_value))
        if not image_path.is_absolute():
            image_path = root / image_path
        if image_path.exists():
            return image_path
    return find_image(images_dir, image_id)


def find_start_index(
    source_entries: list[dict[str, Any]],
    target_by_id: dict[str, dict[str, Any]],
    config: AnnotationSessionConfig,
) -> int:
    """Find the first index to show."""
    if config.image_id is not None:
        for index, entry in enumerate(source_entries):
            if str(entry["id"]) == config.image_id:
                return index
        raise ValueError(f"image_id={config.image_id!r} was not found in split={config.split!r}.")

    start_index = min(max(config.start_index, 0), max(len(source_entries) - 1, 0))
    if config.start_after_last_completed:
        last_completed_index = find_last_completed_index(source_entries, target_by_id, config)
        if last_completed_index is not None:
            return (last_completed_index + 1) % len(source_entries)

    if not config.skip_completed:
        return start_index
    return advance_index(source_entries, target_by_id, start_index - 1, 1, skip_completed=True, config=config)


def find_last_completed_index(
    source_entries: list[dict[str, Any]],
    target_by_id: dict[str, dict[str, Any]],
    config: AnnotationSessionConfig,
) -> int | None:
    """Return the last completed source index in dataset order."""
    for index in range(len(source_entries) - 1, -1, -1):
        image_id = str(source_entries[index]["id"])
        if is_completed_id(image_id, target_by_id, config):
            return index
    return None


def advance_index(
    source_entries: list[dict[str, Any]],
    target_by_id: dict[str, dict[str, Any]],
    start: int,
    step: int,
    *,
    skip_completed: bool,
    config: AnnotationSessionConfig,
) -> int:
    """Advance to the next or previous entry."""
    if not source_entries:
        return 0
    index = start
    for _ in range(len(source_entries)):
        index = (index + step) % len(source_entries)
        image_id = str(source_entries[index]["id"])
        if not skip_completed or not is_completed_id(image_id, target_by_id, config):
            return index
    return index


def render(
    image_bgr: np.ndarray,
    kps: list[list[float | None]],
    *,
    image_id: str,
    index: int,
    total: int,
    completed_count: int,
    ui: UiState,
    config: AnnotationSessionConfig,
    visibility: list[int],
) -> np.ndarray:
    """Render the current annotation frame."""
    canvas = image_bgr.copy()
    arr = to_numpy(kps)

    for start_idx, end_idx in GROUND_DRAW_PAIRS:
        if start_idx < arr.shape[0] and end_idx < arr.shape[0]:
            draw_line(canvas, arr[start_idx], arr[end_idx], color=(80, 80, 80), thickness=1)

    for start_idx, end_idx in NET_DRAW_PAIRS:
        draw_line(canvas, arr[start_idx], arr[end_idx], color=(0, 220, 255), thickness=2)

    for idx in range(min(GROUND_KEYPOINT_COUNT, arr.shape[0])):
        color = (80, 220, 255) if idx in config.editable_indices else (170, 170, 170)
        draw_point(canvas, arr[idx], color=color, radius=3, thickness=-1)

    draw_point(canvas, arr[NET_CENTER_IDX], color=(255, 180, 40), radius=6, thickness=2)
    for idx in range(min(COURT_KP20_COUNT, arr.shape[0])):
        if idx not in config.editable_indices and idx >= GROUND_KEYPOINT_COUNT:
            continue
        if not is_finite_idx(arr, idx):
            if idx in config.editable_indices:
                continue
            continue
        color = keypoint_color(idx, visibility, config)
        radius = 8 if idx == ui.selected_keypoint else 5
        thickness = 3 if idx == ui.selected_keypoint else 2
        draw_point(canvas, arr[idx], color=color, radius=radius, thickness=thickness)
        label_xy = tuple(np.round(arr[idx] + np.array([8.0, -8.0])).astype(np.int32).tolist())
        cv2.putText(canvas, str(idx), label_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, lineType=cv2.LINE_AA)

    header = (
        f"{index + 1}/{total} {image_id} | completed {completed_count}/{total} | "
        f"selected {ui.selected_keypoint}: {COURT_KP_NAMES[ui.selected_keypoint]} "
        f"vis={visibility_label(visibility[ui.selected_keypoint])} "
        f"mode={visibility_label(ui.input_visibility)}"
    )
    draw_text_box(canvas, [header], origin=(12, 24))

    if ui.show_help:
        help_lines = [
            "click: set selected KP | drag near a KP: adjust nearest point | c: Complete",
            "arrow keys select KP | 1-9 select | any 4 ground KPs auto-fill ground KPs",
            "v visible mode | o toggle occluded mode",
            "x out-of-frame | del clear | g center | n/p move | r reset | h help | q quit",
        ]
        draw_text_box(canvas, help_lines, origin=(12, 54))

    display, scale = resize_for_display(canvas, config.max_display_width, config.max_display_height)
    ui.display_scale = scale
    draw_complete_button(
        display,
        enabled=is_complete(
            kps,
            config.required_indices,
            visibility=visibility,
            annotation_format=config.annotation_format,
        ),
        ui=ui,
    )
    return display


def keypoint_color(
    idx: int,
    visibility: list[int],
    config: AnnotationSessionConfig,
) -> tuple[int, int, int]:
    """Return a display color for a keypoint."""
    if idx not in config.editable_indices:
        return (170, 170, 170)
    state = visibility[idx] if idx < len(visibility) else 0
    if state == 2:
        return (0, 170, 255)
    if state == 3:
        return (160, 120, 120)
    return (0, 255, 0)


def visibility_label(value: int) -> str:
    """Return a compact visibility label."""
    return {
        0: "not_labeled",
        1: "visible",
        2: "occluded",
        3: "out_of_frame",
    }.get(value, "unknown")


def draw_point(
    canvas: np.ndarray,
    point: np.ndarray,
    *,
    color: tuple[int, int, int],
    radius: int,
    thickness: int,
) -> None:
    """Draw one finite point."""
    if not np.isfinite(point).all():
        return
    xy = tuple(np.round(point).astype(np.int32).tolist())
    cv2.circle(canvas, xy, radius, color, thickness, lineType=cv2.LINE_AA)


def draw_line(
    canvas: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw one finite line."""
    if not (np.isfinite(start).all() and np.isfinite(end).all()):
        return
    cv2.line(
        canvas,
        tuple(np.round(start).astype(np.int32).tolist()),
        tuple(np.round(end).astype(np.int32).tolist()),
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_complete_button(display: np.ndarray, *, enabled: bool, ui: UiState) -> None:
    """Draw the Complete button on the display image."""
    height, width = display.shape[:2]
    button_w = 154
    button_h = 44
    margin = 14
    x1 = max(margin, width - button_w - margin)
    y1 = margin
    x2 = min(width - margin, x1 + button_w)
    y2 = min(height - margin, y1 + button_h)
    ui.complete_button_rect = (x1, y1, x2, y2)

    fill = (42, 150, 72) if enabled else (82, 82, 82)
    border = (230, 255, 230) if enabled else (180, 180, 180)
    cv2.rectangle(display, (x1, y1), (x2, y2), fill, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(display, (x1, y1), (x2, y2), border, 2, lineType=cv2.LINE_AA)
    text = "Complete"
    text_size, _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
    text_x = x1 + (x2 - x1 - text_size[0]) // 2
    text_y = y1 + (y2 - y1 + text_size[1]) // 2
    cv2.putText(display, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)


def point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int] | None) -> bool:
    """Return whether a point is inside a rectangle."""
    if rect is None:
        return False
    x, y = point
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_text_box(canvas: np.ndarray, lines: list[str], origin: tuple[int, int]) -> None:
    """Draw a small translucent text box."""
    if not lines:
        return
    x, y = origin
    line_height = 22
    widths = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for line in lines]
    box_w = max(widths) + 18
    box_h = line_height * len(lines) + 10
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x - 8, y - 20), (x - 8 + box_w, y - 20 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, canvas, 0.42, 0.0, dst=canvas)
    for offset, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (x, y + line_height * offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            lineType=cv2.LINE_AA,
        )


def resize_for_display(
    image: np.ndarray,
    max_width: int,
    max_height: int,
) -> tuple[np.ndarray, float]:
    """Resize an image for display."""
    height, width = image.shape[:2]
    scale = min(float(max_width) / float(width), float(max_height) / float(height), 1.0)
    if scale >= 0.999:
        return image, 1.0
    resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def set_mouse_callback(
    window_name: str,
    kps: list[list[float | None]],
    visibility: list[int],
    ui: UiState,
    config: AnnotationSessionConfig,
) -> None:
    """Register mouse controls for click placement and drag adjustment."""

    def on_mouse(event: int, x: int, y: int, flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if point_in_rect((x, y), ui.complete_button_rect):
                ui.pending_action = "complete"
                return
            ui.mouse_down_display_xy = (x, y)
            ui.dragging_idx = nearest_editable_keypoint(kps, ui, (x, y))
            ui.drag_started = False
            return

        if event == cv2.EVENT_MOUSEMOVE and ui.mouse_down_display_xy is not None and ui.dragging_idx is not None:
            if not (flags & cv2.EVENT_FLAG_LBUTTON):
                return
            dx = x - ui.mouse_down_display_xy[0]
            dy = y - ui.mouse_down_display_xy[1]
            if ui.drag_started or float(np.hypot(dx, dy)) >= ui.drag_start_threshold_px:
                ui.drag_started = True
                ui.selected_keypoint = ui.dragging_idx
                kps[ui.dragging_idx] = display_to_original_xy((x, y), ui)
                visibility[ui.dragging_idx] = ui.input_visibility
                ui.auto_filled_indices.discard(ui.dragging_idx)
                maybe_auto_fill_ground_keypoints(kps, visibility, ui, config, changed_idx=ui.dragging_idx)
            return

        if event == cv2.EVENT_LBUTTONUP:
            if ui.mouse_down_display_xy is None:
                return
            if ui.drag_started and ui.dragging_idx is not None:
                kps[ui.dragging_idx] = display_to_original_xy((x, y), ui)
                visibility[ui.dragging_idx] = ui.input_visibility
                ui.auto_filled_indices.discard(ui.dragging_idx)
                maybe_auto_fill_ground_keypoints(kps, visibility, ui, config, changed_idx=ui.dragging_idx)
                ui.selected_keypoint = ui.dragging_idx
            elif not point_in_rect((x, y), ui.complete_button_rect):
                clicked_idx = ui.dragging_idx if ui.input_visibility == 2 else None
                changed_idx = clicked_idx if clicked_idx is not None else ui.selected_keypoint
                ui.selected_keypoint = changed_idx
                if clicked_idx is None:
                    kps[changed_idx] = display_to_original_xy((x, y), ui)
                visibility[changed_idx] = ui.input_visibility
                ui.auto_filled_indices.discard(changed_idx)
                auto_filled = maybe_auto_fill_ground_keypoints(kps, visibility, ui, config, changed_idx=changed_idx)
                if clicked_idx is None:
                    if auto_filled:
                        ui.selected_keypoint = first_missing_keypoint(kps, ui.editable_indices, visibility)
                    else:
                        advance_selected_keypoint(ui)
            ui.mouse_down_display_xy = None
            ui.dragging_idx = None
            ui.drag_started = False

    cv2.setMouseCallback(window_name, on_mouse)


def display_to_original_xy(point: tuple[int, int], ui: UiState) -> list[float]:
    """Convert display coordinates to original image coordinates."""
    x, y = point
    return [float(x) / max(ui.display_scale, 1e-6), float(y) / max(ui.display_scale, 1e-6)]


def nearest_editable_keypoint(
    kps: list[list[float | None]],
    ui: UiState,
    display_xy: tuple[int, int],
) -> int | None:
    """Return the nearest editable keypoint within the drag radius."""
    arr = to_numpy(kps)
    best_idx: int | None = None
    best_dist = float("inf")
    cursor = np.asarray(display_xy, dtype=np.float32)
    for idx in ui.editable_indices:
        if not is_finite_idx(arr, idx):
            continue
        display_point = arr[idx] * float(ui.display_scale)
        dist = float(np.linalg.norm(display_point - cursor))
        if dist < best_dist:
            best_idx = idx
            best_dist = dist
    if best_idx is not None and best_dist <= ui.drag_radius_px:
        return best_idx
    return None


def advance_selected_keypoint(ui: UiState, *, step: int = 1) -> None:
    """Advance selected keypoint through the editable order."""
    selected_pos = ui.editable_indices.index(ui.selected_keypoint)
    next_pos = (selected_pos + step) % len(ui.editable_indices)
    ui.selected_keypoint = ui.editable_indices[next_pos]


def _number_keys_for_editable_indices(editable_indices: tuple[int, ...]) -> set[int]:
    max_numeric_keys = min(len(editable_indices), 9)
    return {ord(str(index + 1)) for index in range(max_numeric_keys)}
