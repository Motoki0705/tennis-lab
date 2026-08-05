"""OpenCV review session for predicted YouTube ball candidates."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from src.utils.io import load_json, save_json_atomic, utc_now_iso

JSONDict = dict[str, Any]
VISIBLE_STATES = {"visible", "occluded"}
NULL_STATES = {"absent", "out_of_frame"}
SUPPORTED_STATES = VISIBLE_STATES | NULL_STATES
LEFT_KEYS = {81, 65361, 2424832}
RIGHT_KEYS = {83, 65363, 2555904}


@dataclass(frozen=True)
class ZoomConfig:
    """Prediction-centered zoom settings."""

    key: str = "z"
    factor: float = 4.0


@dataclass(frozen=True)
class FinalizeConfig:
    """Candidate finalization settings."""

    key: str = "f"
    overwrite: bool = False


@dataclass(frozen=True)
class BallAnnotationSessionConfig:
    """Runtime settings for candidate annotation."""

    root: Path
    video_id: str
    candidate_id: str | None
    start_index: int | None
    window_name: str
    max_display_width: int
    max_display_height: int
    point_radius: int
    point_thickness: int
    max_balls_per_frame: int
    zoom: ZoomConfig
    finalize: FinalizeConfig


@dataclass(frozen=True)
class ViewTransform:
    """Mapping between original-image and displayed coordinates."""

    origin_x: int
    origin_y: int
    scale: float


@dataclass
class UiState:
    """Mutable state for one candidate."""

    document: JSONDict
    document_path: Path
    current_index: int
    selected_ball_index: int = 0
    zoom_enabled: bool = False
    transform: ViewTransform = ViewTransform(0, 0, 1.0)
    dirty: bool = False


def run_annotation_session(config: BallAnnotationSessionConfig) -> int:
    """Review predicted candidates for one video and finalize completed clips."""
    _validate_config(config)
    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
    print(
        "[annotate_youtube_ball] keys: click/drag=select/move, shift+click=add, "
        "tab=next ball, a/delete=remove, v=visible, o=occluded, "
        "x=out-of-frame, t=role, r=reset, c=complete, "
        f"{config.zoom.key}=zoom, {config.finalize.key}=finalize, "
        "arrows=navigate, q=quit"
    )
    requested_candidate = config.candidate_id
    requested_start_index = config.start_index
    try:
        while True:
            candidate_path = _next_candidate_path(
                root=config.root,
                video_id=config.video_id,
                candidate_id=requested_candidate,
            )
            if candidate_path is None:
                print(
                    f"[annotate_youtube_ball] no pending candidates for {config.video_id}"
                )
                return 0
            requested_candidate = None
            result = _run_candidate(
                candidate_path,
                config,
                start_index=requested_start_index,
            )
            requested_start_index = None
            if result == "quit":
                return 0
    finally:
        cv2.destroyWindow(config.window_name)


def _run_candidate(
    document_path: Path,
    config: BallAnnotationSessionConfig,
    *,
    start_index: int | None,
) -> str:
    document = load_json(document_path)
    _normalize_document_schema(document)
    if document.get("status") not in {"pseudo_labeled", "annotating"}:
        raise ValueError(
            f"Candidate {document.get('clip_id')} is not ready for annotation; "
            f"status={document.get('status')!r}. Run mode=predict first."
        )
    frames = document.get("frames", [])
    if not frames:
        raise ValueError(f"No frames in candidate: {document_path}")
    document["status"] = "annotating"
    document["max_balls_per_frame"] = config.max_balls_per_frame
    stored_cursor = int(document.get("cursor_index", 0))
    current_index = stored_cursor if start_index is None else int(start_index)
    state = UiState(
        document=document,
        document_path=document_path,
        current_index=min(max(current_index, 0), len(frames) - 1),
    )
    cv2.setMouseCallback(config.window_name, _mouse_callback, state)
    _save_state(state)
    print(
        f"[annotate_youtube_ball] candidate={document['clip_id']} "
        f"completed={_completed_count(frames)}/{len(frames)}"
    )

    while True:
        frame = frames[state.current_index]
        image_path = document_path.parent / str(frame["file_name"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read candidate frame: {image_path}")
        canvas, transform = _render(image, frame, state, config)
        state.transform = transform
        cv2.imshow(config.window_name, canvas)
        key = cv2.waitKeyEx(20)

        if state.dirty:
            _save_state(state)
        if key < 0:
            continue
        if key in RIGHT_KEYS:
            _move_cursor(state, 1)
            continue
        if key in LEFT_KEYS:
            _move_cursor(state, -1)
            continue
        ascii_key = key & 0xFF
        character = chr(ascii_key).lower()
        if ascii_key == 9:
            _cycle_selected_ball(state)
            continue
        if character == "q":
            _save_state(state)
            return "quit"
        if character == "c":
            _complete_current_frame(state)
            continue
        if character == config.zoom.key.lower():
            if _zoom_point(frame, state.selected_ball_index) is None:
                print(
                    "[annotate_youtube_ball] no ball or model prediction available for zoom"
                )
            else:
                state.zoom_enabled = not state.zoom_enabled
            continue
        if character == config.finalize.key.lower():
            try:
                clip_id = finalize_candidate(state.document_path, config)
            except (FileExistsError, ValueError) as error:
                print(f"[annotate_youtube_ball] finalize failed: {error}")
                continue
            print(f"[annotate_youtube_ball] finalized {clip_id}")
            return "next"
        if character == "s":
            _save_state(state)
            continue
        if character == "v":
            _set_selected_ball_state(state, "visible")
        elif character == "o":
            _set_selected_ball_state(state, "occluded")
        elif character == "a" or ascii_key in {8, 127}:
            _remove_selected_ball(state)
        elif character == "x":
            _set_selected_ball_state(state, "out_of_frame")
        elif character == "t":
            _cycle_selected_ball_role(state)
        elif character == "r":
            _reset_to_predictions(frame)
            state.selected_ball_index = 0
        else:
            continue
        state.dirty = True


def _mouse_callback(
    event: int,
    x: int,
    y: int,
    flags: int,
    userdata: Any | None,
) -> None:
    if not isinstance(userdata, UiState):
        raise TypeError("Annotation mouse callback requires UiState userdata.")
    is_drag = event == cv2.EVENT_MOUSEMOVE and bool(flags & cv2.EVENT_FLAG_LBUTTON)
    if event != cv2.EVENT_LBUTTONDOWN and not is_drag:
        return
    state = userdata
    frame = state.document["frames"][state.current_index]
    transform = state.transform
    width = max(int(state.document["width"]), 1)
    height = max(int(state.document["height"]), 1)
    original_x = float(
        np.clip(
            transform.origin_x + x / max(transform.scale, 1e-8),
            0,
            width - 1,
        )
    )
    original_y = float(
        np.clip(
            transform.origin_y + y / max(transform.scale, 1e-8),
            0,
            height - 1,
        )
    )
    balls = frame.setdefault("balls", [])
    add_ball = bool(flags & cv2.EVENT_FLAG_SHIFTKEY) or not balls
    if event == cv2.EVENT_LBUTTONDOWN and add_ball:
        if len(balls) >= state.document.get("max_balls_per_frame", 16):
            print("[annotate_youtube_ball] maximum balls per frame reached")
            return
        balls.append(
            {
                "ball_id": _next_ball_id(balls),
                "prediction_id": None,
                "x": original_x,
                "y": original_y,
                "state": "visible",
                "role": "target",
                "confidence": None,
                "label_source": "manual",
            }
        )
        state.selected_ball_index = len(balls) - 1
    elif event == cv2.EVENT_LBUTTONDOWN:
        state.selected_ball_index = _nearest_ball_index(
            balls,
            point=(original_x, original_y),
        )
    if not balls:
        return
    ball = balls[state.selected_ball_index]
    ball["x"] = original_x
    ball["y"] = original_y
    if ball.get("state") not in VISIBLE_STATES:
        ball["state"] = "visible"
    ball["confidence"] = None
    ball["label_source"] = "manual"
    frame["review_status"] = "pending"
    state.dirty = True


def _render(
    image: np.ndarray,
    frame: JSONDict,
    state: UiState,
    config: BallAnnotationSessionConfig,
) -> tuple[np.ndarray, ViewTransform]:
    crop, origin_x, origin_y = _view_crop(
        image,
        prediction=_zoom_point(frame, state.selected_ball_index),
        zoom_enabled=state.zoom_enabled,
        zoom_factor=config.zoom.factor,
    )
    scale = min(
        config.max_display_width / max(crop.shape[1], 1),
        config.max_display_height / max(crop.shape[0], 1),
    )
    if not state.zoom_enabled:
        scale = min(scale, 1.0)
    canvas = cv2.resize(
        crop,
        (
            max(1, int(round(crop.shape[1] * scale))),
            max(1, int(round(crop.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    transform = ViewTransform(origin_x=origin_x, origin_y=origin_y, scale=scale)

    predictions = _prediction_candidates(frame)
    for prediction_index, prediction in enumerate(predictions, start=1):
        _draw_cross(
            canvas,
            _to_display((float(prediction["x"]), float(prediction["y"])), transform),
            color=(255, 180, 40),
            size=config.point_radius + 3,
            thickness=config.point_thickness,
        )
        _draw_index(
            canvas,
            _to_display((float(prediction["x"]), float(prediction["y"])), transform),
            prediction_index,
            color=(255, 180, 40),
        )
    balls = frame.get("balls", [])
    for ball_index, ball in enumerate(balls):
        if ball.get("x") is None or ball.get("y") is None:
            continue
        point = _to_display((float(ball["x"]), float(ball["y"])), transform)
        selected = ball_index == state.selected_ball_index
        color = _ball_color(ball)
        cv2.circle(
            canvas,
            point,
            config.point_radius + (3 if selected else 0),
            color,
            config.point_thickness + (1 if selected else 0),
            cv2.LINE_AA,
        )
        _draw_index(canvas, point, ball_index + 1, color=color)

    selected_ball = _selected_ball(frame, state.selected_ball_index)

    lines = [
        (
            f"{state.document['clip_id']} frame {state.current_index + 1}/"
            f"{len(state.document['frames'])} source={frame['source_frame_index']}"
        ),
        (
            f"balls={len(balls)} selected={_selected_ball_label(state, balls)} "
            f"state={None if selected_ball is None else selected_ball.get('state')} "
            f"role={None if selected_ball is None else selected_ball.get('role')} "
            f"review={frame.get('review_status')} "
            f"zoom={'on' if state.zoom_enabled else 'off'}"
        ),
        "blue cross=model | circle=label | shift+click add | tab select | drag move",
        (
            f"v visible | o occluded | a/delete remove | x out | t role | r reset | "
            f"{config.zoom.key} zoom | {config.finalize.key} finalize | q quit"
        ),
    ]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 92), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0.0)
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (12, 20 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas, transform


def _view_crop(
    image: np.ndarray,
    *,
    prediction: tuple[float, float] | None,
    zoom_enabled: bool,
    zoom_factor: float,
) -> tuple[np.ndarray, int, int]:
    if not zoom_enabled or prediction is None:
        return image, 0, 0
    height, width = image.shape[:2]
    crop_width = max(32, min(width, int(round(width / zoom_factor))))
    crop_height = max(32, min(height, int(round(height / zoom_factor))))
    center_x = int(round(prediction[0]))
    center_y = int(round(prediction[1]))
    x0 = min(max(center_x - crop_width // 2, 0), width - crop_width)
    y0 = min(max(center_y - crop_height // 2, 0), height - crop_height)
    return image[y0 : y0 + crop_height, x0 : x0 + crop_width], x0, y0


def finalize_candidate(
    document_path: Path,
    config: BallAnnotationSessionConfig,
) -> str:
    """Move a fully reviewed staging candidate into the final clip dataset."""
    document = load_json(document_path)
    incomplete = [
        str(frame["frame_id"])
        for frame in document["frames"]
        if frame.get("review_status") != "completed"
    ]
    if incomplete:
        preview = ", ".join(incomplete[:5])
        suffix = "" if len(incomplete) <= 5 else f", ... ({len(incomplete)} total)"
        raise ValueError(f"incomplete frames: {preview}{suffix}")
    errors = [
        f"{frame['frame_id']}: {error}"
        for frame in document["frames"]
        if (error := frame_completion_error(frame)) is not None
    ]
    if errors:
        raise ValueError("; ".join(errors[:5]))

    candidate_dir = document_path.parent
    target_dir = config.root / "frames" / config.video_id / candidate_dir.name
    if target_dir.exists() and not config.finalize.overwrite:
        raise FileExistsError(f"final clip exists: {target_dir}")
    _write_label_csv(candidate_dir / "Label.csv", document["frames"])
    clip_document = {
        "schema_name": "ball_youtube_clip_v2",
        "clip_id": document["clip_id"],
        "video_id": document["video_id"],
        "split": document["split"],
        "annotation_status": "completed",
        "start_frame_index": document["start_frame_index"],
        "end_frame_index": document["end_frame_index"],
        "frame_count": document["frame_count"],
        "fps": document["fps"],
        "width": document["width"],
        "height": document["height"],
        "frames": document["frames"],
        "source": document["source"],
        "prediction": document.get("prediction"),
        "annotation": {
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        },
    }
    save_json_atomic(clip_document, candidate_dir / "clip.json")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    candidate_dir.replace(target_dir)
    _register_final_clip(config.root, clip_document, target_dir)
    return str(document["clip_id"])


def frame_completion_error(frame: JSONDict) -> str | None:
    """Validate one reviewed frame."""
    balls = frame.get("balls")
    if balls is None:
        balls = [frame["ball"]] if "ball" in frame else []
    if not isinstance(balls, list):
        return "balls must be a list"
    seen_ids: set[str] = set()
    for index, ball in enumerate(balls):
        error = _ball_completion_error(ball)
        if error is not None:
            return f"ball[{index}]: {error}"
        ball_id = str(ball.get("ball_id", ""))
        if not ball_id:
            return f"ball[{index}]: ball_id is required"
        if ball_id in seen_ids:
            return f"duplicate ball_id {ball_id!r}"
        seen_ids.add(ball_id)
    return None


def _ball_completion_error(ball: JSONDict) -> str | None:
    state = str(ball.get("state", "unreviewed"))
    if state not in SUPPORTED_STATES:
        return f"unsupported ball state {state!r}"
    x = ball.get("x")
    y = ball.get("y")
    if state in VISIBLE_STATES:
        if x is None or y is None:
            return f"state={state} requires x/y coordinates"
        if not np.isfinite(float(x)) or not np.isfinite(float(y)):
            return "x/y coordinates must be finite"
    elif x is not None or y is not None:
        return f"state={state} requires null x/y"
    role = str(ball.get("role", "target"))
    if role not in {"target", "secondary", "distractor"}:
        return f"unsupported role {role!r}"
    return None


def _complete_current_frame(state: UiState) -> None:
    frame = state.document["frames"][state.current_index]
    error = frame_completion_error(frame)
    if error is not None:
        print(f"[annotate_youtube_ball] cannot complete {frame['frame_id']}: {error}")
        return
    for ball in frame.get("balls", []):
        if ball.get("label_source") == "pseudo":
            ball["label_source"] = "human_approved_pseudo"
    frame["review_status"] = "completed"
    frame["reviewed_at"] = utc_now_iso()
    state.dirty = True
    _save_state(state)
    _move_cursor(state, 1)


def _set_selected_ball_state(state: UiState, ball_state: str) -> None:
    frame = state.document["frames"][state.current_index]
    ball = _selected_ball(frame, state.selected_ball_index)
    if ball is None:
        return
    ball["state"] = ball_state
    if ball_state in NULL_STATES:
        ball["x"] = None
        ball["y"] = None
    ball["confidence"] = None
    ball["label_source"] = "manual"
    frame["review_status"] = "pending"


def _reset_to_predictions(frame: JSONDict) -> None:
    frame["balls"] = [
        {
            "ball_id": f"b{index:03d}",
            "prediction_id": candidate.get("prediction_id"),
            "x": candidate.get("x"),
            "y": candidate.get("y"),
            "state": "visible",
            "role": "target",
            "confidence": candidate.get("confidence"),
            "label_source": "pseudo",
        }
        for index, candidate in enumerate(_prediction_candidates(frame), start=1)
    ]
    frame["rejected_prediction_ids"] = []
    frame["review_status"] = "pending"


def _remove_selected_ball(state: UiState) -> None:
    frame = state.document["frames"][state.current_index]
    balls = frame.get("balls", [])
    if not balls:
        return
    index = min(state.selected_ball_index, len(balls) - 1)
    removed = balls.pop(index)
    prediction_id = removed.get("prediction_id")
    if prediction_id is not None:
        rejected = frame.setdefault("rejected_prediction_ids", [])
        if prediction_id not in rejected:
            rejected.append(prediction_id)
    state.selected_ball_index = min(index, max(len(balls) - 1, 0))
    frame["review_status"] = "pending"


def _cycle_selected_ball(state: UiState) -> None:
    frame = state.document["frames"][state.current_index]
    balls = frame.get("balls", [])
    if balls:
        state.selected_ball_index = (state.selected_ball_index + 1) % len(balls)


def _cycle_selected_ball_role(state: UiState) -> None:
    frame = state.document["frames"][state.current_index]
    ball = _selected_ball(frame, state.selected_ball_index)
    if ball is None:
        return
    roles = ("target", "secondary", "distractor")
    current = str(ball.get("role", "target"))
    ball["role"] = (
        roles[(roles.index(current) + 1) % len(roles)] if current in roles else "target"
    )
    ball["label_source"] = "manual"
    frame["review_status"] = "pending"


def _register_final_clip(root: Path, clip: JSONDict, target_dir: Path) -> None:
    split = str(clip["split"])
    annotation_path = root / "annotations" / f"{split}.json"
    payload = cast(
        JSONDict,
        (
            load_json(annotation_path)
            if annotation_path.exists()
            else {"schema_name": "ball_youtube_dataset_v1", "split": split, "items": []}
        ),
    )
    item = {
        "clip_id": clip["clip_id"],
        "clip_path": str(target_dir.relative_to(root)),
        "dataset_entry": str(target_dir.relative_to(root)),
        "annotation_path": str((target_dir / "clip.json").relative_to(root)),
        "label_csv": str((target_dir / "Label.csv").relative_to(root)),
        "video_id": clip["video_id"],
        "frame_count": clip["frame_count"],
        "start_frame_index": clip["start_frame_index"],
        "end_frame_index": clip["end_frame_index"],
        "annotation_status": "completed",
        "source": {"type": "youtube"},
    }
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise TypeError(f"Annotation registry items must be a list: {annotation_path}")
    items_by_id = {str(existing["clip_id"]): existing for existing in items}
    items_by_id[str(clip["clip_id"])] = item
    payload["items"] = list(items_by_id.values())
    save_json_atomic(payload, annotation_path)
    entries = sorted(
        str(existing["dataset_entry"])
        for existing in cast(list[JSONDict], payload["items"])
    )
    _write_text_atomic(
        root / "annotations" / f"{split}.txt",
        "".join(f"{entry}\n" for entry in entries),
    )


def _write_label_csv(path: Path, frames: list[JSONDict]) -> None:
    fields = [
        "file name",
        "instance id",
        "prediction id",
        "visibility",
        "x-coordinate",
        "y-coordinate",
        "ball state",
        "role",
        "label source",
        "source frame index",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in frames:
            balls = frame.get("balls", [])
            if not balls:
                writer.writerow(
                    {
                        "file name": frame["file_name"],
                        "instance id": "",
                        "prediction id": "",
                        "visibility": 0,
                        "x-coordinate": 0.0,
                        "y-coordinate": 0.0,
                        "ball state": "absent",
                        "role": "",
                        "label source": "manual",
                        "source frame index": frame["source_frame_index"],
                    }
                )
                continue
            for ball in balls:
                visible = str(ball["state"]) in VISIBLE_STATES
                writer.writerow(
                    {
                        "file name": frame["file_name"],
                        "instance id": ball["ball_id"],
                        "prediction id": ball.get("prediction_id") or "",
                        "visibility": 1 if visible else 0,
                        "x-coordinate": float(ball["x"]) if visible else 0.0,
                        "y-coordinate": float(ball["y"]) if visible else 0.0,
                        "ball state": ball["state"],
                        "role": ball.get("role", "target"),
                        "label source": ball.get("label_source", "manual"),
                        "source frame index": frame["source_frame_index"],
                    }
                )


def _next_candidate_path(
    *,
    root: Path,
    video_id: str,
    candidate_id: str | None,
) -> Path | None:
    staging_dir = root / "staging" / video_id
    paths = sorted(staging_dir.glob("clip_*/candidate.json"))
    if candidate_id is None:
        return paths[0] if paths else None
    for path in paths:
        document = load_json(path)
        if candidate_id in {path.parent.name, str(document.get("clip_id"))}:
            return path
    raise FileNotFoundError(
        f"candidate_id={candidate_id!r} was not found under {staging_dir}"
    )


def _zoom_point(
    frame: JSONDict,
    selected_ball_index: int,
) -> tuple[float, float] | None:
    ball = _selected_ball(frame, selected_ball_index)
    if ball is not None:
        point = _finite_point(ball)
        if point is not None:
            return point
    candidates = _prediction_candidates(frame)
    return _finite_point(candidates[0]) if candidates else None


def _finite_point(value: JSONDict) -> tuple[float, float] | None:
    x = value.get("x")
    y = value.get("y")
    if x is None or y is None:
        return None
    if not np.isfinite(float(x)) or not np.isfinite(float(y)):
        return None
    return float(x), float(y)


def _prediction_candidates(frame: JSONDict) -> list[JSONDict]:
    predictions = frame.get("predictions")
    if isinstance(predictions, dict):
        candidates = predictions.get("candidates", [])
        if isinstance(candidates, list):
            return candidates
    prediction = frame.get("prediction")
    return (
        [prediction]
        if isinstance(prediction, dict) and _finite_point(prediction)
        else []
    )


def _selected_ball(frame: JSONDict, selected_ball_index: int) -> JSONDict | None:
    balls = frame.get("balls", [])
    if not balls:
        return None
    return cast(JSONDict, balls[min(max(selected_ball_index, 0), len(balls) - 1)])


def _nearest_ball_index(
    balls: list[JSONDict],
    *,
    point: tuple[float, float],
) -> int:
    distances = []
    for ball in balls:
        ball_point = _finite_point(ball)
        distance = (
            float("inf")
            if ball_point is None
            else float(np.hypot(ball_point[0] - point[0], ball_point[1] - point[1]))
        )
        distances.append(distance)
    return int(np.argmin(distances)) if distances else 0


def _next_ball_id(balls: list[JSONDict]) -> str:
    used = {str(ball.get("ball_id")) for ball in balls}
    number = 1
    while f"b{number:03d}" in used:
        number += 1
    return f"b{number:03d}"


def _selected_ball_label(state: UiState, balls: list[JSONDict]) -> str:
    if not balls:
        return "none"
    return f"{min(state.selected_ball_index, len(balls) - 1) + 1}/{len(balls)}"


def _ball_color(ball: JSONDict) -> tuple[int, int, int]:
    if ball.get("role") == "distractor":
        return 160, 160, 160
    if ball.get("state") == "occluded":
        return 0, 165, 255
    if ball.get("role") == "secondary":
        return 80, 220, 255
    return 80, 255, 80


def _draw_index(
    image: np.ndarray,
    point: tuple[int, int],
    index: int,
    *,
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        image,
        str(index),
        (point[0] + 8, point[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _normalize_document_schema(document: JSONDict) -> None:
    """Upgrade single-ball candidate documents in memory."""
    for frame in document.get("frames", []):
        if "predictions" not in frame:
            legacy_prediction = frame.get("prediction")
            candidates = []
            if isinstance(legacy_prediction, dict) and _finite_point(legacy_prediction):
                candidates.append(
                    {
                        "prediction_id": "p001",
                        "rank": 1,
                        "x": legacy_prediction["x"],
                        "y": legacy_prediction["y"],
                        "confidence": legacy_prediction.get("confidence"),
                    }
                )
            frame["predictions"] = {
                "frame_id": frame.get("frame_id"),
                "method": "legacy_top1",
                "candidates": candidates,
                "prediction_count": 1 if candidates else 0,
            }
        if "balls" not in frame:
            legacy_ball = frame.pop("ball", None)
            if (
                isinstance(legacy_ball, dict)
                and legacy_ball.get("state") in VISIBLE_STATES
                and _finite_point(legacy_ball) is not None
            ):
                frame["balls"] = [
                    {
                        "ball_id": "b001",
                        "prediction_id": "p001"
                        if _prediction_candidates(frame)
                        else None,
                        "x": legacy_ball["x"],
                        "y": legacy_ball["y"],
                        "state": legacy_ball["state"],
                        "role": "target",
                        "confidence": legacy_ball.get("confidence"),
                        "label_source": legacy_ball.get("label_source", "pseudo"),
                    }
                ]
            else:
                frame["balls"] = []
        frame.setdefault("rejected_prediction_ids", [])


def _to_display(
    point: tuple[float, float], transform: ViewTransform
) -> tuple[int, int]:
    return (
        int(round((point[0] - transform.origin_x) * transform.scale)),
        int(round((point[1] - transform.origin_y) * transform.scale)),
    )


def _draw_cross(
    image: np.ndarray,
    point: tuple[int, int],
    *,
    color: tuple[int, int, int],
    size: int,
    thickness: int,
) -> None:
    x, y = point
    cv2.line(image, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def _move_cursor(state: UiState, step: int) -> None:
    state.current_index = min(
        max(state.current_index + step, 0),
        len(state.document["frames"]) - 1,
    )
    state.selected_ball_index = 0
    state.document["cursor_index"] = state.current_index
    state.dirty = True
    _save_state(state)


def _save_state(state: UiState) -> None:
    state.document["cursor_index"] = state.current_index
    save_json_atomic(state.document, state.document_path)
    state.dirty = False


def _completed_count(frames: list[JSONDict]) -> int:
    return sum(frame.get("review_status") == "completed" for frame in frames)


def _format_confidence(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _validate_config(config: BallAnnotationSessionConfig) -> None:
    if len(config.zoom.key) != 1 or len(config.finalize.key) != 1:
        raise ValueError("zoom.key and finalize.key must each be one character.")
    if config.zoom.key.lower() == config.finalize.key.lower():
        raise ValueError("zoom.key and finalize.key must differ.")
    if config.zoom.factor <= 1.0:
        raise ValueError("zoom.factor must be greater than 1.")
    if config.max_balls_per_frame <= 0:
        raise ValueError("max_balls_per_frame must be positive.")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
