"""Import the original-frame Meiji event labels and camera-relative hints."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from numpy.typing import NDArray

from src.tennis_scene.ball_event_schema import EVENT_STATUS, EVENT_SUBTYPES
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.schema import SceneResult
from src.utils.geometry.triangulation import PinholeCamera

ANNOTATION_SCHEMA_FILENAME = "meiji_event_labels.v1.schema.json"
ANNOTATION_CLIP_ID = re.compile(r"video_[0-9]{3}/clip_[0-9]{3}")
_HINT_REFERENCE = re.compile(
    r"(?:player_hint|near\s*/\s*far|基準視点)[^。\n]{0,64}?cam\s*([0-9]+)",
    re.IGNORECASE,
)
_CAMERA_DIGEST = re.compile(
    r"\b(?:clip_[0-9]{3}_)?(cam[0-3])(?:\.mp4)?\s*=\s*([0-9a-f]{64})(?![0-9a-f])"
)
_LABELER_NAMES = frozenset({
    "ChatGPT",
    "ChatGPT (visual multiview review)",
    "ChatGPT / four-view visual annotation",
    "ChatGPT GPT-6 Astra Pro / multiview visual annotation",
    "ChatGPT GPT-6 Astra Pro / multiview visual review",
    "ChatGPT GPT-6 Astra Pro visual review",
    "ChatGPT multiview visual annotation",
    "ChatGPT visual annotation",
    "ChatGPT visual annotation (4-camera review)",
    "ChatGPT visual multiview review",
    "ChatGPT visual review",
    "ChatGPT-GPT-6-Astra-Pro",
    "ChatGPT（4視点の元映像によるイベント注釈）",
})


def annotation_path(directory: Path, clip_id: str) -> Path:
    if ANNOTATION_CLIP_ID.fullmatch(clip_id) is None:
        raise ValueError(f"Annotation method requires a video_NNN/clip_NNN clip ID: {clip_id!r}")
    return directory / f"{clip_id.split('/')[1]}_ball_events.json"


def _read_pinned_json(asset: Mapping[str, str]) -> dict[str, Any]:
    path = Path(asset["path"])
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != asset["sha256"]:
        raise ValueError(f"Event annotation asset changed: {path}")
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError(f"Expected an event annotation JSON object: {path}")
    return document


@dataclass(frozen=True)
class BallAnnotations:
    events: list[dict[str, Any]]
    coverage: NDArray[np.bool_]
    columns: dict[str, NDArray[Any]]
    provenance: dict[str, Any]
    evidence: dict[str, Any]


def read_ball_annotations(
    scene: SceneResult, source: ClipSource, *,
    annotation: Mapping[str, str], schema: Mapping[str, str],
) -> BallAnnotations:
    """Validate structure and scene correspondence before encoding any labels."""
    document = _read_pinned_json(annotation)
    schema_document = _read_pinned_json(schema)
    Draft202012Validator.check_schema(schema_document)
    error = next(Draft202012Validator(schema_document).iter_errors(document), None)
    if error is not None:
        raise ValueError(f"Invalid annotation {source.clip_id} {error.json_path}: {error.message}")
    identity = (document["clip_id"], document["fps"], document["frames"])
    if identity != (source.clip_id, scene.fps, scene.num_frames):
        raise ValueError(f"Annotation clip_id/fps/frames disagree with the scene: {source.clip_id}")
    if document["schema"] != "meiji_event_labels.v1" or document["benchmark"] != "multiview-v2":
        raise ValueError("Unsupported event annotation schema or benchmark")

    notes = document.get("notes", "")
    camera_hashes = {video.camera_id: video.sha256 for video in source.videos}
    if any(camera_hashes.get(camera) != digest for camera, digest in _CAMERA_DIGEST.findall(notes)):
        raise ValueError(f"Annotation video SHA-256 disagrees with the scene: {source.clip_id}")
    events = sorted(document["events"], key=lambda row: (row["frame"], row["kind"]))
    if len({event["id"] for event in events}) != len(events):
        raise ValueError("Annotation event IDs must be unique")
    if any(
        not 0 <= event["frame"] < min(scene.num_frames, 2**31)
        or not 0 <= event["frame_uncertainty"] <= 255
        for event in events
    ):
        raise ValueError("Annotation event frame or frame_uncertainty is out of range")
    if any(not set(event.get("cameras", ())) <= camera_hashes.keys() for event in events):
        raise ValueError("Annotation evidence refers to a camera outside the scene")

    coverage: NDArray[np.bool_] = np.ones(scene.num_frames, dtype=np.bool_)
    intervals = document["unlabelable"]
    for interval in intervals:
        start, end = int(interval["start"]), int(interval["end"])
        if not 0 <= start <= end < scene.num_frames:
            raise ValueError("Annotation unlabelable interval is outside the scene")
        coverage[start : end + 1] = False

    hints = [event.get("player_hint") if event["kind"] == "shot" else None for event in events]
    hint_sides: dict[str | None, int] = {None: 0}
    reference: dict[str, Any] = {}
    if any(hint is not None for hint in hints):
        if set(_HINT_REFERENCE.findall(notes)) - {"0"}:
            raise ValueError("Annotation player_hint uses a reference other than cam0")
        center_y = _cam0_court_y(scene)
        near = -1 if center_y < 0 else 1
        hint_sides.update({"near": near, "far": -near})
        reference = {"camera_id": "cam0", "camera_center_y_m": center_y, "near": near, "far": -near}

    labeler = document["labeler"]
    return BallAnnotations(
        events,
        coverage,
        {
            "ball_event_status": np.asarray([EVENT_STATUS.index(e["status"]) for e in events], dtype=np.uint8),
            "ball_event_frame_uncertainty": np.asarray([e["frame_uncertainty"] for e in events], dtype=np.uint8),
            "ball_event_in_point": np.asarray([e["in_point"] for e in events], dtype=np.bool_),
            "ball_event_subtype": np.asarray([EVENT_SUBTYPES.index(e.get("subtype")) for e in events], dtype=np.uint8),
            "ball_event_side": np.asarray([hint_sides[hint] for hint in hints], dtype=np.int8),
        },
        {
            "annotation_sha256": annotation["sha256"],
            "annotation_schema_sha256": schema["sha256"],
            "labeler": "chatgpt" if labeler in _LABELER_NAMES else labeler.strip(),
            "labeler_original": labeler,
        },
        {"events": events, "notes": notes, "unlabelable": intervals, "player_hint_reference": reference},
    )


def _cam0_court_y(scene: SceneResult) -> float:
    reference = scene.metadata.get("court_reference")
    if not isinstance(reference, dict):
        raise ValueError("Annotation player_hint requires a calibrated court reference")
    ids, fits = reference.get("camera_ids"), reference.get("camera_fits")
    if (
        not isinstance(ids, list) or not isinstance(fits, list)
        or len(ids) != len(fits) or ids.count("cam0") != 1
    ):
        raise ValueError("Annotation player_hint requires exactly one cam0 court calibration")
    fit = fits[ids.index("cam0")]
    camera = PinholeCamera(
        "cam0",
        np.asarray(fit["K"], dtype=np.float64),
        np.asarray(fit["R"], dtype=np.float64),
        np.asarray(fit["t"], dtype=np.float64),
    )
    y = float(camera.center[1])
    if abs(y) <= 1e-6:
        raise ValueError("cam0 center lies on the net plane and has no near court side")
    if "camera_center_court_m" in fit and not np.allclose(
        camera.center, fit["camera_center_court_m"], rtol=0, atol=1e-6,
    ):
        raise ValueError("cam0 camera center disagrees with its court calibration")
    return y
