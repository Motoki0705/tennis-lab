"""Import the original-frame Meiji event labels and camera-relative hints."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from numpy.typing import NDArray

from src.tennis_scene.ball_event_schema import (
    EVENT_CLASSES,
    EVENT_CLASSES_V4,
    EVENT_STATUS,
    EVENT_SUBTYPES,
    EVENT_SUBTYPES_V4,
)
from src.tennis_scene.pipeline.contracts import ClipSource
from src.tennis_scene.schema import SceneResult
from src.utils.geometry.triangulation import PinholeCamera

ANNOTATION_SCHEMA_FILENAME = "meiji_event_labels.v1.schema.json"
ANNOTATION_SCHEMA_FILENAME_V1 = ANNOTATION_SCHEMA_FILENAME
ANNOTATION_SCHEMA_FILENAME_V2 = "meiji_event_labels.v2.schema.json"
ANNOTATION_CLIP_ID = re.compile(r"video_[0-9]{3}/clip_[0-9]{3}")
MEIJI_V2_MANIFEST_SHA256 = "0d11010fa2a10d73ba4b03ba01f107c90641868fbb8dd86978503cc761d4ac44"
MEIJI_V3_MANIFEST_SHA256 = "790f05016e31ff4eb3fa8fc7d681d1a22ff0540928672e2a274a64543cd2a28a"
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


def annotation_schema_version(directory: Path) -> Literal[1, 2]:
    """Identify the explicit Meiji event annotation schema version in a directory."""
    has_v1 = (directory / ANNOTATION_SCHEMA_FILENAME_V1).is_file()
    has_v2 = (directory / ANNOTATION_SCHEMA_FILENAME_V2).is_file()
    if has_v1 and has_v2:
        raise ValueError(f"Ambiguous event annotation directory has both v1 and v2 schemas: {directory}")
    if has_v2:
        return 2
    if has_v1:
        return 1
    raise FileNotFoundError(
        f"Missing event annotation schema ({ANNOTATION_SCHEMA_FILENAME_V1} or {ANNOTATION_SCHEMA_FILENAME_V2}): {directory}"
    )


def annotation_schema_filename(version: Literal[1, 2]) -> str:
    return ANNOTATION_SCHEMA_FILENAME_V2 if version == 2 else ANNOTATION_SCHEMA_FILENAME_V1


def meiji_v2_manifest_asset(path: Path) -> dict[str, str]:
    """Bind the manifest that supplies the annotated video identities."""
    return {"path": str(path), "sha256": MEIJI_V2_MANIFEST_SHA256}


def meiji_v3_manifest_asset(path: Path) -> dict[str, str]:
    """Bind the multiview-v3 manifest that supplies the pinned media and v1 annotation hashes."""
    return {"path": str(path), "sha256": MEIJI_V3_MANIFEST_SHA256}


def read_meiji_v2_manifest(asset: Mapping[str, str]) -> dict[str, Any]:
    if asset["sha256"] != MEIJI_V2_MANIFEST_SHA256:
        raise ValueError("Event annotation requires the fixed Meiji multiview-v2 manifest SHA-256")
    document = _read_pinned_json(asset)
    if document["benchmark_id"] != "meiji_4cam/multiview-v2":
        raise ValueError("Expected the Meiji multiview-v2 benchmark")
    return document


def read_meiji_v3_manifest(asset: Mapping[str, str]) -> dict[str, Any]:
    if asset["sha256"] != MEIJI_V3_MANIFEST_SHA256:
        raise ValueError("Event annotation v2 requires the fixed Meiji multiview-v3 manifest SHA-256")
    document = _read_pinned_json(asset)
    if document["benchmark_id"] != "meiji_4cam/multiview-v3":
        raise ValueError("Expected the Meiji multiview-v3 benchmark")
    return document


@dataclass(frozen=True)
class BallAnnotations:
    events: list[dict[str, Any]]
    coverage: NDArray[np.bool_]
    columns: dict[str, NDArray[Any]]
    provenance: dict[str, Any]
    evidence: dict[str, Any]
    scene_schema_version: int = 3
    classes: tuple[str, ...] = EVENT_CLASSES


def read_ball_annotations(
    scene: SceneResult, source: ClipSource, *,
    annotation: Mapping[str, str], schema: Mapping[str, str], benchmark_manifest: Mapping[str, str],
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

    schema_name = document["schema"]
    if schema_name == "meiji_event_labels.v1":
        if document["benchmark"] != "multiview-v2":
            raise ValueError("Unsupported event annotation schema or benchmark")
        benchmark = read_meiji_v2_manifest(benchmark_manifest)
        benchmark_label = "multiview-v2"
        scene_version = 3
    elif schema_name == "meiji_event_labels.v2":
        if document["benchmark"] != "multiview-v3":
            raise ValueError("Unsupported event annotation schema or benchmark")
        benchmark = read_meiji_v3_manifest(benchmark_manifest)
        benchmark_label = "multiview-v3"
        scene_version = 4
    else:
        raise ValueError("Unsupported event annotation schema or benchmark")

    camera_hashes = {video.camera_id: video.sha256 for video in source.videos}
    matches = [row for row in benchmark["clips"] if row["clip_id"] == source.clip_id]
    if len(matches) != 1:
        raise ValueError(f"Annotation clip must be present once in the pinned Meiji {benchmark_label} manifest: {source.clip_id}")
    row = matches[0]
    expected_hashes = {camera["camera_id"]: camera["sha256"] for camera in row["cameras"]}
    if camera_hashes != expected_hashes:
        raise ValueError(f"Annotation video SHA-256 or camera set disagrees with the pinned Meiji {benchmark_label} manifest: {source.clip_id}")
    if (source.fps, source.num_frames) != (row["fps"], row["frames"]):
        raise ValueError(f"Annotation timeline disagrees with the pinned Meiji {benchmark_label} manifest: {source.clip_id}")
    if scene_version == 4:
        expected_v1_sha = row["ball_events"]["annotation_sha256"]
        if document.get("source_v1_annotation_sha256") != expected_v1_sha:
            raise ValueError(f"v2 annotation source_v1_annotation_sha256 disagrees with multiview-v3: {source.clip_id}")

    notes = document.get("notes", "")
    if any(camera_hashes.get(camera) != digest for camera, digest in _CAMERA_DIGEST.findall(notes)):
        raise ValueError(f"Annotation video SHA-256 disagrees with the scene: {source.clip_id}")
    classes = EVENT_CLASSES_V4 if scene_version == 4 else EVENT_CLASSES
    events = sorted(document["events"], key=lambda row: (row["frame"], classes.index(row["kind"])))
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
    subtypes_enum = EVENT_SUBTYPES_V4 if scene_version == 4 else EVENT_SUBTYPES
    columns: dict[str, NDArray[Any]] = {
        "ball_event_status": np.asarray([EVENT_STATUS.index(e["status"]) for e in events], dtype=np.uint8),
        "ball_event_frame_uncertainty": np.asarray([e["frame_uncertainty"] for e in events], dtype=np.uint8),
        "ball_event_subtype": np.asarray([subtypes_enum.index(e.get("subtype")) for e in events], dtype=np.uint8),
        "ball_event_side": np.asarray([hint_sides[hint] for hint in hints], dtype=np.int8),
    }
    if scene_version == 3:
        columns["ball_event_in_point"] = np.asarray([e["in_point"] for e in events], dtype=np.bool_)

    provenance: dict[str, Any] = {
        "annotation_sha256": annotation["sha256"],
        "annotation_schema_sha256": schema["sha256"],
        "labeler": "chatgpt" if labeler in _LABELER_NAMES else labeler.strip(),
        "labeler_original": labeler,
    }
    evidence: dict[str, Any] = {
        "events": events,
        "notes": notes,
        "unlabelable": intervals,
        "player_hint_reference": reference,
    }
    if scene_version == 4:
        provenance["source_v1_annotation_sha256"] = document["source_v1_annotation_sha256"]
        evidence["excluded_events"] = document.get("excluded_events", [])

    return BallAnnotations(
        events,
        coverage,
        columns,
        provenance,
        evidence,
        scene_schema_version=scene_version,
        classes=classes,
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
