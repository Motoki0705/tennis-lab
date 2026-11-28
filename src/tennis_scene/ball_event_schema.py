"""SceneResult's sparse ball-event columns and metadata contract."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_refiner_3d.events import EVENT_CLASSES

if TYPE_CHECKING:
    from src.tennis_scene.schema import SceneResult

EventSource = Literal["annotation", "rule", "refiner"]
EVENT_SOURCES = ("annotation", "rule", "refiner")
EVENT_STATUS = ("certain", "probable", "uncertain")
EVENT_SUBTYPES = ("serve", "groundstroke", "volley", "smash", "let_cord", "net_fault", "other", None)
EVENT_SUBTYPES_BY_KIND = (
    (0, 1, 2, 3, 6, 7),  # shot
    (7,),  # bounce
    (4, 5, 6, 7),  # net
)

EVENT_CLASSES_V4 = ("shot", "bounce", "net", "unknown")
EVENT_SUBTYPES_V4 = (
    "serve_overhead",
    "serve_underhand",
    "groundstroke_forehand",
    "groundstroke_backhand",
    "volley_forehand",
    "volley_backhand",
    "volley_center",
    "smash",
    "service_let",
    "net_in",
    "net_fault",
    "bounce",
    "unknown",
)
EVENT_SUBTYPES_BY_KIND_V4 = (
    (0, 1, 2, 3, 4, 5, 6, 7),  # shot
    (11,),  # bounce
    (8, 9, 10),  # net
    (12,),  # unknown
)

_COMMON = {
    "ball_event_frame": np.int32,
    "ball_event_kind": np.uint8,
    "ball_event_coverage": np.bool_,
}
_ANNOTATION_V3 = {
    "ball_event_status": np.uint8,
    "ball_event_frame_uncertainty": np.uint8,
    "ball_event_in_point": np.bool_,
    "ball_event_subtype": np.uint8,
    "ball_event_side": np.int8,
}
_ANNOTATION_V4 = {
    "ball_event_status": np.uint8,
    "ball_event_frame_uncertainty": np.uint8,
    "ball_event_subtype": np.uint8,
    "ball_event_side": np.int8,
}
_PREDICTION = {"ball_event_score": np.float32}
BALL_EVENT_FIELDS = tuple((*_COMMON, *_ANNOTATION_V3, *_PREDICTION))


def event_classes_for_version(scene_version: int) -> tuple[str, ...]:
    if scene_version == 3:
        return tuple(str(name) for name in EVENT_CLASSES)
    if scene_version == 4:
        return EVENT_CLASSES_V4
    raise ValueError(f"Unsupported ball-event scene schema version: {scene_version}")


def event_subtypes_for_version(scene_version: int) -> tuple[str | None, ...]:
    if scene_version == 3:
        return EVENT_SUBTYPES
    if scene_version == 4:
        return EVENT_SUBTYPES_V4
    raise ValueError(f"Unsupported ball-event scene schema version: {scene_version}")


def ball_event_metadata(
    source: EventSource,
    provenance: dict[str, Any],
    *,
    scene_version: int = 3,
) -> dict[str, Any]:
    if scene_version == 3:
        return {
            "version": 1,
            "source": source,
            "classes": list(EVENT_CLASSES),
            "statuses": list(EVENT_STATUS),
            "subtypes": list(EVENT_SUBTYPES),
            "provenance": provenance,
        }
    if scene_version == 4:
        if source != "annotation":
            raise ValueError("SceneResult v4 ball events require source='annotation'")
        return {
            "version": 2,
            "source": source,
            "classes": list(EVENT_CLASSES_V4),
            "statuses": list(EVENT_STATUS),
            "subtypes": list(EVENT_SUBTYPES_V4),
            "provenance": provenance,
        }
    raise ValueError(f"Unsupported ball-event scene schema version: {scene_version}")


def validate_scene_ball_events(scene: SceneResult, *, version: int) -> None:
    present = {name for name in BALL_EVENT_FIELDS if getattr(scene, name) is not None}
    if version < 3:
        if present or "ball_events" in scene.metadata:
            raise ValueError("Ball-event columns and metadata require SceneResult v3 or v4")
        return
    if version not in (3, 4):
        raise ValueError(f"Unsupported SceneResult schema version for ball events: {version}")

    metadata = scene.metadata.get("ball_events")
    if not isinstance(metadata, dict):
        raise ValueError(f"v{version} requires metadata.ball_events")
    source = metadata.get("source")
    if source not in EVENT_SOURCES:
        raise ValueError("Unknown ball-event source")
    if version == 4 and source != "annotation":
        raise ValueError("SceneResult v4 ball events require source='annotation'")
    provenance = metadata.get("provenance")
    if (
        not isinstance(provenance, dict)
        or type(metadata.get("version")) is not int
        or metadata != ball_event_metadata(source, provenance, scene_version=version)
    ):
        raise ValueError("Invalid ball-event version, provenance or enum order")
    _validate_provenance(source, provenance)

    if source == "annotation":
        annotation_cols = _ANNOTATION_V4 if version == 4 else _ANNOTATION_V3
        required = {**_COMMON, **annotation_cols}
    else:
        required = {**_COMMON, **_PREDICTION}
    if present != set(required):
        raise ValueError(
            f"Ball-event columns disagree with source {source}: "
            f"missing={sorted(required.keys() - present)}, extra={sorted(present - required.keys())}"
        )
    arrays: dict[str, NDArray[Any]] = {}
    for name, dtype in required.items():
        value = getattr(scene, name)
        if not isinstance(value, np.ndarray) or value.ndim != 1 or value.dtype != dtype:
            raise ValueError(f"{name} must be a one-dimensional {dtype.__name__} array")
        arrays[name] = value
    frames = arrays["ball_event_frame"]
    kinds = arrays["ball_event_kind"]
    classes = event_classes_for_version(version)
    subtypes_by_kind = EVENT_SUBTYPES_BY_KIND_V4 if version == 4 else EVENT_SUBTYPES_BY_KIND
    for name, value in arrays.items():
        length = scene.num_frames if name == "ball_event_coverage" else len(frames)
        if len(value) != length:
            raise ValueError(f"{name} must have length {length}")
    if np.any((frames < 0) | (frames >= scene.num_frames)):
        raise ValueError("Ball-event frame is outside the scene timeline")
    if np.any(kinds >= len(classes)):
        raise ValueError("Unknown ball-event kind")
    if np.any(frames[1:] < frames[:-1]):
        raise ValueError("Ball events must be in frame order")
    keys = frames.astype(np.int64) * len(classes) + kinds
    if len(np.unique(keys)) != len(frames):
        raise ValueError("Duplicate ball-event kind and frame")

    if source == "annotation":
        if np.any(arrays["ball_event_status"] >= len(EVENT_STATUS)):
            raise ValueError("Unknown ball-event status")
        subtypes = arrays["ball_event_subtype"]
        for kind, allowed in enumerate(subtypes_by_kind):
            if not np.isin(subtypes[kinds == kind], allowed).all():
                raise ValueError("Unknown or incompatible ball-event subtype")
        sides = arrays["ball_event_side"]
        if not np.isin(sides, [-1, 0, 1]).all() or np.any((kinds != 0) & (sides != 0)):
            raise ValueError("Invalid ball-event hitter side")
    else:
        scores = arrays["ball_event_score"]
        if not np.all((scores >= 0) & (scores <= 1)):
            raise ValueError("Ball-event scores must be finite and in [0, 1]")


def _validate_provenance(source: EventSource, provenance: dict[str, Any]) -> None:
    commit = provenance.get("code_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("Ball-event provenance requires the complete execution commit")
    digest_key = {"annotation": "annotation_sha256", "refiner": "checkpoint_sha256"}.get(source)
    if digest_key is not None:
        digest = provenance.get(digest_key)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Ball-event provenance requires {digest_key}")
    if source == "annotation" and any(
        not isinstance(provenance.get(name), str) or not provenance[name].strip()
        for name in ("labeler", "labeler_original")
    ):
        raise ValueError("Annotation provenance requires the normalized and original labeler")
    if source == "rule" and not isinstance(provenance.get("thresholds"), dict):
        raise ValueError("Rule provenance requires rule thresholds")
