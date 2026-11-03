"""Strict reader for ``tennis_lab_ball3d_user_reference.v1`` scoring truth."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from ._validation import array, digest, fields, freeze, integer, number, string, strings
from .contracts import (
    ReferenceIdentity,
    ReferenceKind,
    ReferencePosition,
    UserReference,
)

SCHEMA = "tennis_lab_ball3d_user_reference.v1"
_KINDS = {"observed", "estimated", "excluded_extrapolation", "occluded", "unknown"}
_ROOT_FIELDS = (
    "schema created_at_utc clip_id camera_ids num_frames fps source_size_wh "
    "video_sha256 provenance counts_by_camera notes positions events estimation_segments"
)
_SEGMENT_FIELDS = (
    "camera_id gap_id start_frame end_frame frames method quality events_in_gap "
    "max_model_spread_px masked_validation_mean_px masked_validation_p90_px "
    "validation_scenario validation_cases"
)


def sha256_file(path: Path) -> str:
    """Hash bytes without decoding video or interpreting CSV content."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON number: {token}")


def _identity(data: dict[str, Any], expected: ReferenceIdentity) -> ReferenceIdentity:
    size = array(data["source_size_wh"], "source_size_wh")
    if len(size) != 2:
        raise ValueError("source_size_wh: expected [width, height]")
    identity = ReferenceIdentity(
        clip_id=data["clip_id"],
        camera_ids=tuple(array(data["camera_ids"], "camera_ids")),
        num_frames=data["num_frames"],
        fps=data["fps"],
        source_size_wh=(size[0], size[1]),
        video_sha256=strings(data["video_sha256"], "video_sha256"),
    )
    for name in (
        "clip_id",
        "camera_ids",
        "num_frames",
        "source_size_wh",
        "video_sha256",
    ):
        if getattr(identity, name) != getattr(expected, name):
            raise ValueError(f"Reference identity {name} mismatch")
    # Same absolute FPS tolerance as pipeline ClipSource; no rate conversion.
    if abs(identity.fps - expected.fps) > 1e-6:
        raise ValueError("Reference identity fps mismatch")
    return identity


def _camera_frame(
    camera: Any, frame: Any, identity: ReferenceIdentity
) -> tuple[str, int]:
    camera = string(camera, "camera_id")
    if camera not in identity.camera_ids:
        raise ValueError(f"Unknown camera_id: {camera}")
    frame = integer(frame, "frame_index")
    if frame >= identity.num_frames:
        raise ValueError(f"frame_index outside reference timeline: {frame}")
    return camera, frame


def _provenance(
    value: Any, root: Path, identity: ReferenceIdentity
) -> tuple[set[str], dict[str, Any]]:
    provenance = fields(value, "kind instruction exclusion source_files", "provenance")
    if provenance["kind"] != "user_supplied_reference":
        raise ValueError("provenance.kind must be user_supplied_reference")
    string(provenance["instruction"], "provenance.instruction")
    exclusion = fields(
        provenance["exclusion"],
        "camera_id frame_start_inclusive frame_end_inclusive source_flag reason",
        "provenance.exclusion",
    )
    _camera_frame(exclusion["camera_id"], exclusion["frame_start_inclusive"], identity)
    _camera_frame(exclusion["camera_id"], exclusion["frame_end_inclusive"], identity)
    if (
        exclusion["frame_start_inclusive"] > exclusion["frame_end_inclusive"]
        or exclusion["source_flag"] != "extrapolated_unverified"
        or exclusion["reason"] != "user_excluded_tail_extrapolation"
    ):
        raise ValueError("Invalid provenance.exclusion")
    names: set[str] = set()
    paths: set[Path] = set()
    for item in array(provenance["source_files"], "provenance.source_files"):
        source = fields(item, "filename path sha256 size_bytes", "source file")
        name = string(source["filename"], "source filename")
        relative = Path(string(source["path"], "source path"))
        path = (root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(root.resolve())
            or relative.name != name
            or Path(name).name != name
        ):
            raise ValueError("Invalid source file path")
        if name in names or path in paths:
            raise ValueError("duplicate source file")
        names.add(name)
        paths.add(path)
        if path.stat().st_size != integer(source["size_bytes"], "source size_bytes"):
            raise ValueError(f"source file size mismatch: {name}")
        if sha256_file(path) != digest(source["sha256"], "source SHA-256"):
            raise ValueError(f"source file SHA-256 mismatch: {name}")
    if not names:
        raise ValueError("provenance.source_files must not be empty")
    return names, exclusion


def _source_file(value: Any, names: set[str]) -> str:
    name = string(value, "source_file")
    if name not in names:
        raise ValueError(f"Unknown source_file: {name}")
    return name


def _position(
    value: Any, identity: ReferenceIdentity, names: set[str], exclusion: dict[str, Any]
) -> ReferencePosition:
    row = fields(
        value,
        "camera_id frame_index source_status reference_kind valid_for_scoring uv_px source_file estimation",
        "position",
    )
    camera, frame = _camera_frame(row["camera_id"], row["frame_index"], identity)
    kind = string(row["reference_kind"], "reference_kind")
    if kind not in _KINDS:
        raise ValueError(f"Unknown reference_kind: {kind}")
    status = string(row["source_status"], "source_status")
    expected_status = (
        "unknown" if kind in ("estimated", "excluded_extrapolation") else kind
    )
    if status != expected_status:
        raise ValueError("source_status contradicts reference_kind")
    valid = row["valid_for_scoring"]
    if type(valid) is not bool or valid != (kind in ("observed", "estimated")):
        raise ValueError(
            "valid_for_scoring contradicts reference_kind scoring contract"
        )
    uv: tuple[float, float] | None = None
    if valid:
        xy = array(row["uv_px"], "uv_px")
        if len(xy) != 2:
            raise ValueError("uv_px must contain exactly two coordinates")
        uv = number(xy[0], "uv_px"), number(xy[1], "uv_px")
        if not all(
            0 <= p <= size - 1
            for p, size in zip(uv, identity.source_size_wh, strict=True)
        ):
            raise ValueError("uv_px outside native pixel bounds [0, size-1]")
    elif row["uv_px"] is not None:
        raise ValueError("Invalid scoring row must have uv_px=null")
    estimation = (
        None if row["estimation"] is None else strings(row["estimation"], "estimation")
    )
    excluded = (
        camera == exclusion["camera_id"]
        and exclusion["frame_start_inclusive"]
        <= frame
        <= exclusion["frame_end_inclusive"]
    )
    if excluded != (kind == "excluded_extrapolation"):
        raise ValueError("reference_kind contradicts provenance.exclusion")
    flag = (estimation or {}).get("estimation_flag")
    if (flag == "extrapolated_unverified") != excluded:
        raise ValueError("estimation flag contradicts provenance.exclusion")
    if kind == "observed" and estimation not in (None, {"position_source": "observed"}):
        raise ValueError("Observed reference cannot claim estimation provenance")
    if kind in ("unknown", "occluded") and estimation not in (
        None,
        {"position_source": "unavailable"},
    ):
        raise ValueError("Unavailable reference cannot claim estimation provenance")
    if kind in ("estimated", "excluded_extrapolation"):
        if not estimation or not (
            estimation.get("estimation_method")
            or (
                estimation.get("position_source") == "estimated"
                and estimation.get("estimate_method")
            )
        ):
            raise ValueError("Estimated reference requires estimation provenance")
        if estimation.get("position_source") not in (None, "estimated"):
            raise ValueError(
                "Estimated reference cannot claim observed estimation provenance"
            )
    return ReferencePosition(
        camera,
        frame,
        status,
        cast(ReferenceKind, kind),
        valid,
        uv,
        _source_file(row["source_file"], names),
        freeze(estimation),
    )


def _segments(value: Any, identity: ReferenceIdentity) -> tuple[Mapping[str, str], ...]:
    segments: list[Mapping[str, str]] = []
    keys: set[tuple[str, str]] = set()
    for item in array(value, "estimation_segments"):
        row = strings(fields(item, _SEGMENT_FIELDS, "segment"), "segment")
        parsed: dict[str, int] = {}
        for name in (
            "gap_id",
            "start_frame",
            "end_frame",
            "frames",
            "validation_cases",
        ):
            text = row[name]
            if not text.isascii() or not text.isdecimal():
                raise ValueError(f"segment {name}: expected an integer string")
            parsed[name] = int(text)
        try:
            _camera_frame(row["camera_id"], parsed["start_frame"], identity)
            _camera_frame(row["camera_id"], parsed["end_frame"], identity)
        except ValueError as exc:
            raise ValueError(f"segment: {exc}") from exc
        if (
            parsed["frames"] < 1
            or parsed["frames"] != parsed["end_frame"] - parsed["start_frame"] + 1
        ):
            raise ValueError("Invalid segment interval")
        key = row["camera_id"], row["gap_id"]
        if key in keys:
            raise ValueError("duplicate segment camera/gap_id")
        keys.add(key)
        segments.append(freeze(row))
    return tuple(segments)


def load_user_reference(
    path: Path, *, expected: ReferenceIdentity, expected_sha256: str
) -> UserReference:
    """Verify a pinned JSON, all source CSV bytes and caller identity, then read.

    ``expected`` must describe the evaluation inputs, not be inferred from this
    reference itself. Video hashes are compared here; callers verify actual video
    bytes with ``sha256_file`` when binding their inputs. No video is decoded.
    Missing files and malformed/unsupported data raise; nothing is repaired.
    """
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if checksum != digest(expected_sha256, "expected_sha256"):
        raise ValueError("reference SHA-256 mismatch")
    data = fields(
        json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite),
        _ROOT_FIELDS,
        "reference",
    )
    if data["schema"] != SCHEMA:
        raise ValueError(f"Unsupported reference schema: {data['schema']}")
    string(data["created_at_utc"], "created_at_utc")
    for note in array(data["notes"], "notes"):
        string(note, "note")
    identity = _identity(data, expected)
    names, exclusion = _provenance(data["provenance"], path.parent, identity)
    positions: dict[tuple[str, int], ReferencePosition] = {}
    for value in array(data["positions"], "positions"):
        position = _position(value, identity, names, exclusion)
        key = position.camera_id, position.frame_index
        if key in positions:
            raise ValueError(f"duplicate camera/frame: {key}")
        positions[key] = position
    if len(positions) != len(identity.camera_ids) * identity.num_frames:
        raise ValueError(
            "Reference requires the full camera/frame timeline, including missing rows"
        )
    counts = fields(
        data["counts_by_camera"], " ".join(identity.camera_ids), "counts_by_camera"
    )
    for camera in identity.camera_ids:
        declared = counts[camera]
        if not isinstance(declared, dict) or any(k not in _KINDS for k in declared):
            raise ValueError("Invalid counts_by_camera reference_kind")
        for count in declared.values():
            integer(count, "counts_by_camera")
        actual = Counter(
            p.reference_kind for p in positions.values() if p.camera_id == camera
        )
        if Counter(declared) != actual:
            raise ValueError(f"counts_by_camera mismatch: {camera}")
    events: list[Mapping[str, Any]] = []
    for item in array(data["events"], "events"):
        row = fields(item, "camera_id frame_index event note source_file", "event")
        _camera_frame(row["camera_id"], row["frame_index"], identity)
        if row["event"] not in ("hit", "bounce"):
            raise ValueError("Unknown event type")
        string(row["note"], "event note", empty=True)
        _source_file(row["source_file"], names)
        events.append(freeze(row))
    segments = _segments(data["estimation_segments"], identity)
    for segment in segments:
        for frame in range(int(segment["start_frame"]), int(segment["end_frame"]) + 1):
            if positions[segment["camera_id"], frame].reference_kind != "estimated":
                raise ValueError("segment must refer to estimated positions")
    metadata = {
        k: v
        for k, v in data.items()
        if k not in ("positions", "events", "estimation_segments")
    }
    return UserReference(
        identity,
        checksum,
        MappingProxyType(positions),
        tuple(events),
        segments,
        freeze(metadata),
    )
