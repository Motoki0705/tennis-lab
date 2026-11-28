"""Validate version 1 manual point and serve-attempt supplements."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any, Literal, get_args

from src.tennis_scene.analytics.ingest.input_files import decode_json, json_object
from src.tennis_scene.analytics.ingest.score_adapter import POINT_KEY, ScoredClip
from src.tennis_scene.analytics.ingest.v4_reader import SERVE_SUBTYPES
from src.tennis_scene.analytics.metrics import (
    TERMINAL_CAUSES_BY_SOURCE,
    ErrorAssessment,
    TerminalCause,
)
from src.tennis_scene.generate_dataset.manifest import split_clip_id

_REQUIRED = {"clip_id", "point_key", "reviewer"}
_OPTIONAL = {"terminal_cause", "error_assessment", "serve_attempts", "note"}
_ATTEMPT_FIELDS = {
    "serve_number",
    "replay_index",
    "result",
    "source_event_id",
    "fault_cause",
}
ServeResult = Literal["in", "fault", "let", "unknown"]


@dataclass(frozen=True)
class ManualServe:
    result: ServeResult
    serve_number: int | None
    replay_index: int | None
    source_event_id: str | None
    fault_cause: str | None


@dataclass(frozen=True)
class ManualPoint:
    clip_id: str
    point_key: str
    reviewer: str
    note: str | None
    terminal_cause: TerminalCause | None
    error_assessment: ErrorAssessment | None
    serve_attempts: tuple[ManualServe, ...]


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def _parse_attempt(data: Any) -> ManualServe:
    if (
        not isinstance(data, dict)
        or "result" not in data
        or data.keys() - _ATTEMPT_FIELDS
    ):
        raise ValueError("Invalid serve attempt fields")
    result = data["result"]
    if result not in get_args(ServeResult):
        raise ValueError("Invalid serve attempt result")
    number, replay = data.get("serve_number"), data.get("replay_index")
    if number is not None and (type(number) is not int or number not in (1, 2)):
        raise ValueError("serve_number must be 1, 2 or null")
    if replay is not None and (type(replay) is not int or not 0 <= replay < 2**31):
        raise ValueError("replay_index must be a nonnegative 32-bit integer or null")
    event, cause = data.get("source_event_id"), data.get("fault_cause")
    if event is not None:
        event = _text(event, "source_event_id")
    if cause is not None:
        cause = _text(cause, "fault_cause")
        if result != "fault":
            raise ValueError("fault_cause requires a fault result")
    return ManualServe(result, number, replay, event, cause)


def _parse_row(data: Any) -> ManualPoint:
    if (
        not isinstance(data, dict)
        or not _REQUIRED <= data.keys() <= _REQUIRED | _OPTIONAL
    ):
        raise ValueError("Invalid manual supplement fields")
    clip_id = _text(data["clip_id"], "clip_id")
    split_clip_id(clip_id)
    if data["point_key"] != POINT_KEY:
        raise ValueError(f"point_key must be {POINT_KEY!r}")
    cause = data.get("terminal_cause")
    if cause is not None and cause not in TERMINAL_CAUSES_BY_SOURCE["manual_confirmed"]:
        raise ValueError("Invalid manual terminal cause")
    error = data.get("error_assessment")
    if error is not None and error not in get_args(ErrorAssessment):
        raise ValueError("Invalid error assessment")
    attempts = data.get("serve_attempts", [])
    if not isinstance(attempts, list):
        raise ValueError("serve_attempts must be an array")
    if cause is None and error is None and not attempts:
        raise ValueError(
            "A supplement must supply a cause, assessment or serve attempt"
        )
    note = data.get("note")
    return ManualPoint(
        clip_id,
        POINT_KEY,
        _text(data["reviewer"], "reviewer"),
        _text(note, "note") if note not in (None, "") else None,
        cause,
        error,
        tuple(_parse_attempt(item) for item in attempts),
    )


def _csv_rows(raw: bytes) -> list[dict[str, Any]]:
    reader = csv.DictReader(StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    fields = reader.fieldnames or []
    if (
        len(fields) != len(set(fields))
        or not _REQUIRED <= set(fields) <= _REQUIRED | _OPTIONAL
    ):
        raise ValueError("Invalid manual supplement CSV header")
    rows = []
    for raw_row in reader:
        if None in raw_row or any(value is None for value in raw_row.values()):
            raise ValueError(f"Incomplete manual supplement CSV row {reader.line_num}")
        row: dict[str, Any] = {
            key: value for key, value in raw_row.items() if value != ""
        }
        if "serve_attempts" in row:
            row["serve_attempts"] = decode_json(
                row["serve_attempts"].encode("utf-8"),
                f"CSV line {reader.line_num} serve_attempts",
            )
        rows.append(row)
    return rows


def read_supplement(path: Path | None) -> tuple[str | None, dict[str, ManualPoint]]:
    if path is None:
        return None, {}
    raw = path.read_bytes()
    if path.suffix.lower() == ".json":
        document = json_object(raw, str(path))
        if (
            document.keys() != {"version", "points"}
            or type(document["version"]) is not int
            or document["version"] != 1
        ):
            raise ValueError("Expected manual supplement JSON version 1")
        rows = document["points"]
        if not isinstance(rows, list):
            raise ValueError("Manual supplement points must be an array")
    elif path.suffix.lower() == ".csv":
        try:
            rows = _csv_rows(raw)
        except csv.Error as error:
            raise ValueError(f"Invalid manual supplement CSV: {error}") from error
    else:
        raise ValueError("Manual supplement must be .json or .csv")
    points = {}
    for i, row in enumerate(rows, 1):
        try:
            point = _parse_row(row)
            if point.clip_id in points:
                raise ValueError(
                    f"Duplicate supplement for {point.clip_id}/{point.point_key}"
                )
        except ValueError as error:
            raise ValueError(f"Manual supplement record {i}: {error}") from error
        points[point.clip_id] = point
    return sha256(raw).hexdigest(), points


def _last_serve_numbers(attempts: tuple[ManualServe, ...]) -> set[int]:
    """Check that the supplied prefix or excerpt can be one legal serve sequence."""
    # A position gives the serve number, minimum replay index and whether
    # that index is exact. A partial recording can begin on either serve.
    positions = {(1, 0, False), (2, 0, False)}
    for index, attempt in enumerate(attempts):
        compatible = set()
        for number, replay, exact in positions:
            if attempt.serve_number not in (None, number):
                continue
            if attempt.replay_index is None:
                compatible.add((number, replay, exact))
            elif attempt.replay_index >= replay and (
                not exact or attempt.replay_index == replay
            ):
                compatible.add((number, attempt.replay_index, True))
        if not compatible:
            raise ValueError(
                f"Manual serve attempt {index + 1} conflicts with the preceding serve results, "
                "serve numbers or replay indices"
            )
        if index == len(attempts) - 1:
            return {number for number, _, _ in compatible}
        positions = {
            (number, replay + 1, exact)
            for number, replay, exact in compatible
            if attempt.result in ("let", "unknown")
        }
        if attempt.result in ("fault", "unknown") and any(
            number == 1 for number, _, _ in compatible
        ):
            positions.add((2, 0, True))
    return set()


def validate_supplement(
    point: ManualPoint, scored: ScoredClip, *, derived_cause: TerminalCause
) -> None:
    score = scored.confirmed
    if score is None:
        raise ValueError(f"{point.clip_id}: a supplement requires a confirmed point")
    cause = point.terminal_cause
    if cause == "double_fault" and score.server == score.winner:
        raise ValueError(
            f"{point.clip_id}: a double fault must lose the point for the server"
        )
    event_frames = {
        event.source_event_id: event.frame
        for event in scored.v4.events
        if event.in_point and event.subtype in SERVE_SUBTYPES
    }
    frame = -1
    for attempt in point.serve_attempts:
        if attempt.source_event_id is not None:
            new_frame = event_frames.get(attempt.source_event_id)
            if new_frame is None or new_frame <= frame:
                raise ValueError(
                    f"{point.clip_id}: manual attempt {attempt.source_event_id} must reference "
                    "an in-point V4 serve in frame order"
                )
            frame = new_frame
    if not point.serve_attempts:
        return

    numbers = _last_serve_numbers(point.serve_attempts)
    last = point.serve_attempts[-1]
    if cause == "double_fault":
        faults = {
            attempt.serve_number
            for attempt in point.serve_attempts
            if attempt.result == "fault"
        }
        if faults != {1, 2}:
            raise ValueError(
                f"{point.clip_id}: double_fault with a supplied serve sequence requires "
                "confirmed faults on both serve 1 and serve 2"
            )
    effective_cause = cause or derived_cause
    if last.result == "fault" and numbers == {2}:
        if score.server == score.winner:
            raise ValueError(
                f"{point.clip_id}: a second serve fault must lose the point for the server"
            )
        if effective_cause not in ("unknown", "double_fault"):
            raise ValueError(
                f"{point.clip_id}: a second serve fault conflicts with terminal cause {effective_cause}"
            )
    if effective_cause == "double_fault" and last.result == "in":
        raise ValueError(
            f"{point.clip_id}: a legal serve conflicts with terminal cause double_fault"
        )
