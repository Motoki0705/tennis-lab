"""The fixed inputs and file format used by this Meiji comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

BENCHMARK_ID = "meiji_4cam/multiview-v2"
CLASSES = ("shot", "bounce", "net")
METHODS = ("refiner3", "rule")
TOLERANCES = (2, 5, 10)
PRIMARY_TOLERANCE = 5


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")


def read_json(path: Path, expected: str | None = None) -> dict[str, Any]:
    raw = path.read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, encoding="utf-8", mode="w", delete=False
    ) as file:
        temporary_path = Path(file.name)
        try:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            file.write("\n")
            file.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


PINS = read_json(Path(__file__).with_name("inputs.json"))


def load_labels(labels_parent: Path, clip: dict[str, Any], role: str) -> dict[str, Any]:
    key = clip["clip_id"]
    name = key.rsplit("/", 1)[-1]
    number = name.removeprefix("clip_")
    path = (
        labels_parent
        / f"parent-1722-issue-2542-label-{number}"
        / "tennis_scene/analyze/event-eval-labels"
        / name
        / "labels.json"
    )
    label = read_json(path, PINS["label_sha256"][key])
    expected = {
        "schema": "meiji_event_labels.v1",
        "benchmark": "multiview-v2",
        "clip_id": key,
        "fps": clip["fps"],
        "frames": clip["frames"],
        "role": role,
    }
    if any(label.get(field) != value for field, value in expected.items()):
        raise ValueError(f"Label provenance or frame grid mismatch: {key}")
    seen = set()
    for event in label["events"]:
        if (
            event["id"] in seen
            or event["kind"] not in CLASSES
            or type(event["frame"]) is not int
            or not 0 <= event["frame"] < clip["frames"]
            or event["status"] not in {"certain", "probable", "uncertain"}
            or type(event["in_point"]) is not bool
        ):
            raise ValueError(f"Invalid label event: {key} {event}")
        seen.add(event["id"])
    for interval in label["unlabelable"]:
        start, end = interval["start"], interval["end"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end < clip["frames"]
        ):
            raise ValueError(f"Invalid unlabelable interval: {key} {interval}")
    return label
