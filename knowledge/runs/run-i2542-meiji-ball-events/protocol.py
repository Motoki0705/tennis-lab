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
