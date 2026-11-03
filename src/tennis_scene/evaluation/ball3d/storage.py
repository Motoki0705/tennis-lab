"""Checksummed typed result files, without pickle or dynamic type imports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar, cast

from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256

T = TypeVar("T")


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Artifact path escapes its directory")
    return path


def read_json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise ValueError(f"Nonfinite JSON value: {value}")

    value = json.loads(
        path.read_text(), object_pairs_hook=pairs, parse_constant=invalid
    )
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def save_typed(directory: Path, value: T, output_type: type[T], *, schema: str) -> str:
    """Publish to a new directory; an interrupted publication has no valid manifest."""
    directory.mkdir(parents=True, exist_ok=False)
    payload, arrays = ArtifactCodec(output_type).dump(value, directory)
    write_json_atomic(
        directory / "result.json",
        {"schema": schema, "payload": payload, "arrays": arrays},
    )
    digest = cast(str, dual_sha256(directory / "result.json"))
    (directory / "result.sha256").write_text(digest + "\n")
    return digest


def load_typed(
    directory: Path,
    output_type: type[T],
    *,
    schema: str,
    expected_sha256: str | None = None,
) -> T:
    digest = dual_sha256(directory / "result.json")
    pinned = (directory / "result.sha256").read_text().strip()
    if digest != pinned or (expected_sha256 is not None and digest != expected_sha256):
        raise ValueError("Result manifest checksum mismatch")
    document = read_json(directory / "result.json")
    if set(document) != {"schema", "payload", "arrays"} or document["schema"] != schema:
        raise ValueError("Unsupported result schema")
    return ArtifactCodec(output_type).load(
        document["payload"], directory, document["arrays"]
    )
