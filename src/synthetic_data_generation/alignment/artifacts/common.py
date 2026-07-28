"""Strict helpers shared by immutable alignment artifact types."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode a mapping deterministically for fingerprints."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def artifact_fingerprint(payload: Mapping[str, Any]) -> str:
    """Hash a payload while excluding its declared artifact fingerprint."""
    canonical = {
        key: value for key, value in payload.items() if key != "artifact_fingerprint"
    }
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def validate_artifact_id(value: object, *, artifact_type: str) -> str:
    """Return one validated path-safe immutable artifact identifier."""
    if not isinstance(value, str) or _ARTIFACT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{artifact_type} artifact_id must be path-safe.")
    return value


def validate_sha256(value: object, *, name: str) -> str:
    """Return one strict lower-case SHA-256 digest."""
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lower-case SHA-256 digest.")
    return value


def publish_json_artifact(
    payload: dict[str, Any],
    *,
    output_dir: Path,
    validate: Callable[[dict[str, Any]], None],
    artifact_type: str,
) -> Path:
    """Atomically publish one strict content-addressed JSON artifact."""
    validate(payload)
    manifest = dict(payload)
    fingerprint = artifact_fingerprint(manifest)
    manifest["artifact_fingerprint"] = fingerprint
    artifact_id = validate_artifact_id(
        payload.get("artifact_id"),
        artifact_type=artifact_type,
    )
    destination = output_dir / f"{artifact_id}-{fingerprint[:16]}.json"
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite {artifact_type} artifact: {destination}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{artifact_id}-",
        suffix=".json",
        dir=output_dir,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                manifest,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.rename(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def load_json_artifact(
    path: Path,
    *,
    validate: Callable[[dict[str, Any]], None],
    artifact_type: str,
) -> dict[str, Any]:
    """Strict-load and fingerprint-verify one JSON artifact."""
    with path.open(encoding="utf-8") as handle:
        raw: Any = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{artifact_type} artifact must be a JSON object.")
    payload = dict(raw)
    validate(payload)
    declared = validate_sha256(
        payload.get("artifact_fingerprint"),
        name=f"{artifact_type} artifact_fingerprint",
    )
    expected = artifact_fingerprint(payload)
    if declared != expected:
        raise ValueError(
            f"{artifact_type} artifact fingerprint mismatch: "
            f"declared {declared}, computed {expected}."
        )
    return payload
