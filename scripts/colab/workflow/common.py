"""Shared validation and serialization helpers for the Colab workflow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{7,63}$")
NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
RCLONE_REMOTE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class WorkflowError(RuntimeError):
    """A user-actionable workflow contract error."""


def canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically for hashing and persistence."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    """Return the SHA-256 digest of a canonical JSON value."""

    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a regular file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any, mode: int = 0o600) -> None:
    """Atomically replace a JSON file with deterministic content."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json(value))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject other top-level values."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError(f"cannot read JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise WorkflowError(f"JSON file must contain an object: {path}")
    return value


def strict_relative_path(value: str, label: str) -> str:
    """Validate and normalize a portable, non-escaping relative path."""

    if not isinstance(value, str) or not value or "\\" in value:
        raise WorkflowError(f"{label} must be a non-empty POSIX relative path")
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in raw_parts):
        raise WorkflowError(
            f"{label} must not be absolute or contain '.'/'..': {value}"
        )
    normalized = path.as_posix()
    if normalized.startswith("/"):
        raise WorkflowError(f"{label} escaped its root: {value}")
    return normalized


def validate_run_id(value: str) -> str:
    """Validate a run identifier used in local, VM, and Drive paths."""

    if not RUN_ID_RE.fullmatch(value):
        raise WorkflowError("run id must be 8-64 lowercase letters, digits, or hyphens")
    return value


def ensure_within(root: Path, candidate: Path, label: str) -> Path:
    """Resolve a path and ensure it remains below the requested root."""

    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if (
        resolved_candidate == resolved_root
        or resolved_root not in resolved_candidate.parents
    ):
        raise WorkflowError(f"{label} escaped {resolved_root}: {candidate}")
    return resolved_candidate
