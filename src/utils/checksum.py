"""Independent SHA-256 computation with file identity and length checks."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Protocol, cast

_CHUNK_SIZE = 1024 * 1024


class FileIntegrityError(Exception):
    """Hash computation or file stability could not be verified; never retry silently."""

    def __init__(self, reason: str, *, details: dict[str, object]) -> None:
        self.details = details
        super().__init__(f"{reason}: {json.dumps(details, sort_keys=True)}")


class _Digest(Protocol):
    def update(self, data: bytes) -> None: ...

    def hexdigest(self) -> str: ...


class _SHA256Module(Protocol):
    def sha256(self) -> _Digest: ...


def _stat_identity(stat: os.stat_result) -> dict[str, int]:
    return {
        "dev": stat.st_dev,
        "ino": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def dual_sha256(path: str | Path) -> str:
    """Hash identical immutable chunks using hashlib and CPython's _sha256.

    Reject provider disagreement, changed path/descriptor metadata, and an
    unexpected byte count. Symlinks and hardlinks are followed normally.
    This does not detect the same incorrect bytes supplied to both providers,
    prove general hardware correctness, or replace a trusted expected digest.
    There is no single-provider fallback or automatic retry.
    """
    path = Path(path)
    details: dict[str, object] = {
        "path": str(path),
        "hashlib_sha256": None,
        "cpython_sha256": None,
        "bytes_read": 0,
        "path_before": None,
        "descriptor_before": None,
        "descriptor_after": None,
        "path_after": None,
    }
    try:
        independent = cast(_SHA256Module, importlib.import_module("_sha256")).sha256()
    except (ImportError, AttributeError) as error:
        raise FileIntegrityError(
            "CPython _sha256 provider is required", details=details
        ) from error
    primary = hashlib.sha256()
    length = 0
    try:
        before = _stat_identity(path.stat())
        details["path_before"] = before
        with path.open("rb") as handle:
            descriptor_before = _stat_identity(os.fstat(handle.fileno()))
            details["descriptor_before"] = descriptor_before
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                primary.update(chunk)
                independent.update(chunk)
                length += len(chunk)
                details["bytes_read"] = length
            digest = primary.hexdigest()
            details["hashlib_sha256"] = digest
            details["cpython_sha256"] = independent.hexdigest()
            descriptor_after = _stat_identity(os.fstat(handle.fileno()))
            details["descriptor_after"] = descriptor_after
            after = _stat_identity(path.stat())
            details["path_after"] = after
    except OSError as error:
        raise FileIntegrityError(
            f"File could not be verified ({error})", details=details
        ) from error
    if not before == descriptor_before == descriptor_after == after:
        raise FileIntegrityError("File identity or metadata changed", details=details)
    if length != before["size"]:
        raise FileIntegrityError(
            "File byte count differs from stat size", details=details
        )
    if digest != details["cpython_sha256"]:
        raise FileIntegrityError("SHA-256 providers disagree", details=details)
    return digest
