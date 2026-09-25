"""Bounded JSON-only ZIP storage; intake never extracts or interprets annotations."""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from ..runtime.contracts import loads_json, sha256_file

MAX_ZIP_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 256


def simple_name(name: str, suffix: str) -> str:
    if not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_.()-]{0,199}", name
    ) or not name.endswith(suffix):
        raise ValueError(
            f"expected a plain {suffix} filename (at most 200 ASCII letters, digits, "
            "underscores, dots, hyphens or parentheses; start with a letter, digit or underscore)"
        )
    return name


def inspect_zip(data: bytes) -> list[str]:
    if not data or len(data) > MAX_ZIP_BYTES:
        raise ValueError("ZIP must be nonempty and at most 16 MiB")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if not 1 <= len(members) <= MAX_MEMBERS:
            raise ValueError("ZIP must contain 1..256 JSON files")
        if sum(member.file_size for member in members) > MAX_JSON_BYTES:
            raise ValueError("expanded JSON exceeds 64 MiB")
        names: list[str] = []
        for member in members:
            try:
                simple_name(member.filename, ".json")
            except ValueError as error:
                raise ValueError(
                    f"invalid ZIP member {member.filename!r}: {error}; "
                    "this check applies to entries inside the ZIP, not the submitted ZIP name. "
                    "Put only JSON files directly at the archive root, without directories."
                ) from None
            mode = member.external_attr >> 16
            if member.filename in names or member.flag_bits & 1 or stat.S_ISLNK(mode):
                raise ValueError("duplicate, encrypted or symlink ZIP member")
            if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise ValueError("only stored/deflated ZIP members are supported")
            with archive.open(member) as handle:
                content = handle.read(MAX_JSON_BYTES + 1)
            if len(content) > MAX_JSON_BYTES:
                raise ValueError("JSON exceeds 64 MiB")
            if not isinstance(loads_json(content), dict):
                raise ValueError("each JSON must be an annotation object")
            names.append(member.filename)
        return names


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, artifact_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}\.zip", artifact_id):
            raise ValueError(
                "artifact_id must be the SHA-256 ZIP name returned by save_artifact"
            )
        path = self.root / artifact_id
        if path.is_symlink():
            raise ValueError("artifact symlinks are forbidden")
        return path

    def save(self, filename: str, data: bytes) -> dict[str, Any]:
        simple_name(filename, ".zip")
        digest = hashlib.sha256(data).hexdigest()
        try:
            members = inspect_zip(data)
        except (ValueError, zipfile.BadZipFile) as error:
            raise ValueError(
                f"ZIP validation failed (sha256={digest}, bytes={len(data)}): {error}"
            ) from None
        artifact_id = f"{digest}.zip"
        path = self._path(artifact_id)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".upload-", dir=self.root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
                created = True
            except FileExistsError:
                if path.is_symlink() or sha256_file(path) != digest:
                    raise ValueError(
                        "existing artifact is corrupt; refusing overwrite"
                    ) from None
                created = False
        finally:
            temporary.unlink(missing_ok=True)
        return {
            "artifact_id": artifact_id,
            "sha256": digest,
            "bytes": len(data),
            "members": members,
            "submitted_filename": filename,
            "created": created,
        }

    def list(self, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError("offset >= 0 and 1 <= limit <= 100 required")
        paths = sorted(self.root.glob("*.zip"))
        page = paths[offset : offset + limit]
        return {
            "artifacts": [
                {
                    "artifact_id": path.name,
                    "bytes": self._path(path.name).stat().st_size,
                }
                for path in page
            ],
            "next_offset": offset + limit if offset + limit < len(paths) else None,
        }

    def read(self, artifact_id: str) -> dict[str, Any]:
        path = self._path(artifact_id)
        with path.open("rb") as handle:
            data = handle.read(MAX_ZIP_BYTES + 1)
        if hashlib.sha256(data).hexdigest() + ".zip" != artifact_id:
            raise ValueError("artifact checksum mismatch")
        return {
            "artifact_id": artifact_id,
            "bytes": len(data),
            "members": inspect_zip(data),
        }
