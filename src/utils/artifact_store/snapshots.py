"""Immutable, content-verified snapshots of a quiescent artifact directory."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from src.utils.checksum import dual_sha256
from src.utils.data.verification import file_version

from .contracts import ArtifactStoreError
from .rclone import RcloneArtifactStore

SCHEMA = "verified_artifact_snapshot.v1"
MARKER = "_SNAPSHOT.json"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def _relative(name: str) -> Path:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or path.as_posix() != name
            or any(part in {"", ".", ".."} for part in name.split("/")) or "\\" in name or ":" in name):
        raise ArtifactStoreError("Snapshot file paths must be canonical relative paths")
    return Path(name)


def _identity(path: Path) -> dict[str, Any]:
    before = file_version(path)
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(chunk)
    result = dict(size_bytes=path.stat().st_size, md5=md5.hexdigest(), sha256=dual_sha256(path))
    if before != file_version(path):
        raise ArtifactStoreError("Snapshot input changed while hashing")
    return result


class RcloneSnapshotStore(RcloneArtifactStore):
    """Publish a new generation, then verify every byte before returning a receipt.

    The caller pauses all writers during publication. No existing generation is
    modified, and no remote file is deleted. A partial generation has no trusted
    completion marker. Restoration additionally requires the receipt's pinned
    SHA-256 of that marker and verifies local SHA-256 after downloading.
    """

    def _destination(self, generation: str) -> str:
        if not _NAME.fullmatch(generation) or generation in {".", ".."}:
            raise ArtifactStoreError("Invalid snapshot generation name")
        return f"{self.remote_uri}/{generation}"

    def _inventory(self, uri: str) -> dict[str, dict[str, Any]]:
        try:
            entries = json.loads(self._output("lsjson", uri, "--recursive", "--hash"))
            if not isinstance(entries, list):
                raise ValueError("Expected a list")
            names: set[str] = set()
            files = {}
            for entry in entries:
                name = entry["Path"]
                _relative(name)
                if name in names:
                    raise ArtifactStoreError("Duplicate remote snapshot path")
                names.add(name)
                if not entry["IsDir"]:
                    files[name] = entry
            return files
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactStoreError("Invalid remote snapshot inventory") from error

    def _verify(self, uri: str, expected: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        actual = self._inventory(uri)
        if actual.keys() != expected.keys():
            raise ArtifactStoreError("Remote snapshot file set differs")
        for name, identity in expected.items():
            entry = actual[name]
            hashes = {key.lower(): value for key, value in entry.get("Hashes", {}).items()}
            digest = hashes.get("md5")
            if (entry.get("Size") != identity["size_bytes"] or not digest
                    or digest.lower() != identity["md5"]):
                raise ArtifactStoreError(f"Remote snapshot size/hash differs or is unavailable: {name}")
        return actual

    def publish(self, generation: str) -> dict[str, Any]:
        started = time.monotonic()
        destination = self._destination(generation)
        # Inventory the parent first: a reused or partially uploaded generation
        # must not be overwritten. Missing remote roots are created explicitly.
        self._run("mkdir", self.remote_uri)
        entries = json.loads(self._output("lsjson", self.remote_uri))
        names = [entry["Name"] for entry in entries]
        if len(set(names)) != len(names) or generation in names:
            raise ArtifactStoreError("Snapshot generation already exists or its parent is ambiguous")
        paths = self._paths()
        versions = {path: file_version(path) for path in paths}
        files = {path.relative_to(self.local_root).as_posix(): _identity(path) for path in paths}
        if not files or MARKER in files:
            raise ArtifactStoreError("Snapshot requires files and reserves its completion marker name")
        self._run("copy", str(self.local_root), destination, "--immutable", "--checksum")
        self._verify(destination, files)
        self._unchanged(versions)
        manifest = dict(schema=SCHEMA, generation=generation, files=files)
        with tempfile.TemporaryDirectory(prefix="artifact-snapshot-") as temporary:
            marker = Path(temporary) / MARKER
            marker.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
            marker_identity = _identity(marker)
            self._run("copyto", str(marker), f"{destination}/{MARKER}", "--immutable")
            inventory = self._verify(destination, {**files, MARKER: marker_identity})
        self._unchanged(versions)
        return dict(schema=SCHEMA, generation=generation, remote=destination,
                    manifest_sha256=marker_identity["sha256"], manifest_md5=marker_identity["md5"],
                    files=len(files), size_bytes=sum(item["size_bytes"] for item in files.values()),
                    marker_id=inventory[MARKER].get("ID"), seconds=time.monotonic() - started, status="verified")

    def _paths(self) -> list[Path]:
        if not self.local_root.is_dir() or self.local_root.is_symlink():
            raise ArtifactStoreError("Snapshot root must be a regular directory")
        paths = sorted(self.local_root.rglob("*"))
        if any(path.is_symlink() or not (path.is_dir() or path.is_file()) for path in paths):
            raise ArtifactStoreError("Snapshot tree contains a symlink or a special file")
        return [path for path in paths if path.is_file()]

    def _unchanged(self, versions: dict[Path, Any]) -> None:
        if self._paths() != list(versions) or any(file_version(path) != version for path, version in versions.items()):
            raise ArtifactStoreError("Snapshot inputs changed during publication")

    def restore(self, generation: str, *, manifest_sha256: str) -> dict[str, Any]:
        started = time.monotonic()
        source = self._destination(generation)
        if self.local_root.exists():
            raise ArtifactStoreError("Restore requires a new local directory")
        if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
            raise ArtifactStoreError("Restore requires the saved manifest SHA-256")
        self.local_root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".restoring-", dir=self.local_root.parent) as temporary:
            marker = Path(temporary) / MARKER
            self._run("copyto", f"{source}/{MARKER}", str(marker))
            if dual_sha256(marker) != manifest_sha256:
                raise ArtifactStoreError("Snapshot completion marker SHA-256 differs")
            manifest = json.loads(marker.read_text())
            if manifest.get("schema") != SCHEMA or manifest.get("generation") != generation:
                raise ArtifactStoreError("Unknown snapshot schema or generation")
            files = manifest["files"]
            if not isinstance(files, dict) or not files or MARKER in files:
                raise ArtifactStoreError("Invalid snapshot files")
            for name in files:
                _relative(name)
            self._verify(source, {**files, MARKER: _identity(marker)})
            tree = Path(temporary) / "tree"
            self._run("copy", source, str(tree), "--exclude", "/" + MARKER, "--checksum")
            paths = sorted(path.relative_to(tree).as_posix() for path in tree.rglob("*") if path.is_file())
            if paths != sorted(files) or any(path.is_symlink() for path in tree.rglob("*")):
                raise ArtifactStoreError("Restored snapshot file set differs")
            for name, expected in files.items():
                if _identity(tree / name) != expected:
                    raise ArtifactStoreError(f"Restored snapshot size/hash differs: {name}")
            os.rename(tree, self.local_root)
        return dict(schema=SCHEMA, generation=generation, status="verified", files=len(files),
                    manifest_sha256=manifest_sha256, seconds=time.monotonic() - started)
