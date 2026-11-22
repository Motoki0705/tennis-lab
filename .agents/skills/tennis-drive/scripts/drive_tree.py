"""Finite, verified downloads of closed Drive subtrees.

Remote metadata is read twice per operation, never once per file. Neither the
snapshot nor an existing local directory is trusted without reading its bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from drive_core import (
    CONTENT_HASH_ARGUMENTS,
    CONTENT_HASH_TYPES,
    DriveToolError,
    RcloneBackend,
    _assert_regular_local_tree,
    _normalized_hashes,
    _print_object,
    _resolve_local,
)

Progress = Callable[..., None]


def _relative_name(value: str) -> str:
    name: str = RcloneBackend.normalize_listed_relative(value)
    if name == "." or name != value or "\\" in value:
        raise DriveToolError(f"Invalid subtree entry path: {value!r}")
    return name


def _expected_files(files: Mapping[str, str]) -> dict[str, str]:
    result = {}
    for name, digest in files.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise DriveToolError("Expected manifest requires path/SHA-256 strings")
        name = _relative_name(name)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise DriveToolError(f"Invalid expected SHA-256: {name}")
        result[name] = digest
    return result


def _snapshot(backend: RcloneBackend, source: str) -> dict[str, dict[str, Any]]:
    values = backend.json(
        [
            "lsjson",
            backend.remote_path(source),
            "--recursive",
            "--no-mimetype",
            *CONTENT_HASH_ARGUMENTS,
        ]
    )
    if not isinstance(values, list):
        raise DriveToolError("rclone returned an invalid subtree listing")
    entries: dict[str, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict) or not isinstance(item.get("Path"), str):
            raise DriveToolError("rclone returned an invalid subtree entry")
        name = _relative_name(item["Path"])
        if name in entries:
            raise DriveToolError(f"Ambiguous duplicate subtree path: {name}")
        directory = bool(item.get("IsDir"))
        hashes = _normalized_hashes(item)
        hashes = {key: hashes[key] for key in CONTENT_HASH_TYPES if key in hashes}
        size = item.get("Size")
        if not directory and (type(size) is not int or size < 0 or not hashes):
            raise DriveToolError(f"Cannot verify raw-file size/content hashes: {name}")
        entries[name] = {
            "type": "directory" if directory else "file",
            "id": item.get("ID"),
            "size_bytes": None if directory else size,
            "hashes": {} if directory else hashes,
            "modified": item.get("ModTime"),
        }
    for name in entries:
        for parent in PurePosixPath(name).parents:
            if (
                parent != PurePosixPath(".")
                and entries.get(str(parent), {}).get("type") != "directory"
            ):
                raise DriveToolError(
                    f"Missing or conflicting subtree directory: {parent}"
                )
    return entries


def _check_local(
    local: Path,
    entries: dict[str, dict[str, Any]],
    expected: dict[str, str] | None,
    progress: Progress,
) -> None:
    _assert_regular_local_tree(local)
    if not local.is_dir():
        raise DriveToolError("Subtree destination must be a directory")
    paths = {path.relative_to(local).as_posix(): path for path in local.rglob("*")}
    if paths.keys() != entries.keys():
        raise DriveToolError(
            f"Local subtree file set differs; missing={sorted(entries.keys() - paths.keys())[:10]}, "
            f"extra={sorted(paths.keys() - entries.keys())[:10]}"
        )
    completed_files = completed_bytes = 0
    progress(operation="verifying", completed_files=0, completed_bytes=0)
    for name, entry in sorted(entries.items()):
        path = paths[name]
        if entry["type"] == "directory":
            if not path.is_dir():
                raise DriveToolError(f"Local subtree type changed: {name}")
            continue
        if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
            raise DriveToolError(f"Local subtree type/size changed: {name}")
        hashes = {key: hashlib.new(key) for key in entry["hashes"]}
        if expected is not None:
            hashes.setdefault("sha256", hashlib.sha256())
        read_bytes = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                for hasher in hashes.values():
                    hasher.update(chunk)
                read_bytes += len(chunk)
                progress(
                    operation="verifying",
                    completed_files=completed_files,
                    completed_bytes=completed_bytes + read_bytes,
                    current_path=name,
                )
        if read_bytes != entry["size_bytes"] or any(
            hashes[key].hexdigest() != digest for key, digest in entry["hashes"].items()
        ):
            raise DriveToolError(f"Local subtree content hash changed: {name}")
        if expected is not None and hashes["sha256"].hexdigest() != expected[name]:
            raise DriveToolError(f"Expected subtree SHA-256 changed: {name}")
        completed_files += 1
        completed_bytes += read_bytes
        progress(
            operation="verifying",
            completed_files=completed_files,
            completed_bytes=completed_bytes,
            current_path=name,
        )


def download_tree(
    backend: RcloneBackend,
    source: str,
    destination: Path,
    *,
    expected_files: Mapping[str, str] | None = None,
    expected_id: str | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Download once (or verify existing bytes), then read back remote identity.

    Only a fully verified new tree acquires the destination name. An existing
    destination is checked exactly and never repaired or merged silently.
    """
    report = progress if progress is not None else lambda **kwargs: None
    expected = _expected_files(expected_files) if expected_files is not None else None
    source = backend.normalize_relative(source)
    local = _resolve_local(str(destination), must_exist=False)
    report(operation="listing", completed_files=0, completed_bytes=0)
    root = backend.resolve(source)
    if root is None or not root.get("IsDir"):
        raise DriveToolError("download-tree requires an existing Drive directory")
    if expected_id is not None and root.get("ID") != expected_id:
        raise DriveToolError("Drive directory ID changed")
    before = _snapshot(backend, source)
    files = {name for name, entry in before.items() if entry["type"] == "file"}
    if expected is not None and files != expected.keys():
        raise DriveToolError(
            f"Expected subtree file set differs; missing={sorted(expected.keys() - files)[:10]}, "
            f"extra={sorted(files - expected.keys())[:10]}"
        )
    totals = {
        "total_files": len(files),
        "total_bytes": sum(before[name]["size_bytes"] for name in files),
    }

    def update(**fields: Any) -> None:
        report(**totals, **fields)

    def verify(path: Path) -> None:
        _check_local(path, before, expected, update)
        update(
            operation="readback",
            completed_files=len(files),
            completed_bytes=totals["total_bytes"],
        )
        after_root = backend.resolve(source)
        if (
            after_root is None
            or (after_root.get("ID"), after_root.get("IsDir"))
            != (root.get("ID"), root.get("IsDir"))
            or _snapshot(backend, source) != before
        ):
            raise DriveToolError("Drive subtree changed during download/verification")

    reused = local.exists()
    if reused:
        verify(local)
    else:
        local.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{local.name}.fetching-", dir=local.parent
        ) as temporary:
            staging = Path(temporary)
            update(operation="downloading", completed_files=0, completed_bytes=0)
            backend.run(
                [
                    "copy",
                    backend.remote_path(source),
                    str(staging),
                    "--create-empty-src-dirs",
                    "--immutable",
                    "--stats",
                    "30s",
                ],
                capture_output=False,
                progress=lambda stats: update(
                    operation="downloading",
                    completed_files=int(stats.get("transfers", 0)),
                    completed_bytes=int(stats.get("bytes", 0)),
                ),
            )
            verify(staging)
            if local.exists():
                raise DriveToolError("Subtree destination appeared during download")
            staging.replace(local)
    update(
        operation="completed",
        completed_files=len(files),
        completed_bytes=totals["total_bytes"],
    )
    return {
        "action": "download-tree",
        "rclone_config": str(backend.config_path),
        "source": source,
        "source_id": root.get("ID"),
        "destination": str(local),
        "status": "completed",
        "verified": True,
        "reused": reused,
        **totals,
    }


class ProgressFile:
    """Local-only, bounded status writes; no extra Drive requests or daemon."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.monotonic()
        self.last_write = float("-inf")
        self.value: dict[str, Any] = {}

    def __call__(self, **fields: Any) -> None:
        now = time.monotonic()
        changed = any(self.value.get(key) != value for key, value in fields.items())
        transition = fields.get("operation") != self.value.get("operation")
        stamp = datetime.now(UTC).isoformat()
        self.value.update(fields, updated_at=stamp, elapsed_seconds=now - self.started)
        if changed and fields.get("operation") != "failed":
            self.value["last_progress_at"] = stamp
        if transition or now - self.last_write >= 1:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps({"schema_version": 1, **self.value}, indent=2) + "\n"
            )
            temporary.replace(self.path)
            self.last_write = now


def add_tree_command(subparsers: argparse._SubParsersAction[Any]) -> None:
    parser = subparsers.add_parser(
        "download-tree", help="Download/verify a closed directory in one finite batch."
    )
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument(
        "--expected-manifest",
        type=Path,
        help="JSON {schema_version: 1, files: {relative_path: sha256}}.",
    )
    parser.add_argument("--expected-id")
    parser.add_argument(
        "--progress-file",
        type=Path,
        help="Local JSON progress; at most once/second within an operation.",
    )
    parser.add_argument("--format", choices=("json", "table"), default="json")


def run_tree_command(args: argparse.Namespace, backend: RcloneBackend) -> int:
    if args.progress_file is not None and args.progress_file.resolve().is_relative_to(
        Path(args.destination).expanduser().resolve()
    ):
        raise DriveToolError("--progress-file must lie outside the subtree destination")
    expected = None
    if args.expected_manifest is not None:
        try:
            value = json.loads(args.expected_manifest.read_text())
        except (ValueError, OSError) as error:
            raise DriveToolError("Cannot read expected subtree manifest") from error
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 1
            or not isinstance(value.get("files"), dict)
        ):
            raise DriveToolError(
                "Expected subtree manifest requires schema_version=1 and files"
            )
        expected = value["files"]
    progress = ProgressFile(args.progress_file) if args.progress_file else None
    try:
        receipt = download_tree(
            backend,
            args.source,
            Path(args.destination),
            expected_files=expected,
            expected_id=args.expected_id,
            progress=progress,
        )
    except BaseException:
        if progress:
            progress(operation="failed")
        raise
    _print_object(receipt, args.format)
    return 0
