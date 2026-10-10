"""Bounded management operations and durable inventory receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any

from drive_core import (
    DriveToolError,
    ManifestEntry,
    RcloneBackend,
    _manifest,
    _print_object,
    _resolve_local,
)


def _child(backend: RcloneBackend, path: str) -> str:
    normalized = backend.normalize_relative(path)
    if normalized == ".":
        raise DriveToolError(
            "Management mutations cannot target the configured project root"
        )
    return normalized


def _identity(entry: dict[str, Any], expected: str | None) -> None:
    if expected is not None and entry.get("ID") != expected:
        raise DriveToolError(
            f"Drive ID changed: expected {expected}, found {entry.get('ID')}"
        )


def _receipt(action: str, backend: RcloneBackend, **fields: Any) -> dict[str, Any]:
    return {
        "action": action,
        "remote_root": backend.remote_root,
        "observed_at": datetime.now(UTC).isoformat(),
        **fields,
    }


def _require_hashes(entries: dict[str, ManifestEntry]) -> None:
    missing = [
        name
        for name, (kind, _, hashes) in entries.items()
        if kind == "file" and not hashes
    ]
    if missing:
        raise DriveToolError(f"Cannot verify raw-file content hashes: {missing[:10]}")


def _same_content(
    before: dict[str, ManifestEntry], after: dict[str, ManifestEntry]
) -> bool:
    if before.keys() != after.keys():
        return False
    for name, (kind, size, hashes) in before.items():
        other_kind, other_size, other_hashes = after[name]
        if (kind, size) != (other_kind, other_size):
            return False
        if kind == "file":
            common = hashes.keys() & other_hashes.keys()
            if not common or any(hashes[key] != other_hashes[key] for key in common):
                return False
    return True


def _run_relocate(args: argparse.Namespace, backend: RcloneBackend) -> int:
    source = _child(backend, args.source)
    destination = _child(backend, args.destination)
    if (
        source == destination
        or PurePosixPath(source) in PurePosixPath(destination).parents
    ):
        raise DriveToolError("Destination must differ from and not be inside source")
    entry = backend.resolve(source)
    assert entry is not None
    _identity(entry, args.expected_id)
    if backend.resolve(destination, must_exist=False) is not None:
        raise DriveToolError(
            "Destination already exists; copy/move never merge or overwrite"
        )
    source_type, before = _manifest(backend, backend.remote_path(source))
    _require_hashes(before)
    manifest_digest = hashlib.sha256(
        json.dumps(before, sort_keys=True).encode()
    ).hexdigest()
    payload = _receipt(
        args.command,
        backend,
        source=source,
        destination=destination,
        source_id=entry.get("ID"),
        manifest_sha256=manifest_digest,
        status="planned" if args.dry_run else "completed",
        verified=False,
    )
    if not args.dry_run:
        operation = (
            args.command
            if source_type == "directory"
            else ("copyto" if args.command == "copy" else "moveto")
        )
        # Explicit trash semantics also protect a provider-side move fallback.
        arguments = [
            operation,
            backend.remote_path(source),
            backend.remote_path(destination),
            "--immutable",
            "--drive-use-trash=true",
        ]
        if source_type == "directory":
            arguments.append("--create-empty-src-dirs")
            if args.command == "move":
                arguments.append("--delete-empty-src-dirs")
        backend.run(arguments, capture_output=False)
        after_entry = backend.resolve(destination)
        after_type, after = _manifest(backend, backend.remote_path(destination))
        if source_type != after_type or not _same_content(before, after):
            raise DriveToolError(
                "Destination verification failed; inspect both paths before retrying"
            )
        if args.command == "move":
            remaining = backend.stat(backend.remote_path(source))
            if remaining is not None:
                # Only an empty source directory is removable after verified move.
                if not remaining.get("IsDir"):
                    raise DriveToolError(
                        "Move left its source file; inspect both paths"
                    )
                backend.run(
                    ["rmdir", backend.remote_path(source), "--drive-use-trash=true"]
                )
            if backend.stat(backend.remote_path(source)) is not None:
                raise DriveToolError("Move left a source directory; inspect both paths")
        assert after_entry is not None
        payload.update(destination_id=after_entry.get("ID"), verified=True)
    _print_object(payload, args.format)
    return 0


def _run_mkdir(args: argparse.Namespace, backend: RcloneBackend) -> int:
    path = _child(backend, args.path)
    entry = backend.resolve(path, must_exist=False)
    if entry is not None and not entry.get("IsDir"):
        raise DriveToolError("Directory destination is an existing file")
    if entry is None and not args.dry_run:
        backend.run(["mkdir", backend.remote_path(path)])
        entry = backend.resolve(path)
    _print_object(
        _receipt(
            "mkdir",
            backend,
            path=path,
            status="planned" if args.dry_run else "completed",
            id=entry.get("ID") if entry else None,
        ),
        args.format,
    )
    return 0


def _run_quota(args: argparse.Namespace, backend: RcloneBackend) -> int:
    account = backend.remote_root.split(":", 1)[0] + ":"
    payload = backend.json(["about", account, "--json"])
    if not isinstance(payload, dict):
        raise DriveToolError("rclone returned an invalid quota response")
    _print_object(
        _receipt("quota", backend, scope="account", bytes=payload), args.format
    )
    return 0


def _run_inventory(args: argparse.Namespace, backend: RcloneBackend) -> int:
    path = backend.normalize_relative(args.path)
    backend.resolve(path)
    values = backend.json(
        ["lsjson", backend.remote_path(path), "--recursive", "--hash"]
    )
    if not isinstance(values, list):
        raise DriveToolError("rclone returned an invalid inventory")
    entries: list[dict[str, Any]] = []
    for item in values:
        relative = backend.normalize_relative(str(item["Path"]))
        entries.append(
            {
                "path": relative,
                "id": item.get("ID"),
                "type": "directory" if item.get("IsDir") else "file",
                "size_bytes": item.get("Size"),
                "hashes": item.get("Hashes", {}),
                "modified": item.get("ModTime"),
            }
        )
    entries.sort(key=lambda item: (item["path"], str(item["id"])))
    counts: dict[str, int] = {}
    for item in entries:
        counts[item["path"]] = counts.get(item["path"], 0) + 1
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    output = _resolve_local(args.output, must_exist=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = _receipt(
        "inventory",
        backend,
        schema_version=1,
        path=path,
        entries=entries,
        ambiguous_paths=duplicates,
    )
    # Inventory never rewrites a previous snapshot.
    try:
        with output.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError as error:
        raise DriveToolError(f"Inventory output already exists: {output}") from error
    _print_object(
        _receipt(
            "inventory",
            backend,
            path=path,
            output=str(output),
            entries=len(entries),
            ambiguous_paths=duplicates,
        ),
        args.format,
    )
    return 0


def _run_trash(args: argparse.Namespace, backend: RcloneBackend) -> int:
    path = _child(backend, args.path)
    # Capture configuration internally solely to check backend type; never print it.
    try:
        configuration = backend.json(["config", "dump"])
    except DriveToolError:
        raise DriveToolError(
            "Cannot verify remote type; rclone configuration output withheld"
        ) from None
    remote_name = backend.remote_root.split(":", 1)[0]
    if (
        not isinstance(configuration, dict)
        or configuration.get(remote_name, {}).get("type") != "drive"
    ):
        raise DriveToolError(
            "Trash requires a direct Google Drive remote; other backends are refused"
        )
    entry = backend.resolve(path)
    assert entry is not None
    _identity(entry, args.expected_id)
    if entry.get("IsDir") and not args.recursive:
        raise DriveToolError(
            "Trashing a directory requires --recursive for that exact subtree"
        )
    if not args.dry_run:
        operation = "purge" if entry.get("IsDir") else "deletefile"
        backend.run(
            [operation, backend.remote_path(path), "--drive-use-trash=true"],
            capture_output=False,
        )
        if backend.stat(backend.remote_path(path)) is not None:
            raise DriveToolError(
                "Trash readback still finds the source; inspect before retrying"
            )
    _print_object(
        _receipt(
            "trash",
            backend,
            path=path,
            id=entry.get("ID"),
            status="planned" if args.dry_run else "trashed",
            permanent=False,
        ),
        args.format,
    )
    return 0


def add_management_commands(subparsers: argparse._SubParsersAction[Any]) -> None:
    for name in ("copy", "move"):
        command = subparsers.add_parser(
            name, help=f"Verified Drive-to-Drive {name}; no overwrite."
        )
        command.add_argument("source")
        command.add_argument("destination")
        command.add_argument("--expected-id")
        command.add_argument("--dry-run", action="store_true")
    mkdir = subparsers.add_parser("mkdir", help="Create a project subdirectory.")
    mkdir.add_argument("path")
    mkdir.add_argument("--dry-run", action="store_true")
    subparsers.add_parser(
        "quota", help="Read account quota, including other services and trash."
    )
    inventory = subparsers.add_parser(
        "inventory",
        help="Save a versioned recursive inventory to a new local JSON file.",
    )
    inventory.add_argument("--path", default=".")
    inventory.add_argument("--output", required=True)
    trash = subparsers.add_parser(
        "trash",
        help="Move only an explicitly requested target into Google Drive trash.",
    )
    trash.add_argument("path")
    trash.add_argument("--recursive", action="store_true")
    trash.add_argument("--expected-id")
    trash.add_argument("--dry-run", action="store_true")
    for name in ("copy", "move", "mkdir", "quota", "inventory", "trash"):
        subparsers.choices[name].add_argument(
            "--format", choices=("json", "table"), default="json"
        )


def management_handlers() -> dict[
    str, Callable[[argparse.Namespace, RcloneBackend], int]
]:
    return {
        "copy": _run_relocate,
        "move": _run_relocate,
        "mkdir": _run_mkdir,
        "quota": _run_quota,
        "inventory": _run_inventory,
        "trash": _run_trash,
    }
