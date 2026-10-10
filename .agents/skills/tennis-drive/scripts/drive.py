"""JSON-first CLI for tennis-lab Google Drive storage."""

from __future__ import annotations

import argparse
import fnmatch
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from drive_core import (
    DEFAULT_REMOTE_ROOT,
    DriveToolError,
    OutputFormat,
    RcloneBackend,
    _assert_regular_local_tree,
    _entries_from_rclone,
    _normalized_hashes,
    _print_entries,
    _print_object,
    _resolve_local,
    _verify_paths,
)
from drive_management import add_management_commands, management_handlers


def _run_list(args: argparse.Namespace, backend: RcloneBackend) -> int:
    backend.resolve(args.path)
    target = backend.remote_path(args.path)
    values = backend.json(
        ["lsjson", target, "--max-depth", str(args.max_depth), "--no-mimetype"]
    )
    if not isinstance(values, list):
        raise DriveToolError("rclone returned an unexpected directory listing.")
    entries = _entries_from_rclone(args.path, values)
    _print_entries(entries, output_format=args.format, limit=args.limit)
    return 0


def _run_search(args: argparse.Namespace, backend: RcloneBackend) -> int:
    backend.resolve(args.path)
    target = backend.remote_path(args.path)
    arguments = ["lsjson", target, "--recursive", "--no-mimetype"]
    if args.max_depth is not None:
        arguments.extend(("--max-depth", str(args.max_depth)))
    values = backend.json(arguments)
    if not isinstance(values, list):
        raise DriveToolError("rclone returned an unexpected directory listing.")
    entries = _entries_from_rclone(args.path, values)
    matches = [
        entry
        for entry in entries
        if (args.type == "any" or entry.type == args.type)
        and fnmatch.fnmatchcase(PurePosixPath(entry.path).name, args.name)
    ]
    _print_entries(matches, output_format=args.format, limit=args.limit)
    return 0


def _transfer(
    *,
    backend: RcloneBackend,
    source: str,
    destination: str,
    source_is_directory: bool,
    destination_exists: bool,
    direction: Literal["upload", "download"],
    overwrite: bool,
    dry_run: bool,
    verify: bool,
    local_path: Path,
    drive_relative_path: str,
    output_format: OutputFormat,
) -> int:
    if destination_exists and not overwrite:
        raise DriveToolError(
            f"Destination already exists: {destination}. Pass --overwrite to update it."
        )
    arguments = [
        "copy" if source_is_directory else "copyto",
        source,
        destination,
        "--stats",
        "30s",
    ]
    if source_is_directory:
        arguments.append("--create-empty-src-dirs")
    if not overwrite:
        arguments.append("--immutable")
    if dry_run:
        arguments.extend(("--dry-run", "--verbose"))
    backend.run(arguments, capture_output=False)
    if not dry_run:
        backend.resolve(drive_relative_path)

    verified = False
    if verify and not dry_run:
        result = _verify_paths(
            backend=backend,
            local_path=local_path,
            drive_relative_path=drive_relative_path,
            download_missing_hashes=False,
        )
        allowed_extras = (
            source_is_directory
            and overwrite
            and (
                result.extra_on_drive
                if direction == "upload"
                else result.missing_on_drive
            )
        )
        disallowed_missing = (
            result.missing_on_drive if direction == "upload" else result.extra_on_drive
        )
        if (
            result.local_type != result.drive_type
            or result.changed
            or disallowed_missing
            or result.unverifiable
            or (
                not allowed_extras
                and (result.extra_on_drive or result.missing_on_drive)
            )
        ):
            raise DriveToolError(
                "Transferred content did not pass rclone hash verification."
            )
        verified = True

    payload = {
        "action": direction,
        "source": source,
        "destination": destination,
        "status": "planned" if dry_run else "completed",
        "dry_run": dry_run,
        "verified": verified,
    }
    _print_object(payload, output_format)
    return 0


def _run_upload(args: argparse.Namespace, backend: RcloneBackend) -> int:
    source = _resolve_local(args.source, must_exist=True)
    _assert_regular_local_tree(source)
    normalized_destination = backend.normalize_relative(args.destination)
    if normalized_destination == ".":
        raise DriveToolError("The configured Drive root cannot be overwritten.")
    destination = backend.remote_path(normalized_destination)
    destination_stat = backend.resolve(normalized_destination, must_exist=False)
    if (
        destination_stat is not None
        and bool(destination_stat.get("IsDir")) != source.is_dir()
    ):
        raise DriveToolError("Source and destination types do not match.")
    return _transfer(
        backend=backend,
        source=str(source),
        destination=destination,
        source_is_directory=source.is_dir(),
        destination_exists=destination_stat is not None,
        direction="upload",
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        verify=args.verify,
        local_path=source,
        drive_relative_path=normalized_destination,
        output_format=args.format,
    )


def _run_download(args: argparse.Namespace, backend: RcloneBackend) -> int:
    normalized_source = backend.normalize_relative(args.source)
    source_stat = backend.resolve(normalized_source)
    source = backend.remote_path(normalized_source)
    if source_stat is None:
        raise DriveToolError(f"Drive path does not exist: {args.source}")
    destination = _resolve_local(args.destination, must_exist=False)
    destination_exists = destination.exists()
    source_is_directory = bool(source_stat.get("IsDir", False))
    if destination_exists and destination.is_dir() != source_is_directory:
        raise DriveToolError("Source and destination types do not match.")
    return _transfer(
        backend=backend,
        source=source,
        destination=str(destination),
        source_is_directory=source_is_directory,
        destination_exists=destination_exists,
        direction="download",
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        verify=args.verify,
        local_path=destination,
        drive_relative_path=normalized_source,
        output_format=args.format,
    )


def _run_inspect(args: argparse.Namespace, backend: RcloneBackend) -> int:
    normalized_path = backend.normalize_relative(args.path)
    resolved = backend.resolve(normalized_path)
    target = backend.remote_path(normalized_path)
    stat = backend.stat(target, hashes=True) if args.checksum else resolved
    if stat is None:
        raise DriveToolError(f"Drive path does not exist: {args.path}")
    is_directory = bool(stat.get("IsDir", False))
    if args.checksum and is_directory:
        raise DriveToolError(
            "--checksum is only supported for files; use verify or inventory for directories."
        )
    payload: dict[str, Any] = {
        "path": normalized_path,
        "type": "directory" if is_directory else "file",
        "size_bytes": None if is_directory else int(stat.get("Size", 0)),
        "modified": str(stat.get("ModTime", "")),
        "id": resolved.get("ID") if resolved else None,
        "mime_type": str(stat.get("MimeType", "")),
    }
    if args.checksum:
        payload["hashes"] = _normalized_hashes(stat)
    if is_directory and not args.metadata_only:
        size = backend.json(["size", target, "--json"])
        payload["file_count"] = int(size.get("count", 0))
        payload["size_bytes"] = int(size.get("bytes", 0))
    _print_object(payload, args.format)
    return 0


def _run_verify(args: argparse.Namespace, backend: RcloneBackend) -> int:
    local_path = _resolve_local(args.local_path, must_exist=True)
    _assert_regular_local_tree(local_path)
    normalized_drive_path = backend.normalize_relative(args.drive_path)
    backend.resolve(normalized_drive_path)
    result = _verify_paths(
        backend=backend,
        local_path=local_path,
        drive_relative_path=normalized_drive_path,
        download_missing_hashes=args.download,
    )
    payload = {
        "local_path": str(local_path),
        "drive_path": normalized_drive_path,
        **asdict(result),
    }
    _print_object(payload, args.format)
    return 0 if result.matches else 3


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="json",
        help="Output format (default: JSON).",
    )


def _add_transfer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Update existing files without deleting unrelated destination files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Ask rclone to plan without writing."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare common rclone hashes after transfer.",
    )
    _add_format(parser)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser shared by all shell wrappers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--remote-root",
        default=os.environ.get("TENNIS_LAB_DRIVE_REMOTE", DEFAULT_REMOTE_ROOT),
        help="Constrained rclone project root.",
    )
    parser.add_argument(
        "--rclone-bin",
        default=os.environ.get("RCLONE_BIN", "rclone"),
        help="rclone executable.",
    )
    parser.add_argument("--timeout-seconds", type=_positive_int, default=3600)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List Drive entries.")
    list_parser.add_argument("--path", default=".", help="Drive-relative start path.")
    list_parser.add_argument(
        "--max-depth", type=_positive_int, default=2, help="Maximum traversal depth."
    )
    list_parser.add_argument(
        "--limit", type=_positive_int, default=200, help="Maximum returned entries."
    )
    _add_format(list_parser)

    search_parser = subparsers.add_parser("search", help="Search Drive entry names.")
    search_parser.add_argument(
        "--name", default="*", help="Case-sensitive glob matched against entry names."
    )
    search_parser.add_argument("--path", default=".", help="Drive-relative start path.")
    search_parser.add_argument(
        "--type", choices=("any", "file", "directory"), default="any"
    )
    search_parser.add_argument(
        "--max-depth", type=_positive_int, default=None, help="Maximum traversal depth."
    )
    search_parser.add_argument(
        "--limit", type=_positive_int, default=200, help="Maximum returned entries."
    )
    _add_format(search_parser)

    upload_parser = subparsers.add_parser("upload", help="Copy local data to Drive.")
    upload_parser.add_argument("source", help="Existing local file or directory.")
    upload_parser.add_argument("destination", help="Drive-relative destination path.")
    _add_transfer_arguments(upload_parser)

    download_parser = subparsers.add_parser(
        "download", help="Copy Drive data to the local machine."
    )
    download_parser.add_argument("source", help="Existing Drive-relative source path.")
    download_parser.add_argument("destination", help="Exact local destination path.")
    _add_transfer_arguments(download_parser)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect one Drive entry.")
    inspect_parser.add_argument("path", help="Existing Drive-relative path.")
    inspect_parser.add_argument("--metadata-only", action="store_true", help="Return identity without recursively calculating directory size.")
    inspect_parser.add_argument(
        "--checksum", action="store_true", help="Return hashes exposed by rclone."
    )
    _add_format(inspect_parser)

    verify_parser = subparsers.add_parser(
        "verify", help="Compare local and Drive content using rclone hashes."
    )
    verify_parser.add_argument("local_path", help="Existing local file or directory.")
    verify_parser.add_argument("drive_path", help="Existing Drive-relative path.")
    verify_parser.add_argument(
        "--download",
        action="store_true",
        help="Download files for SHA-256 when no common remote hash exists.",
    )
    _add_format(verify_parser)
    add_management_commands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a Drive utility command and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        backend = RcloneBackend(
            remote_root=args.remote_root,
            executable=args.rclone_bin,
            timeout=args.timeout_seconds,
        )
        handlers = {
            "list": _run_list,
            "search": _run_search,
            "upload": _run_upload,
            "download": _run_download,
            "inspect": _run_inspect,
            "verify": _run_verify,
            **management_handlers(),
        }
        return handlers[args.command](args, backend)
    except (DriveToolError, OSError) as exc:
        _print_object(
            {"ok": False, "command": args.command, "error": str(exc)}, args.format
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
