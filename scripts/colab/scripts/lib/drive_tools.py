"""Safe filesystem operations for the Colab Google Drive shell wrappers."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import mimetypes
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

DEFAULT_DRIVE_ROOT = Path("/content/drive/MyDrive/tennis_lab")
OutputFormat = Literal["table", "json"]
ManifestEntry = tuple[Literal["file", "directory"], int | None, str | None]


class DriveToolError(RuntimeError):
    """Raised when a requested filesystem operation is unsafe or invalid."""


@dataclass(frozen=True)
class Entry:
    """One filesystem entry returned by list or search."""

    path: str
    type: str
    size_bytes: int | None
    modified: str


@dataclass(frozen=True)
class Snapshot:
    """Content snapshot used to verify a file or directory tree."""

    type: Literal["file", "directory"]
    size_bytes: int
    sha256: str
    entries: dict[str, ManifestEntry]


class DriveContext:
    """Resolve Drive-relative paths without permitting root escapes."""

    def __init__(self, root: Path) -> None:
        try:
            resolved_root = root.expanduser().resolve(strict=True)
        except FileNotFoundError as exc:
            raise DriveToolError(
                f"Drive root does not exist: {root}. Mount Google Drive first or set "
                "TENNIS_LAB_DRIVE_ROOT."
            ) from exc
        if not resolved_root.is_dir():
            raise DriveToolError(f"Drive root is not a directory: {resolved_root}")
        self.root = resolved_root

    def resolve(self, relative_path: str, *, must_exist: bool = False) -> Path:
        """Resolve a relative path and verify that it remains below Drive root."""
        requested = Path(relative_path)
        if requested.is_absolute():
            raise DriveToolError(
                f"Drive paths must be relative to {self.root}: {relative_path}"
            )
        if ".." in requested.parts:
            raise DriveToolError(f"Drive paths cannot contain '..': {relative_path}")

        current = self.root
        for part in requested.parts:
            if part in ("", "."):
                continue
            current /= part
            if current.is_symlink():
                raise DriveToolError(
                    f"Drive paths cannot traverse symbolic links: {relative_path}"
                )
        resolved = (self.root / requested).resolve(strict=False)
        if resolved != self.root and self.root not in resolved.parents:
            raise DriveToolError(
                f"Drive path escapes the allowed root: {relative_path}"
            )
        if must_exist and not resolved.exists():
            raise DriveToolError(f"Drive path does not exist: {relative_path}")
        return resolved

    def relative(self, path: Path) -> str:
        """Return a stable POSIX path relative to Drive root."""
        relative = path.relative_to(self.root)
        return "." if relative == Path(".") else relative.as_posix()


def _timestamp(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=UTC).isoformat(timespec="seconds")


def _entry_from_dir_entry(item: os.DirEntry[str], relative_path: str) -> Entry:
    stat_result = item.stat(follow_symlinks=False)
    if item.is_symlink():
        entry_type = "symlink"
        size: int | None = stat_result.st_size
    elif item.is_dir(follow_symlinks=False):
        entry_type = "directory"
        size = None
    elif item.is_file(follow_symlinks=False):
        entry_type = "file"
        size = stat_result.st_size
    else:
        entry_type = "other"
        size = stat_result.st_size
    return Entry(
        path=relative_path,
        type=entry_type,
        size_bytes=size,
        modified=_timestamp(stat_result.st_mtime),
    )


def _walk_entries(
    context: DriveContext, start: Path, max_depth: int | None
) -> Iterable[Entry]:
    start_relative = context.relative(start)

    def visit(directory: Path, relative_directory: str, depth: int) -> Iterable[Entry]:
        try:
            children = sorted(
                os.scandir(directory), key=lambda item: item.name.casefold()
            )
        except OSError as exc:
            raise DriveToolError(
                f"Could not read directory {directory}: {exc}"
            ) from exc

        for child in children:
            relative_path = (
                child.name
                if relative_directory == "."
                else f"{relative_directory}/{child.name}"
            )
            entry = _entry_from_dir_entry(child, relative_path)
            yield entry
            if entry.type == "directory" and (max_depth is None or depth < max_depth):
                yield from visit(Path(child.path), relative_path, depth + 1)

    if start.is_file():
        stat_result = start.stat()
        yield Entry(
            path=start_relative,
            type="file",
            size_bytes=stat_result.st_size,
            modified=_timestamp(stat_result.st_mtime),
        )
        return
    if not start.is_dir():
        raise DriveToolError(f"Unsupported Drive entry type: {start_relative}")
    yield from visit(start, start_relative, 1)


def _limited(entries: Iterable[Entry], limit: int) -> tuple[list[Entry], bool]:
    collected: list[Entry] = []
    for entry in entries:
        if len(collected) == limit:
            return collected, True
        collected.append(entry)
    return collected, False


def _print_entries(
    entries: list[Entry], *, output_format: OutputFormat, truncated: bool
) -> None:
    if output_format == "json":
        print(
            json.dumps(
                {
                    "count": len(entries),
                    "truncated": truncated,
                    "entries": [asdict(entry) for entry in entries],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print("TYPE\tSIZE_BYTES\tMODIFIED\tPATH")
    for entry in entries:
        size = "-" if entry.size_bytes is None else str(entry.size_bytes)
        print(f"{entry.type}\t{size}\t{entry.modified}\t{entry.path}")
    if truncated:
        print("[colab-drive-tools] Results were truncated by --limit.", file=sys.stderr)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_regular_tree(path: Path) -> None:
    if path.is_symlink():
        raise DriveToolError(f"Symbolic links are not supported for transfer: {path}")
    if path.is_file():
        return
    if not path.is_dir():
        raise DriveToolError(
            f"Only regular files and directories are supported: {path}"
        )
    for current_root, directory_names, file_names in os.walk(path):
        current = Path(current_root)
        for name in [*directory_names, *file_names]:
            candidate = current / name
            if candidate.is_symlink():
                raise DriveToolError(
                    f"Symbolic links are not supported for transfer: {candidate}"
                )


def _snapshot(path: Path) -> Snapshot:
    _assert_regular_tree(path)
    if path.is_file():
        size = path.stat().st_size
        digest = _sha256_file(path)
        return Snapshot(
            type="file",
            size_bytes=size,
            sha256=digest,
            entries={".": ("file", size, digest)},
        )

    entries: dict[str, ManifestEntry] = {}
    total_size = 0
    for candidate in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(path).as_posix()
        if candidate.is_dir():
            entries[relative] = ("directory", None, None)
        elif candidate.is_file():
            size = candidate.stat().st_size
            digest = _sha256_file(candidate)
            entries[relative] = ("file", size, digest)
            total_size += size
    manifest = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return Snapshot(
        type="directory",
        size_bytes=total_size,
        sha256=hashlib.sha256(manifest).hexdigest(),
        entries=entries,
    )


def _verify_paths(local_path: Path, drive_path: Path) -> dict[str, Any]:
    local_snapshot = _snapshot(local_path)
    drive_snapshot = _snapshot(drive_path)
    local_names = set(local_snapshot.entries)
    drive_names = set(drive_snapshot.entries)
    changed = sorted(
        name
        for name in local_names & drive_names
        if local_snapshot.entries[name] != drive_snapshot.entries[name]
    )
    missing_on_drive = sorted(local_names - drive_names)
    extra_on_drive = sorted(drive_names - local_names)
    matches = (
        local_snapshot.type == drive_snapshot.type
        and not changed
        and not missing_on_drive
        and not extra_on_drive
    )
    return {
        "matches": matches,
        "local_type": local_snapshot.type,
        "drive_type": drive_snapshot.type,
        "local_size_bytes": local_snapshot.size_bytes,
        "drive_size_bytes": drive_snapshot.size_bytes,
        "local_sha256": local_snapshot.sha256,
        "drive_sha256": drive_snapshot.sha256,
        "changed": changed,
        "missing_on_drive": missing_on_drive,
        "extra_on_drive": extra_on_drive,
    }


def _print_object(payload: dict[str, Any], output_format: OutputFormat) -> None:
    if output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for key, value in payload.items():
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value) or "-"
        else:
            rendered = str(value).lower() if isinstance(value, bool) else str(value)
        print(f"{key}: {rendered}")


def _resolve_local(path_text: str, *, must_exist: bool) -> Path:
    requested = Path(path_text).expanduser()
    if requested.is_symlink():
        raise DriveToolError(f"Local paths cannot be symbolic links: {path_text}")
    resolved = requested.resolve(strict=False)
    if must_exist and not resolved.exists():
        raise DriveToolError(f"Local path does not exist: {path_text}")
    return resolved


def _copy_to_staging(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    prefix = f".{destination.name}.transfer-"
    staging_parent = Path(tempfile.mkdtemp(prefix=prefix, dir=destination.parent))
    staging_path = staging_parent / "payload"
    try:
        if source.is_dir():
            shutil.copytree(source, staging_path)
        else:
            shutil.copy2(source, staging_path)
    except Exception:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise
    return staging_path


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _commit_staged_copy(staging_path: Path, destination: Path) -> None:
    staging_parent = staging_path.parent
    backup_path = staging_parent / "previous"
    moved_existing = False
    try:
        if destination.exists() or destination.is_symlink():
            destination.replace(backup_path)
            moved_existing = True
        staging_path.replace(destination)
    except Exception:
        if moved_existing and backup_path.exists() and not destination.exists():
            backup_path.replace(destination)
        raise
    finally:
        if staging_path.exists():
            _remove_path(staging_path)

    if backup_path.exists() or backup_path.is_symlink():
        _remove_path(backup_path)
    staging_parent.rmdir()


def _transfer(
    *,
    source: Path,
    destination: Path,
    direction: Literal["upload", "download"],
    overwrite: bool,
    dry_run: bool,
    verify: bool,
    output_format: OutputFormat,
) -> int:
    _assert_regular_tree(source)
    if destination == source:
        raise DriveToolError("Source and destination resolve to the same path.")
    if source.is_dir() and (
        source in destination.parents or destination in source.parents
    ):
        raise DriveToolError("Source and destination directory trees overlap.")
    if (destination.exists() or destination.is_symlink()) and not overwrite:
        raise DriveToolError(
            f"Destination already exists: {destination}. Pass --overwrite to replace it."
        )

    payload: dict[str, Any] = {
        "action": direction,
        "source": str(source),
        "destination": str(destination),
        "dry_run": dry_run,
        "verified": False,
    }
    if dry_run:
        payload["status"] = "planned"
        _print_object(payload, output_format)
        return 0

    staging_path = _copy_to_staging(source, destination)
    try:
        if verify:
            if direction == "upload":
                verification = _verify_paths(source, staging_path)
            else:
                verification = _verify_paths(staging_path, source)
            if not verification["matches"]:
                raise DriveToolError("Staged transfer failed checksum verification.")
            payload["verified"] = True
        _commit_staged_copy(staging_path, destination)
    except Exception:
        if staging_path.parent.exists():
            shutil.rmtree(staging_path.parent, ignore_errors=True)
        raise

    payload["status"] = "completed"
    _print_object(payload, output_format)
    return 0


def _run_list(args: argparse.Namespace, context: DriveContext) -> int:
    target = context.resolve(args.path, must_exist=True)
    entries, truncated = _limited(
        _walk_entries(context, target, args.max_depth), args.limit
    )
    _print_entries(entries, output_format=args.format, truncated=truncated)
    return 0


def _run_search(args: argparse.Namespace, context: DriveContext) -> int:
    target = context.resolve(args.path, must_exist=True)

    def matches() -> Iterable[Entry]:
        for entry in _walk_entries(context, target, args.max_depth):
            type_matches = args.type == "any" or entry.type == args.type
            name_matches = fnmatch.fnmatchcase(Path(entry.path).name, args.name)
            if type_matches and name_matches:
                yield entry

    entries, truncated = _limited(matches(), args.limit)
    _print_entries(entries, output_format=args.format, truncated=truncated)
    return 0


def _run_upload(args: argparse.Namespace, context: DriveContext) -> int:
    source = _resolve_local(args.source, must_exist=True)
    destination = context.resolve(args.destination)
    if destination == context.root:
        raise DriveToolError("The Drive root itself cannot be a transfer destination.")
    return _transfer(
        source=source,
        destination=destination,
        direction="upload",
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        verify=args.verify,
        output_format=args.format,
    )


def _run_download(args: argparse.Namespace, context: DriveContext) -> int:
    source = context.resolve(args.source, must_exist=True)
    destination = _resolve_local(args.destination, must_exist=False)
    return _transfer(
        source=source,
        destination=destination,
        direction="download",
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        verify=args.verify,
        output_format=args.format,
    )


def _run_inspect(args: argparse.Namespace, context: DriveContext) -> int:
    target = context.resolve(args.path, must_exist=True)
    snapshot = _snapshot(target) if args.checksum else None
    stat_result = target.stat()
    payload: dict[str, Any] = {
        "path": context.relative(target),
        "type": "directory" if target.is_dir() else "file",
        "size_bytes": snapshot.size_bytes if snapshot else stat_result.st_size,
        "modified": _timestamp(stat_result.st_mtime),
    }
    if target.is_file():
        payload["mime_type"] = (
            mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        )
    if snapshot is not None:
        payload["sha256"] = snapshot.sha256
        payload["file_count"] = sum(
            entry[0] == "file" for entry in snapshot.entries.values()
        )
        payload["directory_count"] = sum(
            entry[0] == "directory" for entry in snapshot.entries.values()
        )
    _print_object(payload, args.format)
    return 0


def _run_verify(args: argparse.Namespace, context: DriveContext) -> int:
    local_path = _resolve_local(args.local_path, must_exist=True)
    drive_path = context.resolve(args.drive_path, must_exist=True)
    payload = _verify_paths(local_path, drive_path)
    payload = {
        "local_path": str(local_path),
        "drive_path": context.relative(drive_path),
        **payload,
    }
    _print_object(payload, args.format)
    return 0 if payload["matches"] else 3


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _add_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format", choices=("table", "json"), default="table", help="Output format."
    )


def _add_transfer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing destination."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show the operation without copying."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify SHA-256 checksums before publishing the copied data.",
    )
    _add_format(parser)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser shared by all shell wrappers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=DEFAULT_DRIVE_ROOT,
        help="Mounted tennis_lab directory (normally set by the shell wrapper).",
    )
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
        "download", help="Copy Drive data to the local runtime."
    )
    download_parser.add_argument("source", help="Existing Drive-relative source path.")
    download_parser.add_argument("destination", help="Exact local destination path.")
    _add_transfer_arguments(download_parser)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect one Drive entry.")
    inspect_parser.add_argument("path", help="Existing Drive-relative path.")
    inspect_parser.add_argument(
        "--checksum",
        action="store_true",
        help="Calculate a SHA-256 file or directory-manifest digest.",
    )
    _add_format(inspect_parser)

    verify_parser = subparsers.add_parser(
        "verify", help="Compare local and Drive content using SHA-256."
    )
    verify_parser.add_argument("local_path", help="Existing local file or directory.")
    verify_parser.add_argument("drive_path", help="Existing Drive-relative path.")
    _add_format(verify_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a Drive utility command and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        context = DriveContext(args.drive_root)
        handlers = {
            "list": _run_list,
            "search": _run_search,
            "upload": _run_upload,
            "download": _run_download,
            "inspect": _run_inspect,
            "verify": _run_verify,
        }
        return handlers[args.command](args, context)
    except (DriveToolError, OSError) as exc:
        print(f"[colab-drive-tools] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
