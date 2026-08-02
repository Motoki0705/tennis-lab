"""Local Google Drive utilities backed by rclone."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

DEFAULT_REMOTE_ROOT = "gdrive:tennis_lab"
OutputFormat = Literal["table", "json"]
ManifestEntry = tuple[Literal["file", "directory"], int | None, dict[str, str]]


class DriveToolError(RuntimeError):
    """Raised when an rclone operation is unsafe or fails."""


@dataclass(frozen=True)
class Entry:
    """One Drive entry returned by list or search."""

    path: str
    type: Literal["file", "directory"]
    size_bytes: int | None
    modified: str


@dataclass(frozen=True)
class Verification:
    """Comparison result for a local path and Drive path."""

    matches: bool
    local_type: str
    drive_type: str
    changed: list[str]
    missing_on_drive: list[str]
    extra_on_drive: list[str]
    unverifiable: list[str]


class RcloneBackend:
    """Execute rclone while constraining all remote paths to one root."""

    def __init__(self, *, remote_root: str, executable: str) -> None:
        if shutil.which(executable) is None:
            raise DriveToolError(
                f"rclone executable not found: {executable}. Install rclone first."
            )
        self.executable = executable
        self.remote_root = self._validate_remote_root(remote_root)

    @staticmethod
    def _validate_remote_root(remote_root: str) -> str:
        if "\n" in remote_root or "\r" in remote_root:
            raise DriveToolError("The rclone remote root cannot contain newlines.")
        if ":" not in remote_root:
            raise DriveToolError(
                f"Invalid rclone remote root: {remote_root}. Expected remote:path."
            )
        remote_name, root_path = remote_root.split(":", maxsplit=1)
        if not remote_name or not root_path or root_path == "/":
            raise DriveToolError(
                "The rclone remote root must include a remote name and a non-root path."
            )
        if ".." in PurePosixPath(root_path).parts or "\\" in root_path:
            raise DriveToolError(f"Unsafe rclone remote root: {remote_root}")
        return f"{remote_name}:{root_path.rstrip('/')}"

    @staticmethod
    def normalize_relative(relative_path: str) -> str:
        """Validate and normalize a Drive-root-relative POSIX path."""
        if any(character in relative_path for character in ("\n", "\r", "\\", ":")):
            raise DriveToolError(f"Unsafe Drive-relative path: {relative_path}")
        path = PurePosixPath(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise DriveToolError(f"Unsafe Drive-relative path: {relative_path}")
        normalized = path.as_posix().rstrip("/")
        return "." if normalized in ("", ".") else normalized

    def remote_path(self, relative_path: str) -> str:
        """Return an rclone path below the configured remote root."""
        normalized = self.normalize_relative(relative_path)
        if normalized == ".":
            return self.remote_root
        return f"{self.remote_root}/{normalized}"

    def run(
        self,
        arguments: Sequence[str],
        *,
        capture_output: bool = True,
        allow_not_found: bool = False,
    ) -> subprocess.CompletedProcess[str] | None:
        """Run rclone, mapping its directory-not-found exit code when requested."""
        try:
            command = [self.executable, *arguments]
            if capture_output:
                result = subprocess.run(
                    command, text=True, capture_output=True, check=False
                )
            else:
                result = subprocess.run(
                    command,
                    text=True,
                    stdout=sys.stderr,
                    stderr=sys.stderr,
                    check=False,
                )
        except OSError as exc:
            raise DriveToolError(f"Could not execute rclone: {exc}") from exc
        if result.returncode == 0:
            return result
        if allow_not_found and result.returncode == 3:
            return None
        message = (result.stderr or result.stdout or "unknown rclone error").strip()
        raise DriveToolError(f"rclone {' '.join(arguments[:2])} failed: {message}")

    def json(self, arguments: Sequence[str]) -> Any:
        """Run rclone and decode its JSON response."""
        result = self.run(arguments)
        assert result is not None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise DriveToolError("rclone returned invalid JSON.") from exc

    def stat(self, location: str, *, hashes: bool = False) -> dict[str, Any] | None:
        """Return one local or remote entry, or None when it does not exist."""
        arguments = ["lsjson", location, "--stat"]
        if hashes:
            arguments.append("--hash")
        result = self.run(arguments, allow_not_found=True)
        if result is None:
            return None
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise DriveToolError("rclone returned invalid stat JSON.") from exc
        if not isinstance(value, dict):
            raise DriveToolError("rclone returned an unexpected stat response.")
        return value


def _relative_output_path(start_path: str, item_path: str) -> str:
    normalized_start = RcloneBackend.normalize_relative(start_path)
    normalized_item = PurePosixPath(item_path).as_posix()
    if normalized_start == ".":
        return normalized_item
    if normalized_item in ("", "."):
        return normalized_start
    return f"{normalized_start}/{normalized_item}"


def _entries_from_rclone(
    start_path: str, values: Iterable[dict[str, Any]]
) -> list[Entry]:
    entries = []
    for value in values:
        is_directory = bool(value.get("IsDir", False))
        size = value.get("Size")
        entries.append(
            Entry(
                path=_relative_output_path(start_path, str(value.get("Path", ""))),
                type="directory" if is_directory else "file",
                size_bytes=None if is_directory else int(size or 0),
                modified=str(value.get("ModTime", "")),
            )
        )
    return sorted(entries, key=lambda entry: entry.path.casefold())


def _print_entries(
    entries: list[Entry], *, output_format: OutputFormat, limit: int
) -> None:
    truncated = len(entries) > limit
    displayed = entries[:limit]
    if output_format == "json":
        print(
            json.dumps(
                {
                    "count": len(displayed),
                    "truncated": truncated,
                    "entries": [asdict(entry) for entry in displayed],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print("TYPE\tSIZE_BYTES\tMODIFIED\tPATH")
    for entry in displayed:
        size = "-" if entry.size_bytes is None else str(entry.size_bytes)
        print(f"{entry.type}\t{size}\t{entry.modified}\t{entry.path}")
    if truncated:
        print("[drive-tools] Results were truncated by --limit.", file=sys.stderr)


def _print_object(payload: dict[str, Any], output_format: OutputFormat) -> None:
    if output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for key, value in payload.items():
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value) or "-"
        elif isinstance(value, dict):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
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


def _assert_regular_local_tree(path: Path) -> None:
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


def _normalized_hashes(value: dict[str, Any]) -> dict[str, str]:
    hashes = value.get("Hashes", {})
    if not isinstance(hashes, dict):
        return {}
    return {
        str(name).casefold().replace("-", ""): str(digest).casefold()
        for name, digest in hashes.items()
        if digest
    }


def _manifest(
    backend: RcloneBackend, location: str
) -> tuple[Literal["file", "directory"], dict[str, ManifestEntry]]:
    stat = backend.stat(location, hashes=True)
    if stat is None:
        raise DriveToolError(f"Path does not exist: {location}")
    if not bool(stat.get("IsDir", False)):
        return "file", {
            ".": ("file", int(stat.get("Size", 0)), _normalized_hashes(stat))
        }

    values = backend.json(["lsjson", location, "--recursive", "--hash"])
    if not isinstance(values, list):
        raise DriveToolError("rclone returned an unexpected directory listing.")
    entries: dict[str, ManifestEntry] = {}
    for value in values:
        path = str(value.get("Path", ""))
        if bool(value.get("IsDir", False)):
            entries[path] = ("directory", None, {})
        else:
            entries[path] = (
                "file",
                int(value.get("Size", 0)),
                _normalized_hashes(value),
            )
    return "directory", entries


def _sha256_local(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_remote(backend: RcloneBackend, remote_path: str) -> str:
    digest = hashlib.sha256()
    process = subprocess.Popen(
        [backend.executable, "cat", remote_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    stdout = process.stdout
    for chunk in iter(lambda: stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    _, stderr = process.communicate()
    if process.returncode != 0:
        raise DriveToolError(
            f"rclone cat failed: {stderr.decode(errors='replace').strip()}"
        )
    return digest.hexdigest()


def _compare_file(
    *,
    local_path: Path,
    drive_path: str,
    local_entry: ManifestEntry,
    drive_entry: ManifestEntry,
    backend: RcloneBackend,
    download_missing_hashes: bool,
) -> Literal["match", "changed", "unverifiable"]:
    if local_entry[0] != drive_entry[0] or local_entry[1] != drive_entry[1]:
        return "changed"
    if local_entry[0] == "directory":
        return "match"

    local_hashes = local_entry[2]
    drive_hashes = drive_entry[2]
    for hash_name in ("sha256", "sha1", "md5"):
        if hash_name in local_hashes and hash_name in drive_hashes:
            return (
                "match"
                if local_hashes[hash_name] == drive_hashes[hash_name]
                else "changed"
            )
    if not download_missing_hashes:
        return "unverifiable"
    return (
        "match"
        if _sha256_local(local_path) == _sha256_remote(backend, drive_path)
        else "changed"
    )


def _verify_paths(
    *,
    backend: RcloneBackend,
    local_path: Path,
    drive_relative_path: str,
    download_missing_hashes: bool,
) -> Verification:
    drive_path = backend.remote_path(drive_relative_path)
    local_type, local_manifest = _manifest(backend, str(local_path))
    drive_type, drive_manifest = _manifest(backend, drive_path)
    local_names = set(local_manifest)
    drive_names = set(drive_manifest)
    changed: list[str] = []
    unverifiable: list[str] = []
    for name in sorted(local_names & drive_names):
        local_item_path = local_path if name == "." else local_path / name
        remote_item_path = drive_path if name == "." else f"{drive_path}/{name}"
        comparison = _compare_file(
            local_path=local_item_path,
            drive_path=remote_item_path,
            local_entry=local_manifest[name],
            drive_entry=drive_manifest[name],
            backend=backend,
            download_missing_hashes=download_missing_hashes,
        )
        if comparison == "changed":
            changed.append(name)
        elif comparison == "unverifiable":
            unverifiable.append(name)
    missing_on_drive = sorted(local_names - drive_names)
    extra_on_drive = sorted(drive_names - local_names)
    matches = (
        local_type == drive_type
        and not changed
        and not missing_on_drive
        and not extra_on_drive
        and not unverifiable
    )
    return Verification(
        matches=matches,
        local_type=local_type,
        drive_type=drive_type,
        changed=changed,
        missing_on_drive=missing_on_drive,
        extra_on_drive=extra_on_drive,
        unverifiable=unverifiable,
    )


def _run_list(args: argparse.Namespace, backend: RcloneBackend) -> int:
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
    arguments = ["copyto", source, destination, "--progress"]
    if not overwrite:
        arguments.append("--immutable")
    if dry_run:
        arguments.extend(("--dry-run", "--verbose"))
    backend.run(arguments, capture_output=False)

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
    destination_stat = backend.stat(destination)
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
    source = backend.remote_path(normalized_source)
    source_stat = backend.stat(source)
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
    target = backend.remote_path(normalized_path)
    stat = backend.stat(target, hashes=args.checksum)
    if stat is None:
        raise DriveToolError(f"Drive path does not exist: {args.path}")
    is_directory = bool(stat.get("IsDir", False))
    if args.checksum and is_directory:
        raise DriveToolError(
            "--checksum is only supported for files; use verify_transfer.sh for directories."
        )
    payload: dict[str, Any] = {
        "path": normalized_path,
        "type": "directory" if is_directory else "file",
        "size_bytes": None if is_directory else int(stat.get("Size", 0)),
        "modified": str(stat.get("ModTime", "")),
        "mime_type": str(stat.get("MimeType", "")),
    }
    if args.checksum:
        payload["hashes"] = _normalized_hashes(stat)
    if is_directory:
        size = backend.json(["size", target, "--json"])
        payload["file_count"] = int(size.get("count", 0))
        payload["size_bytes"] = int(size.get("bytes", 0))
    _print_object(payload, args.format)
    return 0


def _run_verify(args: argparse.Namespace, backend: RcloneBackend) -> int:
    local_path = _resolve_local(args.local_path, must_exist=True)
    _assert_regular_local_tree(local_path)
    normalized_drive_path = backend.normalize_relative(args.drive_path)
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
        "--format", choices=("table", "json"), default="table", help="Output format."
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
        default=DEFAULT_REMOTE_ROOT,
        help="Constrained rclone root (normally set by the shell wrapper).",
    )
    parser.add_argument(
        "--rclone-bin",
        default="rclone",
        help="rclone executable (normally set by the shell wrapper).",
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
        "download", help="Copy Drive data to the local machine."
    )
    download_parser.add_argument("source", help="Existing Drive-relative source path.")
    download_parser.add_argument("destination", help="Exact local destination path.")
    _add_transfer_arguments(download_parser)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect one Drive entry.")
    inspect_parser.add_argument("path", help="Existing Drive-relative path.")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a Drive utility command and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        backend = RcloneBackend(
            remote_root=args.remote_root, executable=args.rclone_bin
        )
        handlers = {
            "list": _run_list,
            "search": _run_search,
            "upload": _run_upload,
            "download": _run_download,
            "inspect": _run_inspect,
            "verify": _run_verify,
        }
        return handlers[args.command](args, backend)
    except DriveToolError as exc:
        print(f"[drive-tools] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
