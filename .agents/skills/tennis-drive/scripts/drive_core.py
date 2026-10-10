"""Local Google Drive utilities backed by rclone."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
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
    id: str | None


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

    def __init__(
        self, *, remote_root: str, executable: str, timeout: int = 3600
    ) -> None:
        if shutil.which(executable) is None:
            raise DriveToolError(
                f"rclone executable not found: {executable}. Install rclone first."
            )
        self.executable = executable
        self.remote_root = self._validate_remote_root(remote_root)
        if timeout < 1:
            raise DriveToolError("timeout must be positive")
        self.timeout = timeout

    @staticmethod
    def _validate_remote_root(remote_root: str) -> str:
        if "\n" in remote_root or "\r" in remote_root:
            raise DriveToolError("The rclone remote root cannot contain newlines.")
        if ":" not in remote_root:
            raise DriveToolError(
                f"Invalid rclone remote root: {remote_root}. Expected remote:path."
            )
        remote_name, root_path = remote_root.split(":", maxsplit=1)
        if (
            not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_. -]*", remote_name)
            or not root_path
            or root_path == "/"
        ):
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

    def resolve(
        self, relative_path: str, *, must_exist: bool = True
    ) -> dict[str, Any] | None:
        """Reject duplicate names in every existing component, including the root.

        rclone stat alone chooses one of same-named Drive objects. A listing is
        required to distinguish a missing target from an ambiguous one. This is
        a preflight/readback check, not an atomic lock against other Drive clients.
        """
        normalized = self.normalize_relative(relative_path)
        remote, root = self.remote_root.split(":", 1)
        components = list(PurePosixPath(root).parts)
        root_components = len(components)
        if normalized != ".":
            components.extend(PurePosixPath(normalized).parts)
        parent = f"{remote}:"
        found: dict[str, Any] | None = None
        for index, name in enumerate(components):
            values = self.json(["lsjson", parent, "--max-depth", "1", "--no-mimetype"])
            if not isinstance(values, list):
                raise DriveToolError("rclone returned an unexpected listing")
            matches = [
                item for item in values if item.get("Name", item.get("Path")) == name
            ]
            if len(matches) > 1:
                identifiers = [item.get("ID") for item in matches]
                raise DriveToolError(
                    f"Ambiguous Drive path {parent}/{name}; IDs={identifiers}"
                )
            if not matches:
                if must_exist or index < root_components:
                    raise DriveToolError(f"Drive path does not exist: {parent}/{name}")
                return None
            found = matches[0]
            if index < len(components) - 1 and not found.get("IsDir"):
                raise DriveToolError(
                    f"Path component is not a directory: {parent}/{name}"
                )
            parent = parent.rstrip("/") + ("" if parent.endswith(":") else "/") + name
        return found

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
                    command,
                    text=True,
                    capture_output=True,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    timeout=self.timeout,
                )
            else:
                result = subprocess.run(
                    command,
                    text=True,
                    stdout=sys.stderr,
                    stderr=sys.stderr,
                    check=False,
                    stdin=subprocess.DEVNULL,
                    timeout=self.timeout,
                )
        except subprocess.TimeoutExpired as exc:
            raise DriveToolError(
                f"rclone operation timed out after {self.timeout}s; inspect destination before retrying"
            ) from exc
        except OSError as exc:
            raise DriveToolError(f"Could not execute rclone: {exc}") from exc
        if result.returncode == 0:
            return result
        if allow_not_found and result.returncode in (3, 4):
            return None
        message = (
            result.stderr
            or result.stdout
            or f"exit code {result.returncode}; see stderr logs"
        ).strip()
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
                id=str(value["ID"]) if value.get("ID") else None,
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
                    "schema_version": 1,
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
        print(
            json.dumps({"schema_version": 1, **payload}, ensure_ascii=False, indent=2)
        )
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
        RcloneBackend.normalize_relative(path)
        if path in entries:
            raise DriveToolError(f"Duplicate path in inventory: {location}/{path}")
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
    # A file-backed stderr prevents deadlock while the checksum streams stdout.
    with (
        tempfile.TemporaryFile() as errors,
        subprocess.Popen(
            [backend.executable, "cat", remote_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=errors,
        ) as process,
    ):
        timer = threading.Timer(backend.timeout, process.kill)
        timer.start()
        try:
            assert process.stdout is not None
            stdout = process.stdout
            for chunk in iter(lambda: stdout.read(1024 * 1024), b""):
                digest.update(chunk)
            returncode = process.wait()
        finally:
            timer.cancel()
        if returncode != 0:
            errors.seek(0)
            raise DriveToolError(
                f"rclone cat failed or timed out: {errors.read().decode(errors='replace').strip()}"
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
