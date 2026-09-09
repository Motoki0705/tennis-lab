"""Create self-contained, secret-safe working-tree snapshots for Colab."""

from __future__ import annotations

import configparser
import gzip
import hashlib
import os
import stat
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from .common import WorkflowError, canonical_json, sha256_file, strict_relative_path

SECRET_NAMES = frozenset(
    {
        ".env",
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "application_default_credentials.json",
        "authorized_user.json",
        "client_secret.json",
        "credentials.json",
        "id_dsa",
        "id_ed25519",
        "id_ecdsa",
        "id_rsa",
        "rclone.conf",
        "service-account.json",
    }
)
SECRET_SUFFIXES = (".key", ".pem", ".p12", ".pfx")


@dataclass(frozen=True)
class Submodule:
    """A submodule URL and immutable gitlink revision."""

    path: str
    url: str
    commit: str


@dataclass(frozen=True)
class Snapshot:
    """A deterministic full working-tree snapshot and its provenance."""

    archive_path: Path
    archive_sha256: str
    content_digest: str
    included_paths: tuple[str, ...]
    deleted_paths: tuple[str, ...]
    excluded_secret_paths: tuple[str, ...]
    submodules: tuple[Submodule, ...]


def _git(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise WorkflowError(f"git {' '.join(args)} failed: {message}")
    return result.stdout


def _nul_paths(value: bytes) -> set[str]:
    try:
        decoded = [item.decode("utf-8") for item in value.split(b"\0") if item]
    except UnicodeDecodeError as error:
        raise WorkflowError("snapshot paths must be valid UTF-8") from error
    return {strict_relative_path(item, "snapshot path") for item in decoded}


def _is_secret(path: str) -> bool:
    lower_parts = tuple(part.lower() for part in PurePosixPath(path).parts)
    lower_name = lower_parts[-1]
    return (
        lower_name in SECRET_NAMES
        or lower_name.startswith(".env.")
        or lower_name.endswith(SECRET_SUFFIXES)
        or lower_name.startswith("client_secret")
        or ("credential" in lower_name and lower_name.endswith(".json"))
        or ("secret" in lower_name and lower_name.endswith(".json"))
        or ("token" in lower_name and lower_name.endswith(".json"))
        or any(part in {".git", ".secrets", "secrets"} for part in lower_parts)
    )


def _submodule_index(repo_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw_entry in _git(repo_root, "ls-files", "--stage", "-z").split(b"\0"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, commit, _stage = metadata.decode("ascii").split(" ")
            path = strict_relative_path(raw_path.decode("utf-8"), "submodule path")
        except (UnicodeDecodeError, ValueError) as error:
            raise WorkflowError("git returned malformed index metadata") from error
        if mode == "160000":
            if len(commit) != 40:
                raise WorkflowError(f"submodule {path} has an invalid gitlink")
            result[path] = commit
    return result


def _verify_submodule_worktrees(
    repo_root: Path, indexed: dict[str, str], urls: dict[str, str]
) -> None:
    for path, commit in sorted(indexed.items()):
        checkout = repo_root / path
        fetch_contract = (
            f"snapshot records only gitlink {commit} and requires that commit to be "
            f"pushed and fetchable from {urls[path]}"
        )
        if checkout.is_symlink():
            raise WorkflowError(
                f"submodule checkout must not be a symlink: {path}; {fetch_contract}"
            )
        if not checkout.exists():
            continue
        if not checkout.is_dir():
            raise WorkflowError(
                f"submodule checkout is not a directory: {path}; {fetch_contract}"
            )
        git_marker = checkout / ".git"
        if git_marker.is_symlink():
            raise WorkflowError(
                f"submodule Git metadata must not be a symlink: {path}; {fetch_contract}"
            )
        if not git_marker.exists():
            try:
                populated = next(checkout.iterdir(), None) is not None
            except OSError as error:
                raise WorkflowError(
                    f"cannot inspect submodule checkout {path}: {error}; {fetch_contract}"
                ) from error
            if populated:
                raise WorkflowError(
                    f"submodule checkout is populated but is not an initialized Git "
                    f"worktree: {path}; {fetch_contract}"
                )
            continue
        head = _git(checkout, "rev-parse", "HEAD").decode("ascii").strip()
        if head != commit:
            raise WorkflowError(
                f"submodule checkout {path} is at {head}, not recorded gitlink "
                f"{commit}; {fetch_contract}"
            )
        status = (
            _git(
                checkout,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            )
            .decode("utf-8", errors="replace")
            .strip()
        )
        if status:
            summary = "; ".join(status.splitlines()[:20])
            raise WorkflowError(
                f"submodule checkout has tracked, untracked, or nested changes: "
                f"{path} ({summary}); {fetch_contract}"
            )


def _submodules(repo_root: Path) -> tuple[Submodule, ...]:
    indexed = _submodule_index(repo_root)
    if not indexed:
        return ()
    config_path = repo_root / ".gitmodules"
    if not config_path.is_file():
        raise WorkflowError("git index has submodules but .gitmodules is missing")
    parser = configparser.ConfigParser(interpolation=None)
    try:
        with config_path.open(encoding="utf-8") as stream:
            parser.read_file(stream)
    except (OSError, UnicodeDecodeError, configparser.Error) as error:
        raise WorkflowError(f"cannot parse .gitmodules: {error}") from error
    urls: dict[str, str] = {}
    for section in parser.sections():
        if not section.startswith('submodule "'):
            continue
        if not parser.has_option(section, "path") or not parser.has_option(
            section, "url"
        ):
            raise WorkflowError(f"{section} must define path and url")
        path = strict_relative_path(parser.get(section, "path"), "submodule path")
        url = parser.get(section, "url").strip()
        if not url or "\x00" in url:
            raise WorkflowError(f"submodule {path} has an invalid URL")
        if path in urls:
            raise WorkflowError(f"duplicate submodule path: {path}")
        urls[path] = url
    missing = set(indexed) - set(urls)
    if missing:
        raise WorkflowError(f"submodules are missing URLs: {sorted(missing)}")
    _verify_submodule_worktrees(repo_root, indexed, urls)
    return tuple(
        Submodule(path=path, url=urls[path], commit=commit)
        for path, commit in sorted(indexed.items())
    )


def _under(path: str, roots: set[str]) -> bool:
    return any(path == root or path.startswith(f"{root}/") for root in roots)


def _entry_metadata(repo_root: Path, relative: str) -> dict[str, object]:
    absolute = repo_root / relative
    file_stat = absolute.lstat()
    if stat.S_ISLNK(file_stat.st_mode):
        target = os.readlink(absolute)
        link = PurePosixPath(target)
        if link.is_absolute():
            raise WorkflowError(f"snapshot symlink must be relative: {relative}")
        resolved_target = (absolute.parent / target).resolve()
        resolved_root = repo_root.resolve()
        if (
            resolved_target == resolved_root
            or resolved_root not in resolved_target.parents
        ):
            raise WorkflowError(f"snapshot symlink escapes repository: {relative}")
        return {
            "path": relative,
            "kind": "symlink",
            "mode": stat.S_IMODE(file_stat.st_mode),
            "target": target,
        }
    if not stat.S_ISREG(file_stat.st_mode):
        raise WorkflowError(f"snapshot path is not a regular file/symlink: {relative}")
    return {
        "path": relative,
        "kind": "file",
        "mode": stat.S_IMODE(file_stat.st_mode),
        "size": file_stat.st_size,
        "sha256": sha256_file(absolute),
    }


def _write_archive(
    repo_root: Path, archive_path: Path, metadata: list[dict[str, object]]
) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        archive_path.open("wb") as raw_stream,
        gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as compressed,
        tarfile.open(
            fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
        ) as archive,
    ):
        for entry in metadata:
            relative = str(entry["path"])
            info = tarfile.TarInfo(relative)
            info.mode = cast(int, entry["mode"])
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            if entry["kind"] == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = str(entry["target"])
                archive.addfile(info)
            else:
                info.size = cast(int, entry["size"])
                with (repo_root / relative).open("rb") as stream:
                    archive.addfile(info, stream)


def create_snapshot(repo_root: Path, archive_path: Path) -> Snapshot:
    """Archive the full tracked/unignored working tree without submodule content."""

    tracked = _nul_paths(_git(repo_root, "ls-files", "--cached", "-z"))
    head_tracked = _nul_paths(
        _git(repo_root, "ls-tree", "-r", "--name-only", "-z", "HEAD")
    )
    untracked = _nul_paths(
        _git(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    )
    submodules = _submodules(repo_root)
    submodule_paths = {item.path for item in submodules}
    candidates = {
        path
        for path in tracked | head_tracked | untracked
        if not _under(path, submodule_paths)
    }
    excluded_secrets = tuple(sorted(path for path in candidates if _is_secret(path)))
    safe_candidates = candidates - set(excluded_secrets)

    included: list[str] = []
    deleted: list[str] = []
    metadata: list[dict[str, object]] = []
    for relative in sorted(safe_candidates):
        absolute = repo_root / relative
        try:
            absolute.lstat()
        except FileNotFoundError:
            if relative in tracked or relative in head_tracked:
                deleted.append(relative)
                metadata.append({"path": relative, "kind": "deleted"})
            continue
        if absolute.is_dir() and not absolute.is_symlink():
            continue
        included.append(relative)
        metadata.append(_entry_metadata(repo_root, relative))

    archived_metadata = [entry for entry in metadata if entry["kind"] != "deleted"]
    _write_archive(repo_root, archive_path, archived_metadata)
    content_digest = hashlib.sha256(canonical_json({"entries": metadata})).hexdigest()
    return Snapshot(
        archive_path=archive_path,
        archive_sha256=sha256_file(archive_path),
        content_digest=content_digest,
        included_paths=tuple(included),
        deleted_paths=tuple(deleted),
        excluded_secret_paths=excluded_secrets,
        submodules=submodules,
    )
