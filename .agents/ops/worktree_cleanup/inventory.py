"""Git registrations, statx birth time, and scope checks without following links."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import stat
import struct
import subprocess
from pathlib import Path
from typing import Any

from .storage import CleanupError


def git(repo: Path, *arguments: str, timeout: float = 60) -> str:
    try:
        proc = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo), *arguments],
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise CleanupError(f"git {arguments[0]} timed out after {timeout}s") from exc
    if proc.returncode:
        raise CleanupError(
            f"git {' '.join(arguments)}: {proc.stderr.decode(errors='replace').strip()}"
        )
    return proc.stdout.decode(errors="surrogateescape")


def main_checkout(cwd: Path) -> tuple[Path, Path]:
    common = Path(
        git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir").strip()
    ).resolve(strict=True)
    first = git(cwd, "worktree", "list", "--porcelain", "-z").split("\0", 1)[0]
    if not first.startswith("worktree ") or common.is_symlink():
        raise CleanupError("cannot identify main checkout")
    repo = Path(first.removeprefix("worktree ")).resolve(strict=True)
    if git(repo, "rev-parse", "--is-bare-repository").strip() != "false":
        raise CleanupError("a main checkout is required")
    return repo, common


def birth_time(fd: int) -> int | None:
    """Linux statx ABI: btime is the timestamp at offset 0x50 in linux/stat.h.

    AT_EMPTY_PATH pins the queried directory to the open descriptor. Unsupported
    libc/kernel/filesystem is the only fallback; I/O and permission errors fail.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    call = getattr(libc, "statx", None)
    if call is None:
        return None
    call.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_void_p,
    ]
    call.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(256)
    if call(fd, b"", 0x1000, 0x800, buffer) != 0:
        code = ctypes.get_errno()
        if code in {errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL}:
            return None
        raise OSError(code, os.strerror(code))
    mask = struct.unpack_from("=I", buffer, 0)[0]
    if not mask & 0x800:
        return None
    seconds, nanos = struct.unpack_from("=qI", buffer, 0x50)
    if seconds < 0 or not 0 <= nanos < 10**9:
        raise CleanupError("invalid statx birth time")
    return int(seconds) * 10**9 + int(nanos)


def identity(path: Path, *, birth: bool = False) -> dict[str, Any]:
    fd = open_directory(path)
    try:
        st = os.fstat(fd)
        result: dict[str, Any] = {"device": st.st_dev, "inode": st.st_ino}
        if birth:
            result["birth_time_ns"] = birth_time(fd)
        return result
    finally:
        os.close(fd)


def registration_id(admin: Path, ident: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"{admin}:{ident['device']}:{ident['inode']}:{ident.get('birth_time_ns')}".encode()
    ).hexdigest()


def open_directory(path: Path) -> int:
    """Open every absolute path component with O_NOFOLLOW, including ancestors."""
    if not path.is_absolute() or ".." in path.parts:
        raise CleanupError("an absolute directory without '..' is required")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            next_fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def quarantine_tree(path: Path, destination: Path, expected: dict[str, Any]) -> None:
    source_fd = open_directory(path.parent)
    try:
        target_fd = open_directory(destination.parent)
        try:
            st = os.stat(path.name, dir_fd=source_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(st.st_mode)
                or {"device": st.st_dev, "inode": st.st_ino} != expected
            ):
                raise CleanupError("worktree identity changed before quarantine")
            # Linux RENAME_NOREPLACE also protects a concurrently created target.
            call = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
            if call is None:
                raise CleanupError("libc renameat2 is required for safe quarantine")
            call.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            call.restype = ctypes.c_int
            if call(
                source_fd,
                os.fsencode(path.name),
                target_fd,
                os.fsencode(destination.name),
                1,
            ):
                raise OSError(ctypes.get_errno(), "cannot quarantine worktree")
            os.fsync(source_fd)
            os.fsync(target_fd)
        finally:
            os.close(target_fd)
    finally:
        os.close(source_fd)


def same_registration(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["device"] == right["device"]
        and left["inode"] == right["inode"]
        and (
            left.get("birth_time_ns") is None
            or right.get("birth_time_ns") is None
            or left["birth_time_ns"] == right["birth_time_ns"]
        )
    )


def scope_error(repo: Path, path: Path) -> str | None:
    root = repo / ".claude/worktrees"
    if (
        not path.is_absolute()
        or path == root
        or not path.is_relative_to(root)
        or ".." in path.parts
    ):
        return "outside main checkout .claude/worktrees"
    for part in (path, *path.parents):
        if part == repo:
            break
        if part.is_symlink():
            return f"symlink in worktree path: {part}"
    if path.exists() and not path.is_dir():
        return "worktree path is not a directory"
    return None


def registrations(repo: Path, common: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for chunk in git(repo, "worktree", "list", "--porcelain", "-z").split("\0\0"):
        if not chunk:
            continue
        fields = dict(field.partition(" ")[::2] for field in chunk.split("\0") if field)
        if "worktree" not in fields:
            raise CleanupError("invalid git worktree list output")
        path = Path(fields["worktree"])
        if path == repo:
            continue
        records.append(
            {
                "path": str(path),
                "head": fields.get("HEAD"),
                "branch": fields.get("branch"),
                "locked": "locked" in fields,
                "scope_error": scope_error(repo, path),
            }
        )
    admins = common / "worktrees"
    if not admins.exists():
        return records
    mapping: dict[str, tuple[Path, dict[str, Any]]] = {}
    for admin in admins.iterdir():
        if admin.is_symlink() or not admin.is_dir():
            raise CleanupError(f"invalid worktree administration directory: {admin}")
        try:
            raw = (admin / "gitdir").read_text().rstrip("\n")
        except FileNotFoundError:
            continue  # concurrent Git creation/removal; checked before mutation
        gitfile = Path(os.path.abspath(admin / raw))
        if gitfile.name != ".git":
            raise CleanupError(f"unexpected gitdir: {admin}")
        if str(gitfile.parent) in mapping:
            raise CleanupError("duplicate worktree administrative paths")
        mapping[str(gitfile.parent)] = (admin, identity(admin, birth=True))
    for row in records:
        item = mapping.get(row["path"])
        if item is None:
            raise CleanupError(f"no administrative directory for {row['path']}")
        admin, ident = item
        row.update(
            admin=str(admin), admin_identity=ident, id=registration_id(admin, ident)
        )
        path = Path(row["path"])
        row["directory_identity"] = (
            identity(path) if not row["scope_error"] and path.exists() else None
        )
    return records


def changes(path: Path, error: str | None) -> dict[str, Any]:
    if error or not path.exists():
        return {
            "discarded_changes": None,
            "changes_error": error or "worktree directory missing",
        }
    try:
        # Include ignored and populated submodules; index flags may hide changes.
        status = git(
            path,
            "status",
            "--porcelain=v1",
            "-z",
            "--ignored=matching",
            "--untracked-files=normal",
            "--ignore-submodules=none",
            timeout=10,
        )
        flags = git(path, "ls-files", "-v", "-z", timeout=10)
        hidden = any(
            entry and (entry[0].islower() or entry[0] == "S")
            for entry in flags.split("\0")
        )
        return {"discarded_changes": bool(status) or hidden, "changes_error": None}
    except CleanupError as exc:
        return {"discarded_changes": None, "changes_error": str(exc)}


def require_directory(path: Path, expected: dict[str, Any]) -> None:
    if identity(path) != expected:
        raise CleanupError(f"directory identity changed: {path}")


def remove_tree(path: Path, expected: dict[str, Any]) -> None:
    """Use the fd-based rmtree implementation; symlink entries are unlinked."""
    import shutil

    require_directory(path, expected)
    if not shutil.rmtree.avoids_symlink_attacks:
        raise CleanupError("fd-based rmtree is required")
    parent = open_directory(path.parent)
    try:
        st = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(st.st_mode)
            or {"device": st.st_dev, "inode": st.st_ino} != expected
        ):
            raise CleanupError("quarantine identity changed before removal")
        shutil.rmtree(path.name, dir_fd=parent)
    finally:
        os.close(parent)
