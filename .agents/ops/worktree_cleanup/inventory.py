"""Git registrations, statx birth time, and scope checks without following links."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import stat
import struct
import subprocess
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

from .storage import CleanupError

TOKEN_FILE = "tennis-lab-expiry-id"


class QuarantineConflict(CleanupError):
    """A rename moved a successor instead of the pinned source directory."""


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
        return opened_identity(fd, birth=birth)
    finally:
        os.close(fd)


def opened_identity(fd: int, *, birth: bool = False) -> dict[str, Any]:
    st = os.fstat(fd)
    result: dict[str, Any] = {"device": st.st_dev, "inode": st.st_ino}
    if birth:
        result["birth_time_ns"] = birth_time(fd)
    return result


def registration_id(admin: Path, ident: dict[str, Any]) -> str:
    return hashlib.sha256(
        f"{admin}:{ident['device']}:{ident['inode']}:{ident.get('birth_time_ns')}:{ident.get('token')}".encode()
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


def read_at(fd: int, name: str) -> str:
    item = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    with os.fdopen(item, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise CleanupError(f"administrative file is not regular: {name}")
        value = stream.read(65537)
    if len(value) > 65536:
        raise CleanupError(f"administrative file is too large: {name}")
    return value.decode(errors="surrogateescape")


def admin_identity(fd: int, *, create: bool = False) -> dict[str, Any]:
    result = opened_identity(fd, birth=True)
    try:
        token = read_at(fd, TOKEN_FILE)
    except FileNotFoundError:
        if not create:
            return {**result, "token": None}
        # Publish a complete write-once token. A crash before link leaves only
        # an unreferenced temp file; a crash after link retains the same token.
        token = uuid.uuid4().hex
        temporary = ".expiry-id-" + uuid.uuid4().hex
        item = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=fd,
        )
        try:
            with os.fdopen(item, "w") as stream:
                stream.write(token)
                stream.flush()
                os.fsync(stream.fileno())
            with suppress(FileExistsError):
                os.link(
                    temporary,
                    TOKEN_FILE,
                    src_dir_fd=fd,
                    dst_dir_fd=fd,
                    follow_symlinks=False,
                )
        finally:
            os.unlink(temporary, dir_fd=fd)
            os.fsync(fd)
        token = read_at(fd, TOKEN_FILE)
    if len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise CleanupError("invalid worktree administration generation token")
    return {**result, "token": token}


def rename_no_replace(source_fd: int, source: str, target_fd: int, target: str) -> None:
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
    if call(source_fd, os.fsencode(source), target_fd, os.fsencode(target), 1):
        raise OSError(ctypes.get_errno(), "cannot rename worktree without replacement")


def quarantine_tree(path: Path, destination: Path, expected: dict[str, Any]) -> None:
    source_fd = open_directory(path.parent)
    try:
        target_fd = open_directory(destination.parent)
        try:
            # Pin the leaf too: its inode cannot be recycled across rename.
            leaf = os.open(
                path.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=source_fd,
            )
            try:
                if opened_identity(leaf) != expected:
                    raise CleanupError("worktree identity changed before quarantine")
                rename_no_replace(source_fd, path.name, target_fd, destination.name)
                os.fsync(source_fd)
                os.fsync(target_fd)
                try:
                    moved = identity(destination)
                except (OSError, CleanupError) as exc:
                    raise QuarantineConflict(
                        "quarantine moved a replacement entry"
                    ) from exc
                if moved != opened_identity(leaf):
                    raise QuarantineConflict("quarantine moved a replacement worktree")
            finally:
                os.close(leaf)
        finally:
            os.close(target_fd)
    finally:
        os.close(source_fd)


def restore_quarantine(path: Path, destination: Path) -> None:
    """Restore a mis-moved successor, without ever overwriting an occupied path."""
    source = open_directory(destination.parent)
    try:
        target = open_directory(path.parent)
        try:
            leaf = os.open(destination.name, os.O_PATH | os.O_NOFOLLOW, dir_fd=source)
            try:
                expected = os.fstat(leaf)
                rename_no_replace(source, destination.name, target, path.name)
                os.fsync(source)
                os.fsync(target)
                actual = os.stat(path.name, dir_fd=target, follow_symlinks=False)
                if (actual.st_dev, actual.st_ino, actual.st_mode) != (
                    expected.st_dev,
                    expected.st_ino,
                    expected.st_mode,
                ):
                    raise CleanupError("replacement changed during quarantine recovery")
            finally:
                os.close(leaf)
        finally:
            os.close(target)
    finally:
        os.close(source)


def same_registration(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["device"] == right["device"]
        and left["inode"] == right["inode"]
        and left.get("token") == right.get("token")
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
    mapping: dict[str, Path] = {}
    for admin in admins.iterdir():
        if admin.is_symlink() or not admin.is_dir():
            raise CleanupError(f"invalid worktree administration directory: {admin}")
        descriptor = open_directory(admin)
        try:
            try:
                raw = read_at(descriptor, "gitdir").rstrip("\n")
            except FileNotFoundError:
                continue
        finally:
            os.close(descriptor)
        gitfile = Path(os.path.abspath(admin / raw))
        if gitfile.name != ".git":
            raise CleanupError(f"unexpected gitdir: {admin}")
        if str(gitfile.parent) in mapping:
            raise CleanupError("duplicate worktree administrative paths")
        mapping[str(gitfile.parent)] = admin
    for row in records:
        item = mapping.get(row["path"])
        if item is None:
            raise CleanupError(f"no administrative directory for {row['path']}")
        admin = item
        path = Path(row["path"])
        descriptor = open_directory(admin)
        leaf: int | None = None
        try:
            raw = read_at(descriptor, "gitdir").rstrip("\n")
            if Path(os.path.abspath(admin / raw)) != path / ".git":
                raise CleanupError(
                    "worktree administrative mapping changed during scan"
                )
            ident = admin_identity(descriptor, create=row["scope_error"] is None)
            directory_identity = None
            if not row["scope_error"] and path.exists():
                leaf = open_directory(path)
                directory_identity = opened_identity(leaf)
                # The pinned admin and root must belong to the same snapshot.
                if (path / ".git").exists():
                    link = read_at(leaf, ".git").strip()
                    if (
                        not link.startswith("gitdir: ")
                        or Path(os.path.abspath(path / link.removeprefix("gitdir: ")))
                        != admin
                    ):
                        raise CleanupError(
                            "worktree root refers to another registration"
                        )
            check = open_directory(admin)
            try:
                if admin_identity(check) != ident:
                    raise CleanupError("administrative generation changed during scan")
            finally:
                os.close(check)
            if leaf is not None and identity(path) != directory_identity:
                raise CleanupError("worktree root changed during scan")
            row.update(
                admin=str(admin),
                admin_identity=ident,
                id=registration_id(admin, ident),
                directory_identity=directory_identity,
            )
        finally:
            if leaf is not None:
                os.close(leaf)
            os.close(descriptor)
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
    """Remove contents through the pinned leaf, not a later path-selected root."""
    parent = open_directory(path.parent)
    try:
        leaf = os.open(
            path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
        )
        try:
            if opened_identity(leaf) != expected:
                raise CleanupError("quarantine identity changed before removal")
            st = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if (
                not stat.S_ISDIR(st.st_mode)
                or {"device": st.st_dev, "inode": st.st_ino} != expected
            ):
                raise CleanupError("quarantine path moved before removal")
            clear_directory(leaf)
            if identity(path) != expected:
                raise CleanupError("quarantine generation replaced during removal")
            # rmdir can remove only an empty entry, never a successor's files.
            os.rmdir(path.name, dir_fd=parent)
        finally:
            os.close(leaf)
    finally:
        os.close(parent)


def clear_directory(fd: int, *, administration: bool = False) -> None:
    import shutil

    if not shutil.rmtree.avoids_symlink_attacks:
        raise CleanupError("fd-based rmtree is required")
    with os.scandir(fd) as entries:
        names = [entry.name for entry in entries]
    if administration:
        # Keep the token until Git's backlink has been removed, so a retry can
        # never create a new token for a half-unregistered old generation.
        names.sort(
            key=lambda name: 2 if name == TOKEN_FILE else 1 if name == "gitdir" else 0
        )
    for name in names:
        try:
            mode = os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode
            if stat.S_ISDIR(mode):
                shutil.rmtree(name, dir_fd=fd)
            else:
                os.unlink(name, dir_fd=fd)
        except FileNotFoundError:
            continue
    os.fsync(fd)
    if os.listdir(fd):
        raise CleanupError("pinned directory is still being written")


def unregister(common: Path, row: dict[str, Any]) -> None:
    """Invalidate only the opened administrative inode, never a path-selected tree.

    Git can prune a missing worktree and reuse its management name during this
    call. Descriptor-relative removal keeps that replacement untouched. Leave
    the empty management directory for Git's ordinary prune/GC: rmdir by name
    would reintroduce the same replacement race.
    """
    if row.get("registration_lost_at"):
        return
    admin = Path(row["admin"])
    if admin.parent != common / "worktrees" or admin.is_symlink():
        raise CleanupError("administrative directory is outside this repository")
    try:
        fd = open_directory(admin)
    except FileNotFoundError:
        return
    try:
        actual = admin_identity(fd)
        if not same_registration(actual, row["admin_identity"]):
            return  # The old registration already disappeared; keep its successor.
        clear_directory(fd, administration=True)
    finally:
        os.close(fd)
