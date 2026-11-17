"""Coordinate checkout replacement with job lifetimes and dispatcher controls."""

from __future__ import annotations

import argparse
import atexit
import fcntl
import importlib
import os
import re
import stat
import subprocess
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

CHECKOUT_ROOT_ENV = "TENNIS_LAB_CHECKOUT_ROOT"
CHECKOUT_FD_ENV = "TENNIS_LAB_CHECKOUT_FD"
_cli_lock: ExitStack | None = None
_cli_context: tuple[Path, int, str] | None = None
_registry_context: ContextVar[tuple[Path, int, str] | None] = ContextVar(
    "checkout_registry", default=None
)
_CAP_FD = "TENNIS_LAB_CHECKOUT_PARENT_FD"
_CAP_DIRECTORY = "TENNIS_LAB_CHECKOUT_PARENT_DIRECTORY"
_registry_stacks: dict[int, ExitStack] = {}


def _git_environment() -> dict[str, str]:
    # Hooks and callers may select another repository. Our explicit root wins.
    return {
        key: value
        for key, value in os.environ.items()
        if key not in {"GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE"}
    }


def _release_cli_lock() -> None:
    global _cli_lock, _cli_context
    if _cli_lock is not None:
        _cli_lock.close()
        _cli_lock = None
        _cli_context = None


def _source_git_directory(
    root: Path, *, include_worktrees: bool = False
) -> Path | None:
    """Locate the bootstrap's Git metadata without invoking caller-supplied tools."""
    marker = root / ".git"
    if marker.is_dir():
        directory = marker.resolve()
    else:
        location = marker.read_text(encoding="utf-8").strip()
        prefix = "gitdir: "
        if (
            not location.startswith(prefix)
            or "\n" in location
            or not location[len(prefix) :]
        ):
            raise ValueError("invalid source checkout gitfile")
        directory = (root / location[len(prefix) :]).resolve()
    if not (directory / "HEAD").is_file():
        raise ValueError("source checkout Git directory has no HEAD")
    common = directory / "commondir"
    if common.exists():
        location = common.read_text(encoding="utf-8").strip()
        if not location or "\n" in location:
            raise ValueError("invalid source checkout commondir")
        shared = (directory / location).resolve()
        if not (shared / "objects").is_dir():
            raise ValueError("source checkout commondir is not a Git directory")
        if shared != directory:
            return shared if include_worktrees else None
    return directory


def _descriptor_mode(directory: Path, descriptor: int) -> str:
    actual = os.fstat(descriptor)
    expected = (directory / "ops-code.lock").stat()
    if not stat.S_ISREG(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
        expected.st_dev,
        expected.st_ino,
    ):
        raise ValueError("not the inherited checkout lock descriptor")
    for line in Path(f"/proc/self/fdinfo/{descriptor}").read_text().splitlines():
        fields = line.split()
        if (
            len(fields) >= 6
            and fields[0] == "lock:"
            and fields[2:4] == ["FLOCK", "ADVISORY"]
            and fields[4] in {"READ", "WRITE"}
        ):
            return fields[4].lower()
    raise ValueError("inherited descriptor does not hold a checkout flock")


def checkout_subprocess_options(
    *, directory: Path | None = None, descriptor: int | None = None
) -> dict[str, Any]:
    """Explicit synchronous lifecycle calls alone inherit the parent's lock."""
    context = _registry_context.get() or _cli_context
    if descriptor is not None:
        if directory is None:
            raise ValueError("checkout descriptor needs its Git directory")
        context = (directory, descriptor, _descriptor_mode(directory, descriptor))
    if context is None:
        return {}
    common, fd, _ = context
    return {
        "pass_fds": (fd,),
        "env": {**os.environ, _CAP_FD: str(fd), _CAP_DIRECTORY: str(common)},
    }


@contextmanager
def registry_checkout_lock(*, blocking: bool = True) -> Iterator[int | None]:
    """Always acquire checkout before registry, including API/worktree callers."""
    directory = _source_git_directory(
        Path(__file__).resolve().parents[3], include_worktrees=True
    )
    assert directory is not None
    existing = _registry_context.get() or _cli_context
    if existing is not None and existing[0] == directory:
        yield existing[1]
        return
    with ExitStack() as stack:
        descriptor = stack.enter_context(
            _file_lock(directory / "ops-code.lock", blocking=False)
        )
        if descriptor is None:
            if not blocking:
                yield None
                return
            print(
                "checkout registry waiting for git-sync before registry acquisition",
                file=sys.stderr,
                flush=True,
            )
            descriptor = stack.enter_context(_file_lock(directory / "ops-code.lock"))
        assert descriptor is not None
        _registry_stacks[descriptor] = stack
        token = _registry_context.set((directory, descriptor, "read"))
        try:
            yield descriptor
        finally:
            _registry_context.reset(token)
            _registry_stacks.pop(descriptor, None)


def source_checkout_barrier() -> Path:
    directory = _source_git_directory(
        Path(__file__).resolve().parents[3], include_worktrees=True
    )
    assert directory is not None
    return directory / "ops-code.blocked"


def guard_interactive_cli(
    *, allow_exclusive: bool = False, recovery_owner: bool = False
) -> None:
    """Pin this module's main checkout before importing operational code.

    Linked worktrees use their own source and do not hold the main checkout.
    The descriptor is process-local: neither exec nor fork may extend its life.
    """
    global _cli_lock, _cli_context
    if _cli_lock is not None:
        return
    root = Path(__file__).resolve().parents[3]
    try:
        directory = _source_git_directory(root)
        inherited_fd = os.environ.pop(_CAP_FD, None)
        inherited_directory = os.environ.pop(_CAP_DIRECTORY, None)
        mode = "read"
        descriptor: int | None
        with ExitStack() as stack:
            if inherited_fd is not None or inherited_directory is not None:
                common = _source_git_directory(root, include_worktrees=True)
                if (
                    inherited_fd is None
                    or inherited_directory is None
                    or common != Path(inherited_directory)
                ):
                    raise ValueError(
                        "inherited checkout context belongs to another repository"
                    )
                descriptor = int(inherited_fd)
                mode = _descriptor_mode(common, descriptor)
                if mode == "write" and not allow_exclusive:
                    raise ValueError(
                        "exclusive checkout context is only for maintenance status/stop"
                    )
                directory = common
                stack.callback(os.close, descriptor)
            else:
                if directory is None:
                    return
                path = directory / "ops-code.lock"
                descriptor = stack.enter_context(_file_lock(path, blocking=False))
                if descriptor is None:
                    print(
                        "checkout CLI waiting for git-sync to release the checkout lock",
                        file=sys.stderr,
                        flush=True,
                    )
                    descriptor = stack.enter_context(_file_lock(path))
            assert directory is not None
            if (
                mode != "write"
                and not recovery_owner
                and (directory / "ops-code.blocked").exists()
            ):
                raise ValueError("checkout recovery pending")
            assert descriptor is not None
            os.set_inheritable(descriptor, False)
            _cli_lock = stack.pop_all()
            _cli_context = (directory, descriptor, mode)
        atexit.register(_release_cli_lock)
    except (OSError, ValueError, OverflowError) as exc:
        print(
            f"checkout CLI refused: {exc}; inspect spin ops status git-sync",
            file=sys.stderr,
        )
        raise SystemExit(75) from exc


def run_cli(
    module: str,
    *,
    root_options: tuple[str, ...] = (),
    finite_commands: tuple[str, ...] = (),
    maintenance_commands: tuple[str, ...] = (),
    release_before_main: bool = False,
) -> None:
    """Thin entries load all operational implementation only after the guard."""
    parser = argparse.ArgumentParser(add_help=False)
    for option in root_options:
        parser.add_argument(option)
    _, remaining = parser.parse_known_args()
    command = remaining[0] if remaining else ""
    guard_interactive_cli(
        allow_exclusive=command in maintenance_commands,
        recovery_owner=release_before_main,
    )
    if command in finite_commands:
        require_checkout_job()
    implementation = importlib.import_module(module)
    if release_before_main:
        _release_cli_lock()
    raise SystemExit(implementation.main())


def _process_entry(root: Path, pid: int) -> str:
    """Describe the executable/module/script, never arbitrary CLI arguments."""
    process = Path(f"/proc/{pid}")
    arguments = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
    for index, argument in enumerate(arguments[1:], start=1):
        if argument == "-m":
            module = arguments[index + 1]
            if re.fullmatch(r"[\w.]+", module):
                return f"python -m {module}"
            break
        if argument == "-c":
            break
        if argument.endswith((".py", ".sh")):
            path = Path(argument)
            if not path.is_absolute():
                path = (process / "cwd").resolve() / path
            if path.is_relative_to(root / ".agents/ops"):
                return str(path.relative_to(root / ".agents/ops"))
        if argument and not argument.startswith("-"):
            break
    return Path(arguments[0]).name or "unavailable"


def checkout_lock_holders(root: Path) -> list[dict[str, Any]]:
    """Read the kernel's live flock owners; no PID files survive a crash."""
    lock = (common_directory(root) / "ops-code.lock").stat()
    identity = (os.major(lock.st_dev), os.minor(lock.st_dev), lock.st_ino)
    holders = []
    for row in Path("/proc/locks").read_text().splitlines():
        fields = row.split()
        if len(fields) < 6 or fields[1] != "FLOCK":
            continue
        device, minor, inode = fields[5].split(":")
        if (int(device, 16), int(minor, 16), int(inode)) != identity:
            continue
        pid = int(fields[4])
        holder: dict[str, Any] = {"pid": pid, "mode": fields[3].lower()}
        if pid < 0:
            holder["entry"] = "inherited lock; original owner exited"
            holders.append(holder)
            continue
        try:
            holder["entry"] = _process_entry(root, pid)
        except FileNotFoundError:
            # The lock can disappear between the kernel snapshot and /proc read.
            continue
        except OSError as exc:
            holder.update(entry="unavailable", inspection_error=str(exc))
        holders.append(holder)
    return sorted(holders, key=lambda holder: holder["pid"])


def require_checkout_job() -> None:
    """Reject CLI jobs without the launcher's inherited, real checkout lock FD."""
    try:
        root = Path(os.environ[CHECKOUT_ROOT_ENV])
        descriptor = int(os.environ[CHECKOUT_FD_ENV])
        if not root.is_absolute():
            raise ValueError("checkout root must be absolute")
        actual = os.fstat(descriptor)
        expected = (common_directory(root) / "ops-code.lock").stat()
        if not stat.S_ISREG(actual.st_mode) or (actual.st_dev, actual.st_ino) != (
            expected.st_dev,
            expected.st_ino,
        ):
            raise ValueError("not the checkout lock descriptor")
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if checkout_barrier(root).exists():
            raise ValueError("checkout recovery pending")
    except (KeyError, ValueError, OSError, OverflowError) as exc:
        print(
            f"checkout job refused: {exc}; use checkout_job.py --repo <main> -- <command...>",
            file=sys.stderr,
        )
        raise SystemExit(75) from exc


def state_root() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    if not base.is_absolute():
        raise ValueError("XDG_STATE_HOME must be absolute")
    return base / "tennis-lab-agents"


def common_directory(root: Path) -> Path:
    common = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        text=True,
        env=_git_environment(),
    ).strip()
    return Path(common)


def checkout_barrier(root: Path) -> Path:
    return common_directory(root) / "ops-code.blocked"


@contextmanager
def checkout_lock(
    root: Path,
    *,
    name: str = "ops-code",
    exclusive: bool = False,
    blocking: bool = True,
) -> Iterator[int | None]:
    """Never unlink these files: all checkouts must refer to the same inode."""
    with _file_lock(
        common_directory(root) / f"{name}.lock",
        exclusive=exclusive,
        blocking=blocking,
    ) as descriptor:
        yield descriptor


@contextmanager
def _file_lock(
    path: Path,
    *,
    exclusive: bool = False,
    blocking: bool = True,
) -> Iterator[int | None]:
    with path.open("a", encoding="utf-8") as stream:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(stream, mode | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield None
        else:
            yield stream.fileno()


def _release_forked_locks() -> None:
    _release_cli_lock()
    for stack in list(_registry_stacks.values()):
        stack.close()
    _registry_stacks.clear()
    _registry_context.set(None)


os.register_at_fork(after_in_child=_release_forked_locks)
