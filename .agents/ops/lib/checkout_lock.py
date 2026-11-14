"""Coordinate checkout replacement with job lifetimes and dispatcher controls."""

from __future__ import annotations

import atexit
import fcntl
import os
import re
import stat
import subprocess
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

CHECKOUT_ROOT_ENV = "TENNIS_LAB_CHECKOUT_ROOT"
CHECKOUT_FD_ENV = "TENNIS_LAB_CHECKOUT_FD"
_cli_lock: ExitStack | None = None


def _git_environment() -> dict[str, str]:
    # Hooks and callers may select another repository. Our explicit root wins.
    return {
        key: value
        for key, value in os.environ.items()
        if key not in {"GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE"}
    }


def _release_cli_lock() -> None:
    global _cli_lock
    if _cli_lock is not None:
        _cli_lock.close()
        _cli_lock = None


def guard_interactive_cli() -> None:
    """Pin this module's main checkout before importing operational code.

    Linked worktrees use their own source and do not hold the main checkout.
    The descriptor is process-local: neither exec nor fork may extend its life.
    """
    global _cli_lock
    if _cli_lock is not None:
        return
    root = Path(__file__).resolve().parents[3]
    try:
        directories = subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "rev-parse",
                "--path-format=absolute",
                "--git-dir",
                "--git-common-dir",
                "--show-toplevel",
            ],
            text=True,
            stderr=subprocess.PIPE,
            env=_git_environment(),
        ).splitlines()
        git_dir, common, top = map(Path, directories)
        if top != root:
            raise ValueError("operational source is outside its checkout root")
        if git_dir != common:
            return
        with ExitStack() as stack:
            descriptor = stack.enter_context(checkout_lock(root, blocking=False))
            if descriptor is None:
                print(
                    "checkout CLI waiting for git-sync to release the checkout lock",
                    file=sys.stderr,
                    flush=True,
                )
                descriptor = stack.enter_context(checkout_lock(root))
            if checkout_barrier(root).exists():
                raise ValueError("checkout recovery pending")
            assert descriptor is not None
            os.set_inheritable(descriptor, False)
            _cli_lock = stack.pop_all()
        atexit.register(_release_cli_lock)
        os.register_at_fork(after_in_child=_release_cli_lock)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(
            f"checkout CLI refused: {exc}; inspect spin ops status git-sync",
            file=sys.stderr,
        )
        raise SystemExit(75) from exc


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
    with (common_directory(root) / f"{name}.lock").open(
        "a", encoding="utf-8"
    ) as stream:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(stream, mode | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield None
        else:
            yield stream.fileno()
