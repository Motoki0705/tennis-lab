"""Kernel-enforced, inherited file access for workers with private input inboxes.

Landlock applies to the CLI itself and every descendant, including view_image and
shell tools. A workspace-write sandbox alone does not restrict host file reads.
There is deliberately no permissive backend when Landlock is unavailable.
"""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
from collections.abc import Iterable
from pathlib import Path

RUNTIME_READ_ROOTS = (
    "/usr/bin",
    "/usr/lib",
    "/usr/lib64",
    "/usr/local/bin",
    "/usr/local/lib",
    "/bin",
    "/lib",
    "/lib64",
    "/etc/ssl/certs",
)


def require_private_paths(paths: Iterable[Path]) -> None:
    """Original inputs must not overlap the trusted executable/library trees."""
    roots = tuple(Path(root).resolve() for root in RUNTIME_READ_ROOTS)
    if any(path.resolve().is_relative_to(root) for path in paths for root in roots):
        raise ValueError(
            "annotation media/repository overlaps a readable system runtime"
        )


def private_home(directory: Path, source: Path) -> Path:
    """Share credentials, never other threads, caches, rules or project config."""
    home = directory / "work" / ".codex"
    home.mkdir(mode=0o700, exist_ok=True)
    auth = source / "auth.json"
    if auth.exists() and not (home / "auth.json").exists():
        shutil.copyfile(auth, home / "auth.json")
        (home / "auth.json").chmod(0o600)
    return home


def restrict_attempt(directory: Path, executable: Path) -> None:
    """Call in the child before exec; no project/media directory is readable."""
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise RuntimeError("private annotation inputs require Linux Landlock")
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long

    def checked(value: int) -> int:
        if value < 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        return value

    # The supported architectures share the generic Landlock syscall numbers.
    abi = checked(syscall(444, 0, 0, 1))
    if abi < 3:
        raise RuntimeError("private annotation inputs require Landlock ABI >= 3")
    read = (1 << 0) | (1 << 2) | (1 << 3)  # execute, read_file, read_dir
    all_access = (1 << 15) - 1  # through REFER and TRUNCATE (ABI 3)

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]

    spec = Ruleset(all_access)
    ruleset = checked(syscall(444, ctypes.byref(spec), ctypes.sizeof(spec), 0))
    try:
        allowed: list[tuple[Path, int]] = [
            (directory / "work", all_access),
            (directory / "input", read),
            (directory / "exchange", read),
            (executable.resolve(strict=True), read),
        ]
        allowed.extend(
            (executable.resolve().with_name(name), read)
            for name in ("codex-code-mode-host", "codex-exec-server")
        )
        # Runtime binaries/libraries are system inputs, never project artifacts.
        allowed.extend((Path(p), read) for p in RUNTIME_READ_ROOTS)
        allowed.extend(
            (Path(p), read)
            for p in (
                "/etc/resolv.conf",
                "/etc/hosts",
                "/etc/nsswitch.conf",
                "/etc/ld.so.cache",
                "/etc/localtime",
            )
        )
        allowed.extend(
            (Path(p), (1 << 1) | (1 << 2)) for p in ("/dev/null", "/dev/urandom")
        )
        for path, rights in allowed:
            if not path.exists():
                continue
            descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                if not path.is_dir():
                    rights &= (1 << 0) | (1 << 1) | (1 << 2) | (1 << 14)
                rule = PathRule(rights, descriptor)
                checked(syscall(445, ruleset, 1, ctypes.byref(rule), 0))
            finally:
                os.close(descriptor)
        checked(libc.prctl(38, 1, 0, 0, 0))  # PR_SET_NO_NEW_PRIVS
        checked(syscall(446, ruleset, 0))
    finally:
        os.close(ruleset)
