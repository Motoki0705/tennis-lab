"""Persistent filesystem IDs require a known UUID source and successful statfs."""

from __future__ import annotations

import ctypes
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.parametrize("case", ["ext4", "zero", "drvfs", "missing", "permission"])
def test_filesystem_id_uses_ext4_fsid_or_reports_an_explicit_failure(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    module = importlib.import_module("worktree_cleanup.inventory")

    class Statfs:
        def __call__(self, fd: int, pointer: Any) -> int:
            assert fd == 42
            result = ctypes.cast(pointer, ctypes.POINTER(module._Statfs)).contents
            result.kind = 0x1021997 if case == "drvfs" else 0xEF53
            result.fsid[:] = [0, 0] if case == "zero" else [0x12345678, 0x9ABCDEF0]
            if case == "permission":
                ctypes.set_errno(13)
                return -1
            return 0

    libc = SimpleNamespace() if case == "missing" else SimpleNamespace(fstatfs=Statfs())
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: libc)
    if case == "ext4":
        assert module.filesystem_id(42) == "ext4-fsid:123456789abcdef0"
    elif case == "permission":
        with pytest.raises(PermissionError, match="filesystem identity"):
            module.filesystem_id(42)
    else:
        with pytest.raises(module.CleanupError, match="filesystem identity"):
            module.filesystem_id(42)


def test_identity_ignores_device_and_distinguishes_inode_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    module = importlib.import_module("worktree_cleanup.inventory")
    stat = SimpleNamespace(st_dev=2112, st_ino=123)
    born = [10]
    monkeypatch.setattr(module.os, "fstat", lambda fd: stat)
    monkeypatch.setattr(
        module, "filesystem_id", lambda fd: "ext4-fsid:0123456789abcdef"
    )
    monkeypatch.setattr(module, "birth_time", lambda fd: born[0])
    first = module.opened_identity(42)
    stat.st_dev = 2096
    assert module.opened_identity(42) == first
    born[0] = 11
    assert module.opened_identity(42) != first
    assert "device" not in first


def test_directory_generation_without_birth_time_cannot_authorize_deletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    module = importlib.import_module("worktree_cleanup.inventory")
    expected = {
        "filesystem_id": "ext4-fsid:0123456789abcdef",
        "inode": 123,
        "birth_time_ns": None,
    }
    with pytest.raises(module.CleanupError, match="birth time is unavailable"):
        module.require_directory(Path("/unopened"), expected)
