"""Unit tests for pidfd helpers in .agents/ops/lib/opsproc.py."""

from __future__ import annotations

import importlib
import os
import select
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".agents/ops"))
opsproc: Any = importlib.import_module("lib.opsproc")


def test_pidfd_libc_fallback_signals_and_waits_for_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    monkeypatch.delattr(signal, "pidfd_send_signal", raising=False)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    descriptor: int | None = None
    try:
        descriptor = opsproc.pidfd_open(child.pid)
        opsproc.pidfd_send_signal(descriptor, signal.SIGTERM)
        assert select.select([descriptor], [], [], 5)[0] == [descriptor]
        assert child.wait(timeout=5) == -signal.SIGTERM
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_pidfd_open_raises_process_lookup_for_missing_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = child.pid
    child.wait(timeout=5)
    with pytest.raises(ProcessLookupError):
        opsproc.pidfd_open(pid)


def test_pidfd_uses_native_os_and_signal_apis_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, int | None]] = []

    def fake_open(pid: int) -> int:
        calls.append(("open", pid, None))
        return 42

    def fake_send(descriptor: int, sig: int) -> None:
        calls.append(("signal", descriptor, sig))

    monkeypatch.setattr(os, "pidfd_open", fake_open, raising=False)
    monkeypatch.setattr(signal, "pidfd_send_signal", fake_send, raising=False)
    descriptor = opsproc.pidfd_open(1234)
    opsproc.pidfd_send_signal(descriptor, signal.SIGTERM)
    assert descriptor == 42
    assert calls == [("open", 1234, None), ("signal", 42, signal.SIGTERM)]
