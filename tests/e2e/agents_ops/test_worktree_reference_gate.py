"""Actual proc references gate isolated Git removal; only test children receive signals."""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_worktree_cleanup import LIFETIME, Lab
from tests.e2e.agents_ops.test_worktree_cleanup import lab as lab

HOLDER = r"""
import ctypes, os, sys
root, kind = sys.argv[1:]
if kind == 'cwd':
    os.chdir(root)
else:
    fd = os.open(root + '/payload', os.O_RDONLY)
    if kind == 'mmap':
        lib = ctypes.CDLL(None, use_errno=True)
        lib.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
        lib.mmap.restype = ctypes.c_void_p
        address = lib.mmap(None, 4096, 1, 2, fd, 0)
        assert address not in {None, ctypes.c_void_p(-1).value}
        os.close(fd)
        assert ctypes.string_at(address, 1) == b'x'
print('ready', flush=True)
sys.stdin.readline()
"""


@pytest.mark.parametrize("stoppable", [False, True])
@pytest.mark.parametrize("kind", ["cwd", "fd", "mmap"])
def test_quarantine_gate_refuses_live_references_and_succeeds_after_stop(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, kind: str, stoppable: bool
) -> None:
    path = lab.add()
    (path / "payload").write_bytes(b"x" * 4096)
    child = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(path), kind],
        cwd=lab.repo,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None and child.stdout.readline() == "ready\n"

        def view() -> dict[int, Any]:
            return {
                pid: node
                for pid, node in lab.real_snapshot().items()
                if pid == child.pid
            }

        monkeypatch.setattr(lab.lifecycle.processes, "snapshot", view)
        monkeypatch.setattr(lab.lifecycle.processes, "agents", lambda root: [])

        def reference_scan(root: Path, **kwargs: Any) -> Any:
            kwargs.setdefault("view", view())
            return lab.real_references(root, **kwargs)

        monkeypatch.setattr(lab.references, "scan", reference_scan)
        attempted = []
        if not stoppable:
            monkeypatch.setattr(
                lab.lifecycle.processes,
                "stop_tree",
                lambda ident, **kw: attempted.append(kw.get("proof")),
            )
        lab.now += LIFETIME
        code, report = lab.cleaner().run(cleanup=True)
        assert report["results"][0]["deleted"] is stoppable
        assert code == int(not stoppable)
        checks = [check for check in report["reference_checks"] if check["references"]]
        assert checks, "the real proc holder must be observed"
        assert any(
            ref["item"].split("/")[0] == ("maps" if kind == "mmap" else kind)
            for check in checks
            for row in check["references"]
            for ref in row["references"]
        )
        if kind == "mmap":
            assert all(
                not ref["item"].startswith("fd/")
                for check in checks
                for row in check["references"]
                for ref in row["references"]
            )
        if stoppable:
            assert child.wait(timeout=5) == -signal.SIGKILL
        else:
            assert attempted and child.poll() is None
            row = next(
                row for row in lab.records().values() if row["phase"] == "quarantined"
            )
            assert Path(row["quarantine"]).is_dir()
            assert "reference gate failed" in report["results"][0]["error"]
            assert str(path) in lab.git("worktree", "list", "--porcelain")
            assert child.stdin is not None
            child.stdin.write("\n")
            child.stdin.flush()
            assert child.wait(timeout=5) == 0
            lab.run(cleanup=True)
        assert not path.exists()
        assert all(
            not Path(row["quarantine"]).exists()
            for row in lab.records().values()
            if row.get("quarantine")
        )
        assert str(path) not in lab.git("worktree", "list", "--porcelain")
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)
