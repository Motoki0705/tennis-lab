"""Reference decisions use proc files, with no signals or host-wide fixtures."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest


class ProcFiles:
    def __init__(self, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.syspath_prepend(
            str(Path(__file__).resolve().parents[3] / ".agents/ops")
        )
        self.module = importlib.import_module("worktree_cleanup.references")
        self.root = tmp / "worktree with spaces"
        self.proc = tmp / "proc"
        self.view: dict[int, Any] = {}
        self.add(11, 1, "20")

    def add(self, pid: int, parent: int, ticks: str) -> None:
        p = self.proc / str(pid)
        p.mkdir(parents=True)
        (p / "fd").mkdir()
        (p / "map_files").mkdir()
        (p / "comm").write_text("python\n")
        (p / "maps").write_text("")
        self.stat(pid, parent, ticks)
        for name, value in (
            ("cwd", "/outside"),
            ("root", "/"),
            ("exe", "/usr/bin/python"),
        ):
            (p / name).symlink_to(value)
        self.view[pid] = {
            "identity": {"pid": pid, "boot_id": "boot", "start_ticks": ticks},
            "parent": parent,
            "uid": os.getuid(),
            "cwd": "/outside",
        }

    def stat(self, pid: int, parent: int, ticks: str, state: str = "S") -> None:
        (self.proc / str(pid) / "stat").write_text(
            f"{pid} (test worker) "
            + " ".join([state, str(parent), *(["0"] * 17), ticks])
        )

    def scan(self, **kwargs: Any) -> dict[str, Any]:
        return dict(
            self.module.scan(self.root, view=self.view, proc=self.proc, **kwargs)
        )


@pytest.fixture
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ProcFiles:
    return ProcFiles(tmp_path, monkeypatch)


@pytest.mark.parametrize("kind", ["cwd", "root", "exe", "fd", "maps", "map_files"])
def test_detects_each_reference_kind_and_deleted_paths(
    files: ProcFiles, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    p = files.proc / "11"
    value = str(files.root / "opened file") + " (deleted)"
    if kind in {"cwd", "root", "exe"}:
        (p / kind).unlink()
        (p / kind).symlink_to(value)
    elif kind == "fd":
        (p / "fd/3").symlink_to(value)
    elif kind == "maps":
        (p / "maps").write_text("1000-2000 r--p 0000 00:00 12 " + value + "\n")
    else:
        (p / "map_files/1000-2000").symlink_to(value)
        original = Path.read_text

        def text(path: Path, *args: Any, **kwargs: Any) -> str:
            if path == p / "maps":
                raise PermissionError("maps")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", text)
    result = files.scan()
    assert result["errors"] == []
    assert len(result["references"]) == 1
    row = result["references"][0]
    assert row["identity"] == files.view[11]["identity"]
    assert row["references"][0]["item"].split("/")[0] == kind
    assert row["references"][0]["path"] == str(files.root / "opened file")


@pytest.mark.parametrize("related", [False, True])
@pytest.mark.parametrize("item", ["cwd", "root", "exe", "fd", "maps"])
def test_opaque_process_only_blocks_when_relation_was_observed(
    files: ProcFiles, monkeypatch: pytest.MonkeyPatch, item: str, related: bool
) -> None:
    original_link, original_text, original_iter = (
        os.readlink,
        Path.read_text,
        Path.iterdir,
    )

    def readlink(path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path) == files.proc / "11" / item:
            raise PermissionError(item)
        return original_link(path, *args, **kwargs)

    def text(path: Path, *args: Any, **kwargs: Any) -> str:
        if item == "maps" and path == files.proc / "11/maps":
            raise PermissionError(item)
        return original_text(path, *args, **kwargs)

    def iterdir(path: Path) -> Any:
        if (item == "fd" and path == files.proc / "11/fd") or (
            item == "maps" and path == files.proc / "11/map_files"
        ):
            raise PermissionError(item)
        return original_iter(path)

    monkeypatch.setattr(os, "readlink", readlink)
    monkeypatch.setattr(Path, "read_text", text)
    monkeypatch.setattr(Path, "iterdir", iterdir)
    result = files.scan(related=[files.view[11]["identity"]] if related else [])
    assert bool(result["errors"]) is related
    assert result["references"] == []
    assert result["uninspectable"][0]["pid"] == 11
    assert result["uninspectable"][0]["comm"] == "python"
    assert result["uninspectable"][0]["related"] is related
    assert item in {r["item"] for r in result["uninspectable"][0]["items"]}


@pytest.mark.parametrize("relation", ["none", "direct", "descendant", "managed_pid"])
def test_foreign_uid_does_not_remove_an_observed_relation(
    files: ProcFiles, monkeypatch: pytest.MonkeyPatch, relation: str
) -> None:
    files.view[11]["uid"] = os.getuid() + 1
    if relation == "descendant":
        files.add(10, 1, "10")
        files.view[10]["cwd"] = str(files.root)
        files.view[11]["parent"] = 10
        files.stat(11, 10, "20")
    original = os.readlink

    def link(path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path) == files.proc / "11/cwd":
            raise PermissionError("foreign")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", link)
    result = files.scan(
        related=[files.view[11]["identity"]] if relation == "direct" else [],
        managed_pids={11} if relation == "managed_pid" else set(),
    )
    assert bool(result["errors"]) is (relation != "none")
    assert bool(result["uninspectable"]) is (relation != "none")


@pytest.mark.parametrize("state", ["Z", "X", "exited"])
def test_ended_processes_never_block_or_need_cwd(
    files: ProcFiles, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    if state == "exited":
        (files.proc / "11/stat").unlink()
    else:
        files.stat(11, 1, "20", state)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("ended processes must not read cwd")

    monkeypatch.setattr(os, "readlink", forbidden)
    result = files.scan(related=[files.view[11]["identity"]])
    assert (
        not result["references"]
        and not result["errors"]
        and not result["uninspectable"]
    )


def test_reused_pid_keeps_its_new_identity_when_it_references_the_worktree(
    files: ProcFiles,
) -> None:
    old = dict(files.view[11]["identity"])
    files.stat(11, 1, "99")
    (files.proc / "11/cwd").unlink()
    (files.proc / "11/cwd").symlink_to(files.root)
    result = files.scan(related=[old])
    assert result["references"][0]["identity"]["start_ticks"] == "99"
    assert result["references"][0]["identity"] != old


@pytest.mark.parametrize(
    "record", ["owner", "dispatcher", "runner", "queue_owner", "queue_pid"]
)
def test_managed_records_make_opaque_pids_related(
    files: ProcFiles, monkeypatch: pytest.MonkeyPatch, record: str
) -> None:
    lifecycle = importlib.import_module("worktree_cleanup.lifecycle")
    stopper = lifecycle.Stopper(files.root.parent, root=files.root.parent / "state")
    ident = files.view[11]["identity"]
    parent: dict[str, Any] = {"session": "codex:test", "generation": 1, "tasks": {}}
    runs = []
    names = set()
    queue = files.root.parent / "queue"
    if record == "owner":
        parent["owner_runtime"] = {
            "session": "codex:test",
            "generation": 1,
            "cwd": str(files.root),
            "process": ident,
        }
    elif record == "dispatcher":
        parent["dispatch"] = {
            "jobs": {
                "job": {"phase": "running", "cwd": str(files.root), "worker": ident}
            }
        }
    elif record == "runner":
        directory = files.root.parent / "runtime"
        directory.mkdir()
        (directory / "result.json").write_text(
            json.dumps({"runner": ident, "process": None})
        )
        runs.append(directory)
    else:
        (queue / "state").mkdir(parents=True)
        suffix = "owner" if record == "queue_owner" else "state"
        (queue / "state" / ("job." + suffix)).write_text(
            "pid=11\nstart_time=20\njob=job.job\n"
        )
        names.add("job.job")
    monkeypatch.setattr(
        stopper.store, "read_data", lambda: {"parents": {"o/r#1": parent}}
    )
    monkeypatch.setattr(stopper, "queue_jobs", lambda root: (queue, names))
    known, pids = stopper.managed_processes(files.root, files.view, runs=runs)
    if record == "queue_pid":
        assert known == [] and pids == {11}, (
            "a PID-only record must not authorize a signal"
        )
    else:
        assert ident in known
    original = os.readlink

    def link(path: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(path) == files.proc / "11/cwd":
            raise PermissionError("opaque managed process")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", link)
    result = files.scan(related=known, managed_pids=pids)
    assert result["errors"] and result["uninspectable"][0]["related"] is True


def test_sibling_prefix_is_not_a_worktree_reference(files: ProcFiles) -> None:
    (files.proc / "11/fd/3").symlink_to(str(files.root) + "-other/file")
    assert files.scan()["references"] == []


@pytest.mark.parametrize("related", [False, True])
def test_stat_permission_failure_stays_visible_without_a_fabricated_identity(
    files: ProcFiles, monkeypatch: pytest.MonkeyPatch, related: bool
) -> None:
    original = Path.read_text

    def text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == files.proc / "11/stat":
            raise PermissionError("opaque stat")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", text)
    monkeypatch.setattr(files.module.processes, "snapshot", lambda: {})
    result = files.module.scan(
        files.root,
        proc=files.proc,
        related=[files.view[11]["identity"]] if related else [],
    )
    assert bool(result["errors"]) is related
    assert result["uninspectable"][0]["pid"] == 11
    assert result["uninspectable"][0]["identity"] is None
    assert result["uninspectable"][0]["items"][0]["item"] == "stat"

    # Public dry-run planning must expose stat failures as well as path failures.
    lifecycle = importlib.import_module("worktree_cleanup.lifecycle")
    stopper = lifecycle.Stopper(files.root.parent, root=files.root.parent / "state")
    known = [files.view[11]["identity"]] if related else []
    monkeypatch.setattr(stopper, "known_processes", lambda root: known)
    monkeypatch.setattr(files.module.processes, "agents", lambda root: [])
    real_scan = files.module.scan
    monkeypatch.setattr(
        files.module,
        "scan",
        lambda root, **kwargs: real_scan(root, proc=files.proc, **kwargs),
    )
    preflight = stopper.preflight(files.root)
    assert preflight["reference_gate"]["uninspectable"][0]["pid"] == 11
    assert bool(preflight["errors"]) is related


@pytest.mark.parametrize("name", ["codex", "claude", "python"])
def test_only_native_agents_supply_session_ownership_from_their_environment(
    files: ProcFiles, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    ident = files.view[11]["identity"]
    original = Path.read_bytes

    def read(path: Path) -> bytes:
        if path == Path("/proc/11/cmdline"):
            return name.encode() + b"\0tool.py\0"
        if path == Path("/proc/11/environ"):
            return b"CODEX_THREAD_ID=session\0CLAUDE_CODE_SESSION_ID=other\0"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(files.module.processes, "process_identity", lambda pid: ident)
    result = files.module.processes.agent_metadata(ident)
    if name == "python":
        assert result is None, "a tool must not claim its parent's inherited session"
    else:
        assert result == {
            "identity": ident,
            "sessions": ["codex:session", "claude:other"],
        }


@pytest.mark.parametrize("ended", ["exited", "Z", "X"])
def test_reference_observed_before_parent_exit_keeps_the_snapshot_child_edge(
    files: ProcFiles, monkeypatch: pytest.MonkeyPatch, ended: str
) -> None:
    files.add(12, 11, "30")
    (files.proc / "11/fd/3").symlink_to(files.root / "held file")
    real_text = Path.read_text
    reads = []

    def text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == files.proc / "11/stat":
            reads.append(True)
            if len(reads) == 2:
                # The edge was captured before C left P, after P's fd was read.
                files.stat(12, 900, "30")
                if ended == "exited":
                    raise FileNotFoundError(str(path))
                files.stat(11, 1, "20", ended)
        return real_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", text)
    result = files.scan()
    assert len(reads) == 2
    assert {row["identity"]["pid"] for row in result["references"]} == {11}
    assert {row["pid"] for row in result["related"]} == {11, 12}
