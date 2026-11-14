"""Replacement and process provenance survive the complete deletion attempt."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.e2e.agents_ops.test_worktree_cleanup import LIFETIME, Lab, NativeForest
from tests.e2e.agents_ops.test_worktree_cleanup import lab as lab


class Interruption(BaseException):
    pass


class RenameRace:
    def __init__(self, lab: Lab, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
        self.lab = lab
        self.path = lab.add()
        self.quarantine: Path | None = None
        self.stopped: list[str] = []
        self.moved_stops: list[Path] = []
        real_rename = lab.inventory.rename_no_replace

        def race(source: int, name: str, target: int, destination: str) -> None:
            if self.quarantine is not None:
                real_rename(source, name, target, destination)
                if mode == "recovery_interrupted":
                    raise Interruption()
                return
            # The old root leaf is already open/pinned. Reuse the original
            # path between its check and the actual rename syscall.
            lab.git("worktree", "remove", "--force", "--force", str(self.path))
            monkeypatch.setattr(lab.inventory, "birth_time", lambda fd: lab.now)
            lab.add("replacement", path=self.path)
            (self.path / "new-work").write_text("preserve this successor")
            self.quarantine = Path(os.readlink(f"/proc/self/fd/{target}")) / destination
            real_rename(source, name, target, destination)
            if mode == "interrupted":
                raise Interruption()
            if mode == "occupied":
                self.path.mkdir()
                (self.path / "obstruction").write_text("another owner's data")

        monkeypatch.setattr(lab.inventory, "rename_no_replace", race)

        def stop(root: Path, stamp: str) -> None:
            if root.exists():
                self.stopped.append((root / "tracked").read_text())

        stopper = SimpleNamespace(
            stop=stop,
            stop_moved=lambda root, original, stamp: self.moved_stops.append(root),
            finish_legacy=lambda *args, **kwargs: None,
        )
        self.cleaner = lab.module.Cleaner(
            lab.repo, directory=lab.state, clock=lambda: lab.now, stopper=stopper
        )
        lab.now += LIFETIME

    def restored(self) -> None:
        assert not self.moved_stops, "a successor was passed to stop_moved"
        assert self.path.is_dir()
        assert (self.path / "new-work").read_text() == "preserve this successor"
        assert self.lab.git("branch", "--show-current", cwd=self.path) == "replacement"
        assert str(self.path) in self.lab.git("worktree", "list", "--porcelain")
        assert all(content == "original\n" for content in self.stopped)
        assert self.quarantine is not None and not self.quarantine.exists()


@pytest.mark.parametrize("mode", ["verify", "interrupted", "recovery_interrupted"])
def test_stat_to_rename_replacement_is_restored_before_stopping_it(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    race = RenameRace(lab, monkeypatch, mode)
    if mode in {"interrupted", "recovery_interrupted"}:
        with pytest.raises(Interruption):
            race.cleaner.run(cleanup=True)
        assert race.path.exists() is (mode == "recovery_interrupted")
    code, report = race.cleaner.run(cleanup=True)
    assert code == 1
    assert report["results"][0]["deleted"] is False
    race.restored()
    assert "replacement preserved" in report["results"][0]["error"]
    assert any(r["phase"] == "superseded" for r in lab.records().values())
    assert race.cleaner.run(cleanup=True)[0] == 0
    race.restored()


def test_quarantine_recovery_keeps_occupied_origin_and_retries_before_any_stop(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    race = RenameRace(lab, monkeypatch, "occupied")
    for _ in range(2):
        code, report = race.cleaner.run(cleanup=True)
        assert code == 1 and report["results"][0]["deleted"] is False
        assert race.quarantine is not None
        assert (race.quarantine / "new-work").read_text() == "preserve this successor"
        assert (race.path / "obstruction").read_text() == "another owner's data"
        assert not race.moved_stops
        assert all(content == "original\n" for content in race.stopped)
        assert any(r["phase"] == "quarantine_conflict" for r in lab.records().values())
    (race.path / "obstruction").unlink()
    race.path.rmdir()
    assert race.cleaner.run(cleanup=True)[0] == 1  # Recovery is never deletion success.
    race.restored()
    assert race.cleaner.run(cleanup=True)[0] == 0


@pytest.mark.parametrize(
    "point",
    ["registry", "registry_missing_ancestor", "discovery", "capture_interruption"],
)
def test_parent_exit_during_stop_preparation_keeps_new_orphan_failure_on_retry(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    path = lab.add()
    forest = NativeForest(lab, monkeypatch)
    parent = dict(forest.nodes[101]["identity"])
    if point == "registry_missing_ancestor":
        forest.nodes[101]["parent"] = 777  # Exited before the baseline read it.
    monkeypatch.setattr(
        forest.proc,
        "agents",
        lambda root: (
            [{"identity": parent, "sessions": []}] if 101 in forest.nodes else []
        ),
    )

    def stopping(*args: Any) -> None:
        if 101 in forest.nodes:
            forest.add(103, 101)
            forest.nodes.pop(101)
            for node in forest.nodes.values():
                if node["parent"] == 101:
                    node["parent"] = 1
            if point == "registry_missing_ancestor":
                forest.nodes[103]["parent"] = 900  # An unknown outside subreaper.

    monkeypatch.setattr(lab.lifecycle.Stopper, "registry", stopping)
    if point in {"discovery", "capture_interruption"}:
        forest.nodes[101]["parent"] = 900

        def discovery(root: Path) -> list[dict[str, Any]]:
            if 101 in forest.nodes:
                stopping()
                forest.nodes[103]["parent"] = 900
                if point == "capture_interruption":
                    raise Interruption()
            return []

        monkeypatch.setattr(forest.proc, "agents", discovery)
    lab.now += LIFETIME
    if point == "capture_interruption":
        with pytest.raises(Interruption):
            lab.run(cleanup=True)
    for _ in range(2):
        code, report = lab.cleaner().run(cleanup=True)
        assert code == 1
        assert report["results"][0]["deleted"] is False
        assert path.is_dir()
        assert 103 in forest.nodes
        assert not any(pid == 103 for pid, _ in forest.signals)
    forest.nodes.pop(103)
    if point == "capture_interruption":
        # An incomplete root capture is insufficient proof even after this
        # particular orphan ends; retries must not silently reset the baseline.
        assert lab.cleaner().run(cleanup=True)[0] == 1
        assert path.is_dir()
    else:
        if point == "discovery":
            # This child was already in the first baseline. Its vanished
            # parent's cwd was never observed, so it must not be guessed safe.
            assert 102 in forest.nodes
            assert lab.cleaner().run(cleanup=True)[0] == 1
            forest.nodes.pop(102)
        lab.run(cleanup=True)
        assert not path.exists()


def test_empty_target_scope_preserves_an_unrelated_new_orphan(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    forest = NativeForest(lab, monkeypatch, newborn="orphan")
    forest.nodes.pop(101)
    forest.nodes.pop(102)
    monkeypatch.setattr(forest.proc, "agents", lambda root: [])
    lab.now += LIFETIME
    lab.run(cleanup=True)
    assert not path.exists()
    assert set(forest.nodes) == {1, 900, 103}
    assert not forest.signals


class SnapshotRace:
    """Run the production snapshot against /proc reads with a scheduled exit."""

    def __init__(
        self,
        lab: Lab,
        monkeypatch: pytest.MonkeyPatch,
        path: Path,
        point: str,
        child_first: bool = False,
        order: list[int] | None = None,
    ) -> None:
        self.point = point
        self.fired = False
        self.reads: dict[int, int] = {}
        self.first_view: dict[int, Any] | None = None
        proc = lab.lifecycle.processes
        scope, identity = proc.scoped_processes, proc.process_identity
        self.forest = NativeForest(lab, monkeypatch)
        for node in self.forest.nodes.values():
            node["cwd"] = "/tmp/outside-worktree"
        if point != "outside":
            self.forest.nodes[101]["cwd"] = str(path)
        self.original = copy.deepcopy(self.forest.nodes)
        if point == "new_child":
            self.forest.nodes.pop(102)
        order = order or ([1, 900, 102, 101] if child_first else [1, 900, 101, 102])
        read_text, stat, iterdir, readlink = (
            Path.read_text,
            Path.stat,
            Path.iterdir,
            os.readlink,
        )
        monkeypatch.setattr(proc, "snapshot", lab.real_snapshot)

        def observe_scope(root: Path, view: dict[int, Any]) -> Any:
            if self.first_view is None:
                self.first_view = copy.deepcopy(view)
            return scope(root, view)

        monkeypatch.setattr(proc, "scoped_processes", observe_scope)
        monkeypatch.setattr(proc, "process_identity", identity)
        monkeypatch.setattr(proc, "agents", lambda root: [])

        def proc_pid(candidate: Any, name: str | None = None) -> int | None:
            parts = Path(candidate).parts
            if (
                len(parts) == (3 if name is None else 4)
                and parts[:2] == ("/", "proc")
                and parts[2].isdecimal()
                and (name is None or parts[3] == name)
            ):
                return int(parts[2])
            return None

        def raw(pid: int) -> str:
            node = self.forest.nodes.get(pid)
            if node is None:
                raise FileNotFoundError(str(pid))
            fields = [
                node.get("state", "R"),
                str(node["parent"]),
                *(["0"] * 17),
                node["identity"]["start_ticks"],
            ]
            return f"{pid} (native worker) " + " ".join(fields)

        def text(candidate: Path, *args: Any, **kwargs: Any) -> str:
            if candidate == Path("/proc/sys/kernel/random/boot_id"):
                return "boot"
            pid = proc_pid(candidate, "stat")
            if pid is None:
                return read_text(candidate, *args, **kwargs)
            self.reads[pid] = self.reads.get(pid, 0) + 1
            value = raw(pid)
            if pid == 101 and not self.fired:
                if point == "stat" and self.reads[pid] == 1:
                    self.exit()
                elif (
                    point
                    in {"identity", "reuse", "zombie", "outside", "identity_permission"}
                    and self.reads[pid] == 2
                ):
                    self.exit()
                    if point == "identity_permission":
                        raise PermissionError(str(pid))
                    return raw(pid)  # None/ENOENT or a different PID generation.
            return value

        def info(candidate: Path, *args: Any, **kwargs: Any) -> Any:
            pid = proc_pid(candidate)
            if pid is None:
                return stat(candidate, *args, **kwargs)
            if pid == 101 and point == "uid" and not self.fired:
                self.exit()
            if pid not in self.forest.nodes:
                raise FileNotFoundError(str(pid))
            return SimpleNamespace(st_uid=self.forest.nodes[pid].get("uid", os.getuid()))

        def cwd(candidate: Any, *args: Any, **kwargs: Any) -> Any:
            pid = proc_pid(candidate, "cwd")
            if pid is None:
                return readlink(candidate, *args, **kwargs)
            if pid not in self.forest.nodes:
                raise FileNotFoundError(str(pid))
            value = str(self.forest.nodes[pid]["cwd"])
            if pid == 101 and point in {"cwd", "new_child"} and not self.fired:
                self.exit()
                if point == "cwd":
                    raise FileNotFoundError(str(pid))
            return value

        monkeypatch.setattr(Path, "read_text", text)
        monkeypatch.setattr(Path, "stat", info)
        monkeypatch.setattr(
            Path,
            "iterdir",
            lambda candidate: (
                iter(
                    [Path(f"/proc/{pid}") for pid in order if pid in self.forest.nodes]
                )
                if candidate == Path("/proc")
                else iterdir(candidate)
            ),
        )
        monkeypatch.setattr(os, "readlink", cwd)

    def exit(self) -> None:
        self.fired = True
        if self.point == "new_child":
            self.forest.add(102, 101)
            self.forest.nodes[102]["cwd"] = "/tmp/outside-worktree"
        if self.point == "zombie":
            self.forest.nodes[101]["state"] = "Z"
        else:
            self.forest.nodes.pop(101)
        for node in self.forest.nodes.values():
            if node["parent"] == 101:
                node["parent"] = 1
        if self.point == "reuse":
            self.forest.add(101, 1)
            self.forest.nodes[101].update(cwd="/tmp/replacement")
            self.forest.nodes[101]["identity"]["start_ticks"] = "10001"


@pytest.mark.parametrize(
    "point,child_first",
    [
        ("identity", False),
        ("cwd", False),
        ("uid", False),
        ("stat", False),
        ("stat", True),
        ("new_child", False),
        ("reuse", False),
        ("zombie", False),
        ("identity_permission", False),
    ],
)
def test_snapshot_exit_boundaries_never_leave_a_live_child_after_deletion(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, point: str, child_first: bool
) -> None:
    path = lab.add()
    race = SnapshotRace(lab, monkeypatch, path, point, child_first)
    lab.now += LIFETIME
    code, report = lab.cleaner().run(cleanup=True)
    assert race.fired, "the scheduled exit must occur inside snapshot's /proc reads"
    assert race.first_view is not None
    assert 101 in race.first_view, "the observed parent stat must survive its exit"
    assert race.first_view[101]["identity"] == race.original[101]["identity"]
    assert race.first_view[101]["parent"] == 1
    if point in {"identity", "zombie", "new_child", "reuse", "identity_permission"}:
        assert race.first_view[101]["cwd"] == str(path)
    if point in {"identity", "zombie"}:
        assert 102 not in race.forest.nodes, (
            "observed child of the target must be stopped"
        )
        assert code == 0 and not path.exists()
        assert (102, race.forest.proc.signal.SIGKILL) in race.forest.signals
    else:
        assert code == 1 and not report["results"][0]["deleted"]
        assert path.is_dir() and 102 in race.forest.nodes
        assert not race.forest.signals, (
            "unconfirmed provenance must never select a process by guess"
        )
        proof = lab.cleaner().stopper.native_proof(path)
        saved = race.forest.proc.read_stop_proof(proof)
        assert race.original[101]["identity"] in saved["unconfirmed"]
        assert lab.cleaner().run(cleanup=True)[0] == 1
        race.forest.nodes.pop(102)
        if point == "reuse":
            race.forest.nodes.pop(101)
        lab.run(cleanup=True)
        assert not path.exists()
    assert 900 in race.forest.nodes


def test_snapshot_exit_of_other_worktree_does_not_block_cleanup(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    race = SnapshotRace(lab, monkeypatch, path, "outside")
    lab.now += LIFETIME
    lab.run(cleanup=True)
    assert race.fired and not path.exists()
    assert 102 in race.forest.nodes and 900 in race.forest.nodes
    assert not race.forest.signals


@pytest.mark.parametrize("job", ["older_branch", "foreign_service", "reused_branch"])
def test_discovery_preserves_an_independent_branch_enumerated_after_a_lost_root(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, job: str
) -> None:
    path = lab.add()
    race = SnapshotRace(lab, monkeypatch, path, "stat", order=[1, 101, 900, 103, 102])
    race.forest.nodes[900]["identity"]["start_ticks"] = "50"
    race.forest.add(103, 900)
    race.forest.nodes[103]["cwd"] = str(lab.repo)
    if job == "foreign_service":
        race.forest.nodes[103].update(parent=1, uid=os.getuid() + 1, cwd=None)
    view = lab.real_snapshot()
    if job == "reused_branch":
        # The child's parent PID can name a replacement rather than this
        # older process. Its observed start time cannot prove independence.
        view[900].update(identity_lost=True, scope_uncertain=True)
    unknown = race.forest.proc.discovery_uncertainty(
        view, [race.original[101]["identity"]]
    )
    assert race.fired and view[101]["identity_lost"]
    assert {item["pid"] for item in unknown} == (
        {101, 102, 103} if job == "reused_branch" else {101, 102}
    )


@pytest.mark.parametrize("state", ["Z", "X"])
def test_snapshot_ignores_other_worktree_zombies_without_reading_their_cwd(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    path = lab.add()
    race = SnapshotRace(lab, monkeypatch, path, "already_dead")
    for pid in (101, 102):
        race.forest.nodes[pid].update(state=state, cwd="/tmp/other-worktree")
    readlink = os.readlink

    def cwd(candidate: Any, *args: Any, **kwargs: Any) -> Any:
        if str(candidate) in {"/proc/101/cwd", "/proc/102/cwd"}:
            pytest.fail("an already terminated zombie has no cwd to classify")
        return readlink(candidate, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", cwd)
    assert set(lab.real_snapshot()) == {1, 900}
    monkeypatch.setattr(os, "readlink", readlink)
    lab.now += LIFETIME
    lab.run(cleanup=True)
    assert not path.exists() and not race.forest.signals
    assert {101, 102} <= race.forest.nodes.keys()


@pytest.mark.parametrize("published", [False, True])
def test_generation_token_is_atomic_and_stable_after_interrupted_publication(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, published: bool
) -> None:
    path = lab.add()
    admin = Path(lab.git("rev-parse", "--absolute-git-dir", cwd=path))
    monkeypatch.setattr(lab.inventory, "birth_time", lambda fd: None)
    real_link = os.link

    def interrupted(*args: Any, **kwargs: Any) -> None:
        if published:
            real_link(*args, **kwargs)
        raise Interruption()

    with monkeypatch.context() as patch:
        patch.setattr(os, "link", interrupted)
        with pytest.raises(Interruption):
            lab.run()
    marker = admin / lab.inventory.TOKEN_FILE
    assert marker.exists() is published
    before = marker.read_text() if published else None
    first = lab.run()["entries"][0]
    token = marker.read_text()
    assert len(token) == 32
    assert before is None or before == token
    lab.now += LIFETIME - 1
    after = lab.run(cleanup=True)["entries"][0]
    assert path.is_dir()
    assert after["created_at"] == first["created_at"]
    assert after["id"] == first["id"]
    assert (admin / lab.inventory.TOKEN_FILE).read_text() == token
    record = lab.records()[first["id"]]
    assert record["admin_identity"]["token"] == token


def test_generation_marker_symlink_never_writes_its_shared_target(lab: Lab) -> None:
    path = lab.add()
    admin = Path(lab.git("rev-parse", "--absolute-git-dir", cwd=path))
    shared = lab.repo.parent / "shared-marker"
    shared.write_text("shared data")
    (admin / lab.inventory.TOKEN_FILE).symlink_to(shared)
    with pytest.raises(OSError):
        lab.run(cleanup=True)
    assert shared.read_text() == "shared data"
    assert path.is_dir()
    assert not (lab.state / "worktrees.json").exists()


@pytest.mark.parametrize("has_birth", [False, True])
def test_migration_seals_old_records_without_reusing_unproven_fallback_age(
    lab: Lab, monkeypatch: pytest.MonkeyPatch, has_birth: bool
) -> None:
    path = lab.add()
    if not has_birth:
        monkeypatch.setattr(lab.inventory, "birth_time", lambda fd: None)
    first = lab.run()["entries"][0]
    state_path = lab.state / "worktrees.json"
    state = json.loads(state_path.read_text())
    old = state["records"].pop(first["id"])
    ident = old["admin_identity"]
    ident.pop("token")
    legacy_id = hashlib.sha256(
        f"{old['admin']}:{ident['device']}:{ident['inode']}:{ident['birth_time_ns']}".encode()
    ).hexdigest()
    old["id"] = legacy_id
    state["records"][legacy_id] = old
    state_path.write_text(json.dumps(state))
    lab.now += LIFETIME
    report = lab.run(cleanup=True)
    current = report["entries"][0]
    assert current["id"] != legacy_id
    assert current["identity_migration"] == "unsealed_generation"
    assert current["selected"] is has_birth
    assert path.exists() is not has_birth
    assert current["created_at"] == (
        first["created_at"] if has_birth else lab.storage.utc(lab.now)
    )


def test_python_process_in_worktree_is_stopped_without_a_native_agent_name(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; print('ready', flush=True); sys.stdin.read()",
        ],
        cwd=path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None and child.stdout.readline() == "ready\n"
        # Exercise real /proc identities, cwd selection, thread states and
        # pidfd signals, with the enumerated PID scope limited to our child.
        monkeypatch.setattr(
            lab.lifecycle.processes,
            "snapshot",
            lambda: {
                pid: node
                for pid, node in lab.real_snapshot().items()
                if pid == child.pid
            },
        )
        lab.now += LIFETIME
        lab.run(cleanup=True)
        assert child.wait(timeout=5) == -lab.lifecycle.processes.signal.SIGKILL
        assert not path.exists()
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)


def test_removal_keeps_a_successor_installed_after_the_root_fd_is_opened(
    lab: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = lab.add()
    expected = lab.inventory.identity(path)
    pinned = path.with_name("pinned-old-root")
    real_clear = lab.inventory.clear_directory

    def replace(fd: int, **kwargs: Any) -> None:
        path.rename(pinned)
        path.mkdir()
        (path / "keep").write_text("new root")
        real_clear(fd, **kwargs)

    monkeypatch.setattr(lab.inventory, "clear_directory", replace)
    with pytest.raises(lab.storage.CleanupError, match="replaced during removal"):
        lab.inventory.remove_tree(path, expected)
    assert (path / "keep").read_text() == "new root"
    assert not (pinned / "tracked").exists()
