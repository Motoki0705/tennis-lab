"""The shared state writer recovers from abandoned files and rejects unsafe entries."""

from __future__ import annotations

import grp
import importlib
import json
import os
import pwd
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"


@pytest.fixture(autouse=True)
def ops_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(OPS))


@pytest.mark.parametrize("writer", ["registry", "agent-index"])
def test_regular_legacy_temporary_does_not_block_write(
    tmp_path: Path, writer: str
) -> None:
    directory = tmp_path / "private"
    directory.mkdir(mode=0o700)
    path = directory / ("registry.json" if writer == "registry" else "index.json")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text('{"interrupted":')
    previous = path.with_name(f".{path.name}.{'1' * 32}.tmp")
    previous.write_text('{"previous":')
    value = {"issues": {"10": {"title": "本文"}}}
    if writer == "registry":
        importlib.import_module("parent.state").Store.write_json(path, value)
    else:
        importlib.import_module("dispatcher.agent_input")._write_index(path, value)
        assert '    "10":{"title":"本文"}' in path.read_text().splitlines()

    assert json.loads(path.read_text()) == value
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert temporary.read_text() == '{"interrupted":'
    assert previous.read_text() == '{"previous":'
    assert set(directory.iterdir()) == {path, temporary, previous}


@pytest.mark.parametrize(
    "entry", ["registry.json", "registry.json.tmp", f".registry.json.{'1' * 32}.tmp"]
)
def test_write_rejects_link_without_touching_target(tmp_path: Path, entry: str) -> None:
    store = importlib.import_module("parent.state").Store
    path = tmp_path / "registry.json"
    link = path.with_name(entry)
    target = tmp_path / "private-record.json"
    target.write_text("preserve me")
    link.symlink_to(target)

    with pytest.raises(OSError, match="symlink"):
        store.write_json(path, {"replacement": True})

    assert link.is_symlink()
    assert target.read_text() == "preserve me"


@pytest.mark.parametrize("entry", ["registry.json", "registry.json.tmp"])
def test_write_rejects_a_file_owned_by_another_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    store = importlib.import_module("parent.state").Store
    path = tmp_path / "registry.json"
    foreign = path.with_name(entry)
    foreign.write_text("foreign state")
    original_stat = os.stat

    def stat_with_foreign_owner(
        name: str | Path | int,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        info = original_stat(name, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if name in (foreign, foreign.name):
            fields = list(info)
            fields[4] = os.geteuid() + 1  # st_uid
            return os.stat_result(fields)
        return info

    monkeypatch.setattr(os, "stat", stat_with_foreign_owner)
    with pytest.raises(OSError, match="wrong_owner"):
        store.write_json(path, {"replacement": True})

    assert foreign.read_text() == "foreign state"
    assert set(tmp_path.iterdir()) == {foreign}


def test_write_creates_private_directories(tmp_path: Path) -> None:
    store = importlib.import_module("parent.state").Store
    path = tmp_path / "state" / "parents" / "record.json"

    store.write_json(path, {"entry": 1})

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.parent.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_read_reopens_state_replaced_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_files = importlib.import_module("lib.state_files")
    path = tmp_path / "registry.json"
    path.write_text('{"version":1,"parents":{}}')
    replacement = tmp_path / "next-registry.json"
    current = '{"version":2,"parents":{}}'
    replacement.write_text(current)
    original_open = os.open

    def open_during_replace(
        name: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        fd = original_open(name, flags, mode, dir_fd=dir_fd)
        if name == path.name and replacement.exists():
            replacement.replace(path)
            assert os.fstat(fd).st_nlink == 0
        return fd

    monkeypatch.setattr(os, "open", open_during_replace)
    assert state_files.read_state_text(path) == current


def test_read_rejects_hardlinked_state(tmp_path: Path) -> None:
    state_files = importlib.import_module("lib.state_files")
    path = tmp_path / "registry.json"
    path.write_text("{}")
    (tmp_path / "alias.json").hardlink_to(path)

    with pytest.raises(state_files.StateFileError, match="hardlink"):
        state_files.read_state_text(path)


def test_write_refuses_symlinked_ancestor_without_creating_directories(
    tmp_path: Path,
) -> None:
    store = importlib.import_module("parent.state").Store
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "symlinked"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError, match="symlink"):
        store.write_json(link / "private" / "registry.json", {})

    assert not list(outside.iterdir())


def _mock_private_group(
    monkeypatch: pytest.MonkeyPatch, uid: int, gid: int
) -> SimpleNamespace:
    account = SimpleNamespace(pw_uid=uid, pw_gid=gid, pw_name="owner")
    accounts = {
        uid: account,
        uid + 1: SimpleNamespace(pw_uid=uid + 1, pw_gid=gid + 1),
    }
    groups = {gid: SimpleNamespace(gr_mem=[account.pw_name])}
    monkeypatch.setattr(pwd, "getpwuid", accounts.__getitem__)
    monkeypatch.setattr(pwd, "getpwall", lambda: list(accounts.values()))
    monkeypatch.setattr(grp, "getgrgid", groups.__getitem__)
    return SimpleNamespace(account=account, accounts=accounts, groups=groups)


def test_open_state_directory_through_user_private_group_ancestors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_files = importlib.import_module("lib.state_files")
    path = tmp_path
    for name, mode in (
        ("home", 0o750),
        (".local", 0o775),
        ("state", 0o775),
        ("tennis-lab-agents", 0o755),
        ("private", 0o700),
    ):
        path /= name
        path.mkdir()
        path.chmod(mode)
    _mock_private_group(monkeypatch, os.geteuid(), path.stat().st_gid)

    with state_files.open_state_directory(path) as directory:
        assert os.path.samestat(os.fstat(directory.fd), path.stat())


@pytest.mark.parametrize(
    "problem",
    [
        "supplementary-member",
        "primary-member",
        "different-primary-group",
        "missing-group",
        "owner-not-enumerated",
    ],
)
def test_write_refuses_unverified_group_writable_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    state_files = importlib.import_module("lib.state_files")
    store = importlib.import_module("parent.state").Store
    shared = tmp_path / "group-writable"
    shared.mkdir()
    shared.chmod(0o775)
    directory = shared / "private"
    directory.mkdir(mode=0o700)
    uid, gid = os.geteuid(), shared.stat().st_gid
    database = _mock_private_group(monkeypatch, uid, gid)
    if problem == "supplementary-member":
        database.groups[gid].gr_mem.append("other-user")
    elif problem == "primary-member":
        database.accounts[uid + 1].pw_gid = gid
    elif problem == "different-primary-group":
        database.account.pw_gid = gid + 1
    elif problem == "missing-group":
        database.groups.clear()
    elif problem == "owner-not-enumerated":
        monkeypatch.setattr(pwd, "getpwall", lambda: [database.accounts[uid + 1]])

    with pytest.raises(state_files.StateFileError, match="shared_write") as exc:
        store.write_json(directory / "registry.json", {})

    assert exc.value.entry == "directory"
    assert str(shared) in str(exc.value)
    assert not list(directory.iterdir())


def test_group_writable_ancestor_must_be_owned_by_current_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_files = importlib.import_module("lib.state_files")
    uid, gid = 1001, 2001
    _mock_private_group(monkeypatch, uid, gid)
    info = SimpleNamespace(st_uid=0, st_gid=gid, st_mode=stat.S_IFDIR | 0o775)
    monkeypatch.setattr(
        state_files, "os", SimpleNamespace(fstat=lambda _fd: info, geteuid=lambda: uid)
    )

    with pytest.raises(state_files.StateFileError, match="shared_write"):
        state_files._check_directory(0, tmp_path, tmp_path / "private", leaf=False)


def test_write_refuses_shared_write_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_files = importlib.import_module("lib.state_files")
    store = importlib.import_module("parent.state").Store
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    directory = shared / "private"
    directory.mkdir(mode=0o700)
    _mock_private_group(monkeypatch, os.geteuid(), shared.stat().st_gid)

    with pytest.raises(state_files.StateFileError, match="shared_write"):
        store.write_json(directory / "registry.json", {})

    assert not list(directory.iterdir())


def test_write_refuses_group_writable_state_directory_with_private_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_files = importlib.import_module("lib.state_files")
    store = importlib.import_module("parent.state").Store
    directory = tmp_path / "private"
    directory.mkdir()
    directory.chmod(0o775)
    _mock_private_group(monkeypatch, os.geteuid(), directory.stat().st_gid)

    with pytest.raises(state_files.StateFileError, match="shared_write"):
        store.write_json(directory / "registry.json", {})

    assert not list(directory.iterdir())


def test_overlapping_writes_commit_complete_files(tmp_path: Path) -> None:
    state_files = importlib.import_module("lib.state_files")
    path = tmp_path / "registry.json"
    path.write_text('{"value": "before"}\n')

    with state_files.atomic_state_writer(path) as outer:
        outer.write('{"value": ')
        with state_files.atomic_state_writer(path) as inner:
            inner.write('{"value": "inner"}\n')
            assert json.loads(path.read_text()) == {"value": "before"}
        assert json.loads(path.read_text()) == {"value": "inner"}
        outer.write('"outer"}\n')

    assert json.loads(path.read_text()) == {"value": "outer"}
    assert set(tmp_path.iterdir()) == {path}


def test_collision_preserves_temporary_owned_by_another_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = importlib.import_module("parent.state").Store
    state_files = importlib.import_module("lib.state_files")
    path = tmp_path / "registry.json"
    token = "1" * 32
    temporary = path.with_name(f".{path.name}.{token}.tmp")
    temporary.write_text("other write")
    monkeypatch.setattr(state_files, "uuid4", lambda: SimpleNamespace(hex=token))

    with pytest.raises(OSError, match="temporary_collision"):
        store.write_json(path, {"value": "new"})

    assert temporary.read_text() == "other write"
    assert not path.exists()
