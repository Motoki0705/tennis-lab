"""Board contracts through temporary SQLite/registry and fake gh/Herdr only."""

from __future__ import annotations

import importlib
import json
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.agents_ops.test_herdr_labels import Labels
from tests.e2e.agents_ops.test_herdr_labels import labels as labels
from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env

OPS = Path(__file__).resolve().parents[3] / ".agents/ops"
NOW = datetime(2026, 10, 22, tzinfo=UTC)


@pytest.fixture
def board(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    monkeypatch.syspath_prepend(str(OPS))
    return SimpleNamespace(
        **{
            name: importlib.import_module("board." + name)
            for name in ("activity", "names", "storage", "service", "errors")
        }
    )


def run_board(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "board", *args],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def private_output_absent(output: str, env: dict[str, str]) -> None:
    for key in (
        "CLAUDE_CODE_SESSION_ID",
        "CODEX_SESSION_ID",
        "CODEX_THREAD_ID",
        "HERDR_PANE_ID",
        "XDG_STATE_HOME",
    ):
        if env.get(key):
            assert env[key] not in output
    assert "Traceback" not in output


def test_catalog_is_about_200_short_distinct_public_names(
    board: SimpleNamespace,
) -> None:
    names = board.names.load_names()
    assert 180 <= len(names) <= 220
    assert all(
        2 <= len(name) <= 9 and name.isascii() and name.isalpha() and name[0].isupper()
        for name in names
    )
    # Independent Levenshtein check: also catches a defect in the catalog validator.
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            row = list(range(len(right) + 1))
            for x, a in enumerate(left.lower(), 1):
                nxt = [x]
                for y, b in enumerate(right.lower(), 1):
                    nxt.append(min(nxt[-1] + 1, row[y] + 1, row[y - 1] + (a != b)))
                row = nxt
            assert row[-1] > 1


@pytest.mark.parametrize(
    "values",
    [
        [],
        ["alice"],
        ["Álice"],
        ["Alice", "ALICE"],
        ["All"],
        ["Ann", "Anna"],
        ["Ben", "Bob", "Jen"],
        ["Bob", "Rob"],
        ["Alice "],
    ],
)
def test_invalid_catalog_is_rejected(board: SimpleNamespace, values: list[str]) -> None:
    with pytest.raises(board.errors.BoardError):
        board.names.validate_names(values)


def test_name_resolution_rejects_reserved_outside_unassigned_and_retired(
    board: SimpleNamespace, tmp_path: Path
) -> None:
    store = board.storage.Store(tmp_path / "board")
    with store.transaction() as connection:
        assigned = board.names.ensure_name(connection, "o/r", 1, "開発", now=NOW)
        assert board.names.resolve_name(connection, assigned.name.lower()) == assigned
        for invalid in ("all", "ALL", "NotInCatalog", board.names.load_names()[1]):
            with pytest.raises(board.errors.BoardError):
                board.names.resolve_name(connection, invalid)
        board.names.retire_name(connection, "o/r", 1, now=NOW)
        with pytest.raises(board.errors.BoardError, match="retired"):
            board.names.resolve_name(connection, assigned.name)


@pytest.mark.parametrize("unread,waiting", [(0, False), (1, False), (0, True)])
def test_reuse_29_and_30_day_boundary_and_pending_state(
    board: SimpleNamespace,
    tmp_path: Path,
    unread: int,
    waiting: bool,
) -> None:
    store = board.storage.Store(tmp_path / "board")
    with store.transaction() as connection:
        first = board.names.ensure_name(
            connection, "o/r", 1, "first", now=NOW, catalog=["Alice"]
        )
        post_id = "0" * 32
        stamp = board.names.timestamp(NOW)
        connection.execute(
            "INSERT INTO posts VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                post_id,
                stamp,
                first.id,
                first.name,
                first.repo,
                first.parent,
                "question" if waiting else "info",
                0,
                "Reuse guard",
                "Keep this history",
                None,
                post_id,
            ),
        )
        connection.execute(
            "INSERT INTO recipients(post_id,assignment_id,name,repo,parent,read_at) VALUES(?,?,?,?,?,?)",
            (
                post_id,
                first.id,
                first.name,
                first.repo,
                first.parent,
                None if unread else stamp,
            ),
        )
        connection.execute(
            "INSERT INTO deliveries(post_id,assignment_id,idempotency_key,state) VALUES(?,?,?,?)",
            (post_id, first.id, "reuse-key", "pending"),
        )
        assert board.activity.summary(connection, first.id).unread_count == unread
        assert board.activity.summary(connection, first.id).awaiting_response == waiting
        board.names.retire_name(connection, "o/r", 1, now=NOW)
        for age in (timedelta(days=29), timedelta(days=30) - timedelta(microseconds=1)):
            with pytest.raises(board.errors.BoardError, match="exhausted"):
                board.names.ensure_name(
                    connection, "o/r", 2, "second", now=NOW + age, catalog=["Alice"]
                )
        if unread or waiting:
            with pytest.raises(board.errors.BoardError, match="exhausted"):
                board.names.ensure_name(
                    connection,
                    "o/r",
                    2,
                    "second",
                    now=NOW + timedelta(days=30),
                    catalog=["Alice"],
                )
        else:
            second = board.names.ensure_name(
                connection,
                "o/r",
                2,
                "second",
                now=NOW + timedelta(days=30),
                catalog=["Alice"],
            )
            assert first.id != second.id
            assert board.names.for_parent(connection, "o/r", 1).retired_at is not None
            assert (
                board.names.resolve_name(connection, "alice", catalog=["Alice"]).parent
                == 2
            )


def test_resume_preserves_name_until_it_is_reused(
    board: SimpleNamespace, tmp_path: Path
) -> None:
    store = board.storage.Store(tmp_path / "board")
    with store.transaction() as connection:
        first = board.names.ensure_name(
            connection, "o/r", 1, "first", now=NOW, catalog=["Alice", "Bob"]
        )
        board.names.retire_name(connection, "o/r", 1, now=NOW)
        resumed = board.names.ensure_name(
            connection,
            "O/R",
            1,
            "resumed",
            now=NOW + timedelta(days=1),
            catalog=["Alice", "Bob"],
        )
        assert (resumed.id, resumed.name) == (first.id, first.name)
        board.names.retire_name(connection, "o/r", 1, now=NOW + timedelta(days=1))
        board.names.ensure_name(
            connection,
            "o/r",
            2,
            "second",
            now=NOW + timedelta(days=31),
            catalog=["Alice", "Bob"],
        )
        resumed = board.names.ensure_name(
            connection,
            "o/r",
            1,
            "resumed",
            now=NOW + timedelta(days=31),
            catalog=["Alice", "Bob"],
        )
        assert resumed.name == "Bob"
        assert (
            board.names.resolve_name(
                connection, "Alice", catalog=["Alice", "Bob"]
            ).parent
            == 2
        )


def test_private_permissions_wal_and_rollback(
    board: SimpleNamespace, tmp_path: Path
) -> None:
    store = board.storage.Store(tmp_path / "board")
    previous_mask = os.umask(0)
    try:
        with store.transaction() as connection:
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            board.names.ensure_name(connection, "o/r", 1, "first", now=NOW)
            assert store.directory.stat().st_mode & 0o777 == 0o700
            for file in store.directory.iterdir():
                assert file.stat().st_mode & 0o777 == 0o600
    finally:
        os.umask(previous_mask)
    with pytest.raises(RuntimeError), store.transaction() as connection:
        board.names.ensure_name(connection, "o/r", 2, "rolled back", now=NOW)
        raise RuntimeError("abort transaction")
    with store.read() as connection:
        assert board.names.for_parent(connection, "o/r", 1) is not None
        assert board.names.for_parent(connection, "o/r", 2) is None
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM name_assignments")


def test_parallel_processes_allocate_unique_ordered_names(
    board: SimpleNamespace, parent_env: ParentEnv
) -> None:
    script = """from board.storage import Store, directory
from board.names import ensure_name
from datetime import datetime, UTC
import os, sys
with Store(directory(os.environ)).transaction() as c:
    value = ensure_name(c, 'o/r', int(sys.argv[1]), 'parallel', now=datetime.now(UTC))
print(value.name)
"""

    def allocate(parent: int) -> str:
        result = subprocess.run(
            [sys.executable, "-c", script, str(parent)],
            env=parent_env.env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, "parallel allocation failed"
        return result.stdout.strip()

    with ThreadPoolExecutor(max_workers=4) as pool:
        names = list(pool.map(allocate, range(1, 9)))
    assert len(set(names)) == 8
    assert set(names) == set(board.names.load_names()[:8])


def test_version_migration_is_atomic_and_unknown_or_corrupt_is_not_replaced(
    board: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = board.storage.Store(tmp_path / "board")
    with store.transaction() as connection:
        board.names.ensure_name(connection, "o/r", 1, "kept", now=NOW)
    base_version = board.storage.SCHEMA_VERSION

    def posting_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE extension_witness(id INTEGER PRIMARY KEY, sender INTEGER REFERENCES name_assignments(id))"
        )

    monkeypatch.setattr(
        board.storage, "MIGRATIONS", (*board.storage.MIGRATIONS, posting_schema)
    )
    monkeypatch.setattr(board.storage, "SCHEMA_VERSION", base_version + 1)
    with pytest.raises(board.errors.BoardError, match="migration"), store.read():
        pass
    with store.transaction() as connection:
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0] == base_version + 1
        )
        assert board.names.for_parent(connection, "o/r", 1).identifier == "kept"
        connection.execute("INSERT INTO extension_witness(sender) VALUES(1)")

    def broken_schema(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE should_rollback(id INTEGER)")
        raise RuntimeError("migration interrupted")

    monkeypatch.setattr(
        board.storage, "MIGRATIONS", (*board.storage.MIGRATIONS, broken_schema)
    )
    monkeypatch.setattr(board.storage, "SCHEMA_VERSION", base_version + 2)
    with pytest.raises(RuntimeError), store.transaction():
        pass
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0] == base_version + 1
        )
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='should_rollback'"
            ).fetchall()
            == []
        )
        connection.execute("PRAGMA user_version=999")
    original = store.path.read_bytes()
    with pytest.raises(board.errors.BoardError, match="unknown"), store.transaction():
        pass
    assert store.path.read_bytes() == original


@pytest.mark.parametrize("contents", [b"broken sqlite database", b""])
def test_corrupt_or_unversioned_storage_is_preserved(
    board: SimpleNamespace, parent_env: ParentEnv, contents: bytes
) -> None:
    store = board.storage.Store(board.storage.directory(parent_env.env))
    store.directory.mkdir(parents=True)
    store.path.write_bytes(contents)
    result = parent_env.cli("register", ok=None)
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["parent_action_applied"] is True
    assert data["assignment"] == "active"
    assert data["board_sync"]["status"] == "error"
    assert store.path.read_bytes() == contents
    response = run_board(parent_env.env, "whoami")
    assert response.returncode == 2
    assert response.stdout == ""
    private_output_absent(
        result.stdout + result.stderr + response.stderr, parent_env.env
    )


def test_unknown_version_cli_exit_and_non_destructive_rejection(
    board: SimpleNamespace, parent_env: ParentEnv
) -> None:
    parent_env.cli("register")
    store = board.storage.Store(board.storage.directory(parent_env.env))
    with sqlite3.connect(store.path) as connection:
        connection.execute("PRAGMA user_version=999")
    original = store.path.read_bytes()
    response = run_board(parent_env.env, "agents")
    assert response.returncode == 2
    assert "unknown board schema version" in response.stderr
    assert response.stdout == ""
    assert store.path.read_bytes() == original


def test_parent_lifecycle_names_and_board_failure_recovery(
    board: SimpleNamespace, parent_env: ParentEnv
) -> None:
    p = parent_env
    first = json.loads(p.cli("register").stdout)["board_sync"]
    assert first["status"] == "assigned"
    assert json.loads(p.cli("register").stdout)["board_sync"]["status"] == "retained"
    assert json.loads(run_board(p.env, "whoami").stdout)["name"] == first["name"]
    for action in ("release", "handoff"):
        result = p.cli(action, "--note-file", p.note())
        assert json.loads(result.stdout)["board_sync"]["name"] == first["name"]
        assert json.loads(run_board(p.env, "agents").stdout) == {"agents": []}
        assert run_board(p.env, "whoami").returncode == 2
        p.cli("register") if action == "release" else p.cli(
            "handoff", "--accept", "--note-file", p.note()
        )
    second = {"CLAUDE_CODE_SESSION_ID": "private-second-parent-owner"}
    p.cli("handoff", "--note-file", p.note())
    accepted = p.cli("handoff", "--accept", "--note-file", p.note(), extra=second)
    assert json.loads(accepted.stdout)["board_sync"]["name"] == first["name"]
    assert run_board(p.env, "whoami").returncode == 2
    assert (
        json.loads(run_board({**p.env, **second}, "whoami").stdout)["name"]
        == first["name"]
    )
    ended = p.cli("release", "--ended", "--note-file", p.note(), extra=second)
    assert json.loads(ended.stdout)["board_sync"]["status"] == "retired"
    with board.storage.Store(board.storage.directory(p.env)).read() as connection:
        assert board.names.for_parent(connection, "o/r", 1).retired_at is not None
    p.cli("register", parent="2")
    with board.storage.Store(board.storage.directory(p.env)).read() as connection:
        assert board.names.for_parent(connection, "o/r", 2).name != first["name"]


@pytest.mark.parametrize("title", ["人間が作成した単独タスク", "[Bug] 単独タスク"])
def test_standalone_parent_has_a_name_and_null_identifier(
    board: SimpleNamespace, parent_env: ParentEnv, title: str
) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["1"].update(
        title=title, labels=[{"name": "bug"}], body="単独タスクを実装する。"
    )
    p.save(state)
    first = json.loads(p.cli("register").stdout)["board_sync"]
    assert first["status"] == "assigned"
    store = board.storage.Store(board.storage.directory(p.env))
    with store.read() as connection:
        assignment = board.names.for_parent(connection, "o/r", 1)
        assert assignment.identifier is None
        assert assignment.public()["identifier"] is None
        assert board.names.resolve_name(connection, first["name"]) == assignment
    roster = run_board(p.env, "agents")
    assert roster.returncode == 0
    entry = json.loads(roster.stdout)["agents"][0]
    assert entry["identifier"] is None
    assert entry["name"] == first["name"]
    assert entry["name_status"] == "assigned"
    identity = run_board(p.env, "whoami")
    assert identity.returncode == 0
    assert json.loads(identity.stdout)["name"] == first["name"]
    # Promotion to a parent and back keeps the assignment, updating only metadata.
    for role_labels, title, identifier in (
        ([{"name": "issue:parent"}], "[開発] 親に移行", "開発"),
        ([{"name": "bug"}], "単独タスクに戻す", None),
    ):
        state = p.state()
        state["issues"]["1"].update(title=title, labels=role_labels)
        p.save(state)
        assert (
            json.loads(p.cli("register").stdout)["board_sync"]["status"] == "retained"
        )
        with store.read() as connection:
            updated = board.names.for_parent(connection, "o/r", 1)
            assert updated.id == assignment.id
            assert updated.identifier == identifier
    private_output_absent(roster.stdout + identity.stdout, p.env)


@pytest.mark.parametrize(
    "title", ["識別名のない親", "[閉じていない親", "[開発 #2] 自身の識別名がない親"]
)
def test_malformed_parent_title_is_not_treated_as_standalone(
    board: SimpleNamespace, parent_env: ParentEnv, title: str
) -> None:
    p = parent_env
    state = p.state()
    state["issues"]["1"]["title"] = title
    p.save(state)
    result = p.cli("register", ok=None)
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["parent_action_applied"] is True
    assert data["board_sync"]["status"] == "error"
    store = board.storage.Store(board.storage.directory(p.env))
    if store.path.exists():
        with store.read() as connection:
            assert board.names.for_parent(connection, "o/r", 1) is None
    assert run_board(p.env, "whoami").returncode == 2
    private_output_absent(result.stdout + result.stderr, p.env)


def test_sync_fetches_issue_without_holding_board_write_lock(
    board: SimpleNamespace, parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = parent_env
    first = json.loads(p.cli("register").stdout)["board_sync"]
    store = board.storage.Store(board.storage.directory(p.env))

    def fetch(_github: object, number: int) -> dict[str, object]:
        assert number == 1
        # Another writer must be able to acquire the lock during GitHub I/O.
        with sqlite3.connect(store.path, timeout=0) as writer:
            writer.execute("BEGIN IMMEDIATE")
        return {"title": "[同期後] 親", "labels": [{"name": "issue:parent"}]}

    monkeypatch.setattr(board.service.GitHub, "issue", fetch)
    result = board.service.sync_parent(p.env, "o/r", 1)
    assert result["status"] == "retained"
    assert result["name"] == first["name"]
    with store.read() as connection:
        assert board.names.for_parent(connection, "o/r", 1).identifier == "同期後"


@pytest.mark.parametrize("change", ["session", "status", "generation", "removed"])
def test_sync_rejects_assignment_changes_during_github_fetch(
    board: SimpleNamespace,
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    p = parent_env
    p.cli("register")
    store = board.storage.Store(board.storage.directory(p.env))
    with store.read() as connection:
        original = board.names.for_parent(connection, "o/r", 1)
    registry_path = (
        Path(p.env["XDG_STATE_HOME"])
        / "tennis-lab-agents/parent-sessions/registry.json"
    )

    def fetch(_github: object, number: int) -> dict[str, object]:
        assert number == 1
        state = json.loads(registry_path.read_text())
        if change == "removed":
            del state["parents"]["o/r#1"]
        else:
            state["parents"]["o/r#1"][change] = {
                "session": "claude:private-replacement-owner",
                "status": "ended",
                "generation": 2,
            }[change]
        registry_path.write_text(json.dumps(state))
        return {"title": "[同期後] 親", "labels": [{"name": "issue:parent"}]}

    monkeypatch.setattr(board.service.GitHub, "issue", fetch)
    with pytest.raises(board.errors.BoardError, match="assignment changed"):
        board.service.sync_parent(p.env, "o/r", 1)
    with store.read() as connection:
        assert board.names.for_parent(connection, "o/r", 1) == original


def test_name_failure_does_not_undo_parent_or_skip_display(
    parent_env: ParentEnv, board: SimpleNamespace
) -> None:
    p = parent_env
    store = board.storage.Store(board.storage.directory(p.env))
    with store.transaction() as connection:
        connection.execute("PRAGMA user_version=999")
    response = p.cli("register", ok=None)
    data = json.loads(response.stdout)
    assert response.returncode == 1
    assert data["parent_action_applied"] is True
    assert data["display_sync"]["status"] == "not_configured"
    assert data["board_sync"]["status"] == "error"
    assert json.loads(p.cli("status").stdout)["assignment"] == "active"
    with sqlite3.connect(store.path) as connection:
        connection.execute(f"PRAGMA user_version={board.storage.SCHEMA_VERSION}")
    repaired = p.cli("register")
    assert json.loads(repaired.stdout)["board_sync"]["status"] == "assigned"
    assert len(p.state()["comments"]["1"]) == 1


@pytest.mark.parametrize(
    "extra",
    [
        {"CLAUDE_CODE_SESSION_ID": ""},
        {"CODEX_SESSION_ID": "ambiguous-private"},
        {
            "CLAUDE_CODE_SESSION_ID": "",
            "CODEX_SESSION_ID": "root-private",
            "CODEX_THREAD_ID": "child-private",
        },
        {"CLAUDE_CODE_PARENT_SESSION_ID": "nested-private"},
        {"CLAUDE_CODE_AGENT_ID": "subagent-private"},
    ],
)
def test_shared_identity_rejection_precedes_state_or_network(
    parent_env: ParentEnv, extra: dict[str, str]
) -> None:
    p = parent_env
    for args in (("whoami",),):
        result = run_board({**p.env, **extra}, *args)
        assert result.returncode == 2
        private_output_absent(result.stdout + result.stderr, {**p.env, **extra})
    assert p.cli("register", ok=False, extra=extra).returncode == 2
    assert p.calls() == []
    assert not (Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/board").exists()


def test_codex_root_identity_and_unassigned_human_reading(
    parent_env: ParentEnv,
) -> None:
    env = {
        **parent_env.env,
        "CLAUDE_CODE_SESSION_ID": "",
        "CODEX_SESSION_ID": "private-codex-root",
        "CODEX_THREAD_ID": "private-codex-root",
    }
    parent_env.cli("register", extra=env)
    result = run_board(env, "whoami")
    assert result.returncode == 0
    assert json.loads(result.stdout)["parent"] == 1
    private_output_absent(result.stdout + result.stderr, env)
    env["CODEX_SESSION_ID"] = ""
    env["CODEX_THREAD_ID"] = ""
    assert run_board(env, "whoami").returncode == 2
    roster = run_board(env, "agents")
    assert roster.returncode == 0
    assert json.loads(roster.stdout)["agents"][0]["engine"] == "codex"


def prepare_roster(board: SimpleNamespace, labels: Labels) -> tuple[str, str]:
    identities = ("claude:test-owner", "codex:private-second-owner")
    labels.set_registry(
        {
            "o/r#42": labels.owner(identity=identities[0]),
            "o/r#43": labels.owner(identity=identities[1]),
            "o/r#44": labels.owner(status="released", identity="claude:released-owner"),
            "o/r#45": labels.owner(
                status="handoff_pending", identity="codex:handoff-owner"
            ),
            "o/r#46": labels.owner(status="ended", identity="claude:ended-owner"),
        }
    )
    with board.storage.Store(
        board.storage.directory(labels.env)
    ).transaction() as connection:
        for number in (42, 43):
            board.names.ensure_name(
                connection, "o/r", number, "開発" + str(number), now=NOW
            )
    owners = labels.registry.parent.parent / "herdr-labels/owners"
    for identity, terminal in zip(identities, ("term-a", "term-b"), strict=True):
        (owners / (board.service.owner_token(identity) + ".json")).write_text(
            json.dumps(
                {
                    "version": 1,
                    "scope": board.service.owner_token("/fake/herdr.sock"),
                    "terminal_id": terminal,
                }
            )
        )
    state = labels.read()
    for pane, identity, status in zip(
        ("w1:p1", "w1:p2"), identities, ("working", "idle"), strict=True
    ):
        state["panes"][pane].update(
            state=status, tokens={"tl_owner": board.service.owner_token(identity)}
        )
    state["panes"]["w1:child"] = {
        "terminal_id": "term-child",
        "agent": "codex",
        "state": "working",
    }
    labels.write(state)
    return identities


def test_roster_joins_only_active_parents_and_never_mutates_registry(
    board: SimpleNamespace, labels: Labels
) -> None:
    prepare_roster(board, labels)
    before = labels.registry.read_bytes()
    result = run_board(labels.env, "agents")
    assert result.returncode == 0, "roster command failed"
    entries = json.loads(result.stdout)["agents"]
    assert [(row["parent"], row["engine"], row["state"]) for row in entries] == [
        (42, "claude", "working"),
        (43, "codex", "idle"),
    ]
    assert all(
        row["unread_count"] == 0 and row["last_activity"] is None for row in entries
    )
    assert labels.registry.read_bytes() == before
    private_output_absent(result.stdout + result.stderr, labels.env)
    assert "term-a" not in result.stdout and "term-b" not in result.stdout
    assert all("report-metadata" not in call for call in labels.read()["calls"])


@pytest.mark.parametrize("database", ["missing", "empty", "mixed", "retired"])
def test_roster_includes_unassigned_active_parents(
    board: SimpleNamespace, labels: Labels, database: str
) -> None:
    labels.set_registry(
        {
            "o/r#42": labels.owner(),
            "o/r#43": labels.owner(identity="codex:private-second-owner"),
        }
    )
    store = board.storage.Store(board.storage.directory(labels.env))
    if database != "missing":
        with store.transaction() as connection:
            if database in {"mixed", "retired"}:
                board.names.ensure_name(connection, "o/r", 42, "開発42", now=NOW)
            if database == "retired":
                board.names.ensure_name(connection, "o/r", 43, "開発43", now=NOW)
                board.names.retire_name(connection, "o/r", 43, now=NOW)
    before = labels.registry.read_bytes()
    result = run_board(labels.env, "agents")
    assert result.returncode == 0
    entries = json.loads(result.stdout)["agents"]
    assert [row["parent"] for row in entries] == [42, 43]
    if database in {"mixed", "retired"}:
        assert entries[0]["name"] == board.names.load_names()[0]
        assert entries[0]["name_status"] == "assigned"
        assert entries[0]["identifier"] == "開発42"
        entries = entries[1:]
    for entry in entries:
        assert entry["name"] is None
        assert entry["name_status"] == "unassigned"
        assert entry["identifier"] is None
        assert entry["unread_count"] == 0
        assert entry["last_activity"] is None
    response = run_board(
        {
            **labels.env,
            "CLAUDE_CODE_SESSION_ID": "",
            "CODEX_SESSION_ID": "private-second-owner",
            "CODEX_THREAD_ID": "private-second-owner",
        },
        "whoami",
    )
    assert response.returncode == 2
    assert response.stdout == ""
    assert labels.registry.read_bytes() == before
    assert store.path.exists() == (database != "missing")
    if database != "missing":
        with store.read() as connection:
            assignment = board.names.for_parent(connection, "o/r", 43)
            if database == "retired":
                assert assignment.retired_at is not None
            else:
                assert assignment is None
    private_output_absent(result.stdout + result.stderr + response.stderr, labels.env)
    assert "private-second-owner" not in result.stdout + response.stderr


@pytest.mark.parametrize("status", ["idle", "working", "blocked", "done"])
def test_runtime_states_and_post_activity_boundary(
    board: SimpleNamespace, labels: Labels, status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_roster(board, labels)
    state = labels.read()
    state["panes"]["w1:p1"]["state"] = status
    labels.write(state)
    monkeypatch.setattr(
        board.activity,
        "summary",
        lambda connection, assignment_id: board.activity.Activity(
            3, True, NOW.isoformat()
        ),
    )
    entries = board.service.agents(labels.env)["agents"]
    assert entries[0]["state"] == status
    assert entries[0]["unread_count"] == 3
    assert entries[0]["last_activity"] == NOW.isoformat()


@pytest.mark.parametrize(
    "cause",
    [
        "missing_owner",
        "bad_owner",
        "other_scope",
        "replaced_terminal",
        "foreign_owner",
        "unavailable",
        "bad_runtime",
        "native_mismatch",
        "ended_runtime",
    ],
)
def test_runtime_unknown_when_it_cannot_be_verified(
    board: SimpleNamespace, labels: Labels, cause: str
) -> None:
    identities = prepare_roster(board, labels)
    owner_path = (
        labels.registry.parent.parent
        / "herdr-labels/owners"
        / (board.service.owner_token(identities[0]) + ".json")
    )
    state = labels.read()
    if cause == "missing_owner":
        owner_path.unlink()
    elif cause == "bad_owner":
        owner_path.write_text("not json")
    elif cause == "other_scope":
        owner = json.loads(owner_path.read_text())
        owner["scope"] = "other-server"
        owner_path.write_text(json.dumps(owner))
    elif cause == "replaced_terminal":
        state["panes"]["w1:p1"]["terminal_id"] = "new-terminal"
    elif cause == "foreign_owner":
        state["panes"]["w1:p1"]["tokens"]["tl_owner"] = "someone-else"
    elif cause == "bad_runtime":
        state["panes"]["w1:p1"]["state"] = "future-state"
    elif cause == "native_mismatch":
        state["panes"]["w1:p1"]["agent_session"] = {
            "agent": "claude",
            "kind": "id",
            "value": "private-replacement-session",
        }
    elif cause == "ended_runtime":
        root = labels.registry.parent.parent / "herdr-labels/panes"
        root.mkdir()
        key = board.service.owner_token(
            board.service.owner_token("/fake/herdr.sock") + ":term-a"
        )
        (root / (key + ".json")).write_text(
            json.dumps(
                {
                    "version": 1,
                    "owner": board.service.owner_token(identities[0]),
                    "state": "ended",
                }
            )
        )
    else:
        executable = Path(labels.env["PATH"].split(os.pathsep)[0]) / "herdr"
        executable.write_text("#!/bin/sh\necho 'private runtime details' >&2\nexit 3\n")
        executable.chmod(0o755)
    labels.write(state)
    result = run_board(labels.env, "agents")
    assert result.returncode == 0
    assert json.loads(result.stdout)["agents"][0]["state"] == "unknown"
    assert "private runtime details" not in result.stderr
    if cause != "unavailable":
        assert json.loads(result.stdout)["agents"][1]["state"] == "idle"


@pytest.mark.parametrize(
    "argument",
    ["/private/invalid-command", "--sender=private-session", "--state=/private/state"],
)
def test_cli_argument_errors_do_not_echo_private_inputs(
    parent_env: ParentEnv, argument: str
) -> None:
    result = run_board(parent_env.env, "whoami", argument)
    assert result.returncode == 2
    assert argument not in result.stderr
    assert result.stdout == ""
    private_output_absent(result.stderr, parent_env.env)


def test_public_identifier_rejects_private_id_or_path(parent_env: ParentEnv) -> None:
    p = parent_env
    for identifier in (
        p.env["CLAUDE_CODE_SESSION_ID"],
        "/private/identifier",
        "調査 `/private/identifier`",
    ):
        state = p.state()
        state["issues"]["1"]["title"] = f"[{identifier}] 名前"
        p.save(state)
        result = p.cli("register", ok=None)
        assert result.returncode == 1
        assert json.loads(result.stdout)["board_sync"]["status"] == "error"
        assert identifier not in result.stdout + result.stderr


def test_exhausted_catalog_is_a_partial_parent_failure(
    board: SimpleNamespace, parent_env: ParentEnv
) -> None:
    store = board.storage.Store(board.storage.directory(parent_env.env))
    with store.transaction() as connection:
        connection.executemany(
            "INSERT INTO name_assignments(name,repo,parent,identifier,assigned_at) VALUES(?,?,?,?,?)",
            [
                (name, "other/repo", index, "occupied", board.names.timestamp(NOW))
                for index, name in enumerate(board.names.load_names(), 1)
            ],
        )
    result = parent_env.cli("register", ok=None)
    assert result.returncode == 1
    data = json.loads(result.stdout)
    assert data["parent_action_applied"] is True
    assert data["assignment"] == "active"
    assert "exhausted" in data["board_sync"]["error"]
    assert run_board(parent_env.env, "whoami").returncode == 2


def test_corrupt_database_is_rejected_even_with_empty_roster(
    board: SimpleNamespace, parent_env: ParentEnv
) -> None:
    store = board.storage.Store(board.storage.directory(parent_env.env))
    store.directory.mkdir(parents=True)
    store.path.write_bytes(b"corrupt database with no registered parents")
    result = run_board(parent_env.env, "agents")
    assert result.returncode == 2
    assert result.stdout == ""
    private_output_absent(result.stderr, parent_env.env)


def test_timezone_is_required_and_ended_callback_cannot_resurrect_name(
    board: SimpleNamespace, parent_env: ParentEnv
) -> None:
    with pytest.raises(board.errors.BoardError, match="timezone"):
        board.names.timestamp(datetime(2026, 10, 22))
    parent_env.cli("register")
    parent_env.cli("release", "--ended", "--note-file", parent_env.note())
    result = board.service.sync_parent(parent_env.env, "o/r", 1)
    assert result["status"] == "retired"
    with board.storage.Store(
        board.storage.directory(parent_env.env)
    ).read() as connection:
        assert board.names.for_parent(connection, "o/r", 1).retired_at is not None


def test_cleared_display_uses_current_pane_record(
    board: SimpleNamespace, labels: Labels
) -> None:
    identities = prepare_roster(board, labels)
    state = labels.read()
    state["panes"]["w1:p1"]["tokens"] = {}
    state["panes"]["w1:p1"]["agent_session"] = {
        "agent": "claude",
        "kind": "id",
        "value": "test-owner",
    }
    labels.write(state)
    root = labels.registry.parent.parent / "herdr-labels/panes"
    root.mkdir()
    key = board.service.owner_token(
        board.service.owner_token("/fake/herdr.sock") + ":term-a"
    )
    (root / (key + ".json")).write_text(
        json.dumps(
            {
                "version": 1,
                "owner": board.service.owner_token(identities[0]),
                "state": "active",
            }
        )
    )
    result = run_board(labels.env, "agents")
    assert result.returncode == 0
    assert json.loads(result.stdout)["agents"][0]["state"] == "working"
