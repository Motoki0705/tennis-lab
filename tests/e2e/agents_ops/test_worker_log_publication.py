"""Real Git and public entry contracts with an offline GitHub transport."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.support.agents_ops.worker_logs import LogEnv, git, journal, schema, sources


@pytest.fixture
def logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LogEnv:
    return LogEnv(tmp_path, monkeypatch)


class Crash(BaseException):
    """Emulate process loss without allowing the publisher's error handler to run."""


def test_gitignore_tracks_only_public_worker_log_files(tmp_path: Path) -> None:
    git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_bytes(
        (Path(__file__).resolve().parents[3] / ".gitignore").read_bytes()
    )
    directory = ".agents/logs/parents/parent-1/child-10/d-example/0001-initial"
    allowed = {
        f"{directory}/{name}" for name in ("instruction.md", "final.md", "meta.json")
    }
    ignored = {
        ".agents/logs/raw.md",
        ".agents/logs/instruction.md",
        ".agents/logs/final.md",
        ".agents/logs/meta.json",
        ".agents/logs/notes/meta.json",
        ".agents/logs/parents/raw.md",
        ".agents/logs/parents/parent-1/private.txt",
        f"{directory}/raw.md",
        f"{directory}/stderr.log",
        f"{directory}/events.json",
        f"{directory}/payload.txt",
        f"{directory}/code.py",
        f"{directory}/README.md",
    }
    for name in allowed | ignored:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("fixture\n")
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin", "-z"],
        cwd=tmp_path,
        input="\0".join(sorted(allowed | ignored)) + "\0",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert set(result.stdout.rstrip("\0").split("\0")) == ignored
    git(tmp_path, "add", ".")
    assert set(git(tmp_path, "ls-files").splitlines()) == {".gitignore", *allowed}


@pytest.mark.parametrize(
    "point",
    [
        "before_commit",
        "after_commit",
        "after_push",
        "create_response_lost",
        "merge_response_lost",
        "closed_unmerged",
    ],
)
def test_publication_recovers_after_interruption(
    logs: LogEnv, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    path = logs.ready()
    publisher = logs.publisher(path)
    before = logs.main_files()
    if point in {"before_commit", "after_commit", "after_push"}:
        real = publisher.git
        fired = False

        def interrupted(*args: str, **kwargs: Any) -> bytes:
            nonlocal fired
            if not fired and point == "before_commit" and args[0] == "add":
                fired = True
                raise Crash()
            result: bytes = real(*args, **kwargs)
            if not fired and args[0] == {
                "after_commit": "commit",
                "after_push": "push",
            }.get(point):
                fired = True
                raise Crash()
            return result

        monkeypatch.setattr(publisher, "git", interrupted)
        with pytest.raises(Crash):
            publisher.publish()
        monkeypatch.setattr(publisher, "git", real)
        assert logs.main_files() == before
        if point == "before_commit":
            assert publisher.publish()["error"] == "worktree_dirty"
            row = journal.read(path)
            tree = publisher.worktree(row)
            # Explicit repair of this fixture's interrupted worktree only.
            git(tree, "reset", "--hard", row["base"])
            git(tree, "clean", "-fd")
    else:
        logs.github.failure = point
        first = publisher.publish()
        assert first["saved_to_main"] is False
        if point == "merge_response_lost":
            saved = journal.read(path)
            premature = {**saved, "state": "merged"}
            journal.atomic(path, premature)
            corrupted = path.read_bytes()
            with pytest.raises(journal.PublicationError, match="journal_invalid"):
                journal.read(path)
            assert path.read_bytes() == corrupted
            journal.atomic(path, saved)
        if point == "closed_unmerged":
            assert first["error"] == "closed_unmerged"
            assert publisher.publish()["saved_to_main"] is False
            assert logs.main_files() == before and logs.github.calls == ["create"]
            return
    result = publisher.publish(retry=True)
    assert result["state"] == "merged", result
    assert result["saved_to_main"] is True
    assert logs.github.calls.count("create") == logs.github.calls.count("merge") == 1
    assert publisher.publish()["saved_to_main"] is True
    assert logs.github.calls.count("create") == logs.github.calls.count("merge") == 1
    row = journal.read(path)
    assert set(logs.main_files()) - set(before) == {
        row["directory"] + "/" + n for n in schema.FILENAMES
    }
    assert "[skip ci]" not in git(logs.upstream, "log", "-3", "--format=%s")


def test_two_parents_append_and_merge_without_shared_index(logs: LogEnv) -> None:
    paths = [logs.ready(parent=1, child=10), logs.ready(parent=2, child=20)]
    before = set(logs.main_files())
    # Both turns were captured before either publication advanced main.
    for path in paths:
        assert logs.publisher(path).publish()["state"] == "merged"
    added = set(logs.main_files()) - before
    assert len(added) == 6 and all(p.startswith(".agents/logs/parents/") for p in added)
    assert not any(p.endswith("index.md") for p in added)
    assert logs.github.calls == ["create", "merge", "create", "merge"]


def test_concurrent_retry_reuses_same_pr(logs: LogEnv) -> None:
    path = logs.ready()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: logs.publisher(path).publish(), range(2)))
    assert all(r["state"] == "merged" for r in results), results
    assert logs.github.calls == ["create", "merge"]


@pytest.mark.parametrize("change", ["remote_head", "full_diff"])
def test_publication_gate_rechecks_git_and_remote(logs: LogEnv, change: str) -> None:
    path = logs.ready()
    publisher = logs.publisher(path)
    before = logs.main_files()
    if change == "remote_head":
        logs.github.change_head_on_second_get = True
    else:
        logs.github.mergeable = None
        assert publisher.publish()["state"] == "pending"
        row = journal.read(path)
        tree = publisher.worktree(row)
        (tree / "code.py").write_text("unexpected = 2\n")
        git(tree, "add", "code.py")
        git(tree, "commit", "--amend", "--no-edit")
        row["head"] = git(tree, "rev-parse", "HEAD")
        logs.github.rows[row["pr_number"]]["head"]["sha"] = row["head"]
        journal.atomic(path, row)
    result = publisher.publish()
    assert result["error"] == (
        "pr_identity_changed" if change == "remote_head" else "unsafe_diff"
    )
    assert logs.main_files() == before
    assert "merge" not in logs.github.calls


@pytest.mark.parametrize(
    "failure", ["failed", "missing_final", "prompt_changed", "unsafe_input"]
)
def test_runner_failure_is_not_log_success(logs: LogEnv, failure: str) -> None:
    credential = "ghp_" + "x" * 30
    path, proof = logs.turn(
        instruction=credential if failure == "unsafe_input" else "調査指示。"
    )
    row = journal.read(path)
    directory = Path(row["source"]["run_directory"])
    if failure == "failed":
        proof["record"].update(result="failed", turn_end_reason="cli_error")
    elif failure == "missing_final":
        (directory / "final.md").unlink()
    elif failure == "prompt_changed":
        (directory / "prompt.md").write_text("different prompt")
    result = sources.receive(path, proof)
    assert result["saved_to_main"] is False
    assert (
        result["state"]
        == {
            "failed": "worker_failed",
            "missing_final": "missing_final",
            "prompt_changed": "blocked",
            "unsafe_input": "blocked",
        }[failure]
    )
    assert "files" not in journal.read(path)
    assert not logs.github.calls
    assert credential not in json.dumps(result)
    assert not any(p.startswith(".agents/logs/") for p in logs.main_files())


def test_standalone_worker_uses_same_log_entry(
    logs: LogEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential = "opaque-fixture-credential"
    monkeypatch.setenv("worker_api_key", credential)
    instruction = (
        "\n実際に送った指示\r\n"
        + str(logs.root / ".claude/worktrees/worker-7-7/code.py")
        + f" private-parent-session {credential}\r\n"
    )
    path = logs.ready(parent=7, child=7, instruction=instruction)
    service: Any = importlib.import_module("worker_logs.service")
    session = logs.sessions[0]
    store = type(
        "Store",
        (),
        {
            "directory": logs.state,
            "read_data": lambda self: {"parents": {"o/r#7": session.row}},
        },
    )()
    github = type("GitHub", (), {"repo": "o/r"})()
    monkeypatch.setattr(
        service, "Publisher", lambda root, target: logs.publisher(target)
    )
    before = path.read_bytes()
    journal.atomic(
        logs.state / "registry.json", {"version": 1, "parents": {"o/r#7": session.row}}
    )
    environment = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
    environment.update(
        CODEX_SESSION_ID="private-parent-session",
        CODEX_THREAD_ID="private-parent-session",
        PYTHONPATH=str(Path(__file__).resolve().parents[3] / ".agents/ops"),
    )
    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "parent",
            "--repo",
            "o/r",
            "--parent",
            "7",
            "--state-dir",
            str(logs.state),
            "publish-worker-log",
            "--child",
            "7",
            "--dry-run",
        ],
        cwd=logs.root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["logs"][0]["dry_run"] is True
    assert path.read_bytes() == before and not logs.github.calls
    preview = service.command(
        github,
        store,
        7,
        7,
        "codex:private-parent-session",
        publish_logs=True,
        dry_run=True,
    )
    assert path.read_bytes() == before and not logs.github.calls
    assert preview["logs"][0]["directory"].startswith(
        ".agents/logs/parents/parent-7/child-7/"
    )
    result = service.command(
        github, store, 7, 7, "codex:private-parent-session", publish_logs=True
    )
    assert result["logs"][0]["saved_to_main"] is True
    row = journal.read(path)
    meta = json.loads(row["files"]["meta.json"])
    assert meta["parent"] == meta["issue"] == 7
    assert (
        row["files"]["instruction.md"]
        == "\n実際に送った指示\r\n./code.py [private identifier] [private identifier]\r\n"
    )
    assert "親の計画判断" not in row["files"]["instruction.md"]
    public = json.dumps(logs.github.rows, ensure_ascii=False) + json.dumps(result)
    for name in logs.main_files():
        public += git(logs.upstream, "show", "main:" + name)
    assert str(logs.root) not in public and "private-parent-session" not in public
    assert credential not in public
