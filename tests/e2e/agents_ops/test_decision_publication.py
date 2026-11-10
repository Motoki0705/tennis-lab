"""Public publication entry with real Git, fake GitHub, and no AI execution."""

from __future__ import annotations

import copy
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.support.agents_ops.decision_logs import history, result, schema
from tests.support.agents_ops.decision_publication import (
    DecisionEnv,
    navigation,
    publication,
)
from tests.support.agents_ops.worker_logs import git
from tests.support.agents_ops.worker_logs import journal as worker_journal
from tests.support.agents_ops.worker_logs import schema as worker_schema


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DecisionEnv:
    return DecisionEnv(tmp_path, monkeypatch)


class Crash(BaseException):
    """Process death skips the publisher's ordinary error handler."""


def test_batch_publication_replay_and_no_self_generated_publication_loop(
    env: DecisionEnv,
) -> None:
    rows = history()
    log = env.capture(rows)
    publisher = env.decision_publisher(log)
    first = publisher.publish()
    assert first["state"] == "merged" and first["saved_to_main"], first
    assert env.github.calls == ["create", "merge"]
    assert first["retained_worktrees"] == 0
    assert not list((env.root / ".claude/worktrees").glob("decision-log-*"))
    saved = env.upstream_files()
    assert len(navigation.records(saved)) == 6
    assert all(saved[n] == raw for n, raw in navigation.generate(saved).items())
    assert all(r["publication_status"] == "published" for r in log.status())
    entries = copy.deepcopy(log.read()["entries"])
    env.capture(rows)  # Delivery replay reuses every original record ID.
    again = publisher.publish()
    assert again["saved_to_main"] and again["pr"] == first["pr"], again
    assert log.read()["entries"] == entries
    assert env.github.calls == ["create", "merge"] and env.upstream_files() == saved
    assert "[skip ci]" not in git(env.upstream, "log", "-3", "--format=%s")
    assert publisher.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    "point",
    [
        "before_commit",
        "after_commit",
        "after_push",
        "create_response_lost",
        "merge_response_lost",
        "after_merge_recorded",
    ],
)
def test_publication_reconciles_boundary_response_loss(
    env: DecisionEnv, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    log = env.capture()
    publisher = env.decision_publisher(log)
    if point in {"before_commit", "after_commit", "after_push"}:
        real = publisher.git
        fired = False

        def interrupted(*args: str, **kwargs: Any) -> bytes:
            nonlocal fired
            if not fired and point == "before_commit" and args[0] == "add":
                fired = True
                raise Crash()
            raw: bytes = real(*args, **kwargs)
            if not fired and args[0] == {
                "after_commit": "commit",
                "after_push": "push",
            }.get(point):
                fired = True
                raise Crash()
            return raw

        monkeypatch.setattr(publisher, "git", interrupted)
        with pytest.raises(Crash):
            publisher.publish()
        monkeypatch.setattr(publisher, "git", real)
        if point == "before_commit":
            assert publisher.publish()["error"] == "worktree_dirty"
            batch = json.loads(publisher.path.read_text())["batches"][0]
            tree = publisher.worktree(batch)
            # Only this disposable fixture's interrupted publication is repaired.
            git(tree, "reset", "--hard", batch["revisions"][-1]["base"])
            git(tree, "clean", "-fd")
    elif point == "after_merge_recorded":
        real_note = publisher.note

        def interrupted_note(batch: Any, state: str, **kwargs: Any) -> None:
            if state == "published":
                raise Crash()
            real_note(batch, state, **kwargs)

        monkeypatch.setattr(publisher, "note", interrupted_note)
        with pytest.raises(Crash):
            publisher.publish()
    else:
        env.github.failure = point
        first = publisher.publish()
        assert not first["saved_to_main"]
    recovered = env.decision_publisher(log).publish(retry=True)
    assert recovered["saved_to_main"] and recovered["state"] == "merged", recovered
    assert env.github.calls == ["create", "merge"]
    assert len(navigation.records(env.upstream_files())) == 6


def test_main_advance_regenerates_other_parents_and_reuses_open_pr(
    env: DecisionEnv,
) -> None:
    left, right = env.capture(parent=1), env.capture(parent=2)
    env.github.mergeable = None
    a, b = env.decision_publisher(left), env.decision_publisher(right)
    assert a.publish()["stage"] == b.publish()["stage"] == "mergeability_pending"
    for pr in env.github.rows.values():
        pr["mergeable"] = True
    assert a.publish()["saved_to_main"]
    first = env.upstream_files()
    b_result = b.publish()
    assert b_result["saved_to_main"], b_result
    saved = env.upstream_files()
    assert len(navigation.records(saved)) == 12
    assert all(saved[n] == raw for n, raw in navigation.generate(saved).items())
    assert all(saved[n] == raw for n, raw in first.items() if "/event-" in n)
    index = saved[".agents/logs/parents/README.md"].decode()
    assert "](parent-1/README.md)" in index and "](parent-2/README.md)" in index
    assert env.github.calls.count("create") == 2
    document = json.loads(b.path.read_text())
    assert len(document["batches"][0]["revisions"]) == 2


def test_same_parent_parallel_replay_creates_one_batch(env: DecisionEnv) -> None:
    log = env.capture()
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(lambda _: env.decision_publisher(log).publish(), range(2))
        )
    assert all(r["saved_to_main"] for r in outcomes), outcomes
    assert env.github.calls == ["create", "merge"]


def test_new_captures_wait_for_active_batch_then_publish_together(
    env: DecisionEnv,
) -> None:
    initial = history()
    log = env.capture(initial)
    env.github.mergeable = None
    first = env.decision_publisher(log).publish()
    later = [result(), result()]
    env.capture(later)
    for pr in env.github.rows.values():
        pr["mergeable"] = True
    partial = env.decision_publisher(log).publish()
    assert partial["pr"] == first["pr"] and partial["unpublished"] == 2
    assert not partial["saved_to_main"] and env.github.calls.count("create") == 1
    env.github.mergeable = True
    final = env.decision_publisher(log).publish()
    assert final["saved_to_main"] and env.github.calls.count("create") == 2, final
    assert len(navigation.records(env.upstream_files())) == 8


def test_existing_worker_files_are_immutable_and_legacy_turns_have_navigation(
    env: DecisionEnv,
) -> None:
    worker_path = env.ready()
    assert env.publisher(worker_path).publish()["saved_to_main"]
    original = env.upstream_files()
    log = env.capture()
    value = env.decision_publisher(log).publish()
    assert value["saved_to_main"], value
    saved = env.upstream_files()
    assert all(saved[n] == raw for n, raw in original.items())
    legacy = [
        raw.decode()
        for name, raw in saved.items()
        if name.endswith("/0001-initial/README.md")
        and "導入前・判断未記録" in raw.decode()
    ]
    assert legacy and all("instruction.md" in r and "final.md" in r for r in legacy)
    navigation.validate_links(saved, navigation.generate(saved))


def test_dry_run_is_read_only_and_does_not_prepare_or_publish(env: DecisionEnv) -> None:
    log = env.capture(prepare=True)
    publisher = env.decision_publisher(log)
    original = log.path.read_bytes()
    refs = git(env.root, "show-ref")
    preview = publisher.publish(dry_run=True)
    assert preview["dry_run"] and preview["prepared_records"] == 6
    assert preview["generated_readmes"] == 6
    assert not publisher.path.exists() and log.path.read_bytes() == original
    assert git(env.root, "show-ref") == refs and env.github.calls == []


@pytest.mark.parametrize(
    "fault",
    ["code", "old_event", "readme", "symlink", "remote_head", "pr_files", "marker"],
)
def test_publication_gate_rejects_tampering_before_merge(
    env: DecisionEnv, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    log = env.capture()
    env.github.mergeable = None
    publisher = env.decision_publisher(log)
    pending = publisher.publish()
    assert pending["state"] == "pending", pending
    document = json.loads(publisher.path.read_text())
    batch = document["batches"][0]
    revision = batch["revisions"][-1]
    if fault in {"code", "old_event", "readme", "symlink"}:
        tree = publisher.worktree(batch)
        if fault == "code":
            (tree / "code.py").write_text("injected = True\n")
        elif fault == "old_event":
            target = tree / navigation.event_path(batch["payloads"][0]["record"])
            payload = json.loads(target.read_text())
            payload["record"]["data"]["reason"] = "Changed fixed projection"
            target.write_bytes(schema.json_bytes(payload))
        elif fault == "readme":
            with (tree / ".agents/logs/parents/README.md").open("ab") as stream:
                stream.write(b"arbitrary prose\n")
        else:
            target = tree / ".agents/logs/parents/README.md"
            target.unlink()
            target.symlink_to("code.py")
        git(tree, "add", "--all")
        git(tree, "commit", "--amend", "--no-edit", "-q")
        revision["head"] = git(tree, "rev-parse", "HEAD")
        publisher.document = document
        publisher.save()
        git(tree, "push", "--force", "origin", f"HEAD:refs/heads/{batch['branch']}")
    elif fault == "remote_head":
        git(env.root, "push", "--force", "origin", f"main:refs/heads/{batch['branch']}")
    elif fault == "pr_files":
        real = env.github.files
        monkeypatch.setattr(
            env.github,
            "files",
            lambda n: [*real(n), {"filename": "code.py", "status": "modified"}],
        )
    else:
        env.github.rows[batch["pr_number"]]["body"] = "wrong marker"
    for pr in env.github.rows.values():
        pr["mergeable"] = True
    refused = env.decision_publisher(log).publish(retry=True)
    assert not refused["saved_to_main"] and refused["error"], refused
    assert "merge" not in env.github.calls and not env.upstream_files()


def test_blocked_projection_does_not_hide_safe_records_or_start_an_ai(
    env: DecisionEnv,
) -> None:
    good = result()
    bad = result()
    bad["data"]["summary"] = "ghp_" + "x" * 30
    log = env.capture([good, bad])
    value = env.decision_publisher(log).publish()
    assert (
        value["state"] == "merged"
        and value["blocked"] == 1
        and not value["saved_to_main"]
    ), value
    assert len(navigation.records(env.upstream_files())) == 1
    assert env.github.calls == ["create", "merge"]


def test_existing_published_event_cannot_be_changed_in_a_later_batch(
    env: DecisionEnv,
) -> None:
    old = result()
    log = env.capture([old])
    assert env.decision_publisher(log).publish()["saved_to_main"]
    original = env.upstream_files()[navigation.event_path(old)]
    env.capture([result()])
    env.github.mergeable = None
    publisher = env.decision_publisher(log)
    assert publisher.publish()["stage"] == "mergeability_pending"
    document = json.loads(publisher.path.read_text())
    batch = document["batches"][-1]
    tree = publisher.worktree(batch)
    target = tree / navigation.event_path(old)
    payload = json.loads(target.read_text())
    payload["record"]["data"]["summary"] = "Rewritten earlier observation"
    target.write_bytes(schema.json_bytes(payload))
    # Even an attacker who regenerates the READMEs cannot rewrite old events.
    snapshot = publisher.snapshot(batch["revisions"][-1]["head"])
    snapshot[navigation.event_path(old)] = target.read_bytes()
    for name, raw in navigation.generate(snapshot).items():
        (tree / name).write_bytes(raw)
    git(tree, "add", "--all")
    git(tree, "commit", "--amend", "--no-edit", "-q")
    batch["revisions"][-1]["head"] = git(tree, "rev-parse", "HEAD")
    publisher.document = document
    publisher.save()
    git(tree, "push", "--force", "origin", f"HEAD:refs/heads/{batch['branch']}")
    for pr in env.github.rows.values():
        if pr["state"] == "open":
            pr["mergeable"] = True
    value = env.decision_publisher(log).publish(retry=True)
    assert value["state"] == "blocked" and value["error"] == "record_already_exists", (
        value
    )
    assert env.upstream_files()[navigation.event_path(old)] == original
    assert env.github.calls.count("merge") == 1


@pytest.mark.parametrize("authorized", [True, False])
def test_parent_cli_checks_owner_and_never_captures_its_own_publication(
    env: DecisionEnv,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    authorized: bool,
) -> None:
    cli: Any = importlib.import_module("parent.cli")
    log = env.capture()
    monkeypatch.chdir(env.root)
    monkeypatch.setattr(cli, "session_identity", lambda _: "codex:current-owner")
    monkeypatch.setattr(
        cli.Store,
        "read_data",
        lambda _: {
            "parents": {
                "o/r#1": {
                    "session": "codex:current-owner"
                    if authorized
                    else "codex:another-owner"
                }
            }
        },
    )
    monkeypatch.setattr(publication, "GitHub", lambda _: env.github)

    def forbidden_capture(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Publication entered the capture adapter")

    monkeypatch.setattr(cli.decision_capture, "begin", forbidden_capture)
    args = [
        "--repo",
        "o/r",
        "--parent",
        "1",
        "--state-dir",
        str(env.state),
        "publish-decision-log",
    ]
    code = cli.main(args)
    captured = capsys.readouterr()
    if authorized:
        assert code == 0 and json.loads(captured.out)["saved_to_main"]
        assert cli.main(args) == 0
        assert env.github.calls == ["create", "merge"]
    else:
        assert code == 2 and "assigned parent" in captured.err
        assert env.github.calls == []
    assert len([e for e in log.read()["entries"] if e["event"] == "capture"]) == 6


def test_last_boundary_main_advance_has_explicit_readme_only_recovery(
    env: DecisionEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = worker_journal.read(env.ready(parent=2, child=20))
    worker_files = {n: raw.encode() for n, raw in worker["files"].items()}
    directory = worker_schema.path(worker_schema.validate(worker_files))
    addition = {directory + "/" + n: raw for n, raw in worker_files.items()}
    real_merge = env.github.merge
    fired = False

    def racing(number: int, head: str) -> None:
        nonlocal fired
        if not fired:
            fired = True
            env.advance(addition)
        real_merge(number, head)

    monkeypatch.setattr(env.github, "merge", racing)
    log = env.capture()
    publisher = env.decision_publisher(log)
    first = publisher.publish()
    assert first["stage"] == "readme_refresh_required" and not first["saved_to_main"], (
        first
    )
    assert len(env.github.rows) == 1 and env.github.rows[100]["merged"]
    assert publisher.publish()["stage"] == "readme_refresh_required"
    recovered = env.decision_publisher(log).publish(retry=True)
    assert recovered["saved_to_main"] and len(env.github.rows) == 2, recovered
    assert env.github.rows[101]["changed_files"] == 1  # Exact index regeneration only.
    saved = env.upstream_files()
    assert all(saved[n] == raw for n, raw in addition.items())
    assert all(saved[n] == raw for n, raw in navigation.generate(saved).items())
    assert len(navigation.records(saved)) == 6
    assert env.decision_publisher(log).publish()["saved_to_main"]
    assert env.github.calls.count("create") == 2


@pytest.mark.parametrize(
    "fault", ["invalid_json", "unknown_version", "unknown_field", "symlink"]
)
def test_private_publication_state_is_never_reset_on_corruption(
    env: DecisionEnv, fault: str
) -> None:
    log = env.capture()
    publisher = env.decision_publisher(log)
    original = log.path.read_bytes()
    if fault == "symlink":
        publisher.path.symlink_to(log.path)
    else:
        document = {"version": 1, "repo": "o/r", "parent": 1, "batches": []}
        if fault == "unknown_version":
            document["version"] = 999
        elif fault == "unknown_field":
            document["extra"] = True
        publisher.path.write_text(
            "not JSON" if fault == "invalid_json" else json.dumps(document)
        )
    corrupted = publisher.path.read_bytes()
    with pytest.raises(ValueError):
        publisher.publish(retry=True)
    assert (
        publisher.path.read_bytes() == corrupted and log.path.read_bytes() == original
    )
    assert env.github.calls == [] and not env.upstream_files()
