"""Public publication entry with real Git, fake GitHub, and no AI execution."""

from __future__ import annotations

import copy
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.agents_ops.decision_logs import (
    PublicationError,
    decision,
    history,
    result,
    schema,
)
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


def recovery_owner(
    status: str = "active", *, session: str = "codex:recovery"
) -> dict[str, Any]:
    return {
        "session": session,
        "status": status,
        "generation": 3,
        "tasks": {},
        "reviews": [],
    }


def private_files(directory: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(directory)): p.read_bytes()
        for p in directory.rglob("*")
        if p.is_file()
    }


def unassigned_cli(
    env: DecisionEnv,
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_status: str | None = None,
    issue_state: str = "open",
) -> list[str]:
    cli = importlib.import_module("parent.cli")
    parents = {"o/r#1900": recovery_owner()}
    if target_status:
        parents["o/r#2"] = recovery_owner(target_status, session="codex:previous")
    worker_journal.atomic(
        env.state / "registry.json", {"version": 1, "parents": parents}
    )
    if issue_state == "closed":
        # Cover the split registry used by active parents, as well as legacy v1.
        store = cli.Store(env.state)
        with store.locked():
            store.migrate()
    # The issue exists with this state, but registry-only admission must not
    # query it or try to register/reopen the target before publishing its log.
    issue = {"number": 2, "state": issue_state}

    def forbidden_issue(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"Publication consulted issue lifecycle: {issue}")

    monkeypatch.setattr(
        cli,
        "GitHub",
        lambda *a, **kw: SimpleNamespace(repo="o/r", issue=forbidden_issue),
    )
    monkeypatch.setattr(cli, "session_identity", lambda _: "codex:recovery")
    monkeypatch.setattr(publication, "GitHub", lambda _: env.github)
    monkeypatch.chdir(env.root)
    return [
        "--repo",
        "o/r",
        "--parent",
        "1900",
        "--state-dir",
        str(env.state),
        "publish-decision-log",
        "--unassigned-parent",
        "2",
    ]


@pytest.mark.parametrize(
    "issue_state,target_status",
    [("open", None), ("closed", "ended"), ("open", "released")],
)
def test_unassigned_parent_cli_publishes_open_and_closed_without_assignment(
    env: DecisionEnv,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    issue_state: str,
    target_status: str | None,
) -> None:
    cli = importlib.import_module("parent.cli")
    reconciliation = importlib.import_module("worker_logs.reconciliation")
    log = env.capture(parent=2)
    args = unassigned_cli(
        env, monkeypatch, target_status=target_status, issue_state=issue_state
    )
    before, refs = private_files(env.state), git(env.root, "show-ref")
    assert cli.main(args + ["--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["preview_records"] == 6 and preview["prepared_records"] == 0
    assert ".agents/logs/parents/parent-2/README.md" in preview["preview_readmes"]
    assert private_files(env.state) == before and git(env.root, "show-ref") == refs
    assert env.github.calls == []
    inventory = reconciliation.remaining_publications("O/R", 2, state_dir=env.state)
    assert len(inventory) == 1 and inventory[0]["unpublished_count"] == 6
    status_args = args[: args.index("publish-decision-log")] + ["decision-log-status"]
    status_args[status_args.index("--parent") + 1] = "2"
    assert cli.main(status_args) == 0
    status = json.loads(capsys.readouterr().out)
    assert (
        "--parent RECOVERY_PARENT publish-decision-log --unassigned-parent 2"
        in status["decision_resume_command"]
    )
    assert len(status["records"]) == 6 and private_files(env.state) == before

    assert cli.main(args) == 0
    saved = json.loads(capsys.readouterr().out)
    assert (
        saved["saved_to_main"]
        and saved["parent"] == 2
        and saved["recovery_parent"] == 1900
    )
    actor = saved["publication_actor"]
    assert actor["parent"] == 1900
    files = env.upstream_files()
    records = navigation.records(files)
    assert len(records) == 7 and {r["scope"]["parent"] for r in records} == {2}
    request = next(r for r in records if r["id"] == actor["request_id"])
    assert request["kind"] == "action" and request["data"]["operation"] == "publish"
    assert (
        request["data"]["state"] == "requested"
    )  # No invented acceptance/merge result.
    assert "公開の担当: 親 #1900" in unescape(
        files[navigation.event_path(request).rsplit("/", 1)[0] + "/README.md"].decode()
    )
    assert all(b"codex:recovery" not in raw for raw in files.values())
    assert all(s["publication_actor"] == actor for s in log.status())
    assert next(
        e
        for e in log.read()["entries"]
        if e["event"] == "capture" and e["record"]["id"] == actor["request_id"]
    )["private"]["sessions"] == ["codex:recovery"]
    registry_paths = {
        n for n in before if n == "registry.json" or n.startswith("parents/")
    }
    assert all((env.state / n).read_bytes() == before[n] for n in registry_paths)
    assert reconciliation.remaining_publications("O/R", 2, state_dir=env.state) == []
    complete = private_files(env.state)
    assert cli.main(args + ["--retry"]) == 0
    assert json.loads(capsys.readouterr().out)["saved_to_main"]
    assert private_files(env.state) == complete and env.github.calls == [
        "create",
        "merge",
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "missing_actor",
        "other_actor",
        "ended_actor",
        "released_actor",
        "handoff_actor",
        "active_target",
        "handoff_target",
        "malformed_target",
        "same_parent",
        "negative_target",
        "missing_identity",
    ],
)
def test_unassigned_parent_cli_refuses_ownership_before_any_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fault: str,
) -> None:
    cli = importlib.import_module("parent.cli")
    from tests.support.agents_ops.decision_logs import journal

    log = journal.Journal(tmp_path, "o/r", 2)
    item = result()
    item["scope"]["parent"] = 2
    log.capture("original", item)
    parents = {"o/r#1900": recovery_owner()}
    if fault == "missing_actor":
        parents.clear()
    elif fault == "other_actor":
        parents["o/r#1900"]["session"] = "codex:other"
    elif fault.endswith("_actor"):
        parents["o/r#1900"]["status"] = (
            "handoff_pending"
            if fault == "handoff_actor"
            else fault.removesuffix("_actor")
        )
    elif fault in {"active_target", "handoff_target"}:
        parents["o/r#2"] = recovery_owner(
            "active" if fault == "active_target" else "handoff_pending",
            session="codex:previous",
        )
    elif fault == "malformed_target":
        parents["o/r#2"] = {"status": "unknown"}
    worker_journal.atomic(
        tmp_path / "registry.json", {"version": 1, "parents": parents}
    )
    monkeypatch.setattr(cli, "GitHub", lambda *a, **kw: SimpleNamespace(repo="o/r"))

    def identity(_: Any) -> str:
        if fault == "missing_identity":
            raise ValueError("session identity required")
        return "codex:recovery"

    monkeypatch.setattr(cli, "session_identity", identity)

    def forbidden(*a: Any, **kw: Any) -> None:
        raise AssertionError("Refused request entered Git publication")

    monkeypatch.setattr(publication, "Publisher", forbidden)
    target = (
        "1900"
        if fault == "same_parent"
        else "-2"
        if fault == "negative_target"
        else "2"
    )
    args = [
        "--repo",
        "o/r",
        "--parent",
        "1900",
        "--state-dir",
        str(tmp_path),
        "publish-decision-log",
        "--unassigned-parent",
        target,
    ]
    before = private_files(tmp_path)
    for flags in ([], ["--dry-run"]):
        assert cli.main(args + flags) == 2
        assert capsys.readouterr().err
        assert private_files(tmp_path) == before


@pytest.mark.parametrize(
    "boundary,change",
    [
        ("repository", "target"),
        ("push", "target"),
        ("create", "target"),
        ("merge", "target"),
        ("merge", "actor"),
        ("merge", "generation"),
        ("merge", "target_generation"),
        ("merge", "corrupt"),
    ],
)
def test_unassigned_publication_rechecks_before_external_writes(
    env: DecisionEnv,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    boundary: str,
    change: str,
) -> None:
    cli = importlib.import_module("parent.cli")
    log = env.capture(parent=2, prepare=True)
    args = unassigned_cli(
        env,
        monkeypatch,
        target_status="released" if change == "target_generation" else None,
    )
    registry = env.state / "registry.json"
    original_registry = registry.read_bytes()
    boundary_files: dict[str, bytes] = {}

    def reassign() -> None:
        nonlocal boundary_files
        row = json.loads(registry.read_bytes())
        if change == "corrupt":
            registry.write_text("invalid JSON")
        else:
            if change == "target":
                row["parents"]["o/r#2"] = recovery_owner(session="codex:successor")
            elif change == "target_generation":
                row["parents"]["o/r#2"]["generation"] += 1
            elif change == "actor":
                row["parents"]["o/r#1900"]["status"] = "ended"
            else:
                row["parents"]["o/r#1900"]["generation"] += 1
            worker_journal.atomic(registry, row)
        boundary_files = private_files(env.state)

    target, name = (
        (env.github, "repository")
        if boundary == "repository"
        else (publication.Publisher, "remote")
        if boundary == "push"
        else (env.github, "find")
        if boundary == "create"
        else (publication.policy, "pr_changes")
    )
    original = getattr(target, name)
    fired = False

    def racing(*a: Any, **kw: Any) -> Any:
        nonlocal fired
        value = original(*a, **kw)
        if not fired:
            fired = True
            reassign()
        return value

    monkeypatch.setattr(target, name, racing)
    assert cli.main(args) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["error"] == "log_owner_mismatch" and not refused["saved_to_main"]
    assert fired and private_files(env.state) == boundary_files
    assert "merge" not in env.github.calls
    assert all(s["publication_status"] != "blocked" for s in log.status())
    assert len(navigation.records(env.upstream_files())) == 0
    # Return only this isolated fixture's assignment to retry the saved batch.
    registry.write_bytes(original_registry)
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["saved_to_main"]
    assert env.github.calls.count("create") == env.github.calls.count("merge") == 1


@pytest.mark.parametrize("admission_race", [False, True])
def test_unassigned_publication_reuses_existing_batch_and_records_recovery_actor(
    env: DecisionEnv,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    admission_race: bool,
) -> None:
    cli = importlib.import_module("parent.cli")
    continuation = importlib.import_module("parent.log_publication")
    log = env.capture(parent=2)
    worker = env.decision_publisher(log)
    first: dict[str, Any] = {}
    old_batch: dict[str, Any] = {}

    def prepare_old_batch() -> None:
        nonlocal first, old_batch
        env.github.pause_reads = True
        first = worker.publish()
        old_batch = json.loads(worker.path.read_bytes())["batches"][0]
        env.github.pause_reads = False

    if admission_race:
        check = continuation.require_batch_captures

        def late_batch(*a: Any) -> None:
            check(*a)
            prepare_old_batch()

        monkeypatch.setattr(continuation, "require_batch_captures", late_batch)
    else:
        prepare_old_batch()
    args = unassigned_cli(env, monkeypatch)
    assert cli.main(args + ["--retry"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["saved_to_main"]
    batches = json.loads(worker.path.read_bytes())["batches"]
    assert len(batches) == 2 and all(b["state"] == "merged" for b in batches)
    assert batches[0]["key"] == old_batch["key"]
    assert batches[0]["payloads"] == old_batch["payloads"]
    assert batches[0]["pr_number"] == old_batch["pr_number"]
    assert batches[0]["revisions"] == old_batch["revisions"]
    assert env.github.calls == ["create", "merge", "create", "merge"]
    assert first["pr"] != value["pr"]  # A separate bounded attribution batch.
    assert len(navigation.records(env.upstream_files())) == 7


class Crash(BaseException):
    """Process death skips the publisher's ordinary error handler."""


@pytest.mark.parametrize(
    "damage", ["missing", "empty", "unrelated", "unprepared", "different_envelope"]
)
def test_pending_batch_requires_its_captured_prepared_payloads(
    env: DecisionEnv, damage: str
) -> None:
    reconciliation = importlib.import_module("worker_logs.reconciliation")
    lifecycle = importlib.import_module("worker_logs.lifecycle")
    log = env.capture()
    env.github.pause_reads = True
    assert env.decision_publisher(log).publish()["state"] == "pending"
    env.github.pause_reads = False
    document = log.read()
    if damage == "missing":
        log.path.unlink()
    elif damage == "different_envelope":
        prepared = next(e for e in document["entries"] if e["event"] == "prepare")
        prepared["payload"]["record"]["data"]["reason"] += " Different projection."
        prepared["snapshot_sha256"] = importlib.import_module(
            "decision_logs.journal"
        ).preparation_sha(prepared)
        worker_journal.atomic(log.path, document)
        log.read()  # A valid journal with the same identity and different content.
    else:
        document["entries"] = (
            [e for e in document["entries"] if e["event"] == "capture"]
            if damage == "unprepared"
            else []
        )
        worker_journal.atomic(log.path, document)
        if damage == "unrelated":
            log.capture("unrelated", result())
    before = {str(p): p.read_bytes() for p in env.state.rglob("*") if p.is_file()}
    with pytest.raises(PublicationError, match="input_capture_failed"):
        reconciliation.remaining_publications("o/r", 1, state_dir=env.state)
    assert before == {
        str(p): p.read_bytes() for p in env.state.rglob("*") if p.is_file()
    }
    checkpoint = lifecycle.record_exit(
        "o/r", 1, 1, "ended", state_dir=env.state, root=env.root
    )
    assert checkpoint["remaining_complete"] is False
    assert checkpoint["error"] and checkpoint["resume_command"]
    with pytest.raises(PublicationError, match="input_capture_failed"):
        lifecycle.publish_decisions("o/r", 1, state_dir=env.state, root=env.root)
    assert env.github.calls == ["create"]


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


@pytest.mark.parametrize("status", [405, 409, 403, 404, 422])
def test_http_merge_rejection_is_durable_and_405_rechecks_before_retry(
    env: DecisionEnv, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    merge_rejected = importlib.import_module("worker_logs.publication").MergeRejected
    log = env.capture()
    worker = env.decision_publisher(log)
    merge = env.github.merge

    def refuse(number: int, head: str) -> None:
        raise merge_rejected(
            status, "merge_conflict" if status in {405, 409} else "merge_rejected"
        )

    monkeypatch.setattr(env.github, "merge", refuse)
    result_value = worker.publish()
    batch = json.loads(worker.path.read_text())["batches"][0]
    assert result_value["state"] == "blocked" and not result_value["saved_to_main"]
    assert batch["merge_http_status"] == status and batch["merge_attempted"] is False
    assert batch["stage"] == "merge_rejected"
    monkeypatch.setattr(env.github, "merge", merge)
    retried = worker.publish()
    assert retried["saved_to_main"] is (status == 405)
    assert env.github.calls.count("create") == 1


def test_null_mergeability_proceeds_but_explicit_conflict_stays_blocked(
    env: DecisionEnv,
) -> None:
    first = env.capture(parent=1)
    env.github.mergeable = None
    assert env.decision_publisher(first).publish()["saved_to_main"]
    second = env.capture(parent=2)
    env.github.mergeable = False
    refused = env.decision_publisher(second).publish()
    assert refused["state"] == "blocked" and refused["error"] == "merge_conflict"
    assert env.github.calls.count("merge") == 1


@pytest.mark.parametrize("handoff_after_first_batch", [False, True])
def test_exit_decision_publication_finishes_old_batch_and_fixed_capture_boundary(
    env: DecisionEnv,
    monkeypatch: pytest.MonkeyPatch,
    handoff_after_first_batch: bool,
) -> None:
    lifecycle = importlib.import_module("worker_logs.lifecycle")
    reconciliation = importlib.import_module("worker_logs.reconciliation")
    first, second, concurrent = result(), result(), result()
    log = env.capture([first])
    env.github.pause_reads = True
    assert env.decision_publisher(log).publish()["state"] == "pending"
    env.github.pause_reads = False
    env.capture([second])
    merge = env.github.merge
    current_owner = True

    def arriving_record(number: int, head: str) -> None:
        nonlocal current_owner
        merge(number, head)
        if len([c for c in env.github.calls if c == "merge"]) == 1:
            env.capture([concurrent])
            current_owner = not handoff_after_first_batch

    monkeypatch.setattr(env.github, "merge", arriving_record)
    publisher = publication.Publisher
    monkeypatch.setattr(
        publication,
        "Publisher",
        lambda root, journal, record_ids: publisher(
            root, journal, env.github, record_ids=record_ids
        ),
    )
    saved = lifecycle.publish_decisions(
        "o/r", 1, state_dir=env.state, root=env.root, owner_check=lambda: current_owner
    )
    assert (
        saved["state"] == "merged"
        and not saved["saved_to_main"]
        and env.github.calls
        == ["create", "merge"] * (1 if handoff_after_first_batch else 2)
    ), saved
    assert saved["error"] == (
        "log_owner_mismatch" if handoff_after_first_batch else None
    )
    states = {s["id"]: s["publication_status"] for s in log.status()}
    assert states[first["id"]] == "published"
    assert (states[second["id"]] == "published") is (not handoff_after_first_batch)
    assert states[concurrent["id"]] != "published"
    remaining = reconciliation.remaining_publications("o/r", 1, state_dir=env.state)
    assert len(remaining) == 1 and remaining[0]["kind"] == "decision_log"
    assert remaining[0]["unpublished_count"] == (2 if handoff_after_first_batch else 1)
    assert remaining[0]["batch_key"] is None


@pytest.mark.parametrize("recoverable", [True, False])
def test_exit_replays_frozen_capture_delivery_and_reports_unrecoverable_capture(
    env: DecisionEnv, monkeypatch: pytest.MonkeyPatch, recoverable: bool
) -> None:
    capture = importlib.import_module("decision_logs.capture")
    lifecycle = importlib.import_module("worker_logs.lifecycle")
    reconciliation = importlib.import_module("worker_logs.reconciliation")
    log = reconciliation.decision_journal("o/r", 1, state_dir=env.state)
    op = capture.Operation.begin(env.state, "o/r", 1, "parent.note", "record")
    if recoverable:
        # Drop only the destination journal. The original outbox still freezes
        # IDs and bytes; replay must deliver those records without a new action.
        assert log.path.is_relative_to(env.state) and env.state.is_relative_to(
            env.root.parent
        )
        log.path.unlink()
    else:
        op.mark_failed(lost_source="parent.note.response")
    before = {e["record"]["id"] for e in op.row["entries"]}
    inventory = reconciliation.remaining_publications("o/r", 1, state_dir=env.state)
    assert inventory[0]["unpublished_count"] is None
    publisher = publication.Publisher
    monkeypatch.setattr(
        publication,
        "Publisher",
        lambda root, journal, record_ids: publisher(
            root, journal, env.github, record_ids=record_ids
        ),
    )
    value = lifecycle.publish_decisions("o/r", 1, state_dir=env.state, root=env.root)
    assert value["saved_to_main"] is recoverable
    assert value["error"] == (None if recoverable else "input_capture_failed")
    assert {r["id"] for r in log.status()} == before
    assert env.github.calls == ["create", "merge"]
    remaining = reconciliation.remaining_publications("o/r", 1, state_dir=env.state)
    assert bool(remaining) is (not recoverable)
    if remaining:
        assert (
            remaining[0]["stage"] == "capture_incomplete"
            and remaining[0]["unpublished_count"] is None
        )


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
    env.github.pause_reads = True
    a, b = env.decision_publisher(left), env.decision_publisher(right)
    assert a.publish()["stage"] == b.publish()["stage"] == "pr_create"
    env.github.pause_reads = False
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


@pytest.mark.parametrize(
    "point",
    [
        "before_commit",
        "after_commit",
        "before_push",
        "after_push",
        "after_commit_main_advanced_again",
    ],
)
def test_open_pr_recovers_interrupted_regeneration_without_losing_revisions(
    env: DecisionEnv, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    log = env.capture()
    publisher = env.decision_publisher(log)
    env.github.pause_reads = True
    first = publisher.publish()
    assert first["stage"] == "pr_create", first
    env.github.pause_reads = False
    original = json.loads(publisher.path.read_text())["batches"][0]
    records = copy.deepcopy(log.public_payloads())
    base = env.advance({"upstream-note.txt": b"Independent main advance\n"})
    real_git = publisher.git
    fired = False

    def interrupted(*args: str, **kwargs: Any) -> bytes:
        nonlocal fired
        if not fired and (
            point == "before_commit"
            and args[:2] == ("worktree", "add")
            or point == "before_push"
            and args[0] == "push"
        ):
            fired = True
            raise Crash()
        raw: bytes = real_git(*args, **kwargs)
        if not fired and (
            point.startswith("after_commit")
            and args[0] == "commit"
            or point == "after_push"
            and args[0] == "push"
        ):
            fired = True
            raise Crash()
        return raw

    monkeypatch.setattr(publisher, "git", interrupted)
    with pytest.raises(Crash):
        publisher.publish()
    monkeypatch.setattr(publisher, "git", real_git)
    pending = json.loads(publisher.path.read_text())["batches"][0]
    assert len(pending["revisions"]) == 2
    assert pending["revisions"][0] == original["revisions"][0]
    assert pending["revisions"][1]["base"] == base
    committed = (
        None
        if point == "before_commit"
        else git(publisher.worktree(pending), "rev-parse", "HEAD")
    )
    assert pending["revisions"][1]["head"] == (
        committed if point in {"before_push", "after_push"} else None
    )
    assert env.github.get(original["pr_number"])["head"]["sha"] == (
        committed if point == "after_push" else first["head"]
    )
    if point == "after_commit_main_advanced_again":
        env.advance({"upstream-note.txt": b"Another independent main advance\n"})
    env.github.rows[original["pr_number"]]["mergeable"] = True
    recovered = env.decision_publisher(log).publish(retry=True)
    assert recovered["saved_to_main"] and recovered["state"] == "merged", json.dumps(
        recovered
    )
    assert recovered["pr"] == first["pr"]
    assert env.github.calls == ["create", "merge"]
    final = json.loads(publisher.path.read_text())["batches"][0]
    assert final["revisions"][0] == original["revisions"][0]
    assert len(final["revisions"]) == (
        3 if point == "after_commit_main_advanced_again" else 2
    )
    assert final["revisions"][1]["base"] == base
    if committed is not None:
        assert final["revisions"][1]["head"] == committed
    assert log.public_payloads() == records
    saved = env.upstream_files()
    assert len(navigation.records(saved)) == 6
    assert all(saved[n] == raw for n, raw in navigation.generate(saved).items())
    assert git(env.upstream, "show", "main:upstream-note.txt") == (
        "Another independent main advance"
        if point == "after_commit_main_advanced_again"
        else "Independent main advance"
    )
    assert env.decision_publisher(log).publish()["saved_to_main"]
    assert env.github.calls == ["create", "merge"]


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
    env.github.pause_reads = True
    first = env.decision_publisher(log).publish()
    env.github.pause_reads = False
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
        and "対応イベントなし" in raw.decode()
    ]
    assert legacy and all("instruction.md" in r and "final.md" in r for r in legacy)
    navigation.validate_links(saved, navigation.generate(saved))


def test_typed_termination_after_prior_publication_recovers_without_rewriting_history(
    env: DecisionEnv,
) -> None:
    from tests.support.agents_ops.traceability import (
        observation_record,
        termination_observation,
        worker_observation,
    )

    first = env.capture([observation_record(worker_observation("ended", "absent"))])
    assert env.decision_publisher(first).publish()["saved_to_main"]
    original = env.upstream_files()
    closing = observation_record(termination_observation(), issue=None, delegation=None)
    log = env.capture([closing])
    env.github.failure = "merge_response_lost"
    result = env.decision_publisher(log).publish()
    assert not result["saved_to_main"]
    recovered = env.decision_publisher(log).publish(retry=True)
    assert recovered["saved_to_main"], recovered
    saved = env.upstream_files()
    assert all(
        saved[name] == value for name, value in original.items() if "/event-" in name
    )
    parent = saved[".agents/logs/parents/parent-1/README.md"].decode()
    assert "担当終了" in parent and "最終報告なし" in parent
    assert env.decision_publisher(log).publish()["saved_to_main"]
    assert env.github.calls == ["create", "merge", "create", "merge"]


@pytest.mark.parametrize("prepared", [False, True])
def test_dry_run_is_read_only_and_does_not_prepare_or_publish(
    env: DecisionEnv, prepared: bool
) -> None:
    log = env.capture(prepare=prepared)
    publisher = env.decision_publisher(log)
    original = log.path.read_bytes()
    refs = git(env.root, "show-ref")
    preview = publisher.publish(dry_run=True)
    assert preview["dry_run"] and preview["prepared_records"] == (6 if prepared else 0)
    assert preview["preview_records"] == 6 and preview["preview_blocked"] == []
    assert preview["generated_readmes"] == 6
    assert (
        "Reuse the code check"
        in preview["preview_readmes"][".agents/logs/parents/parent-1/README.md"]
    )
    assert all(
        len(raw.encode()) < navigation.PAGE_BYTES
        for raw in preview["preview_readmes"].values()
    )
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
    env.github.pause_reads = True
    publisher = env.decision_publisher(log)
    pending = publisher.publish()
    env.github.pause_reads = False
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
    dependent = decision([bad])
    log = env.capture([good, bad, dependent])
    original = log.path.read_bytes()
    preview = env.decision_publisher(log).publish(dry_run=True)
    assert preview["preview_records"] == 1 and preview["blocked"] == 2
    assert {r["id"] for r in preview["preview_blocked"]} == {bad["id"], dependent["id"]}
    assert "ghp_" not in json.dumps(preview)
    assert log.path.read_bytes() == original and env.github.calls == []
    value = env.decision_publisher(log).publish()
    assert (
        value["state"] == "merged"
        and value["blocked"] == 2
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
    env.github.pause_reads = True
    publisher = env.decision_publisher(log)
    assert publisher.publish()["stage"] == "pr_create"
    env.github.pause_reads = False
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
    before_repair = env.upstream_files()
    recovered = env.decision_publisher(log).publish(retry=True)
    assert recovered["saved_to_main"] and len(env.github.rows) == 2, recovered
    saved = env.upstream_files()
    delegation = directory.rsplit("/", 1)[0]
    expected_readmes = {
        ".agents/logs/parents/README.md",
        ".agents/logs/parents/parent-2/README.md",
        ".agents/logs/parents/parent-2/child-20/README.md",
        delegation + "/README.md",
        delegation + "/turn-0001/README.md",
        directory + "/README.md",
    }
    assert env.github.rows[101]["changed_files"] == len(expected_readmes)
    assert {
        name
        for name in before_repair.keys() | saved.keys()
        if before_repair.get(name) != saved.get(name)
    } == expected_readmes
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


@pytest.mark.parametrize("point", ["read", "lock", "persist"])
def test_storage_failure_reports_only_a_fixed_error_and_preserves_capture(
    env: DecisionEnv, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    log = env.capture()
    original = [e for e in log.read()["entries"] if e["event"] == "capture"]
    publisher = env.decision_publisher(log)
    if point == "read":
        real_read = Path.read_bytes

        def unreadable(path: Path) -> bytes:
            if path == publisher.path:
                raise PermissionError("private publication storage detail")
            return real_read(path)

        monkeypatch.setattr(Path, "read_bytes", unreadable)
    elif point == "lock":
        real_lock = publication.storage.locked

        def failed_lock(path: Path) -> Any:
            if path == publisher.path:
                raise PermissionError("private publication lock detail")
            return real_lock(path)

        monkeypatch.setattr(publication.storage, "locked", failed_lock)
    else:

        def failed_persistence(*args: Any, **kwargs: Any) -> None:
            raise PermissionError("private publication persistence detail")

        monkeypatch.setattr(publication, "atomic", failed_persistence)
    with pytest.raises(PublicationError, match="^journal_unavailable$"):
        publisher.publish()
    assert [e for e in log.read()["entries"] if e["event"] == "capture"] == original
    assert not publisher.path.exists() and env.github.calls == []


def test_reused_publisher_failure_cannot_overwrite_another_publishers_history(
    env: DecisionEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = env.capture()
    stale = env.decision_publisher(log)
    env.github.pause_reads = True
    assert stale.publish()["stage"] == "pr_create"
    env.github.pause_reads = False
    for pr in env.github.rows.values():
        pr["mergeable"] = True
    assert env.decision_publisher(log).publish()["saved_to_main"]
    completed = stale.path.read_bytes()

    def unavailable_payloads() -> Any:
        raise PublicationError("credential")

    monkeypatch.setattr(log, "public_payloads", unavailable_payloads)
    failed = stale.publish(retry=True)
    assert failed["error"] == "credential" and not failed["saved_to_main"]
    assert stale.path.read_bytes() == completed
    assert all(r["publication_status"] == "published" for r in log.status())
    assert env.github.calls == ["create", "merge"]


def test_late_worker_files_refresh_navigation_without_new_capture_records(
    env: DecisionEnv,
) -> None:
    worker = worker_journal.read(env.ready())
    raw = {n: value.encode() for n, value in worker["files"].items()}
    directory = worker_schema.path(worker_schema.validate(raw))
    log = env.capture()
    assert env.decision_publisher(log).publish()["saved_to_main"]
    journal_before = log.path.read_bytes()
    env.advance({directory + "/" + n: value for n, value in raw.items()})
    stale = env.decision_publisher(log).publish()
    assert stale["stage"] == "readme_refresh_required" and not stale["saved_to_main"]
    refreshed = env.decision_publisher(log).publish(retry=True)
    assert refreshed["saved_to_main"], refreshed
    saved = env.upstream_files()
    assert directory + "/README.md" in saved
    assert all(saved[n] == value for n, value in navigation.generate(saved).items())
    assert (
        len(navigation.records(saved)) == 6 and log.path.read_bytes() == journal_before
    )
    assert env.decision_publisher(log).publish()["saved_to_main"]
    assert env.github.calls == ["create", "merge", "create", "merge"]


def test_renderer_upgrade_uses_the_same_readme_only_publication_exception(
    env: DecisionEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = env.capture()

    def previous_renderer(files: dict[str, bytes]) -> dict[str, bytes]:
        generated: dict[str, bytes] = navigation.generate(files)
        generated[".agents/logs/parents/README.md"] += "\n旧版の派生表示。\n".encode()
        return generated

    with monkeypatch.context() as patch:
        patch.setattr(publication, "generate", previous_renderer)
        patch.setattr(publication.policy, "generate", previous_renderer)
        assert env.decision_publisher(log).publish()["saved_to_main"]
    frozen = log.path.read_bytes()
    old_events = {
        name: raw for name, raw in env.upstream_files().items() if "/event-" in name
    }
    stale = env.decision_publisher(log).publish()
    assert stale["stage"] == "readme_refresh_required"
    repaired = env.decision_publisher(log).publish(retry=True)
    assert repaired["saved_to_main"], repaired
    saved = env.upstream_files()
    assert all(saved[name] == raw for name, raw in old_events.items())
    assert all(saved[name] == raw for name, raw in navigation.generate(saved).items())
    assert log.path.read_bytes() == frozen
    repair = env.github.rows[max(env.github.rows)]
    assert all(
        name.endswith("/README.md")
        for name in (r["filename"] for r in env.github.files(repair["number"]))
    )


def test_renderer_upgrade_does_not_excuse_an_unconfirmed_publication(
    env: DecisionEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = env.capture()
    publisher = env.decision_publisher(log)
    original_note = publisher.note

    def previous_renderer(files: dict[str, bytes]) -> dict[str, bytes]:
        generated: dict[str, bytes] = navigation.generate(files)
        generated[".agents/logs/parents/README.md"] += b"\nEarlier derived layout.\n"
        return generated

    def interrupted_note(batch: Any, status: str, **kwargs: Any) -> None:
        if status == "published":
            raise Crash()
        original_note(batch, status, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(publication, "generate", previous_renderer)
        patch.setattr(publication.policy, "generate", previous_renderer)
        patch.setattr(publisher, "note", interrupted_note)
        with pytest.raises(Crash):
            publisher.publish()
    assert not any(r["publication_status"] == "published" for r in log.status())
    refused = env.decision_publisher(log).publish(retry=True)
    assert (
        not refused["saved_to_main"] and refused["error"] == "prepared_content_changed"
    )
    assert env.github.calls == ["create", "merge"]
