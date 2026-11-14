"""Existing journals and lost-journal PRs through real Git and a GET-only preview."""

from __future__ import annotations

import base64
import copy
import fcntl
import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_session import ParentEnv
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.support.agents_ops.worker_logs import FakeGitHub, LogEnv, git, journal

recovery: Any = importlib.import_module("worker_logs.reconciliation")
orphans: Any = importlib.import_module("worker_logs.orphans")
publication: Any = importlib.import_module("worker_logs.publication")
service: Any = importlib.import_module("worker_logs.service")


class PublicGitHub(FakeGitHub):
    """Expose actual Git objects over the public endpoints used by orphan recovery."""

    def __init__(self, logs: LogEnv) -> None:
        super().__init__(logs.root, logs.upstream)
        self.rows = copy.deepcopy(logs.github.rows)
        self.comments: dict[int, list[dict[str, Any]]] = {}
        self.requests: list[tuple[str, str]] = []

    def get(self, number: int) -> dict[str, Any]:
        value = dict(super().get(number))
        value["commits"] = len(
            git(self.upstream, "rev-list", f"main..{value['head']['sha']}").splitlines()
        )
        if value["merged"]:
            value["commits"] = 1
        return value

    def files(self, number: int) -> list[dict[str, Any]]:
        # GitHub retains a merged PR's diff. A comparison against today's main
        # would incorrectly turn it into an empty list after the merge.
        row = self.rows[number]
        if not row["merged"]:
            return list(super().files(number))
        head = row["head"]["sha"]
        fields = git(
            self.upstream,
            "diff",
            "--name-status",
            "--no-renames",
            "-z",
            head + "^",
            head,
        ).split("\0")
        return [
            {
                "filename": fields[i + 1],
                "status": {"A": "added", "M": "modified", "D": "removed"}[fields[i]],
            }
            for i in range(0, len(fields) - 1, 2)
        ]

    def request(
        self, endpoint: str, method: str = "GET", payload: dict | None = None
    ) -> Any:
        self.requests.append((method, endpoint))
        suffix = endpoint.removeprefix("repos/o/r/").split("?")[0]
        if suffix == "git/ref/heads/main":
            return {"object": {"sha": self.main()}}
        if suffix.startswith("git/commits/"):
            sha = suffix.removeprefix("git/commits/")
            return {
                "sha": sha,
                "tree": {"sha": git(self.upstream, "rev-parse", sha + "^{tree}")},
                "parents": [
                    {"sha": p}
                    for p in git(
                        self.upstream, "rev-list", "--parents", "-n", "1", sha
                    ).split()[1:]
                ],
            }
        if suffix.startswith("git/trees/"):
            rows = []
            for line in git(
                self.upstream, "ls-tree", suffix.removeprefix("git/trees/")
            ).splitlines():
                metadata, name = line.split("\t")
                mode, kind, sha = metadata.split()
                rows.append({"path": name, "type": kind, "mode": mode, "sha": sha})
            return {"truncated": False, "tree": rows}
        if suffix.startswith("git/blobs/"):
            sha = suffix.removeprefix("git/blobs/")
            raw = subprocess.check_output(
                ["git", "cat-file", "blob", sha], cwd=self.upstream
            )
            return {
                "sha": sha,
                "encoding": "base64",
                "content": base64.b64encode(raw).decode(),
                "size": len(raw),
            }
        if suffix.startswith("compare/"):
            base, head = suffix.removeprefix("compare/").split("...")
            files = []
            fields = git(
                self.upstream, "diff", "--no-renames", "--name-status", "-z", base, head
            ).split("\0")
            for i in range(0, len(fields) - 1, 2):
                files.append(
                    {
                        "filename": fields[i + 1],
                        "status": {
                            "A": "added",
                            "M": "modified",
                            "D": "removed",
                            "T": "modified",
                        }[fields[i]],
                    }
                )
            count = int(git(self.upstream, "rev-list", "--count", f"{base}..{head}"))
            return {
                "ahead_by": count,
                "total_commits": count,
                "files": files,
                "merge_base_commit": {
                    "sha": git(self.upstream, "merge-base", base, head)
                },
            }
        if suffix.startswith("pulls/") and suffix.endswith("/commits"):
            row = self.rows[int(suffix.split("/")[1])]
            return [{"sha": row["head"]["sha"]}]
        if suffix.startswith("issues/") and suffix.endswith("/comments"):
            number = int(suffix.split("/")[1])
            if method == "POST":
                self.calls.append("comment")
                self.comments.setdefault(number, []).append(
                    {"id": 1, **(payload or {})}
                )
                return self.comments[number][-1]
            return copy.deepcopy(self.comments.get(number, []))
        if suffix.startswith("pulls/") and method == "PATCH":
            self.calls.append("close")
            self.rows[int(suffix.split("/")[1])]["state"] = "closed"
            return {}
        raise AssertionError((method, endpoint))


@pytest.fixture
def logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LogEnv:
    return LogEnv(tmp_path, monkeypatch)


def public_pr(logs: LogEnv, *, erase: bool = True) -> tuple[Path, PublicGitHub]:
    path = logs.ready()
    logs.github.pause_reads = True
    assert logs.publisher(path).publish()["state"] == "pending"
    logs.github.pause_reads = False
    github = PublicGitHub(logs)
    if erase:
        path.unlink()
    return path, github


def snapshot(path: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }


def test_orphan_preview_then_merge_preserves_public_bytes_and_reuses_pr(
    logs: LogEnv,
) -> None:
    _, github = public_pr(logs)
    worker = orphans.OrphanRecovery(
        "o/r", state_dir=logs.state, root=logs.root, github=github
    )
    before = snapshot(logs.state)
    head = github.rows[100]["head"]["sha"]
    preview = worker.recover(100, dry_run=True)
    assert (
        preview["state"] == "eligible"
        and preview["original_journal_available"] is False
    )
    assert snapshot(logs.state) == before and github.calls == []
    assert all(method == "GET" for method, _ in github.requests)
    assert worker.recover(100, dry_run=False)["saved_to_main"] is True
    assert github.calls == ["merge"]
    assert git(logs.upstream, "merge-base", "--is-ancestor", head, "main") == ""
    assert worker.recover(100, dry_run=False)["saved_to_main"] is True
    assert github.calls == ["merge"]


def test_orphan_lost_merge_response_is_reconciled_without_a_second_merge(
    logs: LogEnv,
) -> None:
    _, github = public_pr(logs)
    github.failure = "merge_response_lost"
    worker = orphans.OrphanRecovery(
        "o/r", state_dir=logs.state, root=logs.root, github=github
    )
    first = worker.recover(100, dry_run=False)
    assert first["state"] == "pending" and first["error"] == "merge_response_uncertain"
    assert not first["saved_to_main"] and github.calls == ["merge"]
    recovered = worker.recover(100, dry_run=False)
    assert recovered["saved_to_main"] and recovered["error"] is None
    assert github.calls == ["merge"]


def test_orphan_recovery_does_not_close_an_unrelated_pr(logs: LogEnv) -> None:
    _, github = public_pr(logs)
    worker = orphans.OrphanRecovery(
        "o/r", state_dir=logs.state, root=logs.root, github=github
    )
    assert worker.recover(100, dry_run=True)["state"] == "eligible"
    github.rows[100]["head"]["ref"] = "feature/other-work"
    assert worker.recover(100, dry_run=True)["action"] == "out_of_scope"
    assert worker.recover(100, dry_run=False)["state"] == "blocked"
    assert github.rows[100]["state"] == "open" and github.calls == []


def test_private_journal_symlink_does_not_close_a_valid_orphan_or_write_outside_state(
    logs: LogEnv,
) -> None:
    path, github = public_pr(logs)
    outside = logs.root.parent / "outside-turn"
    path.parent.rename(outside)
    path.parent.symlink_to(outside, target_is_directory=True)
    worker = orphans.OrphanRecovery(
        "o/r", state_dir=logs.state, root=logs.root, github=github
    )
    before = snapshot(outside)
    preview = worker.recover(100, dry_run=True)
    assert preview["state"] == "blocked" and preview["action"] == "retry_inspection"
    result = worker.recover(100, dry_run=False)
    assert result["error"] == "private_storage_symlink" and result["state"] == "blocked"
    assert github.calls == [] and github.rows[100]["state"] == "open"
    assert snapshot(outside) == before


def test_orphan_audit_lock_is_never_created_through_a_symlink(
    logs: LogEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, github = public_pr(logs)
    outside = logs.root.parent / "outside-audit"
    outside.mkdir()
    (logs.state / "orphan-worker-prs").symlink_to(outside, target_is_directory=True)
    worker = orphans.OrphanRecovery(
        "o/r", state_dir=logs.state, root=logs.root, github=github
    )
    monkeypatch.setattr(orphans, "OrphanRecovery", lambda *a, **kw: worker)
    value = orphans.recover_orphans(
        "o/r", state_dir=logs.state, root=logs.root, numbers=[100], dry_run=False
    )
    assert value[0]["error"] == "private_storage_symlink"
    assert not list(outside.iterdir()) and github.calls == []


@pytest.mark.parametrize(
    "fault",
    [
        "marker",
        "extra_file",
        "wrong_digest",
        "credential",
        "path",
        "private_identifier",
        "mode",
        "second_commit",
    ],
)
def test_orphan_reinspection_refuses_unsafe_pr_and_comments_before_close(
    logs: LogEnv,
    fault: str,
) -> None:
    path, github = public_pr(logs)
    row = github.rows[100]
    head = row["head"]["sha"]
    directory = json.loads(
        git(
            logs.upstream,
            "show",
            f"{head}:"
            + next(
                n
                for n in git(
                    logs.upstream, "ls-tree", "-r", "--name-only", head
                ).splitlines()
                if n.endswith("/meta.json")
            ),
        )
    )
    log_dir = importlib.import_module("worker_logs.schema").path(directory)
    tree = (
        logs.root
        / ".claude/worktrees"
        / f"worker-log-{directory['delegation_key'][2:]}-0001"
    )
    if fault == "marker":
        row["body"] = "wrong marker"
    elif fault == "extra_file":
        (tree / "extra.txt").write_text("unexpected")
    elif fault in {"wrong_digest", "credential", "path", "private_identifier"}:
        (tree / log_dir / "final.md").write_text(
            "Changed report"
            if fault == "wrong_digest"
            else "ghp_" + "x" * 30
            if fault == "credential"
            else "/private/user/file"
            if fault == "path"
            else "codex:12345678-1234-1234-1234-123456789abc"
        )
        if fault != "wrong_digest":
            meta = json.loads((tree / log_dir / "meta.json").read_text())
            meta["files"]["final.md"] = importlib.import_module(
                "worker_logs.schema"
            ).sha((tree / log_dir / "final.md").read_bytes())
            (tree / log_dir / "meta.json").write_text(json.dumps(meta))
    elif fault == "mode":
        (tree / log_dir / "final.md").chmod(0o755)
    if fault != "marker":
        git(tree, "add", "--all")
        if fault == "second_commit":
            git(tree, "commit", "--allow-empty", "-m", "extra commit")
        else:
            git(tree, "commit", "--amend", "--no-edit")
        new_head = git(tree, "rev-parse", "HEAD")
        git(tree, "push", "--force", "origin", row["head"]["ref"])
        row["body"] = row["body"].replace(head, new_head)
        row["head"]["sha"] = new_head
    worker = orphans.OrphanRecovery(
        "o/r", state_dir=logs.state, root=logs.root, github=github
    )
    before = snapshot(logs.state)
    preview = worker.recover(100, dry_run=True)
    assert preview["state"] == "rejected" and preview["error"]
    assert github.calls == [] and snapshot(logs.state) == before
    refused = worker.recover(100, dry_run=False)
    assert refused["state"] == "closed" and refused["error"]
    assert github.calls == ["comment", "close"] and "merge" not in github.calls
    assert not path.exists()  # Never reconstruct an original journal.


def test_orphan_existing_journal_routes_to_normal_recovery(
    logs: LogEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, github = public_pr(logs, erase=False)
    monkeypatch.setattr(
        service, "Publisher", lambda root, target: logs.publisher(target)
    )
    worker = orphans.OrphanRecovery(
        "o/r", state_dir=logs.state, root=logs.root, github=github
    )
    value = worker.recover(100, dry_run=False)
    assert (
        value["state"] == "journal_present"
        and value["original_journal_available"] is True
    )
    assert journal.read(path)["state"] == "merged" and github.calls == []


def test_foreign_unassigned_journal_replay_and_inventory_are_read_only_until_execution(
    logs: LogEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = logs.ready(parent=1723, child=1727)
    before = snapshot(logs.state)
    inventory = recovery.remaining_publications("o/r", 1723, state_dir=logs.state)
    assert inventory[0]["kind"] == "worker_log" and inventory[0]["issue"] == 1727
    assert (
        inventory[0]["public_key"] == "d-" + journal.read(path)["source"]["public_key"]
    )
    assert str(logs.state) not in json.dumps(inventory)
    preview = recovery.continue_workers("o/r", None, state_dir=logs.state, dry_run=True)
    assert preview[0]["action"] == "continue_publication"
    assert snapshot(logs.state) == before and logs.github.calls == []
    monkeypatch.setattr(
        service, "Publisher", lambda root, target: logs.publisher(target)
    )
    result = recovery.continue_workers("o/r", None, state_dir=logs.state)
    assert result[0]["saved_to_main"] is True
    assert journal.read(path)["source"]["public_key"] == inventory[0]["public_key"][2:]
    assert logs.github.calls == ["create", "merge"]


def test_bulk_recovery_requires_recovery_owner_and_reuses_foreign_journal(
    logs: LogEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = importlib.import_module("parent.state")
    path = logs.ready(parent=1723, child=1727)
    owner = {
        "session": "codex:recovery",
        "status": "active",
        "generation": 1,
        "tasks": {},
        "reviews": [],
    }
    journal.atomic(
        logs.state / "registry.json", {"version": 1, "parents": {"o/r#1900": owner}}
    )
    store = state.Store(logs.state)
    args = SimpleNamespace(
        source_parent=[1723],
        pr=None,
        adopt_orphan_prs=False,
        turn=None,
        delegation_key=None,
        dry_run=True,
    )
    before = snapshot(logs.state)
    preview = recovery.bulk_command(
        "o/r", 1900, None, store=store, args=args, root=logs.root
    )
    assert preview["logs"][0]["owner_status"] == "unassigned"
    assert snapshot(logs.state) == before
    args.dry_run = False
    with pytest.raises(journal.PublicationError, match="log_owner_mismatch"):
        recovery.bulk_command(
            "o/r", 1900, "codex:other", store=store, args=args, root=logs.root
        )
    monkeypatch.setattr(
        service, "Publisher", lambda root, target: logs.publisher(target)
    )
    first = recovery.bulk_command(
        "o/r", 1900, "codex:recovery", store=store, args=args, root=logs.root
    )
    assert first["logs"][0]["saved_to_main"] and first["generation"] == 1
    assert (
        recovery.bulk_command(
            "o/r", 1900, "codex:recovery", store=store, args=args, root=logs.root
        )["logs"]
        == []
    )
    assert journal.read(path)["state"] == "merged"
    assert logs.github.calls == ["create", "merge"]


def test_bulk_recovery_records_generation_change_without_overwriting_successor(
    logs: LogEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = importlib.import_module("parent.state")
    owner = {
        "session": "codex:recovery",
        "status": "active",
        "generation": 1,
        "tasks": {},
        "reviews": [],
    }
    document = {"version": 1, "parents": {"o/r#1900": owner}}
    journal.atomic(logs.state / "registry.json", document)

    def transfer(*args: Any, **kwargs: Any) -> list:
        owner.update(generation=2, session="codex:successor")
        journal.atomic(logs.state / "registry.json", document)
        return []

    monkeypatch.setattr(recovery, "continue_workers", transfer)
    args = SimpleNamespace(
        source_parent=[1723],
        pr=None,
        adopt_orphan_prs=False,
        turn=None,
        delegation_key=None,
        dry_run=False,
    )
    result = recovery.bulk_command(
        "o/r",
        1900,
        "codex:recovery",
        store=state.Store(logs.state),
        args=args,
        root=logs.root,
    )
    assert result["error"] == "log_owner_mismatch" and result["generation"] == 1
    assert state.Store(logs.state).read_data()["parents"]["o/r#1900"] == owner


def test_exit_checkpoint_marks_corrupt_inventory_as_unknown(logs: LogEnv) -> None:
    path = logs.ready()
    path.write_text("invalid JSON")
    lifecycle = importlib.import_module("worker_logs.lifecycle")
    with pytest.raises(journal.PublicationError):
        recovery.remaining_publications("o/r", 1, state_dir=logs.state)
    value = lifecycle.record_exit(
        "o/r", 1, 1, "ended", state_dir=logs.state, root=logs.root
    )
    assert (
        value["remaining_complete"] is False and value["error"] == "journal_unavailable"
    )
    assert value["resume_command"] and value["publication_actor"] == "unassigned"


@pytest.mark.parametrize(
    "status,error",
    [(405, "merge_conflict"), (409, "pr_identity_changed"), (403, "merge_rejected")],
)
def test_definite_merge_rejection_stays_failed_and_only_405_retries(
    logs: LogEnv,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    error: str,
) -> None:
    path = logs.ready()
    merge = logs.github.merge

    def refuse(number: int, head: str) -> None:
        raise publication.MergeRejected(status, error, "private HTTP diagnostic")

    monkeypatch.setattr(logs.github, "merge", refuse)
    first = logs.publisher(path).publish()
    row = journal.read(path)
    assert (
        first["state"] == "blocked"
        and first["error"] == error
        and not first["saved_to_main"]
    )
    assert row["merge_attempted"] is False and row["merge_http_status"] == status
    monkeypatch.setattr(logs.github, "merge", merge)
    monkeypatch.setattr(
        service, "Publisher", lambda root, target: logs.publisher(target)
    )
    resumed = recovery.continue_workers("o/r", 1, state_dir=logs.state)
    assert resumed[0]["saved_to_main"] is (status == 405)
    assert logs.github.calls.count("create") == 1


@pytest.mark.parametrize("status", [200, 405, 409, 403, 404, 422])
def test_http_merge_status_and_sha_are_preserved(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    head = "a" * 40

    def run(command: list[str], **kwargs: Any) -> Any:
        if command[command.index("--method") + 1] == "GET":
            return SimpleNamespace(
                returncode=0, stdout=json.dumps({"head": {"sha": "b" * 40}}).encode()
            )
        assert "--include" in command and json.loads(kwargs["input"])["sha"] == head
        raw = json.dumps(
            {"merged": True, "sha": "c" * 40}
            if status == 200
            else {"message": "rejected"}
        ).encode()
        return SimpleNamespace(
            returncode=0 if status == 200 else 1,
            stdout=f"HTTP/2.0 {status} status\r\nX-Test: yes\r\n\r\n".encode() + raw,
            stderr=b"private error",
        )

    monkeypatch.setattr(publication.subprocess, "run", run)
    if status == 200:
        publication.GitHub("o/r").merge(100, head)
    else:
        with pytest.raises(publication.MergeRejected) as rejected:
            publication.GitHub("o/r").merge(100, head)
        assert rejected.value.status == status
        assert str(rejected.value) == (
            "merge_conflict"
            if status == 405
            else "pr_identity_changed"
            if status == 409
            else "merge_rejected"
        )


def test_409_followup_read_failure_keeps_the_known_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    github = publication.GitHub("o/r")

    def request(endpoint: str, method: str = "GET", payload: Any = None) -> Any:
        if method == "PUT":
            raise publication.GitHubHTTPError(409, "head changed")
        raise journal.PublicationError("github_failed")

    monkeypatch.setattr(github, "request", request)
    with pytest.raises(publication.MergeRejected) as rejected:
        github.merge(100, "a" * 40)
    assert rejected.value.status == 409 and str(rejected.value) == "merge_conflict"


@pytest.mark.parametrize("body", [b'{"merged": false}', b"not JSON"])
def test_http_200_without_merge_confirmation_is_an_uncertain_response(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    monkeypatch.setattr(
        publication.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0, stdout=b"HTTP/2.0 200 OK\r\n\r\n" + body, stderr=b""
        ),
    )
    with pytest.raises(publication.GitHubHTTPError) as uncertain:
        publication.GitHub("o/r").merge(100, "a" * 40)
    assert uncertain.value.status == 200 and str(uncertain.value) == "github_failed"


@pytest.mark.parametrize("failure", ["server", "timeout"])
def test_uncertain_merge_remains_pending_without_blind_retry(
    logs: LogEnv, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = logs.ready()
    calls = []

    def uncertain(number: int, head: str) -> None:
        calls.append(head)
        if failure == "server":
            raise publication.GitHubHTTPError(503, "server response")
        raise subprocess.TimeoutExpired("gh api", 60)

    monkeypatch.setattr(logs.github, "merge", uncertain)
    for _ in range(2):
        value = logs.publisher(path).publish()
        assert (
            value["state"] == "pending" and value["error"] == "merge_response_uncertain"
        )
        assert not value["saved_to_main"]
    assert len(calls) == 1 and journal.read(path)["merge_attempted"] is True
    if failure == "server":
        assert value["merge_http_status"] == 503


@pytest.mark.parametrize(
    "command", ["status", "worker-log-status", "inspect-worker", "publish-worker-log"]
)
def test_read_only_completion_never_creates_or_changes_state(
    logs: LogEnv, command: str
) -> None:
    module = importlib.import_module("parent.log_publication")
    state = importlib.import_module("parent.state")
    logs.ready()
    before = snapshot(logs.state)
    args = SimpleNamespace(command=command, dry_run=command == "publish-worker-log")
    continuation = module.Continuation(args, state.Store(logs.state), "o/r", 1, None)
    assert continuation.complete(args, {"state": "observed"}) == {"state": "observed"}
    assert snapshot(logs.state) == before and logs.github.calls == []


@pytest.mark.parametrize(
    "command",
    [
        "note",
        "release",
        "handoff",
        "run-worker",
        "start-worker",
        "continue-worker",
        "workflow-report",
        "improvement-propose",
        "improvement-extract",
        "improvement-decide",
    ],
)
def test_common_continuation_visits_pending_turn_once_outside_operation_lock(
    logs: LogEnv,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    module = importlib.import_module("parent.log_publication")
    state = importlib.import_module("parent.state")
    path = logs.ready()
    owner = logs.sessions[0].row
    owner.update(
        status="ended"
        if command == "release"
        else "handoff_pending"
        if command == "handoff"
        else "active",
        generation=1,
        reviews=[],
    )
    journal.atomic(
        logs.state / "registry.json", {"version": 1, "parents": {"o/r#1": owner}}
    )
    monkeypatch.setattr(
        service, "Publisher", lambda root, target: logs.publisher(target)
    )
    merge = logs.github.merge

    def unlocked_merge(number: int, head: str) -> None:
        for filename in (
            "registry.lock",
            f"operation-{importlib.import_module('parent.policy').digest('o/r#1')}.lock",
        ):
            with (logs.state / filename).open() as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        merge(number, head)

    monkeypatch.setattr(logs.github, "merge", unlocked_merge)
    monkeypatch.setattr(
        module, "publish_decisions", lambda *a, **kw: {"error": "github_failed"}
    )
    args = SimpleNamespace(command=command, dry_run=False, accept=False)
    continuation = module.Continuation(
        args, state.Store(logs.state), "o/r", 1, owner["session"]
    )
    if command in module.BEFORE_WORKER:
        continuation.sweep()  # Called before the command reserves its next turn.
        assert journal.read(path)["state"] == "merged"
    value = continuation.complete(args, {"parent_action_applied": True})
    assert journal.read(path)["state"] == "merged"
    assert logs.github.calls == ["create", "merge"]
    if command in {"release", "handoff"}:
        checkpoint = value["publication_exit"]
        assert checkpoint["event"] == (
            "ended" if command == "release" else "handoff_pending"
        )
        assert checkpoint["publication_actor"] == "unassigned"
        assert checkpoint["error"] == "github_failed" and checkpoint["resume_command"]
        assert str(logs.state) not in json.dumps(checkpoint)


@pytest.mark.parametrize(
    "command,accept", [("release", False), ("handoff", False), ("handoff", True)]
)
@pytest.mark.parametrize("handoff_after_workers", [False, True])
def test_saved_assignment_exit_is_reconciled_even_when_the_original_command_fails(
    logs: LogEnv,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    accept: bool,
    handoff_after_workers: bool,
) -> None:
    module = importlib.import_module("parent.log_publication")
    state = importlib.import_module("parent.state")
    path = logs.ready()
    owner: dict[str, Any] = {
        "session": "codex:previous",
        "status": "handoff_pending" if accept else "active",
        "generation": 1,
        "tasks": {},
        "reviews": [],
    }
    document = {"version": 1, "parents": {"o/r#1": owner}}
    journal.atomic(logs.state / "registry.json", document)
    identity = "codex:next" if accept else owner["session"]
    args = SimpleNamespace(command=command, accept=accept, ended=True, dry_run=False)
    continuation = module.Continuation(
        args, state.Store(logs.state), "o/r", 1, identity
    )
    owner.update(
        session=identity,
        generation=2 if accept else 1,
        status="active"
        if accept
        else "ended"
        if command == "release"
        else "handoff_pending",
    )
    journal.atomic(logs.state / "registry.json", document)
    applied_status, applied_generation = owner["status"], owner["generation"]
    monkeypatch.setattr(
        service, "Publisher", lambda root, target: logs.publisher(target)
    )
    continue_workers = module.continue_workers

    def workers(*args: Any, **kwargs: Any) -> list:
        value: list = continue_workers(*args, **kwargs)
        if handoff_after_workers:
            owner.update(
                session="codex:successor",
                generation=applied_generation + 1,
                status="active",
            )
            journal.atomic(logs.state / "registry.json", document)
        return value

    def decisions(*args: Any, **kwargs: Any) -> dict:
        assert not handoff_after_workers, "old generation started another publication"
        return {"error": "github_failed"}

    monkeypatch.setattr(module, "continue_workers", workers)
    monkeypatch.setattr(module, "publish_decisions", decisions)
    observed = continuation.failed()
    assert observed["assignment_observed"] == applied_status
    assert "parent_action_applied" not in observed
    assert observed["publication_exit"]["generation"] == applied_generation
    assert observed["publication_exit"]["event"] == (
        "handoff_accepted"
        if accept
        else "ended"
        if command == "release"
        else "handoff_pending"
    )
    assert observed["publication_exit"]["error"] == (
        "log_owner_mismatch" if handoff_after_workers else "github_failed"
    )
    current = state.Store(logs.state).read_data()["parents"]["o/r#1"]
    if handoff_after_workers:
        assert current == owner
    else:
        assert {field: current[field] for field in owner} == owner
        assert current["publication_continuation"]["logs"][0]["saved_to_main"] is True
    assert journal.read(path)["state"] == "merged" and logs.github.calls == [
        "create",
        "merge",
    ]


@pytest.mark.parametrize("handoff_at", ["worker", "decision"])
def test_exit_checks_generation_before_decision_publication_and_after_it_finishes(
    logs: LogEnv, monkeypatch: pytest.MonkeyPatch, handoff_at: str
) -> None:
    module = importlib.import_module("parent.log_publication")
    state = importlib.import_module("parent.state")
    owner: dict[str, Any] = {
        "session": "codex:previous",
        "status": "ended",
        "generation": 1,
        "tasks": {},
        "reviews": [],
    }
    document = {"version": 1, "parents": {"o/r#1": owner}}
    journal.atomic(logs.state / "registry.json", document)
    args = SimpleNamespace(command="release", accept=False, ended=True, dry_run=False)
    continuation = module.Continuation(
        args, state.Store(logs.state), "o/r", 1, owner["session"]
    )
    calls = []

    def handoff() -> None:
        owner.update(session="codex:successor", generation=2, status="active")
        journal.atomic(logs.state / "registry.json", document)

    def workers(*a: Any, **kw: Any) -> list:
        if handoff_at == "worker":
            handoff()
        return []

    def decisions(*a: Any, **kw: Any) -> dict:
        assert handoff_at == "decision", "old generation started another publication"
        calls.append("decision")
        handoff()
        return {"error": None}

    monkeypatch.setattr(module, "continue_workers", workers)
    monkeypatch.setattr(module, "publish_decisions", decisions)
    original = {"assignment": "ended", "parent_action_applied": True}
    value = continuation.complete(args, original)
    assert value["parent_action_applied"] is True and value["assignment"] == "ended"
    checkpoint = value["publication_exit"]
    assert checkpoint["generation"] == 1 and checkpoint["error"] == "log_owner_mismatch"
    assert checkpoint["publication_actor"] == 1 and checkpoint["event"] == "ended"
    assert calls == (["decision"] if handoff_at == "decision" else [])
    assert state.Store(logs.state).read_data()["parents"]["o/r#1"] == owner


@pytest.mark.parametrize(
    "arguments",
    [
        ["--child", "10", "--source-parent", "2"],
        ["--child", "10", "--pr", "100"],
        ["--all-parents", "--pr", "100"],
        ["--all-parents", "--turn", "0"],
        ["--adopt-orphan-prs", "--retry"],
    ],
)
def test_bulk_scope_flags_are_never_silently_ignored(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    cli = importlib.import_module("parent.cli")

    def no_github(*args: Any, **kwargs: Any) -> None:
        pytest.fail("invalid scope reached GitHub/state initialization")

    monkeypatch.setattr(cli, "GitHub", no_github)
    assert (
        cli._main(["--repo", "o/r", "--parent", "1", "publish-worker-log", *arguments])
        == 2
    )


def test_ops_tests_cannot_write_default_state_outside_their_private_root(
    tmp_path: Path,
) -> None:
    state = importlib.import_module("parent.state")
    support = importlib.import_module("tests.support.agents_ops.decision_logs")
    root = state.state_directory()
    assert root.is_relative_to(tmp_path)
    log = support.journal.Journal(root, "o/r", 1)
    record = support.result()
    log.capture(record["id"], record)
    assert log.path.is_relative_to(tmp_path) and log.path.is_file()


@pytest.mark.parametrize(
    "command",
    [
        ["status"],
        ["worker-log-status", "--child", "11"],
        ["decision-log-status"],
        ["publish-decision-log", "--dry-run"],
        ["publish-worker-log", "--child", "11", "--dry-run"],
        ["publish-worker-log", "--all-parents", "--dry-run"],
        ["inspect-worker", "--child", "11"],
        ["improvement-expire", "--dry-run"],
    ],
)
def test_status_and_dry_run_cli_preserve_every_state_file(
    parent_env: ParentEnv,
    command: list[str],
) -> None:
    p = parent_env
    p.agreed()
    directory = Path(p.env["XDG_STATE_HOME"])
    before = snapshot(directory)
    value = p.cli(*command, ok=None)
    expected = (
        2
        if command[0] in {"inspect-worker", "publish-decision-log"}
        or command[:2] == ["publish-worker-log", "--child"]
        else 0
    )
    assert value.returncode == expected, value.stderr
    assert snapshot(directory) == before


@pytest.mark.parametrize(
    "command,event", [("release", "ended"), ("handoff", "handoff_pending")]
)
def test_lifecycle_cli_records_remaining_publication_without_refusing_exit(
    parent_env: ParentEnv,
    command: str,
    event: str,
) -> None:
    p = parent_env
    p.agreed()
    args = [command, "--note-file", p.note("終了・引継ぎと未公開記録の確認。")]
    if command == "release":
        args.append("--ended")
    value = json.loads(p.cli(*args).stdout)
    assert value["assignment"] == event and value["parent_action_applied"] is True
    checkpoint = value["publication_exit"]
    assert checkpoint["event"] == event and checkpoint["generation"] == 1
    assert checkpoint["publication_actor"] == "unassigned"
    assert any(
        r["kind"] == "decision_log" and r["unpublished_count"] > 0
        for r in checkpoint["remaining"]
    )
    assert checkpoint["resume_command"] and checkpoint["observed_at"]
    directory = Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/parent-sessions"
    saved = (
        directory
        / "publication-exits"
        / importlib.import_module("worker_logs.schema").sha(b"o/r")
        / "1"
        / f"1-{event}.json"
    )
    assert json.loads(saved.read_text()) == checkpoint


@pytest.mark.parametrize("case", ["recovered", "unfinished", "fatal", "late"])
def test_worker_receipt_uses_the_same_error_classification(
    logs: LogEnv, case: str
) -> None:
    path, proof = logs.turn()
    source = journal.read(path)["source"]
    directory = Path(source["run_directory"])
    record = proof["record"]
    record.update(
        run_dir=str(directory), exit_code=0, cli_exit_code=0, terminated_pids=[]
    )
    error = {
        "type": "error",
        "message": "authentication failed"
        if case == "fatal"
        else "Reconnecting... 2/5 (workspace routing discovery timed out)",
    }
    record["runtime"]["runtime_errors"] = [error]
    final = (directory / "final.md").read_text()
    start = [
        {"type": "thread.started", "thread_id": record["runtime"]["session_id"]},
        {"type": "turn.started"},
    ]
    end = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": final}},
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    events = start + [error] + end
    if case == "unfinished":
        events = start + [error]
    if case == "late":
        events = start + end + [error]
    (directory / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    result = importlib.import_module("worker_logs.sources").receive(path, proof)
    assert result["state"] == ("ready" if case == "recovered" else "worker_failed")
