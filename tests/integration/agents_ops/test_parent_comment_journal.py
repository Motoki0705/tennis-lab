"""Opaque transport receipts cover every family; domain schemas have separate tests."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.e2e.agents_ops.test_dispatcher_judgement import FakeGitHub, issue


def setup(tmp_path: Path):
    state = importlib.import_module("parent.state")
    writer = importlib.import_module("parent.record_writer")
    journal = importlib.import_module("parent.comment_journal")
    gh: Any = FakeGitHub()
    gh.evidence_directory = tmp_path / "evidence-issues"
    session = SimpleNamespace(
        gh=gh,
        parent=1,
        login="operator",
        store=state.Store(tmp_path),
        public=lambda body: None,
    )
    evidence = gh.evidence_issues().ensure(1, "operator")
    return session, evidence.number, writer, journal


def verify(session, evidence: int, journal) -> None:
    journal.verify(
        session.gh,
        session.store.directory,
        1,
        "operator",
        {1: session.gh.comments(1), evidence: session.gh.comments(evidence)},
        evidence,
    )


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    "kind",
    [
        "agreed",
        "execution_basis",
        "basis_revoked",
        "child_stopped",
        "review",
        "parent-review/result",
        "parent-review/resolution",
        "parent-review/evidence",
        "notification_record",
        "notification",
        "completion",
        "parent-improvement",
    ],
)
def test_every_record_family_is_attested_without_a_type_allowlist(
    tmp_path: Path, version: int, kind: str
) -> None:
    session, evidence, writer, journal = setup(tmp_path)
    body = f"Opaque transport bytes for {kind}; semantic validation is independent."
    destination = 1 if kind == "notification" else evidence
    writer.post_once(session, destination, body, kind)
    if version == 1:
        [path] = list((tmp_path / "comment-outbox").glob("*.json"))
        row = journal.load(path)
        del row["parent"], row["source"]
        row["version"] = 1
        importlib.import_module("worker_logs.journal").atomic(path, row)
    verify(session, evidence, journal)
    original = copy.deepcopy(session.gh.comment_rows[destination])
    for fault in ("deleted", "body", "publisher"):
        session.gh.comment_rows[destination] = copy.deepcopy(original)
        if fault == "deleted":
            session.gh.comment_rows[destination].clear()
        elif fault == "body":
            session.gh.comment_rows[destination][0]["body"] += " remotely edited"
        else:
            session.gh.comment_rows[destination][0]["user"]["login"] = "foreign"
        with pytest.raises(ValueError, match="comment journal"):
            verify(session, evidence, journal)
    session.gh.comment_rows[destination] = original
    verify(session, evidence, journal)


def test_pr_projection_and_other_parent_receipts_keep_their_scope(
    tmp_path: Path,
) -> None:
    session, evidence, writer, journal = setup(tmp_path)
    writer.post_once(session, 100, "PR projection", "projection", projection=True)
    session.gh.issues[2] = issue(2)
    other = SimpleNamespace(**{**vars(session), "parent": 2})
    writer.post_once(other, 2, "Other parent", "other")
    session.gh.comment_rows[2].clear()
    verify(session, evidence, journal)
    session.gh.comment_rows[100].clear()
    with pytest.raises(ValueError, match="comment journal"):
        verify(session, evidence, journal)


def test_unreadable_journal_cannot_be_treated_as_an_empty_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, evidence, writer, journal = setup(tmp_path)
    writer.post_once(session, evidence, "revocation bytes", "revocation")
    directory = tmp_path / "comment-outbox"
    original = Path.iterdir

    def unavailable(path):
        if path == directory:
            raise PermissionError("journal access lost")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", unavailable)
    with pytest.raises(ValueError, match="comment journal directory unavailable"):
        verify(session, evidence, journal)


def test_pending_update_intent_does_not_hide_loss_or_post_commit_edits(
    tmp_path: Path,
) -> None:
    session, evidence, writer, journal = setup(tmp_path)
    digest = importlib.import_module("parent.policy").digest
    atomic = importlib.import_module("worker_logs.journal").atomic
    c = writer.post_once(session, evidence, "pending bytes", "pending")
    path = tmp_path / "comment-updates" / f"{digest(['o/r', evidence, c['id']])}.json"
    row = {
        "version": 1,
        "repo": "o/r",
        "parent": 1,
        "issue": evidence,
        "comment_id": c["id"],
        "publisher": "operator",
        "state": "sent",
        "from_sha256": digest("pending bytes"),
        "to_sha256": digest("final bytes"),
    }
    atomic(path, row)
    verify(session, evidence, journal)
    session.gh.comment_rows[evidence][0]["body"] = "final bytes"
    verify(session, evidence, journal)  # A lost PATCH response is still verifiable.
    writer.post_once(session, evidence, "final bytes", "final")
    session.gh.comment_rows[evidence][0]["body"] = "pending bytes"
    with pytest.raises(ValueError, match="comment journal"):
        verify(session, evidence, journal)
    session.gh.comment_rows[evidence][0]["body"] = "final bytes"
    atomic(path, {**row, "state": "bound"})
    verify(session, evidence, journal)
    session.gh.comment_rows[evidence][0]["body"] = "pending bytes"
    with pytest.raises(ValueError, match="comment journal"):
        verify(session, evidence, journal)
