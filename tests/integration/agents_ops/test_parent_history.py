"""The two public media form one validated, ordered history, never a fallback."""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops"))

history = importlib.import_module("parent.history")
policy = importlib.import_module("parent.policy")
evidence = importlib.import_module("issues.evidence")
parse_history = history.parse_history
digest, marker = policy.digest, policy.marker
is_evidence_issue, require_work_issue = (
    evidence.is_evidence_issue,
    evidence.require_work_issue,
)


def comment(n: int, event: dict, *, login: str = "publisher") -> dict:
    return {
        "id": n,
        "body": marker({"parent": 1, **event}) + "\n\nrecord",
        "user": {"login": login},
    }


def test_both_media_preserve_order_provenance_and_legacy_missing_fields() -> None:
    legacy = [comment(20, {"event": "checking", "revision": 1})]
    agreement = {
        k: k
        for k in ("purpose", "scope", "behavior", "acceptance", "human_confirmation")
    }
    evidence = [
        comment(
            30,
            {
                "event": "agreed",
                "revision": 2,
                "agreement": agreement,
                "agreement_sha256": digest(agreement),
            },
        )
    ]
    legacy.insert(0, comment(10, {"event": "assigned", "generation": 1}))
    human = {
        "id": 40,
        "user": {"login": "publisher"},
        "body": "Please keep this answer.",
    }
    history = parse_history(
        "o/r", 1, "publisher", [human, *legacy], evidence, evidence_issue=9
    )
    assert [r.comment_id for r in history.records] == [10, 20, 30]
    assert [r.source for r in history.records] == [
        "legacy_parent_comment",
        "legacy_parent_comment",
        "evidence_issue",
    ]
    assert all(r.publisher_verified for r in history.records)
    assert history.records[-1].url == "https://github.com/o/r/issues/9#issuecomment-30"
    assert [e["event"] for e in history.session_events()] == [
        "assigned",
        "checking",
        "agreed",
    ]
    assert "comment_inputs" not in history.session_events()[-1]
    assert "requirements_sha256" not in history.session_events()[-1]
    assert history.parent_comments == [human, *legacy]


@pytest.mark.parametrize(
    "changed", [False, True], ids=["same_record", "conflicting_record"]
)
def test_duplicate_logical_record_across_media_is_an_error(changed: bool) -> None:
    old = comment(10, {"event": "checking", "revision": 1})
    new = copy.deepcopy(old)
    new["id"] = 20
    if changed:
        new["body"] = (
            marker({"parent": 1, "event": "checking", "revision": 1, "generation": 2})
            + "\n\nchanged"
        )
    with pytest.raises(ValueError, match="duplicate|conflict"):
        parse_history("o/r", 1, "publisher", [old], [new], evidence_issue=9)


@pytest.mark.parametrize(
    "body,login,diagnostic",
    [
        ("unmarked ordinary comment", "publisher", "unmarked_evidence_comment"),
        (
            marker({"parent": 1, "event": "assigned", "generation": 1}),
            "someone-else",
            "untrusted_evidence_publisher",
        ),
    ],
)
def test_evidence_comments_outside_the_machine_contract_are_diagnosed(
    body: str, login: str, diagnostic: str
) -> None:
    history = parse_history(
        "o/r",
        1,
        "publisher",
        [],
        [{"id": 11, "body": body, "user": {"login": login}}],
        evidence_issue=9,
    )
    assert history.diagnostics[0].code == diagnostic
    assert history.diagnostics[0].url.endswith("/issues/9#issuecomment-11")
    with pytest.raises(ValueError, match=diagnostic):
        history.session_events()


def test_foreign_publisher_on_parent_remains_human_input() -> None:
    foreign = comment(10, {"event": "agreed"}, login="someone-else")
    history = parse_history("o/r", 1, "publisher", [foreign], [], evidence_issue=None)
    assert not history.session_events() and history.parent_comments == [foreign]


def test_conflicting_key_on_the_evidence_medium_is_not_a_new_success() -> None:
    first = comment(10, {"event": "checking", "revision": 1, "generation": 1})
    second = comment(20, {"event": "checking", "revision": 1, "generation": 2})
    with pytest.raises(ValueError, match="conflicting"):
        parse_history("o/r", 1, "publisher", [], [first, second], evidence_issue=9)


@pytest.mark.parametrize(
    "issue",
    [
        {"labels": ["parent-evidence"], "body": "", "state": "closed"},
        {
            "labels": ["issue:parent"],
            "body": "<!-- parent-evidence:v2 {} -->",
            "state": "open",
        },
        {"labels": [], "body": "<!-- parent-evidence:v1 {} -->", "state": "closed"},
    ],
)
def test_label_or_binding_marker_excludes_a_container_even_when_mislabelled(
    issue: dict,
) -> None:
    assert is_evidence_issue(issue)
    with pytest.raises(ValueError, match="evidence"):
        require_work_issue(issue)


@pytest.mark.parametrize(
    "body",
    [
        "<!-- parent-session:v2 {} -->",
        '<!-- parent-session:v1 {"parent": 2, "event": "assigned"} -->',
        '<!-- parent-session:v1 {"parent": 1, "event": "assigned"}',
        "<!-- parent-review:v2 {} -->",
    ],
)
def test_invalid_owned_record_never_falls_back_to_legacy_success(body: str) -> None:
    legacy = comment(10, {"event": "assigned", "generation": 1})
    with pytest.raises(ValueError):
        parse_history(
            "o/r",
            1,
            "publisher",
            [legacy],
            [{"id": 20, "user": {"login": "publisher"}, "body": body}],
            evidence_issue=9,
        )


@pytest.mark.parametrize("missing", ["evidence", "question"])
def test_notification_history_rejects_missing_references(missing: str) -> None:
    notifications = importlib.import_module("parent.notifications")
    source = comment(10, {"event": "assigned", "generation": 1})
    spec = {
        "schema_version": 1,
        "key": "close-approval",
        "kind": "agreement",
        "summary": "Human close approval",
        "evidence": [
            f"https://github.com/o/r/issues/9#issuecomment-{99 if missing == 'evidence' else 10}"
        ],
        "resolves": ["absent-question"] if missing == "question" else [],
        "details": {
            "change": "Close approved",
            "impact": "Unverified work remains",
            "answers": [
                {
                    "source": "conversation:original",
                    "quote": "Close with remaining work",
                }
            ],
        },
    }
    row = comment(
        20,
        {
            "event": "notification_record",
            "key": spec["key"],
            "notification_sha256": digest(spec),
            "notification": spec,
        },
    )
    row["body"] = row["body"].replace("\n\nrecord", "\n\n" + notifications.render(spec))
    with pytest.raises(ValueError, match="evidence missing|unknown question"):
        parse_history("o/r", 1, "publisher", [], [source, row], evidence_issue=9)
