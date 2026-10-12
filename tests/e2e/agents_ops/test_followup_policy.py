"""Pure policy of the follow-up issue resolver (.agents/ops/followup/followup_policy.py)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
ROOT_OPS = ROOT / ".agents/ops"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("review_policy", ROOT_OPS / "review/review_policy.py")
F = _load("followup_policy", ROOT_OPS / "followup/followup_policy.py")
CloseReason = F.CloseReason
IssueComment = F.IssueComment
NextStep = F.NextStep
ResolutionParseError = F.ResolutionParseError
Round = F.Round
SourceAction = F.SourceAction
branch_name = F.branch_name
check_commits = F.check_commits
disposition_table = F.disposition_table
next_step = F.next_step
parse_resolution = F.parse_resolution
recorded_rounds = F.recorded_rounds
rejected_marker = F.rejected_marker
rejection_noted = F.rejection_noted
round_marker = F.round_marker
source_action = F.source_action

SHA = "a" * 40


def _output(*dispositions: str, pr: bool | None = None, **overrides: Any) -> str:
    with_pr = "fixed" in dispositions if pr is None else pr
    raw: dict[str, Any] = {
        "summary": "要約。",
        "findings": [
            {"index": i, "disposition": d, "reason": f"理由{i}"}
            for i, d in enumerate(dispositions, start=1)
        ],
        "pr_title": "fix: 追補" if with_pr else None,
        "pr_body": "本文" if with_pr else None,
        "verification": "pytest 通過",
    }
    raw.update(overrides)
    return json.dumps(raw, ensure_ascii=False)


@pytest.mark.parametrize(
    ("dispositions", "opens_pr", "needs_human", "closes", "close_reason"),
    [
        (("fixed", "declined"), True, False, True, None),
        (("fixed", "needs_human"), True, True, False, None),
        (("needs_human", "declined"), False, True, False, None),
        (("declined", "already_resolved"), False, False, True, CloseReason.NOT_PLANNED),
        (
            ("already_resolved", "already_resolved"),
            False,
            False,
            True,
            CloseReason.COMPLETED,
        ),
    ],
)
def test_actions_follow_from_dispositions(
    dispositions: tuple[str, ...],
    opens_pr: bool,
    needs_human: bool,
    closes: bool,
    close_reason: Any,
) -> None:
    r = parse_resolution(_output(*dispositions), len(dispositions))

    assert (r.opens_pr, r.needs_human, r.closes_issue, r.close_reason) == (
        opens_pr,
        needs_human,
        closes,
        close_reason,
    )


def test_fenced_output_is_accepted() -> None:
    r = parse_resolution("```json\n" + _output("declined") + "\n```", 1)

    assert r.dispositions[0].reason == "理由1"


@pytest.mark.parametrize(
    ("text", "n", "message"),
    [
        ("not json", 1, "not JSON"),
        (_output("declined"), 2, "indices 1..2"),
        (_output("declined", "declined"), 1, "indices 1..1"),
        (_output("fixed", pr=False), 1, "exactly when some finding is fixed"),
        (_output("declined", pr=True), 1, "exactly when some finding is fixed"),
        (_output("fixed", pr_body=None), 1, "both be set or both null"),
        (_output("declined", extra=1), 1, "keys must be exactly"),
        (_output("declined", verification=""), 1, "verification"),
        (
            _output(
                "declined",
                findings=[{"index": 1, "disposition": "maybe", "reason": "x"}],
            ),
            1,
            "disposition must be one of",
        ),
        (
            _output(
                "declined",
                findings=[{"index": True, "disposition": "fixed", "reason": "x"}],
            ),
            1,
            "index must be an integer",
        ),
    ],
)
def test_malformed_output_is_rejected(text: str, n: int, message: str) -> None:
    with pytest.raises(ResolutionParseError, match=message):
        parse_resolution(text, n)


@pytest.mark.parametrize(
    ("dispositions", "commits", "ok"),
    [
        (("fixed",), 1, True),
        (("fixed",), 0, False),
        (("declined",), 0, True),
        (("declined",), 2, False),
    ],
)
def test_commits_must_match_fixed(
    dispositions: tuple[str, ...], commits: int, ok: bool
) -> None:
    r = parse_resolution(_output(*dispositions), len(dispositions))
    if ok:
        check_commits(r, commits)
    else:
        with pytest.raises(ResolutionParseError):
            check_commits(r, commits)


def _comment(body: str, author: str = "bot", association: str = "OWNER") -> Any:
    return IssueComment(author, association, body, "https://example.invalid/c")


def test_rounds_are_read_from_trusted_first_lines_only() -> None:
    r1 = Round(7, 1, SHA, 101)
    r2 = Round(7, 2, SHA, None)
    comments = [
        _comment(round_marker(r1) + "\n本文"),
        _comment(round_marker(Round(7, 9, SHA, 1)), author="mallory"),
        _comment("引用:\n" + round_marker(Round(7, 5, SHA, 1))),
        _comment(round_marker(r2)),
    ]

    assert recorded_rounds(7, comments, trusted_login="bot") == (r1, r2)


@pytest.mark.parametrize(
    "rounds",
    [[Round(7, 2, SHA, None)], [Round(7, 1, SHA, None), Round(7, 1, SHA, None)]],
)
def test_round_gaps_and_repeats_are_errors(rounds: list[Any]) -> None:
    with pytest.raises(ValueError, match="out of sequence"):
        recorded_rounds(
            7, [_comment(round_marker(r)) for r in rounds], trusted_login="bot"
        )


def test_round_marker_for_another_issue_is_an_error() -> None:
    with pytest.raises(ValueError, match="names issue #8"):
        recorded_rounds(
            7, [_comment(round_marker(Round(8, 1, SHA, None)))], trusted_login="bot"
        )


def test_rejection_is_matched_by_issue_round_and_author() -> None:
    latest = Round(7, 2, SHA, 55)
    noted = [_comment(rejected_marker(7, 2, 55))]

    assert rejection_noted(7, latest, noted, trusted_login="bot")
    assert not rejection_noted(
        7, latest, [_comment(rejected_marker(7, 1, 55))], trusted_login="bot"
    )
    assert not rejection_noted(
        7,
        latest,
        [_comment(rejected_marker(7, 2, 55), author="x")],
        trusted_login="bot",
    )


@pytest.mark.parametrize(
    ("latest", "state", "noted", "step"),
    [
        (None, None, False, NextStep.RUN),
        (Round(7, 1, SHA, None), None, False, NextStep.RUN),
        (Round(7, 1, SHA, 5), "OPEN", False, NextStep.WAIT_REVIEW),
        (Round(7, 1, SHA, 5), "MERGED", False, NextStep.RUN),
        (Round(7, 1, SHA, 5), "CLOSED", False, NextStep.NOTE_REJECTION),
        (Round(7, 1, SHA, 5), "CLOSED", True, NextStep.RUN),
    ],
)
def test_next_step(latest: Any, state: str | None, noted: bool, step: Any) -> None:
    assert next_step(latest, state, rejection_noted=noted) is step


def test_next_step_rejects_unknown_state() -> None:
    with pytest.raises(ValueError):
        next_step(Round(7, 1, SHA, 5), "DRAFT", rejection_noted=False)


def test_source_action() -> None:
    assert source_action("OPEN") is SourceAction.DEFER
    assert source_action("MERGED") is SourceAction.PROCESS
    assert source_action("CLOSED") is SourceAction.INVALIDATE
    with pytest.raises(ValueError):
        source_action("UNKNOWN")


def test_branch_name_and_table_escape_untrusted_text() -> None:
    r = parse_resolution(_output("declined"), 1)
    table = disposition_table(r, ["minor a.py:1 (x | y) <!-- m -->"])

    assert branch_name(12, 3) == "codex/followup-i12-r3"
    assert "x \\| y" in table
    assert "<!--" not in table
