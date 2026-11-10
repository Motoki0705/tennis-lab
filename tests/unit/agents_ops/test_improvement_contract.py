"""Approval belongs to one candidate revision and one target, never a label."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import Any

import pytest

from tests.support.agents_ops.improvements import candidate_spec


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / ".agents/ops")
    )
    return importlib.import_module("issues.improvement")


def chain(api: Any, action: str = "approve") -> tuple[Any, list[dict[str, Any]], str]:
    source = api.ImprovementSource(12, "result-access")
    body = (
        source.marker()
        + "\n\n## 全体のスコープ\n\n取得だけ。GPUは対象外。\n\n## 全体の完了条件\n\n- [ ] 同じ結果を取得する\n"
    )
    candidate = api.event(
        source,
        "candidate",
        1,
        None,
        {"source_issue": 13, "spec": candidate_spec(), "legacy": False},
    )
    published = api.event(
        source,
        "published",
        1,
        candidate["id"],
        {"issue": 20, "mode": "created", "draft_sha256": "a" * 64},
    )
    decision = api.event(
        source,
        "decision",
        1,
        published["id"],
        {
            "request_key": "instruction-1",
            "request_sha256": "b" * 64,
            "action": action,
            "issue": 20,
            "human_confirmation": {
                "quote": "この取得改善の範囲で実行してください。",
                "source": "2026-11-11 の人間との会話。候補 I-12-result-access の確認。",
            },
            "reason": "示された範囲を実行する指示。",
            "requirements": api.contract(body),
            "previous_decision": None,
        },
    )
    return source, [candidate, published, decision], body


def comments(api: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"id": i + 100, "user": {"login": "owner"}, "body": api.render_event(row)}
        for i, row in enumerate(rows)
    ]


@pytest.mark.parametrize(
    "action,state",
    [
        ("approve", "approved"),
        ("hold", "held"),
        ("decline", "declined"),
        ("revoke", "revoked"),
    ],
)
def test_only_current_explicit_approval_authorizes_one_issue(
    api: Any, action: str, state: str
) -> None:
    source, rows, body = chain(api, action)
    candidate = api.read_candidates(comments(api, rows), 12, "owner")[source.key]
    assert api.human_state(candidate) == state
    if action == "approve":
        assert api.approved(candidate, source, 20, body)["id"] == rows[-1]["id"]
        with pytest.raises(ValueError, match="different issue"):
            api.approved(candidate, source, 21, body)
    else:
        with pytest.raises(ValueError, match=state):
            api.approved(candidate, source, 20, body)


def test_reproposal_and_second_generation_do_not_inherit_approval(api: Any) -> None:
    source, rows, body = chain(api)
    spec = candidate_spec()
    spec["scope"] = "別の取得対象も追加する。"
    rows.append(
        api.event(
            source,
            "candidate",
            2,
            rows[-1]["id"],
            {"source_issue": 13, "spec": spec, "legacy": False},
        )
    )
    candidate = api.read_candidates(comments(api, rows), 12, "owner")[source.key]
    with pytest.raises(ValueError, match="pending"):
        api.approved(candidate, source, 20, body)
    second = api.ImprovementSource(20, "result-access")
    second_row = api.event(
        second,
        "candidate",
        1,
        None,
        {"source_issue": 20, "spec": candidate_spec(), "legacy": False},
    )
    candidate = api.read_candidates(comments(api, [second_row]), 20, "owner")[
        second.key
    ]
    with pytest.raises(ValueError, match="pending"):
        api.approved(candidate, second, 30, body)


def test_formatting_and_progress_are_safe_but_scope_code_and_unknown_sections_are_not(
    api: Any,
) -> None:
    source, rows, body = chain(api)
    candidate = api.read_candidates(comments(api, rows), 12, "owner")[source.key]
    cosmetic = (
        body.replace("\n\n", "\n\n\n").replace("[ ]", "[x]")
        + "\n## 進捗概要\n\n受入確認中。"
    )
    api.approved(candidate, source, 20, cosmetic)
    for material in (
        body.replace("GPUは対象外", "GPUも実行する"),
        body + "\n## 追加要件\n\n動画100本を処理する。",
        body + "\n```python\ntrain()\n```\n",
    ):
        with pytest.raises(ValueError, match="requirements changed"):
            api.approved(candidate, source, 20, material)


@pytest.mark.parametrize(
    "fault",
    [
        "gap",
        "fork",
        "revision",
        "target",
        "publisher",
        "edited_render",
        "unknown_version",
    ],
)
def test_invalid_history_cannot_be_an_approval(api: Any, fault: str) -> None:
    source, rows, _ = chain(api)
    if fault == "gap":
        rows.pop(1)
    elif fault == "fork":
        rows[-1] = api.event(source, "decision", 1, rows[0]["id"], rows[-1]["data"])
    elif fault == "revision":
        rows[-1] = api.event(source, "decision", 2, rows[1]["id"], rows[-1]["data"])
    elif fault == "target":
        data = {**rows[-1]["data"], "issue": 30}
        rows[-1] = api.event(source, "decision", 1, rows[1]["id"], data)
    raw = comments(api, rows)
    if fault == "publisher":
        raw[-1]["user"]["login"] = "different-user"
        candidate = api.read_candidates(raw, 12, "owner")[source.key]
        assert api.human_state(candidate) == "pending"
        return
    if fault == "edited_render":
        raw[-1]["body"] += "\nこの文章だけを書き換えた。"
    elif fault == "unknown_version":
        raw[-1]["body"] = raw[-1]["body"].replace(
            "parent-improvement:v1", "parent-improvement:v2"
        )
    with pytest.raises(ValueError):
        api.read_candidates(raw, 12, "owner")


def test_replayed_old_approval_does_not_restore_a_revocation(api: Any) -> None:
    source, rows, body = chain(api)
    old_approval = copy.deepcopy(rows[-1])
    rows.append(
        api.event(
            source,
            "decision",
            1,
            rows[-1]["id"],
            {
                **old_approval["data"],
                "request_key": "instruction-2",
                "action": "revoke",
                "previous_decision": old_approval["id"],
            },
        )
    )
    rows.append(old_approval)
    candidate = api.read_candidates(comments(api, rows), 12, "owner")[source.key]
    with pytest.raises(ValueError, match="revoked"):
        api.approved(candidate, source, 20, body)


def test_source_is_independent_of_title_and_quoted_examples(api: Any) -> None:
    source = api.ImprovementSource(12, "result-access")
    assert api.index_sources([(20, source.marker() + "\nrenamed")]) == {source: 20}
    assert api.parse_source("```\n" + source.marker() + "\n```\n") is None
    with pytest.raises(ValueError, match="duplicate"):
        api.index_sources([(20, source.marker()), (21, source.marker())])
    with pytest.raises(ValueError):
        api.parse_source("text\n" + source.marker())
