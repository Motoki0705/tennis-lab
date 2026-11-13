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
        body.replace(
            source.marker(), source.marker() + "\n\n追加で動画100本を処理する。"
        ),
        body.replace("GPUは対象外", "GPUも実行する"),
        body + "\n## 追加要件\n\n動画100本を処理する。",
        body + "\n```python\ntrain()\n```\n",
    ):
        with pytest.raises(ValueError, match="requirements changed"):
            api.approved(candidate, source, 20, material)


@pytest.mark.parametrize("kind", ["prose", "supplement", "fence"])
def test_approved_preamble_is_protected_and_formatting_is_preserved(
    api: Any, kind: str
) -> None:
    source, rows, body = chain(api)
    preamble = "保存済みの結果だけを取得する。 処理済みなら終了する。"
    body = body.replace(source.marker(), source.marker() + "\n\n" + preamble)
    if kind == "supplement":
        body = importlib.import_module(
            "issue_supplement.invariants"
        ).render_supplements(
            body,
            [
                {
                    "after": preamble,
                    "content": "保存済み結果は[README](README.md)を参照できる。",
                }
            ],
        )
    elif kind == "fence":
        body = body.replace(
            preamble, preamble + "\n\n```python\n## 見出しの例\nlimit = 1\n```"
        )
    rows[-1] = api.event(
        source,
        "decision",
        1,
        rows[-2]["id"],
        {**rows[-1]["data"], "requirements": api.contract(body)},
    )
    candidate = api.read_candidates(comments(api, rows), 12, "owner")[source.key]
    cosmetic = body.replace(
        preamble, "保存済みの結果だけを取得する。\n\n  処理済みなら終了する。"
    )
    api.approved(candidate, source, 20, cosmetic)
    changed = body.replace(
        "README.md"
        if kind == "supplement"
        else "limit = 1"
        if kind == "fence"
        else "処理済みなら終了する。",
        "CHANGED.md"
        if kind == "supplement"
        else "limit = 100"
        if kind == "fence"
        else "追加の動画も処理する。",
    )
    with pytest.raises(ValueError, match="requirements changed"):
        api.approved(candidate, source, 20, changed)


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


def test_section_cannot_shadow_the_preamble_contract(api: Any) -> None:
    source, _, body = chain(api)
    body = body.replace(source.marker(), source.marker() + "\n\n前文の作業要件。")
    with pytest.raises(ValueError, match="reserved approval contract heading"):
        api.contract(body + "\n## _preamble\n\n別の条件。\n")


@pytest.mark.parametrize("change", ["decision", "body", "source", "hold"])
def test_completion_applies_only_to_current_approval_and_requirements(
    api: Any, change: str
) -> None:
    source, rows, body = chain(api)
    applies = importlib.import_module("parent.improvements").completion_applies
    saved = {
        "state": "completed",
        "improvement_approval": {
            "parent": 12,
            "key": source.key,
            "revision": 1,
            "issue": 20,
            "decision_id": rows[-1]["id"],
        },
    }
    candidate = api.read_candidates(comments(api, rows), 12, "owner")[source.key]
    assert applies(saved, source, candidate, 20, body)
    # Scope applicability does not turn an explicitly incomplete result into
    # unverified; the caller separately displays its recorded outcome.
    assert applies({**saved, "state": "incomplete"}, source, candidate, 20, body)
    if change in {"decision", "hold"}:
        rows.append(
            api.event(
                source,
                "decision",
                1,
                rows[-1]["id"],
                {
                    **rows[-1]["data"],
                    "request_key": "human-2",
                    "human_confirmation": {
                        "quote": "今の条件で再判断してください。",
                        "source": "新しい人間の発言。",
                    },
                    "previous_decision": rows[-1]["id"],
                    "action": "approve" if change == "decision" else "hold",
                },
            )
        )
        candidate = api.read_candidates(comments(api, rows), 12, "owner")[source.key]
    elif change == "body":
        body = body.replace("GPUは対象外", "GPUも実行する")
    else:
        body = body.replace(
            source.marker(), api.ImprovementSource(13, source.key).marker()
        )
    assert not applies(saved, source, candidate, 20, body)


def approval_requirements() -> dict[str, str]:
    return {
        "目的・背景": "取得操作を減らす。",
        "全体のスコープ": "取得のみ。削除は対象外。",
        "全体方針・合意事項": "保存内容を変更せずに取得する。",
        "全体の完了条件": "- [ ] 取得しても保存内容が変わらない。",
        "依存関係・進め方": "依存なし。取得に失敗したら停止する。",
    }


@pytest.mark.parametrize("fault", ["missing", "empty"])
@pytest.mark.parametrize(
    "heading",
    [
        "目的・背景",
        "全体のスコープ",
        "全体方針・合意事項",
        "全体の完了条件",
        "依存関係・進め方",
    ],
)
def test_approved_execution_requires_every_mandatory_section(
    api: Any, heading: str, fault: str
) -> None:
    requirements = approval_requirements()
    if fault == "missing":
        del requirements[heading]
    else:
        requirements[heading] = " \n"
    with pytest.raises(ValueError, match=heading):
        api.execution_terms(requirements)


def test_approved_execution_retains_preamble_extra_rules_and_stop_conditions(
    api: Any,
) -> None:
    body = (
        api.ImprovementSource(12, "retrieval").marker()
        + "\n\n外部送信は禁止。\n\n"
        + "\n\n".join(f"## {h}\n\n{v}" for h, v in approval_requirements().items())
        + '\n\n## 追加の規約\n\n```python\nmode = "read only"\n```\n'
        + "\n## 進捗概要\n\n表示だけの進捗。\n"
    )
    requirements = api.contract(body)
    terms = api.execution_terms(requirements)
    assert terms["purpose"] == "取得操作を減らす。"
    assert terms["scope"] == "取得のみ。削除は対象外。"
    assert terms["acceptance"] == "- [ ] 取得しても保存内容が変わらない。"
    assert "保存内容を変更せずに取得する。" in terms["behavior"]
    assert "依存なし。取得に失敗したら停止する。" in terms["behavior"]
    assert "外部送信は禁止。" in terms["behavior"]
    assert '```python\nmode = "read only"\n```' in terms["behavior"]
    assert "表示だけの進捗。" not in terms["behavior"]
    # Canonical comments sort object keys; mapping insertion order is not consent.
    assert api.execution_terms(dict(reversed(list(requirements.items())))) == terms


@pytest.mark.parametrize(
    "fault", ["unsorted", "duplicate", "foreign", "version", "misplaced"]
)
def test_grouped_extraction_provenance_rejects_ambiguous_candidates(
    api: Any, fault: str
) -> None:
    proposals = importlib.import_module("issues.improvement_proposals")
    marker = proposals.Extraction(12, 30, ("first", "second")).marker()
    assert proposals.parse_extraction(marker).keys == ("first", "second")
    if fault == "unsorted":
        marker = marker.replace("I-12-first,I-12-second", "I-12-second,I-12-first")
    elif fault == "duplicate":
        marker = marker.replace("I-12-second", "I-12-first")
    elif fault == "foreign":
        marker = marker.replace("I-12-second", "I-13-second")
    elif fault == "version":
        marker = marker.replace("extraction:v1", "extraction:v2")
    else:
        marker = "前文\n" + marker
    with pytest.raises(ValueError):
        proposals.parse_extraction(marker)
