"""Reader-facing acceptance: failures, missing states and bounded navigation."""

from __future__ import annotations

import html
import importlib
import json
import posixpath
import re
from pathlib import Path
from typing import Any

import pytest

from tests.support.agents_ops.decision_acceptance import incident, large_history
from tests.support.agents_ops.decision_logs import (
    action,
    decision,
    observed,
    projection,
    result,
    schema,
)

navigation: Any = importlib.import_module("decision_logs.navigation")
policy: Any = importlib.import_module("decision_logs.publication_policy")
ROOT = ".agents/logs/parents"
REPO = Path(__file__).resolve().parents[3]


def files(rows: list[dict[str, Any]]) -> dict[str, bytes]:
    return {
        navigation.event_path(r): schema.json_bytes(projection.render(r)) for r in rows
    }


@pytest.mark.parametrize(
    "name", ["fetch_only", "recovery_refused", "useful_runtime_failure"]
)
def test_incident_readmes_preserve_failures_changes_and_recorded_reasons(
    name: str,
) -> None:
    rows = incident(name)
    original = files(rows)
    generated = navigation.generate(original)
    text = html.unescape("\n".join(raw.decode() for raw in generated.values()))
    assert "前回との差" in text and "コード" in text
    assert "親の確認未記録" in text
    if name == "fetch_only":
        assert "取得だけの再試行" in text and "不足するCIログ" in text
        assert "コード: 変更なし" in text and "実行条件: 未確認" in text
        assert "保存済みのコード確認" in text and "CIログの取得完了" in text
        assert "新しいAIの起動" not in text
    elif name == "recovery_refused":
        assert "resolveは拒否" in text and "修正後のHEADを正式にレビュー" in text
        assert "新規レビューを要求" in text and "コード: 変更あり" in text
        assert "保存済みmajor指摘" in text
        assert "取得だけの再試行" not in text
    else:
        assert "runtime_error" in text and "major 1件" in text and "failed" in text
        assert "最終報告あり" in text and "未判断" in text
        assert not any(
            row["kind"] == "decision" for row in navigation.records(original)
        )
    for row in rows:
        assert f'id="record-{row["id"]}"' in text
    navigation.validate_links({**original, **generated}, generated)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "public", "note": "ordinary prose"},
        {"status": "unavailable", "code": "not an availability object"},
        {"kind": {"nested": "public"}},
        {
            "kind": "public",
            "url": "https://example.org/opaque",
            "revision": None,
            "content_sha256": "b" * 64,
        },
    ],
)
def test_literal_data_is_never_promoted_to_typed_evidence(
    payload: dict[str, Any],
) -> None:
    row = result()
    row["data"]["summary"] = "adapter: " + json.dumps(payload)
    generated = navigation.generate(files([row]))
    text = "\n".join(raw.decode() for raw in generated.values())
    assert "](https://example.org/opaque)" not in text
    assert f'id="record-{row["id"]}"' in text


@pytest.mark.parametrize(
    "state",
    [
        "decided_not_executed",
        "rejected_before_start",
        "missing_final",
        "unverified",
        "undecided",
    ],
)
def test_readme_states_are_distinct_and_do_not_imply_acceptance(state: str) -> None:
    choice = decision()
    rows = [choice]
    if state in {"decided_not_executed", "rejected_before_start"}:
        rows.append(action(choice, state=state))
    else:
        report = result()
        if state == "unverified":
            report["data"]["final_report"] = observed(
                {"kind": "unavailable", "reason": "private report"}
            )
            rows.append(report)
            choice = decision([report])
            rows.append(choice)
        else:
            rows = [report]
    generated = navigation.generate(files(rows))
    activity = next(v.decode() for p, v in generated.items() if "/activity-" in p)
    assert {
        "decided_not_executed": "判断済み未実行",
        "rejected_before_start": "起動前拒否",
        "missing_final": "記録不足・最終報告",
        "unverified": "親の確認未記録",
        "undecided": "未判断",
    }[state] in activity
    assert "公開待ち" in activity and "decision-log-status" in activity
    assert "受入済み" not in activity


def test_published_real_record_copy_can_be_regenerated_under_existing_exception() -> (
    None
):
    manifest = json.loads(
        (
            REPO / "tests/support/agents_ops/decision_log_first_publication.json"
        ).read_bytes()
    )
    before = {name: (REPO / name).read_bytes() for name in manifest["paths"]}
    # Derived bytes may already use the new renderer after the next real
    # publication. Always exercise a stale-view repair without copying them
    # into a second source of truth or requiring Git history in CI.
    before[f"{ROOT}/README.md"] = b"# Earlier renderer snapshot\n"
    records = navigation.records(before)
    assert len([r for r in records if r["scope"]["parent"] == 1796]) == 205
    generated = navigation.generate(before)
    after = {**before, **generated}
    diff = {
        n: "M" if n in before else "A"
        for n, raw in generated.items()
        if before.get(n) != raw
    }
    policy.verify(before, after, diff, parent=1796, expected={})
    assert navigation.generate(after) == generated
    assert all(after[n] == raw for n, raw in before.items() if "/event-" in n)
    index, parent = (
        generated[f"{ROOT}/README.md"],
        generated[f"{ROOT}/parent-1796/README.md"],
    )
    assert len(index) < 16 * 1024 and len(parent) < 48 * 1024
    assert max(map(len, generated.values())) < 192 * 1024
    text = "\n".join(raw.decode() for raw in generated.values())
    assert "pr-1876-2" in parent.decode() and "次の行動" in parent.decode()
    assert "&#123;&#x27;" not in text and "&#123;&quot;" not in text
    assert "モデル" in text and "前回との差" in text
    labels = re.findall(r"\[([^\]]+)\]\(activity-[^)]+\)", parent.decode())
    assert labels and all(not re.fullmatch(r"活動 [0-9]+", label) for label in labels)


def test_literal_summary_fields_are_readable_and_cannot_insert_markup() -> None:
    row = result()
    row["data"]["summary"] = (
        "review.resolve: coverage={'covered': True, 'unresolved': ((1769, 'major-1'),)}; detail={\"message\": \"[bad](https://example.org) <b>unsafe</b>\"}"
    )
    generated = navigation.generate(files([row]))
    text = "\n".join(raw.decode() for raw in generated.values())
    assert "確認範囲の充足: はい" in text and "未解決: 1769、major-1" in text
    assert "[bad](" not in text and "<b>unsafe</b>" not in text
    assert "&#123;&#x27;" not in text and "&#123;&quot;" not in text


@pytest.mark.parametrize("many_activities", [False, True])
def test_large_history_keeps_summaries_bounded_and_all_records_reachable(
    many_activities: bool,
) -> None:
    rows = large_history(800, many_activities=many_activities)
    original = files(rows)
    generated = navigation.generate(original)
    assert len(generated[f"{ROOT}/README.md"]) < 16 * 1024
    assert len(generated[f"{ROOT}/parent-1/README.md"]) < 48 * 1024
    assert max(map(len, generated.values())) < 192 * 1024
    text = "\n".join(raw.decode() for raw in generated.values())
    anchors = re.findall(r'<a id="record-([a-f0-9]{32})"', text)
    assert set(anchors) == {r["id"] for r in rows} and len(anchors) == len(rows)
    reached: set[str] = set()
    pending = [f"{ROOT}/README.md"]
    while pending:
        page = pending.pop()
        if page in reached:
            continue
        reached.add(page)
        for target in re.findall(r"\]\(([^)]+)\)", generated[page].decode()):
            name = posixpath.normpath(
                posixpath.join(posixpath.dirname(page), target.partition("#")[0])
            )
            if name in generated and name not in reached:
                pending.append(name)
    assert reached == set(generated)
    assert navigation.generate({**original, **generated}) == generated
    navigation.validate_links({**original, **generated}, generated)


def test_one_oversized_record_keeps_its_anchor_and_explicit_full_record_link() -> None:
    choice = decision()
    choice["data"]["unresolved"] = [
        f"open item {index}: " + "x" * 200 for index in range(100)
    ]
    original = files([choice])
    generated = navigation.generate(original)
    activity = generated[
        navigation.event_path(choice).rsplit("/", 1)[0] + "/README.md"
    ].decode()
    assert "表示省略" in activity and "完全な記録本文" in activity
    assert f'id="record-{choice["id"]}"' in activity
    assert max(map(len, generated.values())) < 192 * 1024
    assert files([choice]) == original
