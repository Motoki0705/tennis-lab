"""Public navigation and the distinct addition-only exception's contracts."""

from __future__ import annotations

import copy
import importlib
import random
from typing import Any

import pytest

from tests.support.agents_ops.decision_logs import (
    PublicationError,
    action,
    decision,
    evidence,
    history,
    projection,
    result,
    schema,
)
from tests.support.agents_ops.worker_logs import record_source, sources
from tests.support.agents_ops.worker_logs import schema as worker_schema

navigation: Any = importlib.import_module("decision_logs.navigation")
policy: Any = importlib.import_module("decision_logs.publication_policy")


def files(rows: list[dict[str, Any]]) -> dict[str, bytes]:
    return {
        navigation.event_path(r): schema.json_bytes(projection.render(r)) for r in rows
    }


def test_deterministic_hierarchy_and_bidirectional_causal_links() -> None:
    rows = history()
    original = files(rows)
    generated = navigation.generate(original)
    shuffled = list(original.items())
    random.Random(42).shuffle(shuffled)
    assert navigation.generate(dict(shuffled)) == generated
    assert navigation.generate({**original, **generated}) == generated
    assert (
        navigation.generate(
            {
                **original,
                **{n: b"stale view" for n in generated},
                ".agents/logs/parents/parent-1/activity-index-activities-9999/README.md": b"unrelated view",
            }
        )
        == generated
    )
    assert len(generated) == 6  # index, parent, activity, child, delegation, turn
    navigation.validate_links({**original, **generated}, generated)
    activity = next(v.decode() for p, v in generated.items() if "/activity-" in p)
    assert "時刻・対象 | 結果 | 親の判断と理由 | 次の行動・実行状態 | 根拠" in activity
    assert "判断済み未実行" in activity and "公開待ち" in activity
    assert (
        "反映範囲" in activity and "前の記録:" in activity and "次の記録:" in activity
    )
    for row in rows:
        assert f'id="record-{row["id"]}"' in activity
    # A report never becomes an inferred verification or accepted outcome.
    assert "親の確認未記録" in activity and "成果受入とは別" in activity
    assert activity.index(f'id="record-{rows[0]["id"]}"') < activity.index(
        f'id="record-{rows[1]["id"]}"'
    )


def test_direct_investigation_parallel_workers_join_and_handoff() -> None:
    worker = history()
    other = result()
    other["scope"].update(issue=20, activity_key=schema.new_id(), delegation=None)
    other["data"]["summary"] = "Parallel worker identified a different failure."
    join = decision([worker[2], other], operation="handoff")
    join["scope"].update(issue=None, delegation=None, activity_key=schema.new_id())
    join["data"].update(
        evaluation="handoff",
        reason="Join both findings and hand the investigation over.",
    )
    handoff = action(join)
    handoff["scope"] = copy.deepcopy(join["scope"])
    followup = result(action=handoff)
    followup["scope"] = copy.deepcopy(join["scope"])
    followup["data"]["summary"] = "The new parent checked both findings."
    rows = [*worker, other, join, handoff, followup]
    generated = navigation.generate(files(rows))
    parent = generated[".agents/logs/parents/parent-1/README.md"].decode()
    # Upper levels summarize activities; the canonical activity pages retain
    # the causal tables and anchors instead of copying every record here.
    assert len({navigation.event_path(r).rsplit("/", 1)[0] for r in rows}) == 3
    for group in {navigation.event_path(r).rsplit("/", 1)[0] for r in rows}:
        assert group.rsplit("/", 1)[-1] + "/README.md)" in parent
    assert "親の直接活動" in parent and "](child-20/README.md)" in parent
    assert "Join both findings" in parent and "未判断" in parent
    canonical = generated[
        navigation.event_path(join).rsplit("/", 1)[0] + "/README.md"
    ].decode()
    assert all(
        f"#record-{r['id']}" in canonical for r in (worker[2], other, handoff, followup)
    )
    navigation.validate_links({**files(rows), **generated}, generated)


def test_prose_cannot_inject_markdown_links_html_or_rows() -> None:
    row = result()
    row["data"]["summary"] = 'result | [extra](https://example.org) <img src="x">\nnext'
    public = files([row])
    generated = navigation.generate(public)
    text = generated[
        navigation.event_path(row).rsplit("/", 1)[0] + "/README.md"
    ].decode()
    assert "[extra](" not in text and '<img src="x">' not in text
    assert "&#124;" in text and "&lt;img" in text and "<br>next" in text


def test_existing_turn_links_match_worker_metadata_without_copying_its_text() -> None:
    source, receipt = record_source()
    worker_files = sources.render(source, receipt)
    directory = worker_schema.path(worker_schema.validate(worker_files))
    inputs = {
        **files(history()),
        **{directory + "/" + n: raw for n, raw in worker_files.items()},
    }
    generated = navigation.generate(inputs)
    turn = generated[directory + "/README.md"].decode()
    assert "導入前・判断未記録" not in turn
    assert (
        "](instruction.md)" in turn and "](final.md)" in turn and "](meta.json)" in turn
    )
    assert receipt["final"] not in turn and source["instruction"] not in turn
    assert "初回要求・合意" in turn


def test_late_initial_worker_log_preserves_the_turns_existing_navigation() -> None:
    row = result()
    row["scope"]["delegation"]["turn"] = 2
    inputs = files([row])
    before = navigation.generate(inputs)
    logical = (
        ".agents/logs/parents/parent-1/child-10/d-" + "a" * 32 + "/turn-0002/README.md"
    )
    assert logical in before
    assert not any("0002-resume" in name for name in before)
    source, receipt = record_source()
    source.update(sequence=2, kind="initial")
    raw = sources.render(source, receipt)
    directory = worker_schema.path(worker_schema.validate(raw))
    after_inputs = {
        **inputs,
        **before,
        **{directory + "/" + n: value for n, value in raw.items()},
    }
    after = navigation.generate(after_inputs)
    assert set(before) <= set(after)
    assert logical in after and directory.endswith("0002-initial")
    assert "](../0002-initial/instruction.md)" in after[logical].decode()
    assert "](../turn-0002/README.md)" in after[directory + "/README.md"].decode()
    assert navigation.generate({**after_inputs, **after}) == after


@pytest.mark.parametrize(
    "fault",
    [
        "old_record",
        "arbitrary_readme",
        "outside",
        "worker_file",
        "delete",
        "other_parent",
        "noncanonical_path",
        "secret",
        "unsafe_link",
    ],
)
def test_publication_exception_rejects_unsafe_changes(fault: str) -> None:
    old = result()
    new = result(previous=old)
    before = files([old])
    before.update(navigation.generate(before))
    expected = files([new])
    after = {**before, **expected}
    after.update(navigation.generate(after))
    diff = {
        n: "M" if n in before else "A"
        for n, raw in after.items()
        if before.get(n) != raw
    }
    if fault == "old_record":
        old["data"]["summary"] = "changed history"
        after.update(files([old]))
    elif fault == "arbitrary_readme":
        after[".agents/logs/parents/README.md"] += b"arbitrary text\n"
    elif fault in {"outside", "worker_file"}:
        name = (
            "code.py"
            if fault == "outside"
            else ".agents/logs/parents/parent-1/child-10/d-example/0001-initial/final.md"
        )
        after[name] = b"extra\n"
        diff[name] = "A"
    elif fault == "delete":
        after.pop(navigation.event_path(old))
        diff[navigation.event_path(old)] = "D"
    elif fault == "other_parent":
        new["scope"]["parent"] = 2
        extra = files([new])
        after.update(extra)
        diff.update(dict.fromkeys(extra, "A"))
    elif fault == "noncanonical_path":
        name = navigation.event_path(new)
        after[name.replace("activity-", "private-")] = after.pop(name)
    else:
        payload = projection.render(new)
        if fault == "secret":
            payload["record"]["data"]["summary"] = "ghp_" + "x" * 30
        else:
            payload["record"]["data"]["evidence"] = [
                {**evidence(), "url": "https://localhost/details"}
            ]
        after[navigation.event_path(new)] = schema.json_bytes(payload)
    with pytest.raises(PublicationError):
        policy.verify(before, after, diff, parent=1, expected=expected)


def test_valid_record_additions_and_exact_generated_readmes_pass() -> None:
    rows = history()
    before = files(rows[:3])
    before.update(navigation.generate(before))
    expected = files(rows[3:])
    after = {**before, **expected}
    generated = navigation.generate(after)
    after.update(generated)
    diff = {
        n: "M" if n in before else "A"
        for n, raw in after.items()
        if before.get(n) != raw
    }
    policy.verify(before, after, diff, parent=1, expected=expected)
    policy.pr_changes(
        [
            {"filename": n, "status": {"A": "added", "M": "modified"}[s]}
            for n, s in diff.items()
        ],
        diff,
    )


@pytest.mark.parametrize(
    "mode,status",
    [
        ("120000", "A"),
        ("100755", "A"),
        ("160000", "A"),
        ("100644", "D"),
        ("100644", "R100"),
    ],
)
def test_raw_diff_rejects_nonregular_files_and_nonadditive_records(
    mode: str, status: str
) -> None:
    raw = f":000000 {mode} {'0' * 40} {'a' * 40} {status}\0README.md\0".encode()
    with pytest.raises(PublicationError, match="unsafe_diff"):
        policy.changes(raw)


@pytest.mark.parametrize("fault", ["none", "duplicate_page", "invalid_page"])
def test_publication_checks_every_github_file_page(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    github = importlib.import_module("decision_logs.publication").GitHub("o/r")
    pages: Any = [
        [{"filename": f"event-{i}.json", "status": "added"} for i in range(100)],
        [{"filename": "README.md", "status": "modified"}],
    ]
    if fault == "duplicate_page":
        pages[1] = pages[0]
    elif fault == "invalid_page":
        pages[1] = {"error": "invalid response"}
    requested = []

    def request(endpoint: str) -> Any:
        requested.append(endpoint)
        return pages[len(requested) - 1]

    monkeypatch.setattr(github, "request", request)
    if fault == "none":
        assert len(github.files(42)) == 101
    else:
        with pytest.raises(PublicationError, match="github_pagination"):
            github.files(42)
    assert requested == [
        f"repos/o/r/pulls/42/files?per_page=100&page={p}" for p in (1, 2)
    ]
