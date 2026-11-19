"""Turn denominators, separate facts, and public-only historical navigation."""

from __future__ import annotations

import copy
import html
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from tests.support.agents_ops.decision_logs import (
    PublicationError,
    action,
    decision,
    evidence,
    projection,
    result,
    schema,
)
from tests.support.agents_ops.traceability import (
    observation_record,
    remaining_publication,
    termination_observation,
    worker_observation,
)
from tests.support.agents_ops.worker_logs import record_source, sources
from tests.support.agents_ops.worker_logs import schema as worker_schema

navigation: Any = importlib.import_module("decision_logs.navigation")
turns: Any = importlib.import_module("decision_logs.turns")
historical: Any = importlib.import_module("decision_logs.historical")
ROOT = ".agents/logs/parents"
REPO = Path(__file__).resolve().parents[3]


def envelopes(rows: list[dict[str, Any]]) -> dict[str, bytes]:
    return {
        navigation.event_path(r): schema.json_bytes(projection.render(r)) for r in rows
    }


def triplet(**updates: Any) -> dict[str, bytes]:
    source, receipt = record_source()
    source.update(updates)
    raw = sources.render(source, receipt)
    directory = worker_schema.path(worker_schema.validate(raw))
    return {directory + "/" + name: value for name, value in raw.items()}


@pytest.fixture(scope="module")
def audit_files() -> dict[str, bytes]:
    manifest = json.loads(
        (REPO / "tests/support/agents_ops/decision_log_audit.json").read_bytes()
    )
    return {name: (REPO / name).read_bytes() for name in manifest["paths"]}


def test_fixed_audit_has_separate_published_and_union_denominators(
    audit_files: dict[str, bytes],
) -> None:
    rows = navigation.records(audit_files)
    inventory = turns.inventory(audit_files, rows)
    metrics = turns.statistics(list(inventory.values()))
    assert {
        k: metrics[k]
        for k in ("published", "matched", "unmatched", "event_only", "total")
    } == {
        "published": 51,
        "matched": 4,
        "unmatched": 47,
        "event_only": 2,
        "total": 53,
    }
    assert {
        (t.key[1], t.key[2], t.key[4])
        for t in inventory.values()
        if t.directory and t.records
    } == {
        (1796, 1816, 1),
        (1796, 1816, 4),
        (1796, 1817, 1),
        (1796, 1817, 2),
    }
    generated = navigation.generate(audit_files)
    root = html.unescape(generated[ROOT + "/README.md"].decode())
    for text in (
        "公開workerログ: 51 ターン",
        "対応イベントあり: 4 ターン",
        "対応イベントなし: 47 ターン",
        "イベントのみ: 2 ターン",
        "和集合: 53 ターン",
    ):
        assert text in root
    assert "導入前・判断未記録" not in root
    assert "導入時期・未捕捉・捕捉失敗・未公開の区別は未確認" in root
    assert all(
        f"{ROOT}/parent-{p}/README.md" in generated
        for p in (1581, 1694, 1698, 1710, 1796, 1797)
    )
    assert navigation.generate({**audit_files, **generated}) == generated


def test_worker_only_input_has_navigation_and_main_files() -> None:
    inputs = triplet()
    generated = navigation.generate(inputs)
    assert ROOT + "/README.md" in generated
    turn = next(v.decode() for p, v in generated.items() if "/turn-0001/" in p)
    assert "instruction.md" in turn and "final.md" in turn and "meta.json" in turn
    assert "最終報告あり" in turn and "main" in turn
    assert "導入前" not in turn


@pytest.mark.parametrize(
    ("execution", "final", "publication", "label"),
    [
        ("ended", "absent", "unknown", "最終報告なし"),
        ("running", "unknown", "unknown", "稼働中"),
        ("unknown", "unknown", "unknown", "終了未確認"),
        ("ended", "present", "pending", "最終報告あり・公開保留"),
    ],
)
def test_four_states_are_counted_once_per_turn(
    execution: str, final: str, publication: str, label: str
) -> None:
    row = observation_record(worker_observation(execution, final, publication))
    another = copy.deepcopy(row)
    another["id"] = schema.new_id()
    another["scope"]["activity_key"] = schema.new_id()
    inputs = envelopes([row, another])
    inventory = turns.inventory(inputs, navigation.records(inputs))
    assert len(inventory) == 1
    item = next(iter(inventory.values()))
    assert (item.execution, item.final, item.publication) == (
        execution,
        final,
        publication,
    )
    generated = navigation.generate(inputs)
    page = next(v.decode() for p, v in generated.items() if "/turn-0001/" in p)
    assert label in page
    assert turns.statistics(list(inventory.values()))["missing_final"] == int(
        final == "absent"
    )


def test_status_command_ended_and_awaiting_final_do_not_prove_worker_end() -> None:
    choice = decision(operation="record")
    ended = action(choice, state="ended")
    report = result(action=ended)
    report["data"]["summary"] = (
        "worker_logs.status: state=captured; stage=awaiting_final"
    )
    report["data"]["final_report"] = {
        "status": "unavailable",
        "reason": "not a final-report observation",
    }
    inputs = envelopes([choice, ended, report])
    item = next(iter(turns.inventory(inputs, navigation.records(inputs)).values()))
    assert item.execution == "unknown" and item.final == "unknown"


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "missing",
        "version",
        "absence_without_end",
        "pr_without_head",
        "foreign_pr",
        "final_disagreement",
        "termination_with_turn",
    ],
)
def test_typed_observations_reject_ambiguous_or_malformed_facts(fault: str) -> None:
    row = observation_record(worker_observation("ended", "absent"))
    obs = row["data"]["observation"]
    if fault == "extra":
        obs["extra"] = True
    elif fault == "missing":
        del obs["final"]
    elif fault == "version":
        obs["version"] = True
    elif fault == "absence_without_end":
        obs["execution"]["state"] = "unknown"
    elif fault in {"pr_without_head", "foreign_pr"}:
        row = observation_record(worker_observation("ended", "present", "public_pr"))
        row["data"]["observation"]["publication"][
            "pr_head" if fault == "pr_without_head" else "pr_url"
        ] = (
            None
            if fault == "pr_without_head"
            else "https://github.com/other/repo/pull/42"
        )
    elif fault == "final_disagreement":
        row["data"]["final_report"] = {"status": "observed", "value": evidence()}
    else:
        row = observation_record(termination_observation())
    with pytest.raises(PublicationError):
        schema.validate_record(row)


def test_same_sequence_in_different_delegations_is_not_collapsed() -> None:
    rows = [observation_record(worker_observation("ended", "absent")) for _ in range(2)]
    rows[1]["scope"]["delegation"]["key"] = "d-" + "b" * 32
    inputs = envelopes(rows)
    metrics = turns.statistics(
        list(turns.inventory(inputs, navigation.records(inputs)).values())
    )
    assert metrics["total"] == metrics["missing_final"] == 2


def test_end_without_final_inspection_does_not_count_absence() -> None:
    row = observation_record(worker_observation("ended"))
    inputs = envelopes([row])
    items = list(turns.inventory(inputs, navigation.records(inputs)).values())
    assert items[0].execution == "ended" and items[0].final == "unknown"
    assert turns.statistics(items)["missing_final"] == 0


@pytest.mark.parametrize("linked", [False, True])
def test_final_recovery_requires_a_causal_update_or_visible_reconciliation(
    linked: bool,
) -> None:
    absent = observation_record(worker_observation("ended", "absent"))
    recovered = observation_record(worker_observation("ended", "present"))
    if linked:
        recovered["data"]["previous_result_id"] = absent["id"]
    inputs = envelopes([absent, recovered])
    item = next(iter(turns.inventory(inputs, navigation.records(inputs)).values()))
    assert item.final == ("present" if linked else "unknown")
    assert item.conflicting is not linked
    assert turns.statistics([item])["missing_final"] == 0


def test_worker_and_event_repository_aliases_match_the_same_turn() -> None:
    row = result()
    row["scope"]["repo"] = "O/R"
    inputs = {**triplet(), **envelopes([row])}
    metrics = turns.statistics(
        list(turns.inventory(inputs, navigation.records(inputs)).values())
    )
    assert metrics["published"] == metrics["matched"] == metrics["total"] == 1


def test_initial_and_resume_folders_cannot_double_count_one_turn() -> None:
    inputs = {
        **triplet(sequence=2, kind="initial"),
        **triplet(sequence=2, kind="resume"),
    }
    with pytest.raises(PublicationError, match="record_turn"):
        turns.inventory(inputs, [])


def test_intermediate_prs_and_later_main_turn_stay_distinct() -> None:
    rows = []
    for sequence, number in ((2, 1884), (3, 1889)):
        obs = worker_observation("ended", "present", "public_pr")
        obs["publication"]["pr_url"] = f"https://github.com/o/r/pull/{number}"
        obs["publication"]["main_revision"] = "e" * 40
        row = observation_record(obs)
        row["scope"]["delegation"]["turn"] = sequence
        rows.append(row)
    inputs = {**triplet(sequence=4, kind="resume"), **envelopes(rows)}
    generated = navigation.generate(inputs)
    for sequence, number in ((2, 1884), (3, 1889)):
        page = next(
            v.decode() for p, v in generated.items() if f"/turn-{sequence:04d}/" in p
        )
        assert (
            f"https://github.com/o/r/pull/{number}" in page
            and "観測時点の公開PR" in page
        )
        assert "https://github.com/o/r/commit/" + "d" * 40 in page
        assert "https://github.com/o/r/commit/" + "e" * 40 in page
        assert "PR HEAD" in page and "照合先main" in page
        assert not any(f"/{sequence:04d}-resume/" in p for p in inputs)
    fourth = next(v.decode() for p, v in generated.items() if "/turn-0004/" in p)
    assert all("../0004-resume/" + n in fourth for n in worker_schema.FILENAMES)
    # Arrival in the public tree supersedes an older observation of an open PR.
    after = {**inputs, **triplet(sequence=2, kind="resume")}
    inventory = turns.inventory(after, navigation.records(after))
    assert next(t for t in inventory.values() if t.key[4] == 2).publication == "main"


def test_historical_explicit_sources_fix_missing_reports_without_rewriting_events(
    audit_files: dict[str, bytes],
) -> None:
    old = dict(audit_files)
    records = historical.records()
    added = envelopes([r for r in records if r["scope"]["parent"] == 1796])
    inputs = {**old, **added}
    inventory = turns.inventory(inputs, navigation.records(inputs))
    for sequence in (2, 3):
        item = next(
            t for t in inventory.values() if t.key[2] == 1816 and t.key[4] == sequence
        )
        assert item.execution == "ended" and item.final == "absent"
    assert turns.statistics(list(inventory.values()))["missing_final"] == 2
    generated = navigation.generate({**inputs, **envelopes(records)})
    parent = html.unescape(generated[ROOT + "/parent-1796/README.md"].decode())
    assert "6786435778" in parent and "6786439334" in parent
    assert "6767080512" in parent and "要照合" in parent
    other = generated[ROOT + "/parent-1864/README.md"].decode()
    assert "6785708290" in other and "6785704100" in other
    real_base = ROOT + "/parent-1797/child-1862/d-d5f6e27822fd494fb71d2e74f10be4f0/"
    for sequence, pr in ((2, 1884), (3, 1889)):
        page = generated[real_base + f"turn-{sequence:04d}/README.md"].decode()
        assert f"https://github.com/Motoki0705/tennis-lab/pull/{pr}" in page
        assert "観測時点の公開PR" in page
    fourth = generated[real_base + "turn-0004/README.md"].decode()
    assert all("../0004-resume/" + name in fourth for name in worker_schema.FILENAMES)
    assert all(inputs[name] == value for name, value in old.items())


def test_main_observation_without_files_does_not_claim_this_head_has_them() -> None:
    row = observation_record(worker_observation("ended", "present", "main"))
    generated = navigation.generate(envelopes([row]))
    page = next(
        raw.decode() for name, raw in generated.items() if "/turn-0001/" in name
    )
    assert "このHEADの3ファイルは未確認" in page
    assert "このHEADに公開3ファイルを確認" not in page


@pytest.mark.parametrize("field", ["resume_command", "decision_resume_command"])
def test_typed_termination_cannot_publish_a_private_resume_path(field: str) -> None:
    row = observation_record(
        termination_observation(**{field: "resume /home/private/state.json"}),
        issue=None,
        delegation=None,
    )
    with pytest.raises(PublicationError, match="absolute_path"):
        projection.render(row)


@pytest.mark.parametrize(
    "fault",
    [
        "bool_count",
        "negative_count",
        "complete_type",
        "bool_actor",
        "missing_field",
        "batch_alias",
    ],
)
def test_termination_inventory_rejects_invalid_producer_fields(fault: str) -> None:
    pending = remaining_publication("decision_log", unpublished_count=None)
    fact = termination_observation(remaining=[pending], remaining_complete=False)
    if fault == "bool_count":
        pending["unpublished_count"] = True
    elif fault == "negative_count":
        pending["unpublished_count"] = -1
    elif fault == "complete_type":
        fact["remaining_complete"] = 1
    elif fault == "bool_actor":
        fact["publication_actor"] = True
    elif fault == "missing_field":
        del pending["issue"]
    elif fault == "batch_alias":
        pending["public_key"] = "b" * 32
    with pytest.raises(PublicationError):
        projection.render(observation_record(fact, issue=None, delegation=None))


def test_termination_is_visible_for_parent_without_workers_and_keeps_pending_owner() -> (
    None
):
    row = observation_record(
        termination_observation(
            event="handoff_pending",
            publication_actor=2,
            decision_resume_command="python -m parent --parent 1 publish-decision-log --retry",
            remaining=[
                remaining_publication(
                    "decision_log",
                    state="pending",
                    stage="pr_open",
                    pr_number=77,
                    pr_head="d" * 40,
                    unpublished_count=3,
                    public_key="a" * 32,
                    batch_key="a" * 32,
                )
            ],
        ),
        issue=None,
        delegation=None,
    )
    generated = navigation.generate(envelopes([row]))
    parent = generated[ROOT + "/parent-1/README.md"].decode()
    assert (
        "引継ぎ待ち" in parent
        and "公開の担当: 2" in parent
        and "publish-decision-log" in parent
    )
    assert "https://github.com/o/r/pull/77" in parent and "3" in parent
