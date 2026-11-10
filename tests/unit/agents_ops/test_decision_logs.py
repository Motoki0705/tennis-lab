"""Causal, missing-data, privacy and grouping contracts without external IO."""

from __future__ import annotations

import copy
import json
from typing import Any
from uuid import UUID

import pytest

from tests.support.agents_ops.decision_logs import (
    AT,
    PublicationError,
    action,
    aggregation,
    correction,
    decision,
    evidence,
    graph,
    history,
    missing,
    observed,
    projection,
    repetitions,
    result,
    safety,
    schema,
)


def test_initial_agreement_and_action_result_chain() -> None:
    rows = history()
    validated = graph.validate_graph(rows)
    assert len(validated) == 6
    assert rows[0]["data"]["inputs"] == []
    assert rows[0]["data"]["basis"][0]["url"].endswith("#issuecomment-123")
    assert rows[-1]["data"]["state"] == "decided_not_executed"
    report = rows[2]
    assert graph.states(rows)[report["id"]] == {
        "decision": "recorded",
        "parent_confirmation": "unrecorded",
        "final_report": "observed",
    }
    assert rows[3]["data"]["result_ids"] == [report["id"]]
    assert report["data"]["action_id"] == rows[1]["id"]


def test_branches_join_and_explicit_parent_verification() -> None:
    left, right = result(), result()
    branch = decision([left])
    joined = decision([left, right])
    branches = [action(branch), action(joined)]
    check = result()
    check["data"].update(
        observer="parent_verification",
        verifies=[left["id"], right["id"]],
        status="partial",
    )
    rows = [left, right, branch, joined, *branches, check]
    assert len(graph.validate_graph(list(reversed(rows)))) == 7
    assert graph.states(rows)[left["id"]]["parent_confirmation"] == "recorded"
    assert graph.states(rows)[right["id"]]["decision"] == "recorded"
    assert graph.states(rows)[check["id"]]["decision"] == "undecided"


@pytest.mark.parametrize("kind", ["result", "decision", "action", "correction"])
@pytest.mark.parametrize("invalid", ["unknown", "missing", "source_mixup"])
def test_each_kind_rejects_missing_unknown_and_mixed_provenance(
    kind: str, invalid: str
) -> None:
    observation = result()
    judgement = decision([observation])
    records = {
        "result": observation,
        "decision": judgement,
        "action": action(judgement),
        "correction": correction(observation),
    }
    record = records[kind]
    if invalid == "unknown":
        record["data"]["extra"] = "cannot silently ignore"
    elif invalid == "missing":
        field = {
            "result": "observer",
            "decision": "source",
            "action": "state",
            "correction": "source",
        }[kind]
        del record["data"][field]
    else:
        field, value = {
            "result": ("observer", "parent_choice"),
            "decision": (
                "source",
                {**record["data"].get("source", {}), "kind": "worker_report"},
            ),
            "action": ("state", "success"),
            "correction": (
                "source",
                {**record["data"].get("source", {}), "kind": "machine"},
            ),
        }[kind]
        record["data"][field] = value
    with pytest.raises(PublicationError):
        schema.validate_record(record)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), True),
        (("schema_version",), 2),
        (("kind",), []),
        (("scope", "parent"), True),
        (("scope", "delegation", "turn"), False),
        (("scope", "unexpected"), "x"),
        (("time", "occurred_at"), {"status": "unavailable", "value": 0}),
        (
            ("time", "received_at"),
            {"status": "observed", "value": AT, "reason": "extra"},
        ),
        (("time", "recorded_at"), "2026-11-10T00:00:00"),
        (("data", "status"), "completed"),
        (("data", "observer"), ["machine"]),
        (("data", "changes", "code"), "unchanged"),
        (
            ("data", "auxiliary", "elapsed_seconds"),
            observed({"value": float("nan"), "source": evidence()}),
        ),
        (
            ("data", "auxiliary", "wait_seconds"),
            observed({"value": True, "source": evidence()}),
        ),
        (("data", "final_report"), missing("")),
        (
            ("data", "evidence"),
            [
                {
                    "kind": "public",
                    "url": "https://example.com/current",
                    "revision": None,
                    "content_sha256": None,
                }
            ],
        ),
        (("data", "verifies"), ["a" * 32]),
    ],
)
def test_strict_nested_types_and_no_invented_baseline(
    path: tuple[str, ...], value: Any
) -> None:
    record = result()
    target = record
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(PublicationError):
        schema.validate_record(record)


@pytest.mark.parametrize("field", ["schema_version", "observer"])
def test_duplicate_json_keys_are_rejected_even_when_values_match(field: str) -> None:
    record = result()
    raw = json.dumps(record)
    fragment = (
        json.dumps(field)
        + ": "
        + json.dumps(1 if field == "schema_version" else "machine")
    )
    raw = raw.replace(fragment, fragment + ", " + fragment, 1)
    with pytest.raises(PublicationError, match="invalid_json_object"):
        schema.validate_record(raw)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "wrong_kind",
        "self_cycle",
        "cycle",
        "other_parent",
        "duplicate",
        "initial_without_basis",
    ],
)
def test_invalid_reference_graphs_fail(fault: str) -> None:
    first, second = result(), result()
    judgement = decision([first, second])
    rows = [first, second, judgement]
    if fault == "missing":
        judgement["data"]["inputs"].append(schema.new_id())
    elif fault == "wrong_kind":
        judgement["data"]["inputs"] = [judgement["id"]]
    elif fault == "self_cycle":
        first["data"]["previous_result_id"] = first["id"]
    elif fault == "cycle":
        first["data"]["previous_result_id"] = second["id"]
        second["data"]["previous_result_id"] = first["id"]
    elif fault == "other_parent":
        second["scope"]["parent"] = 2
    elif fault == "duplicate":
        rows.append(copy.deepcopy(first))
    else:
        judgement["data"].update(inputs=[], basis=[])
    with pytest.raises(PublicationError):
        graph.validate_graph(rows)


@pytest.mark.parametrize(
    "fault",
    [
        "plan",
        "operation",
        "backwards",
        "fork",
        "false_result",
        "result_before_request",
        "two_starts",
    ],
)
def test_action_histories_reject_false_execution_links(fault: str) -> None:
    rows = history()
    if fault == "plan":
        rows[1]["data"]["plan_id"] = schema.new_id()
    elif fault == "operation":
        rows[1]["data"]["operation"] = "merge"
    elif fault == "backwards":
        rows.append(action(rows[0], previous=rows[3], state="running"))
    elif fault == "fork":
        rows.append(action(rows[0], previous=rows[1], state="unknown"))
    elif fault == "false_result":
        unrelated = result()
        rows.append(unrelated)
        rows[3]["data"]["result_ids"] = [unrelated["id"]]
    elif fault == "result_before_request":
        rows[1]["data"]["state"] = "decided_not_executed"
    else:
        rows.append(action(rows[0]))
    with pytest.raises(PublicationError, match="invalid_log_reference"):
        graph.validate_graph(rows)


def test_corrections_append_references_without_replacing_history() -> None:
    original = result()
    old_bytes = schema.json_bytes(original)
    corrected = result(previous=original)
    corrected["data"]["status"] = "partial"
    note = correction(original, corrected)
    rows = graph.validate_graph([original, corrected, note])
    assert schema.json_bytes(rows[original["id"]]) == old_bytes
    assert rows[note["id"]]["data"]["replacement_id"] == corrected["id"]
    note["data"]["replacement_id"] = decision([original])["id"]
    with pytest.raises(PublicationError):
        graph.validate_graph([original, corrected, note])


def test_missing_final_unverified_and_undecided_are_distinct_from_success() -> None:
    record = result()
    parsed = schema.validate_record(schema.json_bytes(record))
    assert parsed["data"]["final_report"] == {
        "status": "unavailable",
        "reason": "missing_final",
    }
    assert parsed["time"]["occurred_at"]["reason"] == "source_clock_unavailable"
    assert graph.states([record])[record["id"]] == {
        "decision": "undecided",
        "parent_confirmation": "unrecorded",
        "final_report": "unavailable",
    }
    assert all(
        v["status"] == "unavailable" for v in parsed["data"]["auxiliary"].values()
    )
    record["data"]["auxiliary"]["wait_seconds"] = observed(
        {"value": 0, "source": evidence()}
    )
    assert (
        schema.validate_record(record)["data"]["auxiliary"]["wait_seconds"]["value"][
            "value"
        ]
        == 0
    )


def test_observed_action_without_a_decision_does_not_invent_one() -> None:
    first = action(decision(), state="requested")
    first["data"].update(
        decision_id=None, plan_id=None, decision_missing_reason="not_supplied_at_entry"
    )
    report = result(action=first)
    ended = copy.deepcopy(first)
    ended["id"] = schema.new_id()
    ended["data"].update(
        previous_action_id=first["id"], state="ended", result_ids=[report["id"]]
    )
    rows = graph.validate_graph([first, report, ended])
    assert [r["kind"] for r in rows.values()] == ["action", "result", "action"]
    assert graph.states(list(rows.values()))[report["id"]]["decision"] == "undecided"
    first["data"]["decision_missing_reason"] = None
    with pytest.raises(PublicationError):
        schema.validate_record(first)


def test_pre_start_rejection_retains_missing_report_and_unobserved_usage() -> None:
    judgement = decision(operation="start_ai")
    requested = action(judgement)
    rejected = action(judgement, previous=requested, state="rejected_before_start")
    report = result(action=rejected)
    report["data"].update(
        status="rejected",
        summary="Preflight rejected the launch before any AI process started.",
    )
    for key in report["data"]["auxiliary"]:
        report["data"]["auxiliary"][key] = missing("not_started")
    final = action(
        judgement, previous=rejected, state="rejected_before_start", results=[report]
    )
    rows = graph.validate_graph([judgement, requested, rejected, report, final])
    assert all(
        r["data"]["state"] != "running" for r in rows.values() if r["kind"] == "action"
    )
    assert graph.states(list(rows.values()))[report["id"]] == {
        "decision": "undecided",
        "parent_confirmation": "unrecorded",
        "final_report": "unavailable",
    }
    assert rows[report["id"]]["data"]["auxiliary"]["usage"] == missing("not_started")


@pytest.mark.parametrize("state", schema.ACTION_STATES)
def test_action_state_round_trip(state: str) -> None:
    record = action(decision(), state=state)
    assert schema.validate_record(schema.json_bytes(record))["data"]["state"] == state


def test_usage_retains_scope_and_missing_effective_model_without_defaults() -> None:
    record = result()
    record["data"]["auxiliary"].update(
        selected_model=observed(
            {
                "model": "gpt-6.1-sol",
                "effort": "max",
                "specified_by": "settings",
                "source": evidence(),
            }
        ),
        effective_model=observed(
            {
                "model": observed("gpt-6.1-sol"),
                "effort": missing("no_event"),
                "source": evidence(),
            }
        ),
        usage=observed(
            {
                "basis": "cumulative",
                "counter_key": schema.new_id(),
                "input_tokens": observed(12),
                "cached_input_tokens": observed(7),
                "output_tokens": missing("not_reported"),
                "total_tokens": missing("not_reported"),
                "source": evidence(),
            }
        ),
    )
    parsed = schema.validate_record(schema.json_bytes(record))["data"]["auxiliary"]
    assert parsed["effective_model"]["value"]["effort"] == missing("no_event")
    assert parsed["usage"]["value"]["basis"] == "cumulative"
    assert parsed["usage"]["value"]["input_tokens"]["value"] == 12
    assert parsed["usage"]["value"]["cached_input_tokens"]["value"] == 7


def test_public_ids_are_uuid4_and_hierarchy_uses_existing_public_delegation() -> None:
    first, second = schema.new_id(), schema.new_id()
    assert first != second and UUID(hex=first).version == UUID(hex=second).version == 4
    record = schema.validate_record(result())
    assert record["scope"]["delegation"] == {"key": "d-" + "a" * 32, "turn": 1}
    # Direct parent work does not need a fictitious child issue.
    record["scope"].update(issue=None, delegation=None, review_key=schema.new_id())
    schema.validate_record(record)


def test_projection_sanitizes_prose_and_missing_reasons_with_auditable_counts() -> None:
    record = result()
    record["data"]["summary"] = "保存 /home/private/report; private-run-value を受信"
    record["data"]["auxiliary"]["usage"] = missing("診断は~/.private/report")
    public = projection.render(record, safety.Guard(["private-run-value"]))
    assert (
        public["record"]["data"]["summary"]
        == "保存 [local path]; [private identifier] を受信"
    )
    assert (
        public["record"]["data"]["auxiliary"]["usage"]["reason"] == "診断は[local path]"
    )
    assert public["sanitization"] == {
        "version": 2,
        "replacements": 3,
        "local_paths": {"posix": 1, "home": 1},
    }
    assert "/home/private" in record["data"]["summary"]


@pytest.mark.parametrize(
    "fault",
    [
        "secret",
        "uuid",
        "session",
        "control",
        "encoded_url",
        "metadata",
        "evidence_identity",
    ],
)
def test_projection_blocks_unsafe_text_and_metadata(fault: str) -> None:
    record = result()
    value = {
        "secret": "ghp_" + "x" * 30,
        "uuid": "01234567-89ab-4def-8123-456789abcdef",
        "session": "session_id=unknown-private-id",
        "control": "visible\x1b[0m",
        "encoded_url": "https://example.com/?file=%2Fhome%2Fprivate%2Fx",
        "metadata": "token=private-value",
        "evidence_identity": "private-run-value",
    }[fault]
    if fault == "metadata":
        record["data"]["auxiliary"]["effective_model"] = observed(
            {"model": observed(value), "effort": missing(), "source": evidence()}
        )
    elif fault == "evidence_identity":
        record["data"]["evidence"][0]["url"] = "https://example.com/" + value
    else:
        record["data"]["summary"] = value
    with pytest.raises(PublicationError):
        projection.render(record, safety.Guard(["private-run-value"]))


@pytest.mark.parametrize("kind", ["state_check", "fetch_retry"])
def test_unchanged_checks_group_without_erasing_last_failure(kind: str) -> None:
    rows = repetitions(kind)
    prefix: list[dict[str, Any]] = []
    if kind == "fetch_retry":
        judgement = decision(operation="retry_fetch")
        requested = action(judgement)
        for row in rows:
            row["data"]["action_id"] = requested["id"]
        prefix = [judgement, requested]
    original = copy.deepcopy(rows)
    groups = aggregation.aggregate([*prefix, *rows])
    assert [g["count"] for g in groups] == [1] * (len(prefix) + 1) + [2]
    assert groups[-1] == {
        "record_ids": [rows[1]["id"], rows[2]["id"]],
        "count": 2,
        "first_at": "2026-11-10T08:01:00+00:00",
        "last_at": "2026-11-10T08:03:00+00:00",
        "last_record_id": rows[2]["id"],
    }
    assert rows == original and rows[2]["data"]["status"] == "failure"


@pytest.mark.parametrize(
    "boundary",
    [
        "code",
        "requirements",
        "instruction",
        "conditions",
        "verified_scope",
        "unconfirmed",
        "status",
        "target",
        "condition_key",
        "decision",
        "later_branch",
        "new_ai",
        "selected_model",
    ],
)
def test_grouping_preserves_condition_changes_and_decision_ai_boundaries(
    boundary: str,
) -> None:
    rows = repetitions()
    if boundary in schema.CHANGE_FIELDS:
        rows[2]["data"]["changes"][boundary] = "changed"
    elif boundary == "unconfirmed":
        rows[2]["data"]["changes"]["code"] = "unconfirmed"
    elif boundary == "status":
        rows[2]["data"]["status"] = "success"
    elif boundary == "target":
        rows[2]["data"]["repeat"]["target"]["content_sha256"] = "c" * 64
    elif boundary == "condition_key":
        rows[2]["data"]["repeat"]["condition_key"] = schema.new_id()
    elif boundary == "decision":
        rows.insert(2, decision([rows[1]]))
    elif boundary == "later_branch":
        rows.extend([decision([rows[1]]), decision([rows[1]])])
    elif boundary == "new_ai":
        judgement = decision([rows[1]], operation="continue_ai")
        request = action(judgement)
        rows[2]["data"].update(action_id=request["id"], repeat=None)
        rows[2:2] = [judgement, request]
    else:
        rows[2]["data"]["auxiliary"]["selected_model"] = observed(
            {
                "model": "gpt-6-astra",
                "effort": "max",
                "specified_by": "human_instruction",
                "source": evidence(),
            }
        )
    assert all(g["count"] == 1 for g in aggregation.aggregate(rows))


def test_ai_run_cannot_be_disguised_as_an_aggregatable_read() -> None:
    judgement = decision(operation="start_ai")
    request = action(judgement)
    report = result(action=request)
    report["data"]["repeat"] = repetitions()[1]["data"]["repeat"]
    with pytest.raises(PublicationError, match="invalid_log_reference"):
        graph.validate_graph([judgement, request, report])
