"""Broad-review output validation and read-only compatibility for old records."""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops"))
policy = importlib.import_module("lib.ops_review.parent_review_policy")
marker, parse_output, records = policy.marker, policy.parse_output, policy.records


def output() -> dict:
    return {
        "verdict": "major",
        "summary": "Fix required",
        "major": [
            {
                "prs": [11, 12],
                "title": "Incorrect result",
                "detail": "Static analysis shows an invalid result",
            }
        ],
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "keys",
        "verdict",
        "empty",
        "unknown_pr",
        "duplicate_pr",
        "boolean_pr",
        "empty_detail",
        "finding_keys",
    ],
)
def test_invalid_or_inconsistent_results_cannot_approve(mutation: str) -> None:
    value = output()
    if mutation == "keys":
        value["coverage"] = "reviewed"
    elif mutation == "verdict":
        value["verdict"] = "approve"
    elif mutation == "empty":
        value["major"] = []
    elif mutation == "unknown_pr":
        value["major"][0]["prs"] = [13]
    elif mutation == "duplicate_pr":
        value["major"][0]["prs"] = [11, 11]
    elif mutation == "boolean_pr":
        value["major"][0]["prs"] = [11, True]
    elif mutation == "empty_detail":
        value["major"][0]["detail"] = " "
    else:
        value["major"][0]["severity"] = "minor"
    with pytest.raises(ValueError):
        parse_output(json.dumps(value), [11, 12])


def test_output_has_only_approve_or_major_and_rejects_duplicate_keys() -> None:
    assert parse_output(json.dumps(output()), [11, 12]) == output()
    value = {"verdict": "approve", "summary": "Reviewed", "major": []}
    assert parse_output(json.dumps(value), [11, 12]) == value
    with pytest.raises(ValueError, match="duplicate"):
        parse_output(
            '{"verdict":"major","verdict":"approve","summary":"x","major":[]}', [11]
        )


def test_cross_finding_keeps_reviewed_prs_and_deduplicates_references() -> None:
    value = output()
    value["major"][0]["prs"] = [13, 12, 15, 13, 11, 15]
    finding = parse_output(json.dumps(value), [11, 12])["major"][0]
    assert finding == {**output()["major"][0], "prs": [12, 11], "related_prs": [13, 15]}


@pytest.mark.parametrize(
    "kind", ["result", "adoption", "pending", "pr", "resolution", "evidence"]
)
def test_historical_records_are_display_only(kind: str) -> None:
    value = {
        "schema_version": 1,
        "kind": kind,
        "review_run_id": "legacy",
        "parent_issue": 1,
        "request_sha256": "historical value retained",
        "coverage": "reviewed",
    }
    before = copy.deepcopy(value)
    comment = {"user": {"login": "operator"}, "body": marker(value)}
    assert records([comment], "operator") == [before]
    assert records([comment], "other") == []
    assert (
        records([{**comment, "body": "quoted\n" + comment["body"]}], "operator") == []
    )
    assert value == before


def test_new_schema_is_strict_and_has_no_obsolete_evidence_fields() -> None:
    path = Path(__file__).resolve().parents[3] / ".agents/ops/review/result_schema.json"
    schema = json.loads(path.read_text())
    assert set(schema["properties"]) == {"verdict", "summary", "major"}
    assert schema["additionalProperties"] is False
    assert schema["properties"]["major"]["items"]["additionalProperties"] is False


def test_legacy_status_displays_archived_order_without_digest_gate() -> None:
    parent_policy = importlib.import_module("parent.policy")
    request = {"review_run_id": "legacy", "schema_version": 1}
    history = [
        {
            "event": "review",
            "review_run_id": "legacy",
            "state": "complete",
            "request_sha256": "archived-digest",
        },
        {"event": "review", "review_run_id": "legacy", "state": "result_received"},
        {"event": "review", "review_run_id": "other", "state": "instructed"},
    ]
    before = copy.deepcopy(history)
    assert parent_policy.legacy_review_state(history, request) == "result_received"
    assert history == before
