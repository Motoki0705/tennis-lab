"""Repair final reports are a strict protocol consumed by the pipeline."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops"))
repair = importlib.import_module("lib.repair_turn")


def report() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "repair_id": "ci-attempt-1",
        "pr": 42,
        "outcome": "repaired",
        "head_sha": "a" * 40,
        "summary": "Corrected the failing import and pushed the fix.",
        "verification": ["Focused tests passed. CI is pending."],
        "parent_request": None,
    }


@pytest.mark.parametrize("outcome", ["repaired", "needs_parent"])
def test_repair_report_outcomes(outcome: str) -> None:
    value = report()
    value.update(
        outcome=outcome,
        parent_request="Resolve the dependency ordering."
        if outcome == "needs_parent"
        else None,
    )
    assert (
        repair.parse_report(json.dumps(value), repair_id="ci-attempt-1", pr=42) == value
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing",
        "outcome",
        "repair_id",
        "pr",
        "bool_pr",
        "float_version",
        "sha",
        "sha_newline",
        "summary",
        "verification",
        "parent_request",
        "needs_parent",
        "prose",
        "fence",
        "duplicate",
        "nan",
    ],
)
def test_malformed_repair_report_is_an_error(mutation: str) -> None:
    value = report()
    changes: dict[str, Any] = {
        "outcome": "success",
        "repair_id": "other",
        "pr": 43,
        "bool_pr": True,
        "float_version": 1.0,
        "sha": "abc",
        "sha_newline": "a" * 40 + "\n",
        "summary": " ",
        "verification": [False],
        "parent_request": "unexpected",
        "needs_parent": "needs_parent",
    }
    if mutation == "extra":
        value["extra"] = True
    elif mutation == "missing":
        del value["verification"]
    elif mutation in changes:
        key = {
            "bool_pr": "pr",
            "float_version": "schema_version",
            "sha": "head_sha",
            "sha_newline": "head_sha",
            "needs_parent": "outcome",
        }.get(mutation, mutation)
        value[key] = changes[mutation]
    text = json.dumps(value)
    if mutation == "prose":
        text = "repaired and pushed successfully"
    elif mutation == "fence":
        text = "```json\n" + text + "\n```"
    elif mutation == "duplicate":
        text = text.replace('"pr": 42', '"pr": 42, "pr": 42')
    elif mutation == "nan":
        text = text.replace('"schema_version": 1', '"schema_version": NaN')
    with pytest.raises(ValueError, match="invalid repair final report"):
        repair.parse_report(text, repair_id="ci-attempt-1", pr=42)
