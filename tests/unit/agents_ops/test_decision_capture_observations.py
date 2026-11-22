"""Observed settings and counters stay independent of selected/default values."""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from tests.support.agents_ops.decision_logs import schema

observations = importlib.import_module("decision_logs.observations")


@pytest.mark.parametrize("basis", ["turn", "cumulative"])
def test_runner_counters_preserve_zero_missing_and_basis(basis: str) -> None:
    auxiliary = observations.runtime_auxiliary(
        {
            "contract": {
                "settings": {"selected": {"model": "chosen-model", "effort": "max"}}
            },
            "runtime": {
                "effective": {
                    "model": {"status": "unavailable", "reason": "no transcript"}
                },
                "usage": {
                    "status": "observed",
                    "source": "result",
                    "basis": basis,
                    "value": {
                        "input_tokens": 13,
                        "cached_input_tokens": 0,
                        "output_tokens": 7,
                    },
                },
            },
        },
        schema.new_id(),
    )
    schema.auxiliary(auxiliary)
    assert auxiliary["selected_model"]["value"]["model"] == "chosen-model"
    effective = auxiliary["effective_model"]["value"]
    assert effective["model"] == {"status": "unavailable", "reason": "no transcript"}
    assert effective["effort"]["status"] == "unavailable"
    usage = auxiliary["usage"]["value"]
    assert usage["basis"] == basis
    assert usage["input_tokens"]["value"] == 13
    assert usage["cached_input_tokens"] == {"status": "observed", "value": 0}
    assert usage["output_tokens"]["value"] == 7
    assert usage["total_tokens"]["status"] == "unavailable"
    assert auxiliary["elapsed_seconds"]["status"] == "unavailable"


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"status": "unavailable", "reason": "runtime supplied none"},
        {
            "status": "observed",
            "source": "unknown counter scope",
            "value": {"input_tokens": 50},
        },
    ],
)
def test_missing_or_unscoped_usage_is_not_an_observed_zero(usage: Any) -> None:
    value = {"runtime": {"usage": usage}} if usage is not None else {}
    auxiliary = observations.runtime_auxiliary(value, schema.new_id())
    schema.auxiliary(auxiliary)
    assert auxiliary["usage"]["status"] == "unavailable"
    assert auxiliary["effective_model"]["status"] == "unavailable"


def test_duration_requires_two_source_times_and_preserves_zero() -> None:
    result = observations.runtime_auxiliary(
        {"started_at": 12.5, "finished_at": 12.5}, schema.new_id()
    )
    assert result["elapsed_seconds"]["value"]["value"] == 0
    for value in (
        {"started_at": 12.5},
        {"finished_at": 14.0},
        {"started_at": True, "finished_at": 2.0},
    ):
        assert (
            observations.runtime_auxiliary(value, schema.new_id())["elapsed_seconds"][
                "status"
            ]
            == "unavailable"
        )


def test_a_materialized_runner_flag_does_not_invent_a_parent_choice() -> None:
    value = {
        "contract": {
            "settings": {
                "selected": {"model": "fixed-model", "effort": "max"},
                "overrides": {"model": "fixed-model"},
            }
        }
    }
    saved_setting = observations.runtime_auxiliary(value, schema.new_id())
    assert saved_setting["selected_model"]["value"]["specified_by"] == "settings"
    explicit_choice = observations.runtime_auxiliary(
        value, schema.new_id(), "parent_choice"
    )
    assert explicit_choice["selected_model"]["value"]["specified_by"] == "parent_choice"
