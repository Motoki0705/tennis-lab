"""Keep measured qualification, reuse replay, and policy projections distinct."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from worker_logs.journal import atomic, locked

from .catalog import DATA, MODELS, catalog, read
from .execution import forecast, load, observations
from .metrics import summarize


def configure_selection(
    state: Path, specification: dict[str, Any], *, data: Path = DATA
) -> None:
    expected = {case["case_id"] for case in catalog(data / "manifest.json")["cases"]}
    if (
        set(specification) != {"schema_version", "source", "choices"}
        or specification["schema_version"] != 1
        or not specification["source"].strip()
    ):
        raise ValueError("selection needs its explicit parent source")
    choices = specification["choices"]
    if len(choices) != len(expected) or {row["case_id"] for row in choices} != expected:
        raise ValueError("selection must cover exactly the approved seven cases")
    if any(
        set(row) != {"case_id", "model", "reason"}
        or row["model"] not in MODELS
        or not row["reason"].strip()
        for row in choices
    ):
        raise ValueError("selection needs allowed models and a reason per case")
    with locked(state / "evaluation.json"):
        value = load(state, data=data)
        if value["selection"] == specification:
            return
        if value["runs"] or value["selection"] is not None:
            raise ValueError(
                "selection is frozen before the first comparison; no post-hoc choice"
            )
        value["selection"] = specification
        atomic(state / "evaluation.json", value)


def parent_subset(
    adjudications: dict[str, Any] | None, rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if adjudications is None:
        return None
    identifiers = {row["run_id"] for row in rows}
    return {
        **adjudications,
        "judgments": [
            judgment
            for judgment in adjudications["judgments"]
            if judgment["run_id"] in identifiers
        ],
    }


def report(
    state: Path, adjudications: dict[str, Any] | None = None, *, data: Path = DATA
) -> dict[str, Any]:
    value = load(state, data=data)
    rows = observations(state, value)
    gold = read(data / "gold.json")
    qualification = summarize(rows, gold, adjudications)
    baseline = [
        {**row, "condition": "astra_fixed"}
        for row in rows
        if row["role"] == "paired" and row["model"] == "gpt-6-astra"
    ]
    selected = []
    if value["selection"]:
        choices = {
            row["case_id"]: row["model"] for row in value["selection"]["choices"]
        }
        selected = [
            {**row, "condition": "reuse_select", "episode_id": row["case_id"]}
            for row in rows
            if row["role"] == "additional"
            or row["role"] == "paired"
            and row["model"] == choices[row["case_id"]]
        ]
    projection = summarize(selected, gold, parent_subset(adjudications, selected))
    reference = summarize(baseline, gold, parent_subset(adjudications, baseline))
    base_cost = reference["conditions"]["astra_fixed"]
    selected_cost = projection["conditions"]["reuse_select"]
    complete = len(baseline) == len(gold["cases"]) and len(
        {row["case_id"] for row in selected if row["role"] == "paired"}
    ) == len(gold["cases"])
    improvements: dict[str, float | None] = {
        "processed_input_plus_output": None,
        "agent_seconds": None,
    }
    if complete:
        if (
            base_cost["usage"]["status"]
            == selected_cost["usage"]["status"]
            == "observed"
        ):
            before = sum(
                base_cost["usage"][key]["observed_sum"]
                for key in ("input_tokens", "output_tokens")
            )
            after = sum(
                selected_cost["usage"][key]["observed_sum"]
                for key in ("input_tokens", "output_tokens")
            )
            improvements["processed_input_plus_output"] = (
                1 - after / before if before else None
            )
        if (
            base_cost["agent_seconds"]["missing"]
            == selected_cost["agent_seconds"]["missing"]
            == 0
        ):
            before = base_cost["agent_seconds"]["observed_sum"]
            improvements["agent_seconds"] = (
                1 - selected_cost["agent_seconds"]["observed_sum"] / before
                if before
                else None
            )
    return {
        "schema_version": 1,
        "scope": "private paired model qualification; no new formal review or publication is represented",
        "qualification": qualification,
        "execution_evidence": [
            {
                key: row[key]
                for key in (
                    "run_id",
                    "case_id",
                    "model",
                    "role",
                    "runtime_status",
                    "output_status",
                    "runtime_settings",
                    "runtime_sha256",
                    "output_sha256",
                    "counter",
                    "elapsed_seconds",
                    "agent_seconds",
                    "proof_error",
                )
            }
            for row in rows
        ],
        "budget": forecast(value, rows, data=data),
        "conditions": {
            "astra_fixed": {
                "scope": "measured fresh Astra samples",
                "results": reference,
            },
            "reuse_fixed": {
                "scope": "quality references the same fresh Astra results; no fresh LLM required to replay reuse/recovery",
                "quality_source_run_ids": [row["run_id"] for row in baseline],
                "historical_evidence_report": "cases/historical_reuse.json",
                "new_ai_invocations_in_replay": 0,
                "runtime_latency_saved_in_this_comparison": None,
            },
            "reuse_select": {
                "scope": "counterfactual selection from paired samples, including all actual additional calls; not additive to physical usage",
                "selection": value["selection"],
                "results": projection,
                "requires_parent_joint_adjudication": any(
                    row["role"] == "additional" for row in selected
                ),
            },
        },
        "adoption": {
            "paired_matrix_complete": complete,
            "selection_improvements": improvements,
            "minimum_improvement": value["protocol"]["minimum_improvement"],
            "default_policy_change_authorized": False,
            "formal_acceptance_completed": False,
            "decision": "parent_required",
        },
    }
