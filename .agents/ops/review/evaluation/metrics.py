"""Parent-adjudicated detection and observed resource use, without guessed zeros."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .catalog import digest

TOKENS = ("input_tokens", "cached_input_tokens", "output_tokens")
CONDITIONS = ("astra_fixed", "reuse_fixed", "reuse_select")


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("schema_version") != 1 or row.get("condition") not in CONDITIONS:
            raise ValueError("unsupported comparison observation")
        for key in ("elapsed_seconds", "agent_seconds"):
            duration = row.get(key)
            if duration is not None and (
                type(duration) not in (int, float)
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise ValueError(
                    "duration must be observed finite non-negative seconds or missing"
                )
        key = row["run_id"]
        if key in unique and unique[key] != row:
            raise ValueError("same observation ID has conflicting evidence")
        unique[key] = row
    return list(unique.values())


def usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counters: dict[str, dict[str, Any]] = {}
    owners: dict[str, tuple[str, str]] = {}
    missing: list[str] = []
    totals = {name: 0 for name in TOKENS}
    absent = {name: 0 for name in TOKENS}
    cumulative: list[str] = []
    for row in unique_rows(rows):
        if row["agent_invocations"] == 0:
            if row.get("counter") is not None:
                raise ValueError("a no-AI observation cannot charge source usage again")
            continue
        counter = row.get("counter")
        if counter is None:
            missing.append(row["run_id"])
            continue
        key = counter["key"]
        if key in counters:
            if counters[key] != counter or owners[key] != (
                row["case_id"],
                row["model"],
            ):
                raise ValueError("same source counter has conflicting usage")
            continue
        counters[key] = counter
        owners[key] = (row["case_id"], row["model"])
        if counter["basis"] != "turn":
            cumulative.append(key)
            continue
        for name in TOKENS:
            value = counter.get(name)
            if value is None:
                absent[name] += 1
            elif type(value) is not int or value < 0:
                raise ValueError("invalid observed token count")
            else:
                totals[name] += value
        if (
            counter.get("cached_input_tokens") is not None
            and counter.get("input_tokens") is not None
            and counter["cached_input_tokens"] > counter["input_tokens"]
        ):
            raise ValueError("cached input exceeds total input")
    return {
        "status": "not_started"
        if not rows
        else "partial"
        if missing or cumulative or any(absent.values())
        else "observed",
        "unique_counters": len(counters),
        "missing_run_ids": missing,
        "excluded_non_turn_counters": cumulative,
        **{
            name: {
                "observed_sum": totals[name],
                "missing_counters": absent[name] + len(missing) + len(cumulative),
            }
            for name in TOKENS
        },
        # input already contains cached input. Reasoning is already in output.
    }


def findings(row: dict[str, Any]) -> list[dict[str, Any]]:
    output = row.get("output")
    if output is None:
        return []
    return [*output["prs"][0]["findings"], *output["cross_findings"]]


def candidate_table(
    rows: list[dict[str, Any]], gold: dict[str, Any]
) -> list[dict[str, Any]]:
    """Path overlap is only a candidate, never a match or an automated verdict."""
    truth = {c["case_id"]: c["findings"] for c in gold["cases"]}
    return [
        {
            "run_id": row["run_id"],
            "output_sha256": row["output_sha256"],
            "case_id": row["case_id"],
            "finding": finding,
            "candidate_gold_ids": [
                f["id"]
                for f in truth[row["case_id"]]
                if finding["path"] is not None and finding["path"] == f["path"]
            ],
            "parent_verdict": "pending",
        }
        for row in unique_rows(rows)
        for finding in findings(row)
    ]


def runtime_succeeded(row: dict[str, Any]) -> bool:
    return bool(
        row["agent_invocations"] == 1
        and row["runtime_status"] == "completed"
        or row["agent_invocations"] == 0
        and row["runtime_status"] == "not_invoked"
    )


def judged(
    row: dict[str, Any], expected: list[dict[str, Any]], judgment: dict[str, Any] | None
) -> dict[str, Any]:
    emitted = {finding["id"]: finding for finding in findings(row)}
    truth = {finding["id"]: finding for finding in expected}
    if len(emitted) != len(findings(row)) or len(truth) != len(expected):
        raise ValueError("duplicate finding or gold IDs")
    matched, classified = set(), set()
    false, unknown, new = 0, 0, 0
    if judgment is not None:
        if (
            judgment["output_sha256"] != row["output_sha256"]
            or judgment["adjudicator"] != "parent"
        ):
            raise ValueError(
                "adjudication does not identify this exact output and parent"
            )
        for match in judgment["matches"]:
            if (
                match["finding_id"] not in emitted
                or match["gold_id"] not in truth
                or not match["reason"].strip()
                or not match["evidence"].strip()
            ):
                raise ValueError("invalid parent finding match")
            if match["finding_id"] in classified:
                raise ValueError("duplicate match or multiple gold IDs for one finding")
            matched.add(match["gold_id"])
            classified.add(match["finding_id"])
        for item in judgment["other_findings"]:
            if (
                item["finding_id"] not in emitted
                or item["finding_id"] in classified
                or not item["reason"].strip()
                or not item["evidence"].strip()
            ):
                raise ValueError("conflicting or unsupported parent classification")
            classified.add(item["finding_id"])
            if item["classification"] == "false":
                false += 1
            elif item["classification"] == "unconfirmed":
                unknown += 1
            elif item["classification"] == "valid_new":
                new += 1
            else:
                raise ValueError("unknown parent classification")
        if set(emitted) != classified:
            raise ValueError("parent adjudication leaves unclassified findings")
    exclusion = (
        "runtime_failure_excluded"
        if not runtime_succeeded(row)
        else "format_failure_excluded"
        if row["output_status"] != "valid"
        else None
    )
    if exclusion is not None:
        return {
            "status": exclusion,
            "true_positives": None,
            "false_negatives": None,
            "false_positives": None,
            "unconfirmed_findings": None,
            "additional_valid_findings": None,
            "by_severity": {
                severity: {
                    "expected": sum(f["severity"] == severity for f in expected),
                    "detected": None,
                    "missed": None,
                }
                for severity in ("blocker", "major")
            },
        }
    if judgment is None:
        return {
            "status": "awaiting_parent",
            "true_positives": None,
            "false_negatives": None,
            "false_positives": None,
            "unconfirmed_findings": None,
            "by_severity": {
                severity: {
                    "expected": sum(f["severity"] == severity for f in expected),
                    "detected": None,
                    "missed": None,
                }
                for severity in ("blocker", "major")
            },
        }
    return {
        "status": "parent_adjudicated",
        "true_positives": len(matched),
        "false_negatives": len(truth) - len(matched),
        "false_positives": false,
        "unconfirmed_findings": unknown,
        "additional_valid_findings": new,
        "by_severity": {
            severity: {
                "expected": sum(f["severity"] == severity for f in expected),
                "detected": sum(
                    f["severity"] == severity and f["id"] in matched for f in expected
                ),
                "missed": sum(
                    f["severity"] == severity and f["id"] not in matched
                    for f in expected
                ),
            }
            for severity in ("blocker", "major")
        },
    }


def summarize(
    rows: list[dict[str, Any]],
    gold: dict[str, Any],
    adjudications: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = unique_rows(rows)
    truth = {case["case_id"]: case["findings"] for case in gold["cases"]}
    judgments = {}
    if adjudications is not None:
        if adjudications.get("schema_version") != 1 or adjudications[
            "gold_sha256"
        ] != digest(gold):
            raise ValueError("adjudications reference a different answer key")
        for item in adjudications["judgments"]:
            if item["run_id"] in judgments:
                raise ValueError("duplicate parent adjudication")
            judgments[item["run_id"]] = item
        if set(judgments) - {row["run_id"] for row in rows}:
            raise ValueError("parent adjudicated an unknown observation")
    cases = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row["case_id"] not in truth
            or row["agent_invocations"] not in (0, 1)
            or type(row["agent_invocations"]) is not int
        ):
            raise ValueError("invalid comparison case or invocation count")
        groups[row["condition"]].append(row)
        cases.append(
            {
                "run_id": row["run_id"],
                "case_id": row["case_id"],
                "model": row["model"],
                "condition": row["condition"],
                **judged(row, truth[row["case_id"]], judgments.get(row["run_id"])),
            }
        )
    conditions = {}
    for name in CONDITIONS:
        group = groups[name]
        episodes: dict[str, set[str]] = defaultdict(set)
        for row in group:
            calls = episodes[row["episode_id"]]
            if row["agent_invocations"]:
                calls.add(
                    row["counter"]["key"] if row.get("counter") else row["run_id"]
                )
        eligible = [
            row
            for row in group
            if runtime_succeeded(row) and row["output_status"] == "valid"
        ]
        group_scores = [score for score in cases if score["condition"] == name]
        code_unknown = sum(
            not runtime_succeeded(row)
            or row.get("output") is None
            or any(p["coverage"] != "reviewed" for p in row["output"]["prs"])
            for row in group
        )
        durations = [row.get("elapsed_seconds") for row in group]
        conditions[name] = {
            "status": "measured" if group else "not_started",
            "observations": len(group),
            "quality_eligible_observations": len(eligible),
            "runtime_excluded_observations": sum(
                not runtime_succeeded(row) for row in group
            ),
            "format_excluded_observations": sum(
                runtime_succeeded(row) and row["output_status"] != "valid"
                for row in group
            ),
            "agent_invocations": sum(map(len, episodes.values())),
            "additional_invocations": sum(
                max(len(calls) - 1, 0) for calls in episodes.values()
            ),
            "additional_launch_rate": ratio(
                sum(len(calls) > 1 for calls in episodes.values()), len(episodes)
            ),
            "format_failures": sum(
                row["output_status"] in {"invalid", "missing"} for row in group
            ),
            "runtime_failures": sum(
                row["agent_invocations"] == 1 and row["runtime_status"] != "completed"
                for row in group
            ),
            "failed_observations": sum(
                row["output_status"] != "valid"
                or row["runtime_status"] not in {"completed", "not_invoked"}
                for row in group
            ),
            "code_unconfirmed_rate": ratio(code_unknown, len(group)),
            "evidence_waits": sum(
                len(p.get("evidence_waits", []))
                for row in group
                if row.get("output")
                for p in row["output"]["prs"]
            ),
            "elapsed_seconds": {
                "observed_sum": sum(t for t in durations if t is not None),
                "missing": sum(t is None for t in durations),
            },
            "agent_seconds": {
                "observed_sum": sum(
                    row["agent_seconds"]
                    for row in {
                        row["counter"]["key"]
                        if row.get("counter")
                        else row["run_id"]: row
                        for row in group
                        if row["agent_invocations"]
                    }.values()
                    if row.get("agent_seconds") is not None
                ),
                "missing": sum(
                    row.get("agent_seconds") is None
                    for row in group
                    if row["agent_invocations"]
                ),
            },
            "usage": usage(group),
            "detection": {
                key: {
                    "confirmed_sum": sum(
                        score[key]
                        for score in cases
                        if score["condition"] == name and score[key] is not None
                    ),
                    "pending_runs": sum(
                        score["status"] == "awaiting_parent"
                        for score in cases
                        if score["condition"] == name
                    ),
                }
                for key in ("true_positives", "false_negatives", "false_positives")
            },
            "unconfirmed_finding_rate": ratio(
                sum(
                    score["unconfirmed_findings"]
                    for score in group_scores
                    if score["unconfirmed_findings"] is not None
                ),
                sum(len(findings(row)) for row in eligible),
            )
            if not any(score["status"] == "awaiting_parent" for score in group_scores)
            else None,
        }
    return {
        "schema_version": 1,
        "status": "not_started"
        if not rows
        else "awaiting_parent"
        if any(c["status"] == "awaiting_parent" for c in cases)
        else "incomplete"
        if any(
            c["status"] in {"runtime_failure_excluded", "format_failure_excluded"}
            for c in cases
        )
        else "scored",
        "conditions": conditions,
        "detection": cases,
        "physical_usage": usage(rows),
        "candidate_matches": candidate_table(rows, gold),
        "default_policy_change_authorized": False,
    }
