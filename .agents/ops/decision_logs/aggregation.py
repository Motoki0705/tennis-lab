"""Lossless display groups. Raw journal records are never coalesced."""

from __future__ import annotations

from typing import Any

from .graph import validate_graph


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group adjacent unchanged machine checks/retrievals, preserving boundaries.

    The caller assigns a random condition key only to identical conditions.
    A decision, action, correction, condition/target/status change or AI run is
    always a boundary. No usage counters or missing durations are added up.
    """
    validate_graph(records)
    groups: list[dict[str, Any]] = []

    def compatible(before: dict[str, Any], after: dict[str, Any]) -> bool:
        if before["kind"] != "result" or after["kind"] != "result":
            return False
        left, right = before["data"], after["data"]
        return (
            left["repeat"] is not None
            and left["repeat"] == right["repeat"]
            and before["scope"] == after["scope"]
            and left["observer"] == right["observer"] == "machine"
            and left["status"] == right["status"]
            and right["previous_result_id"] == before["id"]
            and all(
                v == "unchanged" for r in (left, right) for v in r["changes"].values()
            )
            and left["action_id"] == right["action_id"]
            and left["verifies"] == right["verifies"] == []
            and left["final_report"] == right["final_report"]
            and all(
                left["auxiliary"][k] == right["auxiliary"][k]
                for k in ("selected_model", "effective_model")
            )
        )

    previous = None
    for record in records:
        if previous is not None and compatible(previous, record):
            group = groups[-1]
            group["record_ids"].append(record["id"])
            group["count"] += 1
            group["last_at"] = record["time"]["recorded_at"]
            group["last_record_id"] = record["id"]
        else:
            groups.append(
                {
                    "record_ids": [record["id"]],
                    "count": 1,
                    "first_at": record["time"]["recorded_at"],
                    "last_at": record["time"]["recorded_at"],
                    "last_record_id": record["id"],
                }
            )
        previous = record
    return groups
