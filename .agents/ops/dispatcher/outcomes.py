"""Read the judgement result contract before scheduling or reserving work."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from parent.policy import nonempty, number, object_value

from .judgement import RESULT_VERSION
from .runtime import load

RESULTS = {"recorded", "deferred", "preview", "failed"}


def validate_result(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = object_value(value)
    if (
        set(result)
        != {
            "schema_version",
            "result",
            "input_sha256",
            "decisions",
            "deferred",
            "candidates",
            "publications",
            "diagnostics",
        }
        or type(result["schema_version"]) is not int
        or result["schema_version"] != RESULT_VERSION
        or result["result"] not in RESULTS
        or any(
            not isinstance(result[key], list)
            for key in (
                "decisions",
                "deferred",
                "candidates",
                "publications",
                "diagnostics",
            )
        )
    ):
        raise ValueError("unknown or malformed judgement result contract")
    deferred = []
    for raw in result["deferred"]:
        item = object_value(raw)
        if set(item) != {"issue", "stage", "reason"} or item["stage"] not in {
            "preflight",
            "publication",
            "final_check",
        }:
            raise ValueError("invalid deferred subject")
        deferred.append(number(item["issue"]))
        nonempty(item["reason"], "deferred reason")
    if deferred != sorted(set(deferred)):
        raise ValueError("duplicate or unordered deferred subjects")
    decisions = {number(d["issue"]): d for d in result["decisions"]}
    if len(decisions) != len(result["decisions"]) or set(decisions) & set(deferred):
        raise ValueError("duplicate or deferred judgement decisions")
    if result["result"] != "recorded" and result["candidates"]:
        raise ValueError("only recorded judgement results may carry candidates")
    if result["result"] == "deferred" and (not deferred or decisions):
        raise ValueError("deferred result requires only deferred subjects")
    for candidate in result["candidates"]:
        if (
            decisions.get(number(candidate["parent"]), {}).get("disposition")
            != "startable"
        ):
            raise ValueError("candidate must have a fresh startable decision")
    return result


def read_result(directory: Path) -> dict[str, Any]:
    return validate_result(load(directory / "result.json"))
