"""Parse the final line of an ordinary implementation worker's report."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from lib.agent_events import parse_json

PREFIX = "PARENT_RESULT"
HEAD = re.compile(r"[0-9a-f]{40}")
JOB = re.compile(r"(?:0-[0-9]{19}_[0-9TZ-]+_[0-9]+|[0-9]+_[0-9]+)_[A-Za-z0-9._-]+\.job")
LOCAL_JOB = re.compile(r"[0-9]{8}T[0-9]{6}Z_[0-9a-f]{12}_[A-Za-z0-9._-]+")
MAX_LOCAL_JOB_LEN = 200


def is_valid_local_job_id(job: Any) -> bool:
    return (
        isinstance(job, str)
        and len(job) <= MAX_LOCAL_JOB_LEN
        and ".." not in job
        and bool(LOCAL_JOB.fullmatch(job))
    )


def parse_report(text: str) -> dict[str, Any]:
    lines = text.rstrip().splitlines()
    marked = [i for i, line in enumerate(lines) if line.lstrip().startswith(PREFIX)]
    if len(marked) != 1:
        raise ValueError(f"{PREFIX}: expected exactly one declaration")
    if marked[0] != len(lines) - 1 or not lines[-1].startswith(PREFIX + " "):
        raise ValueError(f"{PREFIX}: declaration must be the final unindented line")
    try:
        report = parse_json(lines[-1][len(PREFIX) + 1 :])
    except (RecursionError, ValueError) as exc:
        raise ValueError(f"{PREFIX}: invalid JSON") from exc
    if (
        not isinstance(report, dict)
        or set(report) != {"schema_version", "state", "wait"}
        or type(report["schema_version"]) is not int
        or report["schema_version"] != 1
    ):
        raise ValueError(f"{PREFIX}: invalid fields or schema_version")
    state, wait = report["state"], report["wait"]
    if state in ("done", "needs_parent") and wait is None:
        return report
    if state != "waiting" or not isinstance(wait, dict):
        raise ValueError(f"{PREFIX}: invalid state or wait")
    if wait.get("kind") == "ci":
        _validate_ci(wait)
    elif wait.get("kind") == "queue":
        _validate_job_ids(
            wait,
            lambda job: isinstance(job, str) and bool(JOB.fullmatch(job)),
            "queue requires unique training-queue job_ids",
        )
    elif wait.get("kind") == "local":
        _validate_job_ids(
            wait,
            is_valid_local_job_id,
            "local requires unique local job_ids",
        )
    else:
        raise ValueError(f"{PREFIX}: unknown wait kind")
    return report


def _validate_job_ids(
    wait: dict[str, Any], valid_id: Callable[[Any], bool], message: str
) -> None:
    jobs = wait.get("job_ids")
    if (
        set(wait) != {"kind", "job_ids"}
        or not isinstance(jobs, list)
        or not jobs
        or any(not valid_id(job) for job in jobs)
        or len(set(jobs)) != len(jobs)
    ):
        raise ValueError(f"{PREFIX}: {message}")


def _validate_ci(wait: dict[str, Any]) -> None:
    expected = {"kind", "pr", "head", "resume_on_success"}
    if wait.get("resume_on_success") is True:
        expected.add("next_step")
    if (
        set(wait) != expected
        or type(wait["pr"]) is not int
        or wait["pr"] < 1
        or not isinstance(wait["head"], str)
        or not HEAD.fullmatch(wait["head"])
        or type(wait["resume_on_success"]) is not bool
        or (
            wait["resume_on_success"]
            and (
                not isinstance(wait["next_step"], str) or not wait["next_step"].strip()
            )
        )
    ):
        raise ValueError(f"{PREFIX}: invalid CI target or continuation")
