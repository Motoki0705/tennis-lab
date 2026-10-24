"""Conservative failure signatures and evidence-based classification hypotheses."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from .store import encode

ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
TIMESTAMP = re.compile(r"^\d{4}-\d\d-\d\dT\S+\s+")
TEST = re.compile(r"\b(?:FAILED|ERROR)\s+(.+?\.py(?:::.+?)?)(?:\s+-\s+(.*))?$")
ERROR = re.compile(
    r"(?:\b(?:\w*Error|\w*Exception):|##\[error\]|\berror:|\bE\s+assert\b)", re.I
)
DEPENDENCY_STEP = re.compile(
    r"install|setup|set up|checkout|fetch|dependenc|download", re.I
)
NETWORK = re.compile(
    r"could not resolve host|temporary failure in name resolution|failed to download|"
    r"failed to fetch|connection (?:timed out|reset)|503 service unavailable|502 bad gateway|"
    r"no space left on device",
    re.I,
)
RUNNER = re.compile(
    r"runner (?:has lost communication|has received a shutdown signal)|lost communication with the (?:server|runner)",
    re.I,
)


def normalize(value: str) -> str:
    value = re.sub(r"0x[0-9a-fA-F]+", "0x<address>", value)
    value = re.sub(r"(?<=\.py):\d+(?::\d+)?", ":<line>", value)
    return " ".join(value.split())


def signatures(
    workflow: str, job: dict[str, Any], log: str | None, source: dict[str, str]
) -> list[dict[str, Any]]:
    lines = [
        TIMESTAMP.sub("", ANSI.sub("", line)).strip()
        for line in (log or "").splitlines()
    ]
    steps = [
        s["name"]
        for s in job["steps"]
        if s["conclusion"] not in (None, "success", "skipped", "neutral")
    ]
    infra = []
    for index, line in enumerate(lines, 1):
        if RUNNER.search(line) or (
            any(DEPENDENCY_STEP.search(step) for step in steps) and NETWORK.search(line)
        ):
            infra.append({"source": source, "line": index, "excerpt": line})
    found: dict[str, dict[str, Any]] = {}

    def add(
        test_id: str | None, summary: str, line: int | None, specific: bool
    ) -> None:
        identity = {
            "workflow": workflow,
            "job": job["name"],
            "test_id": test_id,
            "error": normalize(summary),
        }
        identifier = (
            "ci-failure-v1:" + hashlib.sha256(encode(identity).encode()).hexdigest()
        )
        evidence: dict[str, Any] = {
            "source": source,
            "line": line,
            "excerpt": summary,
            "kind": "log" if line is not None else "job_result",
        }
        if identifier in found:
            return
        found[identifier] = {
            "signature": identifier,
            **identity,
            "specific": specific,
            "failed_steps": steps,
            "evidence": [evidence],
            "infrastructure_evidence": infra,
        }

    for index, line in enumerate(lines, 1):
        if match := TEST.search(line):
            add(match[1], match[2] or "pytest failure (no summary)", index, True)
    if not found:
        candidates = [
            (i, line)
            for i, line in enumerate(lines, 1)
            if (ERROR.search(line) or NETWORK.search(line) or RUNNER.search(line))
            and "Process completed with exit code" not in line
        ]
        if candidates:
            index, summary = candidates[-1]
            add(
                None,
                summary,
                index,
                bool(re.search(r"\S+\.py|\b[A-Z]+\d{3}\b", summary)),
            )
        else:
            add(
                None,
                f"{job['conclusion']}: {', '.join(steps) or 'no failed step reported'}",
                None,
                False,
            )
    return list(found.values())


def occurrences(commit: dict[str, Any]) -> list[dict[str, Any]]:
    failures = []
    for workflow in commit["fact"]["workflows"]:
        for run in workflow["runs"]:
            for job in run["jobs"]:
                for failure in job["failures"]:
                    failures.append(
                        {
                            **failure,
                            "sha": commit["sha"],
                            "run_id": run["run_id"],
                            "attempt": run["attempt"],
                            "job_id": job["id"],
                            "job_url": job["url"],
                            "job_conclusion": job["conclusion"],
                        }
                    )
        failures.extend(workflow["failures"])
    return failures


def retry_successes(
    commit: dict[str, Any], occurrence: dict[str, Any]
) -> list[dict[str, Any]]:
    recovered = []
    for workflow in commit["fact"]["workflows"]:
        if workflow["workflow"] != occurrence["workflow"]:
            continue
        for run in workflow["runs"]:
            if (
                run["run_id"] != occurrence["run_id"]
                or run["attempt"] <= occurrence["attempt"]
            ):
                continue
            for job in run["jobs"]:
                if (
                    job["name"] == occurrence["job"]
                    and job["status"] == "completed"
                    and job["conclusion"] == "success"
                ):
                    recovered.append(
                        {
                            "sha": commit["sha"],
                            "run_id": run["run_id"],
                            "attempt": run["attempt"],
                            "job_id": job["id"],
                            "source": job["source"],
                        }
                    )
    return recovered


def classify(commits: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_sha = {c["sha"]: occurrences(c) for c in commits}
    results = {}
    for index, commit in enumerate(commits):
        failures = []
        for occurrence in by_sha[commit["sha"]]:
            category = "unclassified"
            confidence = "low"
            reasons: list[dict[str, Any]] = [{"rule": "insufficient_evidence"}]
            infrastructure = occurrence["infrastructure_evidence"]
            recovered = retry_successes(commit, occurrence)
            adjacent = []
            counterevidence = []
            for neighbor_index in (index - 1, index + 1):
                if not 0 <= neighbor_index < len(commits):
                    continue
                neighbor = commits[neighbor_index]
                for other in by_sha[neighbor["sha"]]:
                    if (
                        other["signature"] == occurrence["signature"]
                        and not other["infrastructure_evidence"]
                        and other["job_conclusion"] == "failure"
                    ):
                        if success := retry_successes(neighbor, other):
                            counterevidence.extend(success)
                            continue
                        adjacent.append(
                            {
                                "sha": other["sha"],
                                "run_id": other["run_id"],
                                "attempt": other["attempt"],
                                "job_id": other["job_id"],
                                "evidence": other["evidence"],
                            }
                        )
            if infrastructure:
                category, confidence = "runner_infrastructure", "high"
                reasons = [
                    {
                        "rule": "explicit_runner_or_failed_dependency_step",
                        "evidence": infrastructure,
                    }
                ]
            elif recovered:
                category, confidence = "flaky", "medium"
                reasons = [
                    {
                        "rule": "same_sha_same_run_later_attempt_job_success",
                        "evidence": recovered,
                    }
                ]
            elif (
                adjacent
                and not counterevidence
                and occurrence["specific"]
                and occurrence["job_conclusion"] == "failure"
            ):
                category, confidence = "code_regression", "medium"
                reasons = [
                    {
                        "rule": "same_specific_signature_on_adjacent_main_shas",
                        "evidence": adjacent,
                    }
                ]
            elif counterevidence:
                reasons = [
                    {
                        "rule": "adjacent_signature_has_contradictory_retry_evidence",
                        "evidence": counterevidence,
                    }
                ]
            failures.append(
                {
                    **occurrence,
                    "classification": {
                        "category": category,
                        "confidence": confidence,
                        "assessment": "hypothesis",
                        "reasons": reasons,
                    },
                }
            )
        results[commit["sha"]] = failures
    return results
