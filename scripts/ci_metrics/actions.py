"""Offline, attempt-specific Actions accounting; never infer success from absence."""

from __future__ import annotations

import json
import math
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .runner import write_json


def fetch_attempts(
    repo: str, run_id: int, output: Path, attempt: int | None = None
) -> list[dict[str, Any]]:
    """GET only; retain raw attempt responses and never use latest-attempt jobs."""

    def get(endpoint: str) -> Any:
        return json.loads(subprocess.check_output(["gh", "api", endpoint], text=True))

    output.mkdir(parents=True, exist_ok=True)
    endpoint = f"repos/{repo}/actions/runs/{run_id}"
    latest = get(endpoint)
    records = []
    for number in (
        [attempt] if attempt is not None else range(1, latest["run_attempt"] + 1)
    ):
        run = get(f"{endpoint}/attempts/{number}")
        jobs = []
        page = 1
        while True:
            response = get(
                f"{endpoint}/attempts/{number}/jobs?per_page=100&page={page}"
            )
            jobs.extend(response["jobs"])
            if len(jobs) >= response["total_count"]:
                break
            if not response["jobs"]:
                raise ValueError("jobs pagination ended before total_count")
            page += 1
        records.append(
            {
                "run": run,
                "jobs": jobs,
                "fetched_at": time.time(),
                "source": f"https://api.github.com/{endpoint}/attempts/{number}",
            }
        )
        write_json(output / "actions-raw.json", records)
    return records


def _seconds(start: Any, end: Any) -> float | None:
    if not start or not end:
        return None
    try:
        value = (
            datetime.fromisoformat(end.replace("Z", "+00:00"))
            - datetime.fromisoformat(start.replace("Z", "+00:00"))
        ).total_seconds()
    except (ValueError, TypeError):
        return None
    return value if value >= 0 else None


def _phase(name: str) -> str:
    if name.startswith("Post ") or name == "Complete job":
        return "post"
    if name == "Checkout":
        return "checkout"
    if name.startswith("Set up uv") or name == "Install locked CPU environment":
        return "uv_cache_and_install"
    if name == "Install system dependencies":
        return "apt"
    if name == "Test":
        return "pytest"
    if "Lint" in name:
        return "lint"
    return "other"


def summarize_attempts(
    records: list[dict[str, Any]], checks: list[str]
) -> dict[str, Any]:
    if not records:
        raise ValueError("no attempt records supplied")
    if not checks or len(set(checks)) != len(checks):
        raise ValueError("observed check names must be nonempty and unique")
    attempts = []
    seen: set[tuple[int, int]] = set()
    for record in records:
        run = record["run"]
        jobs = record["jobs"]
        key = (int(run["id"]), int(run["run_attempt"]))
        if key in seen:
            raise ValueError(f"duplicate run/attempt: {key}")
        seen.add(key)
        issues = list(record.get("issues", []))
        if not jobs:
            issues.append("missing jobs")
        if len({job["id"] for job in jobs}) != len(jobs):
            issues.append("duplicate jobs")
        origin = record.get("requested_at") or (
            run.get("created_at") if key[1] == 1 else run.get("run_started_at")
        )
        queue_origin = (
            "explicit requested_at"
            if record.get("requested_at")
            else "created_at"
            if key[1] == 1
            else "run_started_at proxy; retry request time unavailable"
        )
        job_rows = []
        observed: dict[str, list[dict[str, Any]]] = {name: [] for name in checks}
        for job in jobs:
            identity_matches = (
                job.get("run_attempt") == key[1]
                and job.get("run_id") == key[0]
                and bool(run.get("head_sha"))
                and job.get("head_sha") == run["head_sha"]
            )
            if job.get("run_attempt") != key[1]:
                issues.append(f"job attempt mismatch: {job['id']}")
            if job.get("run_id") != key[0]:
                issues.append(f"job run mismatch/missing: {job['id']}")
            if not run.get("head_sha") or job.get("head_sha") != run["head_sha"]:
                issues.append(f"job head mismatch/missing: {job['id']}")
            duration = _seconds(job.get("started_at"), job.get("completed_at"))
            if duration is None and job.get("conclusion") != "skipped":
                issues.append(f"missing/invalid job timestamps: {job['id']}")
            steps = []
            if "steps" not in job:
                issues.append(f"missing job steps: {job['id']}")
            for step in job.get("steps", []):
                seconds = _seconds(step.get("started_at"), step.get("completed_at"))
                if seconds is None and step.get("conclusion") != "skipped":
                    issues.append(
                        f"missing/invalid step timestamps: {job['id']}:{step['name']}"
                    )
                steps.append(
                    {
                        "name": step["name"],
                        "phase": _phase(step["name"]),
                        "status": step.get("status"),
                        "conclusion": step.get("conclusion"),
                        "seconds": seconds,
                    }
                )
            row = {
                "id": job["id"],
                "name": job["name"],
                "status": job.get("status"),
                "conclusion": job.get("conclusion"),
                "identity_matches": identity_matches,
                "runner_s": duration,
                "queue_s": _seconds(job.get("created_at"), job.get("started_at")),
                "completion_from_attempt_origin_s": _seconds(
                    origin, job.get("completed_at")
                )
                if identity_matches
                else None,
                "steps": steps,
            }
            job_rows.append(row)
            if job["name"] in observed:
                observed[job["name"]].append(row)
        for name, matching in observed.items():
            if (
                len(matching) != 1
                or matching[0]["completion_from_attempt_origin_s"] is None
            ):
                issues.append(f"missing/duplicate/incomplete observed check: {name}")
        completion = [
            rows[0]["completion_from_attempt_origin_s"]
            for rows in observed.values()
            if len(rows) == 1
            and rows[0]["completion_from_attempt_origin_s"] is not None
        ]
        runner_observed = sum(
            row["runner_s"] for row in job_rows if row["runner_s"] is not None
        )
        runner_complete = bool(job_rows) and all(
            row["identity_matches"]
            and (row["runner_s"] is not None or row["conclusion"] == "skipped")
            for row in job_rows
        )
        complete = not issues and run.get("status") == "completed"
        successful = (
            complete
            and run.get("conclusion") == "success"
            and all(
                rows[0]["status"] == "completed" and rows[0]["conclusion"] == "success"
                for rows in observed.values()
            )
        )
        starts = [job["started_at"] for job in jobs if job.get("started_at")]
        phase_totals: dict[str, float] = {}
        for row in job_rows:
            for step in row["steps"]:
                if step["seconds"] is not None:
                    phase_totals[step["phase"]] = (
                        phase_totals.get(step["phase"], 0.0) + step["seconds"]
                    )
        attempts.append(
            {
                "run_id": key[0],
                "attempt": key[1],
                "commit": run.get("head_sha"),
                "url": run.get("html_url"),
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "complete_measurement": complete,
                "successful_observed_checks": successful,
                "issues": issues,
                "attempt_origin": origin,
                "queue_origin": queue_origin,
                "dispatch_to_first_job_s": _seconds(origin, min(starts))
                if starts
                else None,
                "observed_check_completion_s": max(completion)
                if len(completion) == len(checks)
                else None,
                "all_jobs_runner_s": runner_observed if runner_complete else None,
                "available_jobs_runner_s": runner_observed,
                "phase_step_s": phase_totals,
                "jobs": job_rows,
                "cache_condition": record.get("cache_condition", "unknown"),
            }
        )
    missing_attempts: list[dict[str, int]] = []
    for run_id in sorted({key[0] for key in seen}):
        latest = max(attempt for ident, attempt in seen if ident == run_id)
        missing_attempts.extend(
            {"run_id": run_id, "attempt": i}
            for i in range(1, latest + 1)
            if (run_id, i) not in seen
        )
    samples = sorted(
        row["observed_check_completion_s"]
        for row in attempts
        if row["successful_observed_checks"]
    )
    return {
        "schema_version": 1,
        "observed_check_names": checks,
        "attempt_count": len(attempts),
        "conclusions": dict(
            Counter(row["conclusion"] or row["status"] or "unknown" for row in attempts)
        ),
        "missing_attempts": missing_attempts,
        "attempts": attempts,
        "successful_completion_distribution": {
            "n": len(samples),
            "median_s": statistics.median(samples) if samples else None,
            "p95_s": samples[math.ceil(0.95 * len(samples)) - 1] if samples else None,
            "p95_method": "nearest rank: sorted[ceil(0.95*n)-1]",
            "p95_is_reference_only": len(samples) < 20,
            "comparison_scope": "descriptive only; do not infer improvement across commits, runners, cache states or test sets",
        },
        "all_attempts_runner_s": sum(row["all_jobs_runner_s"] for row in attempts)
        if not missing_attempts
        and all(row["all_jobs_runner_s"] is not None for row in attempts)
        else None,
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Actions attempt measurements",
        "",
        f"Attempt conclusions: `{summary['conclusions']}`. Missing attempts: `{summary['missing_attempts']}`.",
        f"Successful observed-check completion: `{summary['successful_completion_distribution']}`.",
        "Observed check timing is a measurement, not a merge gate.",
        "",
        "| Run | Attempt | Conclusion | Complete data | Check wait s | All runner s |",
        "|---|---:|---|---|---:|---:|",
    ]
    for item in summary["attempts"]:
        lines.append(
            f"| {item['run_id']} | {item['attempt']} | {item['conclusion']} | {item['complete_measurement']} | {item['observed_check_completion_s']} | {item['all_jobs_runner_s']} |"
        )
    return "\n".join(lines) + "\n"
