"""Pin main, discover its first-parent chain, and refresh every observed SHA."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from . import classify
from .github import FetchError, GitHub, positive_int, required, sha
from .store import Store

SUCCESS = ("success", "skipped", "neutral")
EVENTS = ("push", "workflow_dispatch", "schedule")


def discover(
    api: GitHub, config: dict[str, Any], saved: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    endpoint = f"{api.prefix}/git/ref/heads/main"
    ref, _ = api.json(endpoint)
    try:
        required(ref, "ref", "object")
        if ref["ref"] != "refs/heads/main" or ref["object"]["type"] != "commit":
            raise ValueError("main ref did not identify a commit")
        head = sha(ref["object"]["sha"])
        boundary = saved[-1]["sha"] if saved else config["start_sha"]
        current = head
        commits = []
        seen: set[str] = set()
        while not saved or current != boundary:
            if current in seen:
                raise ValueError("cycle in main ancestry")
            seen.add(current)
            endpoint = f"{api.prefix}/commits/{current}"
            row, source = api.json(endpoint)
            required(row, "sha", "parents", "html_url", "commit")
            if sha(row["sha"]) != current or not isinstance(row["parents"], list):
                raise ValueError("commit identity or parent list mismatch")
            parent = sha(row["parents"][0]["sha"]) if row["parents"] else None
            commits.append(
                {
                    "sha": current,
                    "parent_sha": parent,
                    "url": row["html_url"],
                    "committed_at": row["commit"]["committer"]["date"],
                    "source": source,
                }
            )
            if not saved and current == boundary:
                break
            if parent is None:
                raise ValueError(
                    "saved cursor/start SHA is not on main's first-parent chain; possible history rewrite"
                )
            current = parent
        return head, list(reversed(commits))
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        raise FetchError(endpoint, str(exc)) from exc


def validate_status(row: dict[str, Any]) -> None:
    if not isinstance(row["status"], str) or not row["status"]:
        raise ValueError("missing status")
    if row["status"] == "completed" and (
        not isinstance(row["conclusion"], str) or not row["conclusion"]
    ):
        raise ValueError("completed result has no conclusion")


def collect_job(
    api: GitHub,
    workflow: str,
    row: dict[str, Any],
    job_source: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    required(row, "id", "name", "status", "conclusion", "html_url", "steps")
    identifier = positive_int(row["id"])
    validate_status(row)
    if not isinstance(row["name"], str) or not isinstance(row["steps"], list):
        raise ValueError("invalid job name/steps")
    steps = []
    for step in row["steps"]:
        required(step, "number", "name", "status", "conclusion")
        validate_status(step)
        steps.append(
            {key: step[key] for key in ("number", "name", "status", "conclusion")}
        )
    job = {
        "id": identifier,
        "name": row["name"],
        "status": row["status"],
        "conclusion": row["conclusion"],
        "url": row["html_url"],
        "source": job_source,
        "steps": steps,
        "log": None,
        "failures": [],
    }
    errors = []
    if row["status"] == "completed" and row["conclusion"] not in SUCCESS:
        log = None
        source = {"endpoint": f"{api.prefix}/actions/jobs/{identifier}/logs"}
        try:
            cached = api.store.job_log(identifier)
            if cached:
                source = cached
                log = api.store.object(source["object"])
            else:
                log, source = api.get(source["endpoint"])
                api.store.remember_log(identifier, source)
            job["log"] = source
        except FetchError as exc:
            errors.append(exc.record())
        job["failures"] = classify.signatures(
            workflow, job, log, source if log is not None else job_source
        )
    return job, errors


def collect_workflow(api: GitHub, commit_sha: str, workflow: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "workflow": workflow,
        "status": "not_collected",
        "runs": [],
        "failures": [],
        "sources": [],
        "errors": [],
    }
    endpoint = f"{api.prefix}/actions/workflows/{workflow}/runs"
    try:
        rows, sources = api.pages(
            f"actions/workflows/{workflow}/runs",
            "workflow_runs",
            branch="main",
            head_sha=commit_sha,
        )
        result["sources"] = sources
        for row in sorted(rows, key=lambda r: r["id"]):
            required(
                row, "id", "head_sha", "head_branch", "event", "path", "run_attempt"
            )
            if row["head_sha"] != commit_sha:
                raise ValueError("run list contains a different head SHA")
            if row["head_branch"] != "main" or row["event"] not in EVENTS:
                continue
            if not isinstance(row["path"], str):
                raise ValueError("workflow path is not a string")
            if row["path"].split("@", 1)[0] != f".github/workflows/{workflow}":
                raise ValueError("workflow path mismatch")
            run_id = positive_int(row["id"])
            latest_attempt = positive_int(row["run_attempt"])
            for attempt in range(1, latest_attempt + 1):
                endpoint = f"{api.prefix}/actions/runs/{run_id}/attempts/{attempt}"
                try:
                    metadata, source = api.json(endpoint)
                    required(
                        metadata,
                        "id",
                        "run_attempt",
                        "head_sha",
                        "head_branch",
                        "event",
                        "path",
                        "status",
                        "conclusion",
                        "html_url",
                    )
                    if (
                        metadata["id"],
                        metadata["run_attempt"],
                        metadata["head_sha"],
                        metadata["head_branch"],
                        metadata["event"],
                        metadata["path"],
                    ) != (
                        run_id,
                        attempt,
                        commit_sha,
                        "main",
                        row["event"],
                        row["path"],
                    ):
                        raise ValueError("run attempt identity mismatch")
                    validate_status(metadata)
                    run: dict[str, Any] = {
                        "run_id": run_id,
                        "attempt": attempt,
                        "event": metadata["event"],
                        "status": metadata["status"],
                        "conclusion": metadata["conclusion"],
                        "url": metadata["html_url"],
                        "source": source,
                        "jobs": [],
                        "job_sources": [],
                    }
                    result["runs"].append(run)
                    endpoint += "/jobs"
                    jobs, refs = api.pages(
                        f"actions/runs/{run_id}/attempts/{attempt}/jobs", "jobs"
                    )
                    run["job_sources"] = refs
                    job_sources = {
                        j["id"]: refs[index // 100] for index, j in enumerate(jobs)
                    }
                    job_names: set[str] = set()
                    for job_row in sorted(jobs, key=lambda j: j["id"]):
                        if (
                            job_row.get("run_id") != run_id
                            or job_row.get("head_sha") != commit_sha
                            or job_row.get("run_attempt") != attempt
                        ):
                            raise ValueError("job belongs to a different run/SHA")
                        job, errors = collect_job(
                            api, workflow, job_row, job_sources[job_row["id"]]
                        )
                        if job["name"] in job_names:
                            raise ValueError(
                                "duplicate job name; cannot match retry jobs unambiguously"
                            )
                        job_names.add(job["name"])
                        run["jobs"].append(job)
                        result["errors"].extend(errors)
                    if metadata["conclusion"] == "success" and any(
                        j["conclusion"] not in SUCCESS and j["status"] == "completed"
                        for j in run["jobs"]
                    ):
                        raise ValueError(
                            "run reports success but a completed job failed"
                        )
                    if (
                        not jobs
                        and metadata["status"] == "completed"
                        and metadata["conclusion"] == "success"
                    ):
                        raise ValueError("successful run returned no jobs")
                    if (
                        metadata["status"] == "completed"
                        and metadata["conclusion"] not in SUCCESS
                        and not any(j["failures"] for j in run["jobs"])
                    ):
                        placeholder = {
                            "name": "<workflow>",
                            "conclusion": metadata["conclusion"],
                            "steps": [],
                        }
                        for failure in classify.signatures(
                            workflow, placeholder, None, source
                        ):
                            result["failures"].append(
                                {
                                    **failure,
                                    "sha": commit_sha,
                                    "run_id": run_id,
                                    "attempt": attempt,
                                    "job_id": None,
                                    "job_url": metadata["html_url"],
                                    "job_conclusion": metadata["conclusion"],
                                }
                            )
                except FetchError as exc:
                    result["errors"].append(exc.record())
                except (ValueError, TypeError, KeyError) as exc:
                    result["errors"].append(FetchError(endpoint, str(exc)).record())
        runs = result["runs"]
        if result["errors"]:
            result["status"] = "fetch_error"
        elif not runs:
            result["status"] = "no_run"
        elif any(
            r["status"] != "completed"
            or any(j["status"] != "completed" for j in r["jobs"])
            for r in runs
        ):
            result["status"] = "incomplete"
        else:
            result["status"] = "complete"
    except FetchError as exc:
        result["errors"].append(exc.record())
        result["status"] = "fetch_error"
    except (ValueError, TypeError, KeyError) as exc:
        result["errors"].append(FetchError(endpoint, str(exc)).record())
        result["status"] = "fetch_error"
    return result


def collect_sha(
    api: GitHub, commit: dict[str, Any], workflows: list[str]
) -> dict[str, Any]:
    collected = [
        collect_workflow(api, commit["sha"], workflow) for workflow in workflows
    ]
    errors = [error for workflow in collected for error in workflow["errors"]]
    status = "complete"
    if errors:
        status = "fetch_error"
    elif any(w["status"] != "complete" for w in collected):
        status = "pending"
    conclusions: list[str | None] = []
    for workflow in collected:
        latest: dict[int, dict[str, Any]] = {}
        for run in workflow["runs"]:
            if (
                run["run_id"] not in latest
                or run["attempt"] > latest[run["run_id"]]["attempt"]
            ):
                latest[run["run_id"]] = run
        conclusions.extend(r["conclusion"] for r in latest.values())
    result = None
    if status == "complete":
        result = (
            "success"
            if conclusions and all(c == "success" for c in conclusions)
            else "non_success"
        )
    return {
        "status": status,
        "result": result,
        "workflows": collected,
        "errors": errors,
    }


def publish(store: Store, now: str) -> int:
    commits = store.commits()
    classified = classify.classify(commits)
    count = 0
    with store.db:
        for commit in commits:
            payload = {
                k: v
                for k, v in commit.items()
                if k not in ("fact", "checked_at", "ordinal")
            }
            payload.update(commit["fact"])
            payload["failures"] = classified[commit["sha"]]
            count += store.append("observation", commit["sha"], payload, now)
    return count


def collect(api: GitHub, config: dict[str, Any], now: str) -> dict[str, Any]:
    store = api.store
    appended = 0
    try:
        head, commits = discover(api, config, store.commits())
    except FetchError as exc:
        with store.db:
            store.append(
                "collection",
                None,
                {"status": "fetch_error", "errors": [exc.record()]},
                now,
            )
        return {
            "complete": False,
            "errors": [exc.record()],
            "discovered": 0,
            "appended": appended,
        }
    store.discover(commits, now)
    appended += len(commits)
    for commit in store.commits():
        fact = collect_sha(api, commit, config["workflows"])
        checked_at = datetime.now(UTC).isoformat()
        store.fact(commit["sha"], fact, checked_at)
        appended += publish(store, checked_at)
    latest = store.commits()
    errors = [error for c in latest for error in c["fact"]["errors"]]
    summary = {
        "head_sha": head,
        "cursor_sha": latest[-1]["sha"],
        "discovered": len(commits),
        "observed": len(latest),
        "appended": appended,
        "complete": all(c["fact"]["status"] == "complete" for c in latest),
        "pending": [c["sha"] for c in latest if c["fact"]["status"] != "complete"],
        "errors": errors,
    }
    with store.db:
        store.append(
            "collection",
            None,
            {
                "status": "fetch_error"
                if errors
                else "complete"
                if summary["complete"]
                else "pending",
                "head_sha": head,
                "cursor_sha": latest[-1]["sha"],
                "observed": len(latest),
                "pending": summary["pending"],
                "errors": errors,
            },
            datetime.now(UTC).isoformat(),
        )
    return summary
