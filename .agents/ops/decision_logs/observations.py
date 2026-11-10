"""Adapters for facts already recorded by runner, worker logs and review tools."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from worker_logs.journal import decode
from worker_logs.schema import json_bytes, sha

from .capture import (
    CURRENT,
    RETRIEVAL_ENV,
    RUNTIME_ENV,
    Operation,
    current,
    empty_auxiliary,
    missing,
    observed,
    runtime_environment,
    source,
    warn,
)
from .journal import Journal
from .schema import new_id


def stamp(value: Any) -> str | None:
    if type(value) in (int, float):
        return datetime.fromtimestamp(value, UTC).isoformat()
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value)
        return parsed.isoformat() if parsed.tzinfo is not None else None
    return None


def availability(value: Any, reason: str) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("status") == "observed":
        return observed(value["value"])
    return missing(value.get("reason", reason) if isinstance(value, dict) else reason)


def runtime_auxiliary(
    value: dict[str, Any], counter_key: str, specified_by: str | None = None
) -> dict[str, Any]:
    result = empty_auxiliary()
    evidence = source("runner.result", "result.json", value)
    settings = value.get("contract", {}).get("settings", {})
    selected = settings.get("selected", {})
    if selected.get("model") and selected.get("effort"):
        result["selected_model"] = observed(
            {
                "model": selected["model"],
                "effort": selected["effort"],
                "specified_by": specified_by or "settings",
                "source": source(
                    "runner.result",
                    "contract.settings (selection_sources retained in private snapshot)",
                    settings,
                ),
            }
        )
    runtime = value.get("runtime", {})
    effective = runtime.get("effective")
    if isinstance(effective, dict):
        result["effective_model"] = observed(
            {
                "model": availability(
                    effective.get("model"), "effective model not reported"
                ),
                "effort": availability(
                    effective.get("effort"), "effective effort not reported"
                ),
                "source": evidence,
            }
        )
    start, end = value.get("started_at"), value.get("finished_at")
    if (
        isinstance(start, (int, float))
        and isinstance(end, (int, float))
        and type(start) is not bool
        and type(end) is not bool
        and end >= start
    ):
        result["elapsed_seconds"] = observed({"value": end - start, "source": evidence})
    usage = runtime.get("usage", {})
    if usage.get("status") == "observed" and (
        usage.get("basis") in {"turn", "cumulative"}
        or usage.get("source") in {"turn.completed", "result"}
    ):
        counters = usage["value"]
        # These runner sources report this turn's counters. A cumulative source
        # must explicitly declare its basis; do not guess from field names.
        result["usage"] = observed(
            {
                "basis": usage.get("basis", "turn"),
                "counter_key": counter_key,
                **{
                    key: observed(counters[key])
                    if type(counters.get(key)) is int and counters[key] >= 0
                    else missing(f"{key} not reported")
                    for key in (
                        "input_tokens",
                        "cached_input_tokens",
                        "output_tokens",
                        "total_tokens",
                    )
                },
                "source": evidence,
            }
        )
    elif usage.get("status") == "unavailable":
        result["usage"] = missing(usage.get("reason", "usage unavailable"))
    return result


def runtime_record(
    path: Path, value: dict[str, Any], *, operation: Operation | None = None
) -> None:
    """Only the caller's bound runtime may update its action. No standalone inference."""
    op = operation
    try:
        reference = os.environ.get(RUNTIME_ENV)
        op = operation or (Operation.load(Path(reference)) if reference else current())
        if op is None or op.row["runtime_directory"] != str(path.parent.resolve()):
            return
        terminal = value.get("result") in {"completed", "failed"}
        if not terminal and value.get("agent_spawned") is not True:
            return
        with op.transaction():
            key = (
                "runner:terminal:" + sha(json_bytes(value))
                if terminal
                else "runner:started"
            )
            if any(e["key"] == key for e in op.row["entries"]):
                return
            runtime = value.get("runtime", {})
            for group, values in {
                "runs": [value.get("run_id")],
                "sessions": [runtime.get("session_id"), runtime.get("turn_id")],
                "identifiers": [value.get("contract", {}).get("task_id")],
                "roots": [value.get("contract", {}).get("cwd")],
            }.items():
                op.row["private"][group] = sorted(
                    set(op.row["private"][group])
                    | {v for v in values if isinstance(v, str) and v}
                )
            actions = op.records("action")
            if not terminal:
                if actions[-1]["data"]["state"] == "requested":
                    op.action(
                        "running",
                        "runner.result: agent_spawned observed",
                        at=stamp(value.get("started_at")),
                    )
                op.result(
                    value,
                    artifact="result.json",
                    status="unknown",
                    summary="runner.result: agent started; completion and task acceptance unknown",
                    key=key,
                    auxiliary=runtime_auxiliary(
                        value, op.row["counter_key"], op.row.get("model_specified_by")
                    ),
                    at=stamp(value.get("started_at")),
                )
                return
            spawned = value.get("agent_spawned")
            # Older runner records may lack agent_spawned. A terminal record
            # still describes a run, but cannot prove launch or pre-start rejection.
            state = (
                "rejected_before_start"
                if spawned is False and actions[-1]["data"]["state"] == "requested"
                else "ended"
            )
            reason = value.get("turn_end_reason", "unknown")
            status = (
                "success"
                if value["result"] == "completed"
                else "interrupted"
                if reason in {"stopped", "timeout"}
                else "rejected"
                if state == "rejected_before_start"
                else "failure"
            )
            final_path = path.parent / "final.md"
            final = (
                observed(
                    source(
                        "runner.result",
                        "final.md",
                        {"content_sha256": sha(final_path.read_bytes())},
                    )
                )
                if final_path.is_file() and final_path.stat().st_size
                else missing("missing_final")
            )
            if status == "success" and final["status"] == "unavailable":
                status = "partial"
            op.result(
                value,
                artifact="result.json",
                status=status,
                summary=f"runner.result: {value['result']}; turn_end_reason={reason}; exit_code={value.get('exit_code')}; task acceptance unconfirmed",
                key=key,
                final_report=final,
                auxiliary=runtime_auxiliary(
                    value, op.row["counter_key"], op.row.get("model_specified_by")
                ),
                at=stamp(value.get("finished_at")),
            )
            op.row["closed"] = True
            op.action(
                state,
                f"runner.result: {state}; {reason}",
                at=stamp(value.get("finished_at")),
            )
            op.source_restored("runner.result")
    except Exception:
        if op is not None:
            op.mark_failed(lost_source="runner.result")
        else:
            warn()


def bind_job(store: Any, key: str, row: dict[str, Any], job: dict[str, Any]) -> None:
    if job.get("child") is None:
        return
    try:
        op = current()
        if op is None:
            phase = row["tasks"][str(job["child"])].get("preflight", {})
            scope = {
                "repo": key.rpartition("#")[0],
                "parent": int(key.rpartition("#")[2]),
                "issue": job["child"],
                "activity_key": new_id(),
                "delegation": {
                    "key": "d-" + phase["public_key"],
                    "turn": len(phase["turns"]),
                }
                if phase.get("public_key")
                else None,
                "review_key": None,
            }
            from .parent_commands import capture_for_runtime

            prior = (
                capture_for_runtime(
                    store.directory,
                    scope["repo"],
                    scope["parent"],
                    str(Path(job["resume_from"]).resolve()),
                )
                if job.get("resume_from")
                else []
            )
            op = Operation.begin(
                store.directory,
                scope["repo"],
                scope["parent"],
                "dispatcher.start-child",
                "continue_ai" if job.get("resume_from") else "start_ai",
                scope=scope,
                previous_result_id=prior[-1]["id"] if prior else None,
            )
        env = runtime_environment(
            Path(job["directory"]) / "runtime",
            operation=op,
            actual_operation="continue_ai" if job.get("resume_from") else "start_ai",
        )
        if RUNTIME_ENV in env:
            job["decision_capture"] = env[RUNTIME_ENV]
            bound = Operation.load(Path(env[RUNTIME_ENV]))
            with bound.transaction():
                bound.row.setdefault(
                    "model_specified_by",
                    "parent_choice" if job.get("model") else "settings",
                )
                bound._save()
            bind_worker_task(row["tasks"][str(job["child"])], operation=bound)
    except Exception:
        warn()


def bind_worker_task(
    task: dict[str, Any], *, operation: Operation | None = None
) -> None:
    """Use saved preflight observations; never ask the parent to retype them."""
    op = operation or current()
    if op is None:
        return
    try:
        phase = task.get("preflight", {})
        turn = phase["turns"][-1]
        context = phase.get("context") or {}
        snapshot = {
            "head_at_start": turn.get("head_at_start"),
            "prompt_sha256": turn.get("prompt_sha256"),
            **{
                k: context.get(k)
                for k in ("parent_requirements", "requirements", "dependencies")
            },
        }
        baseline = None
        previous_id = op.row["previous_result_id"]
        if previous_id:
            for path in op.path.parent.glob("*.json"):
                candidate = Operation.load(path)
                if any(
                    e["record"]["id"] == previous_id for e in candidate.row["entries"]
                ):
                    baseline = candidate.row.get("worker_context")
                    break
        with op.transaction():
            op.row["worker_context"] = snapshot
            phase_ids = [
                task.get("delegation_id"),
                phase.get("lineage_id"),
                turn.get("snapshot", {}).get("turn_id"),
                turn.get("snapshot", {}).get("decision_id"),
            ]
            op.row["private"]["identifiers"] = sorted(
                set(op.row["private"]["identifiers"])
                | {v for v in phase_ids if isinstance(v, str) and v}
            )
            if task.get("worktree"):
                op.row["private"]["roots"] = sorted(
                    set(op.row["private"]["roots"]) | {task["worktree"]}
                )
            op.row["private"]["checkout_commit"] = snapshot["head_at_start"]
            if baseline and not op.row["changes_explicit"]:
                changes = op.row["changes"]
                if (
                    snapshot["head_at_start"]
                    and baseline.get("head_at_start")
                    and snapshot["head_at_start"] != baseline["head_at_start"]
                ):
                    changes["code"] = "changed"
                if snapshot["prompt_sha256"] and baseline.get("prompt_sha256"):
                    changes["instruction"] = (
                        "unchanged"
                        if snapshot["prompt_sha256"] == baseline["prompt_sha256"]
                        else "changed"
                    )
                requirement_keys = ("parent_requirements", "requirements")
                if all(snapshot.get(k) and baseline.get(k) for k in requirement_keys):
                    changes["requirements"] = (
                        "unchanged"
                        if all(snapshot[k] == baseline[k] for k in requirement_keys)
                        else "changed"
                    )
                if (
                    snapshot["dependencies"] is not None
                    and baseline.get("dependencies") is not None
                    and snapshot["dependencies"] != baseline["dependencies"]
                ):
                    changes["conditions"] = "changed"
            op._save()
            op.source_restored("worker.context")
    except Exception:
        op.mark_failed(lost_source="worker.context")


def job_environment(job: dict[str, Any]) -> dict[str, str]:
    try:
        reference = job.get("decision_capture")
        return (
            runtime_environment(
                Path(job["directory"]) / "runtime",
                operation=Operation.load(Path(reference)),
            )
            if reference
            else dict(os.environ)
        )
    except Exception:
        warn()
        return dict(os.environ)


def saved_jobs(parents: dict[str, Any]) -> None:
    """Observe durable cancellation/runner-return facts, including no result.json."""
    for row in parents.values():
        for job in (row.get("dispatch") or {}).get("jobs", {}).values():
            reference = job.get("decision_capture")
            if not reference:
                continue
            try:
                op = Operation.load(Path(reference))
                runtime_path = Path(job["directory"]) / "runtime/result.json"
                if runtime_path.exists():
                    runtime_record(
                        runtime_path,
                        decode(runtime_path.read_bytes()),
                        operation=op,
                    )
                elif job.get("cancelled_before_start") or job.get("runner_exit") in {
                    2,
                    127,
                }:
                    with op.transaction():
                        if not op.row["closed"]:
                            op.finish(
                                {
                                    "cancelled_before_start": bool(
                                        job.get("cancelled_before_start")
                                    ),
                                    "runner_exit": job.get("runner_exit"),
                                    "diagnostic": job.get("diagnostic"),
                                },
                                state="rejected_before_start",
                                status="rejected",
                                summary="dispatcher: rejected before start; no runtime result or parent verification",
                                artifact="saved dispatcher job",
                            )
            except Exception:
                warn()


def worker_log(
    path: Path,
    value: dict[str, Any],
    *,
    retry: bool = False,
    entrypoint: str = "worker_logs.publish",
) -> None:
    """Record publication separately, including automatic dispatcher publication."""
    try:
        from worker_logs.journal import read

        row = read(path)
        original = row["source"]
        # Derive the parent state location from the established worker-log path.
        directory = path.parents[6]
        parent = original["parent"]
        repo = original["repo"]
        before = []
        prior_publication = False
        for candidate_path in (
            Journal(directory, repo, parent).path.parent / "captures"
        ).glob("*.json"):
            candidate = Operation.load(candidate_path)
            if candidate.row.get("worker_log_path") == str(path.resolve()):
                before.extend(candidate.records("result"))
                prior_publication |= candidate.row["operation"] in {
                    "publish",
                    "retry_publication",
                }
        previous = (
            max(before, key=lambda r: r["time"]["recorded_at"])["id"]
            if before
            else None
        )
        op = current()
        owned = op is None or op.row["operation"] not in {
            "publish",
            "retry_publication",
        }
        if owned:
            op = Operation.begin(
                directory,
                repo,
                parent,
                entrypoint,
                ("retry_publication" if retry or prior_publication else "publish")
                if entrypoint == "worker_logs.publish"
                else "record",
                issue=original["issue"],
                scope={
                    "repo": repo,
                    "parent": parent,
                    "issue": original["issue"],
                    "activity_key": new_id(),
                    "delegation": {
                        "key": "d-" + original["public_key"],
                        "turn": original["sequence"],
                    },
                    "review_key": None,
                },
                previous_result_id=previous,
            )
        assert op is not None
        with op.transaction():
            op.row["worker_log_path"] = str(path.resolve())
            if not op.row["previous_result_id"] and not op.records("result"):
                op.row["previous_result_id"] = previous
            op._save()
        evidence = []
        final = None
        if value.get("saved_to_main") and row.get("head") and row.get("files"):
            from worker_logs.schema import path as log_path

            meta = decode(row["files"]["meta.json"])
            base = f"https://github.com/{repo}/blob/{row['head']}/{log_path(meta)}"
            evidence = [
                {
                    "kind": "public",
                    "url": base + "/meta.json",
                    "revision": row["head"],
                    "content_sha256": sha(row["files"]["meta.json"].encode()),
                }
            ]
            final = observed(
                {
                    "kind": "public",
                    "url": base + "/final.md",
                    "revision": row["head"],
                    "content_sha256": sha(row["files"]["final.md"].encode()),
                }
            )
        token = CURRENT.set(op)
        try:
            from .capture import observe

            observe(
                value,
                status="success"
                if value.get("saved_to_main")
                else "failure"
                if value.get("error")
                else "partial",
                summary=f"{entrypoint}: delegation_key={value.get('delegation_key')}; turn={value.get('sequence')}; state={value.get('state')}; stage={value.get('stage')}; saved_to_main={value.get('saved_to_main')}; error={value.get('error')}",
                artifact="worker-log journal publication view",
                evidence=evidence,
                final_report=final,
            )
        finally:
            CURRENT.reset(token)
    except Exception:
        warn()


def retrieval(name: str, arguments: Any, response: dict[str, Any]) -> None:
    """Capture supported review retrievals; repeated identical calls are fetch retries."""
    reference = os.environ.get(RETRIEVAL_ENV)
    if not reference:
        return
    try:
        owner = Operation.load(Path(reference))
        fingerprint = sha(json_bytes({"tool": name, "arguments": arguments}))
        previous = []
        for path in owner.path.parent.glob("*.json"):
            candidate = Operation.load(path)
            if (
                candidate.row.get("retrieval_owner") == reference
                and candidate.row.get("retrieval_fingerprint") == fingerprint
            ):
                previous.extend(candidate.records("result"))
        before = (
            max(previous, key=lambda r: r["time"]["recorded_at"]) if previous else None
        )
        op = Operation.begin(
            Path(owner.row["directory"]),
            owner.row["repo"],
            owner.row["parent"],
            "review_tools." + name,
            "retry_fetch" if before else "fetch",
            scope=copy_scope(owner.row["scope"]),
            previous_result_id=before["id"] if before else None,
            changes={
                k: "unchanged" if k == "instruction" else "unconfirmed"
                for k in (
                    "code",
                    "requirements",
                    "instruction",
                    "conditions",
                    "verified_scope",
                )
            }
            if before
            else None,
        )
        with op.transaction():
            op.row.update(retrieval_owner=reference, retrieval_fingerprint=fingerprint)
            op.finish(
                response,
                status="success" if response.get("ok") else "failure",
                summary=f"review_tools.{name}: {'success' if response.get('ok') else 'failure'}; error={(response.get('error') or {}).get('code')}",
                artifact="retrieval received receipt",
            )
    except Exception:
        warn()


def copy_scope(scope: dict[str, Any]) -> dict[str, Any]:
    return {
        **scope,
        "delegation": dict(scope["delegation"]) if scope.get("delegation") else None,
    }
