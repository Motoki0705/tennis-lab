"""Require a main merge and positive execution evidence for the failed target."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ci_watch.classify import ANSI, TIMESTAMP

from .github import GitHub
from .observations import evidence, source_reference
from .store import Store


def chain_order(records: list[dict[str, Any]]) -> dict[str, int]:
    observations = {r["sha"]: r["observation"] for r in records}
    children: dict[str, str] = {}
    roots = []
    for sha, row in observations.items():
        parent = row["parent_sha"]
        if parent not in observations:
            roots.append(sha)
        elif parent in children:
            raise ValueError("CI observations do not form one first-parent main chain")
        else:
            children[parent] = sha
    if len(roots) != 1:
        raise ValueError("CI main observation chain has a gap or cycle")
    order: dict[str, int] = {}
    current = roots[0]
    while current not in order:
        order[current] = len(order)
        if current not in children:
            break
        current = children[current]
    if len(order) != len(observations):
        raise ValueError("CI main observation chain is incomplete")
    return order


def recovery(
    row: dict[str, Any],
    records: list[dict[str, Any]],
    pulls: list[dict[str, Any]],
    gh: GitHub,
    state: Store,
) -> tuple[str, str, dict[str, Any] | None]:
    order = chain_order(records)
    merged = [
        p
        for p in pulls
        if p["merged_at"] is not None and p.get("base", {}).get("ref") == "main"
    ]
    if not merged:
        return "awaiting_fix", "修正PRのmainへのmergeを未確認", None
    tracked = [p for p in merged if p.get("merge_commit_sha") in order]
    if not tracked:
        return (
            "awaiting_main_ci",
            "修正merge SHAが観測対象mainのfirst-parent上で未確認",
            None,
        )
    failure = max(row["occurrences"], key=lambda o: order[o["failure"]["sha"]])[
        "failure"
    ]
    last_failure = max(order[o["failure"]["sha"]] for o in row["occurrences"])
    candidates = sorted(records, key=lambda r: order[r["sha"]], reverse=True)
    reason = "修正merge以降かつ最後の失敗より後のmain CI成功を未確認"
    for record in candidates:
        position = order[record["sha"]]
        fixes = [p for p in tracked if order[p["merge_commit_sha"]] <= position]
        if position <= last_failure or not fixes:
            continue
        observation = record["observation"]
        if observation["status"] != "complete":
            continue
        workflows = [
            w for w in observation["workflows"] if w["workflow"] == failure["workflow"]
        ]
        if len(workflows) != 1 or workflows[0]["status"] != "complete":
            continue
        latest: dict[int, dict[str, Any]] = {}
        for run in workflows[0]["runs"]:
            if (
                run["run_id"] not in latest
                or run["attempt"] > latest[run["run_id"]]["attempt"]
            ):
                latest[run["run_id"]] = run
        if not latest:
            continue
        jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for run in latest.values():
            matching = [j for j in run["jobs"] if j["name"] == failure["job"]]
            if (
                run["status"] != "completed"
                or run["conclusion"] != "success"
                or len(matching) != 1
            ):
                jobs = []
                break
            job = matching[0]
            if (
                job["status"] != "completed"
                or job["conclusion"] != "success"
                or any(
                    [s["conclusion"] for s in job["steps"] if s["name"] == step]
                    != ["success"]
                    for step in failure["failed_steps"]
                )
            ):
                jobs = []
                break
            jobs.append((run, job))
        if not jobs:
            continue
        successes: list[dict[str, Any]] = []
        for run, job in jobs:
            log_ref = None
            if failure["test_id"] is not None:
                log_ref = state.job_log(job["id"])
                if log_ref is None:
                    log_ref = state.remember_log(job["id"], gh.job_log(job["id"]))
                text = state.object(log_ref)
                node = re.escape(failure["test_id"])
                passed = re.compile(
                    rf"(?:^|\s)(?:PASSED\s+{node}|{node}\s+PASSED)(?=\s|$)"
                )
                if not any(
                    passed.search(TIMESTAMP.sub("", ANSI.sub("", line)))
                    for line in text.splitlines()
                ):
                    reason = "job成功は確認したが同じpytest node IDの実行・成功をログで確認できない"
                    successes = []
                    break
            successes.append(
                {
                    "run_id": run["run_id"],
                    "attempt": run["attempt"],
                    "run_url": run["url"],
                    "job_id": job["id"],
                    "job_url": job["url"],
                    "source": job["source"],
                    "test_log_object": log_ref,
                }
            )
        if successes:
            fix = max(fixes, key=lambda p: order[p["merge_commit_sha"]])
            ref = source_reference(Path(row["source_dir"]), record)
            return (
                "recovered",
                "修正merge後のmainで元の失敗対象の成功を確認",
                {
                    "sha": record["sha"],
                    "pr": fix["number"],
                    "merge_sha": fix["merge_commit_sha"],
                    "source": ref,
                    "jobs": successes,
                    "evidence": evidence(
                        ref, record["observed_at"], record["sha"], "main-ci-recovery"
                    ),
                },
            )
    failed_after_merge = any(
        last_failure >= order[p["merge_commit_sha"]] for p in tracked
    )
    return ("regressed" if failed_after_merge else "awaiting_main_ci"), reason, None
