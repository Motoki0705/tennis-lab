"""Child launch selection. Runtime execution and process slots belong to callers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from issues import Issue

from .policy import closing_issues, dependencies, nonempty, progress, raw_section

DEFAULT_LAUNCH_POLICY = {
    "schema_version": 1,
    "max_parallel_children": None,
    "reason": "手動親の既定: 依存が許す全子を並列に起動する。",
}


def launch_policy(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "max_parallel_children", "reason"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("unknown or malformed child launch policy")
    limit = value["max_parallel_children"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("max_parallel_children must be null or a positive integer")
    nonempty(value["reason"], "child launch limit reason")
    return dict(value)


def reason(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def plan_launches(
    children: Mapping[int, Issue],
    tasks: Mapping[str, Any],
    prs: Sequence[Mapping[str, Any]],
    repo: str,
    policy: Mapping[str, Any],
    parent_reasons: Sequence[dict[str, str]] = (),
    overrides: Mapping[int, Sequence[int]] | None = None,
    restart_child: int | None = None,
) -> dict[str, Any]:
    states = progress(children, tasks, overrides)
    matching = {
        n: [p for p in prs if n in closing_issues(p.get("body") or "", repo)]
        for n in children
    }
    reserved = []
    for task_key, reservation in tasks.items():
        if "worker_ended" in reservation:
            ended = reservation["worker_ended"]
            if (
                not isinstance(ended, dict)
                or set(ended) != {"delegation_id", "evidence"}
                or ended["delegation_id"] != reservation.get("delegation_id", "legacy")
            ):
                raise ValueError("malformed or stale worker-end evidence")
            nonempty(ended["evidence"], "worker-end evidence")
        elif reservation.get("delegated"):
            reserved.append(int(task_key))
    reserved.sort()
    limit = policy["max_parallel_children"]
    available = None if limit is None else max(0, limit - len(reserved))

    def ancestors(n: int) -> set[int]:
        deps = (overrides or {}).get(n, dependencies(children[n].body))
        unfinished = [d for d in deps if states[d] != "completed"]
        return set(unfinished).union(*(ancestors(d) for d in unfinished))

    result = []
    for n, child in sorted(children.items()):
        task = tasks.get(str(n), {})
        deps = list((overrides or {}).get(n, dependencies(child.body)))
        reasons = []
        stacked = []
        if states[n] in {"completed", "cancelled"}:
            reasons.append(
                reason(
                    states[n],
                    "完了済み。" if states[n] == "completed" else "取消済み。",
                )
            )
        elif states[n] == "stopped":
            notes = re.findall(
                r"(?m)^reason: (.+)$", raw_section(child.body, "作業状況・停止事項")
            )
            summary = notes[-1].strip() if notes else "停止中"
            reasons.append(
                reason(
                    "stopped",
                    f"{summary}（詳細・再開条件: #{n}）。",
                )
            )
        elif states[n] == "held":
            blocked = [
                f"#{d}" for d in deps if states[d] in {"stopped", "held", "cancelled"}
            ]
            reasons.append(
                reason(
                    "dependency_blocked",
                    "停止・取消された依存先: " + ", ".join(blocked),
                )
            )
        if matching[n]:
            reasons.append(
                reason(
                    "existing_pr",
                    "既存PRを照合して継続する: "
                    + ", ".join(f"#{p['number']}" for p in matching[n]),
                )
            )
        if task.get("delegated") and restart_child != n:
            reasons.append(
                reason(
                    "already_delegated",
                    "委任予約済み。既存worker・worktreeを照合する。",
                )
            )
        for dep in sorted(ancestors(n)):
            if states[dep] != "continue":
                continue
            candidates = [
                p for p in matching[dep] if p["state"] == "open" or p.get("merged_at")
            ]
            if len(candidates) != 1:
                reasons.append(
                    reason(
                        "dependency_pr_unavailable",
                        f"依存先 #{dep}: 利用できるPRが一意でない（{len(candidates)}件）。",
                    )
                )
                continue
            pr = candidates[0]
            head = pr["head"]
            if (
                head["repo"]["full_name"].casefold() != repo.casefold()
                or not isinstance(head["ref"], str)
                or not head["ref"]
                or not isinstance(head["sha"], str)
                or not re.fullmatch(r"[0-9a-f]{40}", head["sha"])
            ):
                reasons.append(
                    reason(
                        "dependency_pr_unsupported",
                        f"依存先 #{dep}: 同repoのbranchと完全なhead SHAが必要。",
                    )
                )
                continue
            stacked.append(
                {
                    "child": dep,
                    "pr": pr["number"],
                    "branch": head["ref"],
                    "head_sha": head["sha"],
                }
            )
        reasons.extend(parent_reasons)
        if not reasons and available is not None:
            if restart_child == n and n in reserved:
                pass  # Replacing a reservation requires inspected worker-end evidence.
            elif available == 0:
                reasons.append(
                    reason(
                        "parallel_limit",
                        f"子の起動予約上限 {limit} 件: {policy['reason']}",
                    )
                )
            else:
                available -= 1
        result.append(
            {
                "child": n,
                "state": states[n],
                "action": "do_not_launch" if reasons else "ready",
                "dependencies": deps,
                "stacked_on": stacked,
                "reasons": reasons,
            }
        )
    return {
        "policy": dict(policy),
        "active_reservations": reserved,
        "available_slots": None if limit is None else max(0, limit - len(reserved)),
        "launchable_children": [t["child"] for t in result if t["action"] == "ready"],
        "tasks": result,
    }
