"""The shared controller/collector/worker/gate API. All writes use short transactions."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

from .model import LIVE, generation, instant, positive, sha, text, utcnow, window
from .store import CycleStore


class CycleAPI:
    def __init__(
        self, store: CycleStore, *, clock: Callable[[], datetime] = utcnow
    ) -> None:
        self.store = store
        self.clock = clock

    def initialize(
        self,
        baseline_sha: str,
        *,
        workflows: list[str] | None = None,
        integrations: dict[str, list[str] | None] | None = None,
    ) -> dict[str, Any]:
        workflows = sorted(set(workflows or ["ci.yml", "knowledge.yml"]))
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+\.ya?ml", w) for w in workflows):
            raise ValueError("invalid workflow filename")
        hooks = integrations or {"observer": None, "workers": None}
        if set(hooks) != {"observer", "workers"} or any(
            v is not None
            and (
                not isinstance(v, list)
                or not v
                or any(not isinstance(a, str) or not a for a in v)
            )
            for v in hooks.values()
        ):
            raise ValueError(
                "integrations must specify observer/workers argv arrays or null"
            )
        with self.store.lock("state"):
            if self.store.read(required=False) is not None:
                raise ValueError("daily_ci is already initialized")
            data = {
                "schema_version": 1,
                "repository": self.store.repository,
                "enabled": False,
                "baseline_sha": sha(baseline_sha),
                "last_success": None,
                "workflows": workflows,
                "integrations": hooks,
                "created_at": self.clock().isoformat(),
                "cycles": {},
            }
            self.store.save(data)
        return data

    def snapshot(self) -> dict[str, Any]:
        with self.store.lock("state"):
            data = self.store.read()
            assert data is not None
        now = self.clock()
        cycle_id, opens, deadline = window(now)
        cycle = data["cycles"].get(cycle_id)
        return {
            **data,
            "current_cycle_id": cycle_id,
            "in_window": opens <= now < deadline,
            "normal_merge_frozen": bool(
                data["enabled"]
                and opens <= now < deadline
                and (cycle is None or cycle["status"] != "success")
            ),
        }

    def active(self, data: dict[str, Any], cycle_id: str) -> dict[str, Any]:
        cycle: dict[str, Any] | None = data["cycles"].get(cycle_id)
        now = self.clock()
        if (
            not data["enabled"]
            or cycle is None
            or cycle["status"] != "active"
            or not instant(cycle["opens_at"]) <= now < instant(cycle["deadline_at"])
            or cycle_id != window(now)[0]
        ):
            raise ValueError("daily_ci cycle does not permit new work")
        return cycle

    def permit_worker(
        self, cycle_id: str, worker_id: str | None = None
    ) -> dict[str, Any]:
        with self.store.transaction() as data:
            cycle = self.active(data, cycle_id)
            if worker_id is not None:
                worker = cycle["workers"].get(worker_id)
                if worker is None or worker["state"] != "registered":
                    raise ValueError("daily_ci worker is unregistered or revoked")
            return {
                "cycle_id": cycle_id,
                "deadline_at": cycle["deadline_at"],
                "target_sha": cycle["target_sha"],
                "generation": len(cycle["generations"]),
            }

    def register_worker(
        self, cycle_id: str, worker_id: str, handle: dict[str, Any]
    ) -> None:
        text(worker_id)
        if not isinstance(handle, dict) or not handle:
            raise ValueError(
                "worker handle must identify its dispatcher/runtime record"
            )
        with self.store.transaction() as data:
            cycle = self.active(data, cycle_id)
            previous = cycle["workers"].get(worker_id)
            if previous and (
                previous["handle"] != handle or previous["state"] != "registered"
            ):
                raise ValueError("worker identity changed or was revoked")
            cycle["workers"].setdefault(
                worker_id,
                {"handle": handle, "state": "registered", "evidence": None},
            )

    def register_repair(
        self, cycle_id: str, pr: int, head_sha: str, signatures: list[str]
    ) -> None:
        positive(pr)
        sha(head_sha)
        if not signatures or any(not isinstance(s, str) or not s for s in signatures):
            raise ValueError(
                "repair PR requires failure/diagnostic signature references"
            )
        with self.store.transaction() as data:
            cycle = self.active(data, cycle_id)
            previous = cycle["repair_prs"].get(str(pr))
            if previous and previous["merge_state"] != "registered":
                raise ValueError(
                    "repair merge already attempted; reconcile its outcome"
                )
            cycle["repair_prs"][str(pr)] = {
                "head_sha": head_sha,
                "signatures": sorted(set(signatures)),
                "registered_at": self.clock().isoformat(),
                "merge_state": "registered",
                "merged_sha": None,
            }

    def merge_reason(self, pr: int, head_sha: str, cycle_id: str | None) -> str | None:
        with self.store.lock("state"):
            data = self.store.read(required=False)
        if data is None:
            return "daily_ci_cycle_unknown" if cycle_id else None
        now = self.clock()
        today, opens, deadline = window(now)
        cycle = data["cycles"].get(today)
        if cycle_id is not None:
            try:
                current = self.active(data, cycle_id)
            except ValueError:
                return "daily_ci_cycle_expired_or_inactive"
            repair = current["repair_prs"].get(str(pr))
            if repair is None or repair["head_sha"] != head_sha:
                return "daily_ci_repair_not_registered"
            if repair["merge_state"] != "registered":
                return "daily_ci_repair_already_attempted"
            return None
        if (
            data["enabled"]
            and opens <= now < deadline
            and (cycle is None or cycle["status"] != "success")
        ):
            return "daily_ci_normal_merge_frozen"
        return None

    def begin_merge(self, cycle_id: str, pr: int, head_sha: str) -> None:
        # Caller holds repository_lock across this check, the existing gate and gh merge.
        reason = self.merge_reason(pr, head_sha, cycle_id)
        if reason:
            raise ValueError(reason)
        with self.store.transaction() as data:
            cycle = self.active(data, cycle_id)
            if any(
                request["state"] == "planned"
                for request in cycle["generations"][0]["requests"].values()
            ):
                raise ValueError("initial daily_ci dispatch has not been attempted")
            cycle["repair_prs"][str(pr)]["merge_state"] = "attempting"

    def finish_merge(
        self, cycle_id: str, pr: int, head_sha: str, merged_sha: str | None
    ) -> None:
        with self.store.transaction() as data:
            cycle = data["cycles"][cycle_id]
            repair = cycle["repair_prs"][str(pr)]
            if repair["head_sha"] != head_sha or repair["merge_state"] != "attempting":
                raise ValueError("repair merge identity mismatch")
            repair["merge_state"] = "merged" if merged_sha else "unknown"
            repair["merged_sha"] = sha(merged_sha) if merged_sha else None
            if merged_sha and cycle["status"] == "active":
                if self.clock() >= instant(cycle["deadline_at"]):
                    cycle["errors"].append("repair_merge_completed_after_deadline")
                    return
                if merged_sha in {g["target_sha"] for g in cycle["generations"]}:
                    raise ValueError("repair verification requires a new main SHA")
                cycle["target_sha"] = merged_sha
                cycle["generations"].append(
                    generation(self.store.repository, cycle, merged_sha, "repair")
                )
            elif merged_sha is None:
                cycle["errors"].append("repair_merge_outcome_unknown")

    def record_result(self, result: dict[str, Any]) -> None:
        required = {
            "cycle_id",
            "generation",
            "workflow",
            "request_id",
            "run_id",
            "attempt",
            "target_sha",
            "checkout_sha",
            "baseline_sha",
            "result",
            "evidence",
        }
        if set(result) != required:
            raise ValueError("daily_ci result fields do not match schema version 1")
        if result["result"] not in {"success", "failure", "incomplete"}:
            raise ValueError("invalid daily_ci result")
        if not isinstance(result["evidence"], dict) or not result["evidence"]:
            raise ValueError("verified manifest/artifact evidence is required")
        with self.store.transaction() as data:
            cycle = self.active(data, result["cycle_id"])
            number = positive(result["generation"])
            if number != len(cycle["generations"]):
                raise ValueError("stale daily_ci generation")
            revision = cycle["generations"][number - 1]
            request = revision["requests"][result["workflow"]]
            if (
                request["state"] != "bound"
                or request["request_id"] != result["request_id"]
                or request["run_id"] != positive(result["run_id"])
                or request["attempt"] != positive(result["attempt"])
                or revision["target_sha"] != sha(result["target_sha"])
                or result["target_sha"] != sha(result["checkout_sha"])
                or cycle["baseline_sha"] != sha(result["baseline_sha"])
                or request["remote_status"] != "completed"
                or (
                    result["result"] == "success"
                    and request["remote_conclusion"] != "success"
                )
            ):
                raise ValueError("daily_ci result identity/checkout/attempt mismatch")
            request["observation"] = result

    def unresolved(self, cycle_id: str, records: list[dict[str, Any]]) -> None:
        if not isinstance(records, list) or any(
            not isinstance(r, dict) or not r.get("signature") or not r.get("evidence")
            for r in records
        ):
            raise ValueError("unresolved records need a signature and evidence")
        with self.store.transaction() as data:
            cycle = self.active(data, cycle_id)
            cycle["unresolved"] = records

    def expire(self) -> list[str]:
        """Revoke first; never wait for repository-merge lock or remote stop replies."""
        changed = []
        with self.store.transaction() as data:
            now = self.clock()
            for cycle in data["cycles"].values():
                if cycle["status"] in LIVE and now >= instant(cycle["deadline_at"]):
                    self.end(cycle, "expired", "deadline_reached")
                    changed.append(cycle["cycle_id"])
        return changed

    def end(self, cycle: dict[str, Any], status: str, reason: str) -> None:
        cycle.update(
            status=status,
            end_reason=reason,
            freeze_released_at=self.clock().isoformat(),
            stop_complete=False,
        )
        for worker in cycle["workers"].values():
            if worker["state"] != "stopped":
                worker["state"] = "stop_requested"
