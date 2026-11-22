"""Durable automatic progression. External effects are supplied by the backend.

Only the controller lock is held while running; this module never holds the
parent registry lock. An intent is durable before each non-repeatable effect.
"""

from __future__ import annotations

import copy
import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from lib.d_capacity import CapacityError
from parent.state import Store, unique_object

POLL_SECONDS = 60
CI_TIMEOUT = 6 * 3600
REPAIR_LIMIT = 3
KNOWLEDGE_MERGE = "knowledge_merge"
INACTIVE_STAGES = {"parent_wait", "completed", "transferred"}


class RateWait(Exception):
    def __init__(self, resume_at: float, reason: str) -> None:
        super().__init__(reason)
        self.resume_at = resume_at


class MergeHold(ValueError):
    def __init__(self, reason: str, *, retry: bool = False) -> None:
        super().__init__(reason)
        self.retry = retry


class Backend(Protocol):
    def local(self, state: dict[str, Any]) -> str | None: ...
    def review(self, state: dict[str, Any], directory: Path) -> dict[str, Any]: ...
    def pr(self, number: int) -> dict[str, Any]: ...
    def ci(self, number: int, head: str, *, main: bool = False) -> dict[str, Any]: ...
    def shape(self, target: dict[str, Any], pr: dict[str, Any]) -> str | None: ...
    def human(self, state: dict[str, Any], target: dict[str, Any]) -> str | None: ...
    def dependency(self, target: dict[str, Any]) -> bool: ...
    def failure(self, target: dict[str, Any], ci: dict[str, Any]) -> dict[str, Any]: ...
    def repair(
        self, state: dict[str, Any], target: dict[str, Any], attempt: dict[str, Any]
    ) -> dict[str, Any]: ...
    def recover_repair(
        self, state: dict[str, Any], target: dict[str, Any], attempt: dict[str, Any]
    ) -> dict[str, Any] | None: ...
    def merge(self, target: dict[str, Any], head: str) -> None: ...
    def issues_completed(self, target: dict[str, Any]) -> bool: ...
    def knowledge_ci(self, pr: dict[str, Any]) -> dict[str, Any]: ...
    def merge_knowledge(
        self,
        state: dict[str, Any],
        target: dict[str, Any],
        ci: dict[str, Any],
        record_intent: Callable[[], None],
    ) -> None: ...
    def confirm_knowledge(self, target: dict[str, Any], pr: dict[str, Any]) -> None: ...
    def issue_states(self, issues: list[int]) -> dict[str, Any]: ...
    def changes(self, state: dict[str, Any]) -> dict[str, Any]: ...
    def notification_changes(
        self, state: dict[str, Any], pr: int | None
    ) -> dict[str, Any]: ...
    def notify(self, state: dict[str, Any], event: dict[str, Any]) -> None: ...


def read(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=unique_object
    )
    if not isinstance(value, dict):
        raise ValueError("progression state must be an object")
    return value


def save(path: Path, value: dict[str, Any]) -> None:
    Store.write_json(path, value)


def review_attempt(execution_path: Path) -> int:
    attempt = read(execution_path).get("attempt", 0)
    if type(attempt) is not int or attempt < 0:
        raise ValueError("invalid saved review launch attempt")
    return attempt


def review_launch_directory(review_dir: Path, attempt: int) -> Path:
    return review_dir / "launches" / str(attempt) if attempt else review_dir


def new_state(
    request: dict[str, Any], *, owner: str, generation: int
) -> dict[str, Any]:
    if request.get("schema_version") != 4 or not request.get("prs"):
        raise ValueError("automatic progression requires a nonempty version 4 request")
    knowledge = request.get("kind") == KNOWLEDGE_MERGE
    return {
        "schema_version": 1,
        "id": request["review_run_id"],
        "request": copy.deepcopy(request),
        "owner": owner,
        "generation": generation,
        "status": "requested",
        "review": {"stage": "not_required", "round": 0}
        if knowledge
        else {"stage": "requested", "round": 1},
        "targets": {
            str(p["number"]): {
                **copy.deepcopy(p),
                "stage": "ci_wait",
                "attempts": [],
                "next_poll": 0,
                "ci_since": None,
                "ci_head": None,
            }
            for p in request["prs"]
        },
        "decisions": [],
        "events": [],
    }


class Engine:
    def __init__(
        self,
        path: Path,
        backend: Backend,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.path, self.backend, self.clock, self.sleep = path, backend, clock, sleep
        self.state = read(path)
        if self.state.get("schema_version") != 1:
            raise ValueError("unknown progression state version")
        numbers = [p["number"] for p in self.state["request"]["prs"]]
        if (
            not numbers
            or len(numbers) != len(set(numbers))
            or set(self.state["targets"]) != {str(n) for n in numbers}
        ):
            raise ValueError("progression targets differ from request")
        if self.state["review"].get("stage") not in {
            "requested",
            "running",
            "accepted",
            "parent_wait",
            "not_required",
        }:
            raise ValueError("unknown review stage")
        self.knowledge = self.state["request"].get("kind") == KNOWLEDGE_MERGE
        for n, target in self.state["targets"].items():
            stage = target["stage"]
            if (
                str(target["number"]) != n
                or stage
                not in INACTIVE_STAGES
                | {
                    "ci_wait",
                    "dependency_wait",
                    "repair_pending",
                    "merge_pending",
                    "post_merge_wait",
                }
                or (stage == "transferred" and not self.knowledge)
            ):
                raise ValueError("unknown progression target or stage")
        confirmation = self.state["request"].get("kind") == "post_merge_confirmation"
        if (self.state["review"]["stage"] == "not_required") != (
            confirmation or self.knowledge
        ):
            raise ValueError("unexpected review omission")
        if confirmation and any(
            t["stage"] not in {"post_merge_wait", "parent_wait", "completed"}
            for t in self.state["targets"].values()
        ):
            raise ValueError("confirmation cannot authorize review, repair or merge")
        if self.knowledge and (
            not self.state["request"].get("evidence", "").strip()
            or any(
                t["stage"] == "repair_pending" or t["attempts"]
                for t in self.state["targets"].values()
            )
        ):
            raise ValueError(
                "knowledge merge requires evidence and forbids repair runs"
            )

    def save(self) -> None:
        save(self.path, self.state)

    def emit(
        self, kind: str, *, pr: int | None = None, reason: str | None = None
    ) -> None:
        event = {
            "id": len(self.state["events"]) + 1,
            "kind": kind,
            "pr": pr,
            "reason": reason,
            "delivered": False,
            "changes": self.backend.notification_changes(self.state, pr),
        }
        self.state["events"].append(event)
        self.save()
        self.deliver()

    def deliver(self) -> None:
        for event in self.state["events"]:
            if not event["delivered"]:
                # Backend delivery is keyed by (progression ID, event ID).
                self.backend.notify(self.state, event)
                event["delivered"] = True
                self.save()

    def pause(
        self, target: dict[str, Any], reason: str, *, resume: str | None = None
    ) -> None:
        target.update(
            resume_stage=resume or target["stage"], stage="parent_wait", reason=reason
        )
        if all(t["stage"] in INACTIVE_STAGES for t in self.state["targets"].values()):
            self.state["status"] = "parent_wait"
        self.emit("parent_wait", pr=target["number"], reason=reason)

    def block_start(self, pr: int, reason: str) -> None:
        if self.state["status"] in {"completed", "transferred"}:
            return
        # No step ran. Preserve each active stage for an explicit continuation
        # and leave already-paused peers and completed targets as they were.
        for target in self.state["targets"].values():
            if target["stage"] not in INACTIVE_STAGES:
                target.update(
                    resume_stage=target["stage"], stage="parent_wait", reason=reason
                )
        self.state["status"] = "parent_wait"
        self.emit("parent_wait", pr=pr, reason=reason)

    def continue_run(
        self,
        evidence: str,
        *,
        prs: list[int] | None = None,
        prepare: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not evidence.strip():
            raise ValueError("continuation requires the parent's instruction")
        if self.state["status"] == "transferred":
            raise ValueError("knowledge merge targets transferred to formal review")
        if prs is not None:
            selected = {str(n) for n in prs}
            if (
                not selected
                or len(selected) != len(prs)
                or not selected <= self.state["targets"].keys()
                or any(
                    self.state["targets"][n]["stage"] != "parent_wait" for n in selected
                )
            ):
                raise ValueError("--prs must select unique PRs awaiting the parent")
            findings = self.state["review"].get("result", {}).get("major", [])
            if any(
                str(n) not in selected for finding in findings for n in finding["prs"]
            ):
                raise ValueError("major continuation must cover all affected PRs")
        reason = self.backend.local(self.state)
        if reason:
            raise ValueError(reason)
        if prepare is not None:
            prepare(self.state)
        self.state["decisions"].append(
            {
                "instruction": evidence,
                "prs": prs,
                "changes": self.backend.changes(self.state),
            }
        )
        review = self.state["review"]
        if (
            review["stage"] == "parent_wait"
            and review.get("result", {}).get("verdict") == "major"
        ):
            review.update(stage="accepted", major_repaired=evidence)
        elif review["stage"] == "parent_wait":
            # Re-enter the same round. Its durable launch intent either yields
            # the saved result or blocks an uncertain second invocation.
            review.update(stage="running")
        for target in self.state["targets"].values():
            if prs is not None and target["number"] not in prs:
                continue
            if target["stage"] == "parent_wait":
                target["stage"] = target.pop("resume_stage")
                target.pop("reason", None)
                target["next_poll"] = 0
                target["ci_since"] = None
        self.state["status"] = "running"
        self.save()

    def run(self) -> dict[str, Any]:
        if self.state["status"] == "transferred":
            return self.state
        self.deliver()
        if self.state["status"] == "completed":
            return self.state
        review = self.state["review"]
        if review["stage"] in {"requested", "running"} and self.state.get(
            "review_history"
        ):
            directory = self.path.parent / f"review-{review['round']}"
            if not (directory / "execution.json").exists():
                prior = self.state["review_history"][-1]
                if prior.get("result", {}).get("verdict") == "major":
                    review = self.state["review"] = copy.deepcopy(prior)
                    self.state["status"] = "parent_wait"
                    self.emit(
                        "major",
                        reason="previous major findings require explicit continuation",
                    )
                    return self.state
        while review["stage"] in {"requested", "running"}:
            try:
                reason = self.backend.local(self.state)
                if reason:
                    self.state["status"] = "parent_wait"
                    self.emit("parent_wait", reason=reason)
                    return self.state
                for target in self.state["targets"].values():
                    reason = self.backend.human(self.state, target)
                    if reason:
                        self.state["status"] = "parent_wait"
                        self.emit("parent_wait", pr=target["number"], reason=reason)
                        return self.state
                review["stage"] = "running"
                self.state["status"] = "running"
                self.save()
                # Re-entry can launch again only after a proven pre-CLI refusal.
                result = self.backend.review(self.state, self.path.parent)
                review.pop("error", None)
                review["result"] = result
                if result["verdict"] == "major":
                    review["changes"] = self.backend.changes(self.state)
                review["stage"] = (
                    "accepted" if result["verdict"] == "approve" else "parent_wait"
                )
                if review["stage"] == "parent_wait":
                    self.state["status"] = "parent_wait"
                    self.emit(
                        "major", reason="major findings require parent instruction"
                    )
                    return self.state
                self.save()
                if result.get("publication_errors"):
                    self.emit(
                        "publication_warning",
                        reason="review result is saved; display publication failed",
                    )
            except RateWait as exc:
                self.wait_rate(exc, review)
            except Exception as exc:
                review.update(stage="parent_wait", error=str(exc))
                self.state["status"] = "parent_wait"
                reason = (
                    "review_capacity_deferred"
                    if isinstance(exc, CapacityError)
                    else "review_failed"
                )
                self.emit("parent_wait", reason=f"{reason}: {exc}")
                return self.state
        if review["stage"] == "parent_wait":
            return self.state
        result = review.get("result", {})
        if (
            review["stage"] != "not_required"
            and result.get("verdict") != "approve"
            and not (result.get("verdict") == "major" and review.get("major_repaired"))
        ):
            raise ValueError("review has no approval or explicit major continuation")
        self.state["status"] = "running"
        self.save()
        while True:
            for target in self.state["targets"].values():
                if (
                    target["stage"] in INACTIVE_STAGES
                    or target["next_poll"] > self.clock()
                ):
                    continue
                try:
                    target.pop("rate_limit", None)
                    self.step(target)
                except RateWait as exc:
                    target.update(
                        next_poll=exc.resume_at,
                        rate_limit={"resume_at": exc.resume_at, "reason": str(exc)},
                    )
                    self.save()
                except CapacityError as exc:
                    self.pause(target, f"repair_capacity_deferred: {exc}")
                except Exception as exc:
                    target["error"] = str(exc)
                    reason = (
                        "recovery_required"
                        if target["stage"] in {"merge_pending", "repair_pending"}
                        else "observation_failed"
                    )
                    self.pause(target, reason)
            active = [
                t
                for t in self.state["targets"].values()
                if t["stage"] not in INACTIVE_STAGES
            ]
            if not active:
                completed = all(
                    t["stage"] in {"completed", "transferred"}
                    for t in self.state["targets"].values()
                )
                self.state["status"] = "completed" if completed else "parent_wait"
                if completed:
                    self.emit("completed")
                else:
                    self.save()
                return self.state
            delay = min(t["next_poll"] for t in active) - self.clock()
            if delay > 0:
                self.sleep(min(POLL_SECONDS, delay))

    def wait_rate(self, error: RateWait, target: dict[str, Any]) -> None:
        target["rate_limit"] = {"resume_at": error.resume_at, "reason": str(error)}
        self.save()
        while self.clock() < error.resume_at:
            self.sleep(min(POLL_SECONDS, error.resume_at - self.clock()))
        target.pop("rate_limit", None)
        self.save()

    def later(self, target: dict[str, Any]) -> None:
        target["next_poll"] = self.clock() + POLL_SECONDS
        self.save()

    def step(self, target: dict[str, Any]) -> None:
        stage = target["stage"]
        if stage in {"merge_pending", "repair_pending", "post_merge_wait"}:
            blocked = self.backend.local(self.state) or self.backend.human(
                self.state, target
            )
            if blocked:
                self.pause(target, blocked)
                return
        if stage == "merge_pending":
            self.confirm_merge(target)
            return
        if stage == "repair_pending":
            report = self.backend.recover_repair(
                self.state, target, target["attempts"][-1]
            )
            if report is None:
                self.pause(target, "recovery_required")
            elif report.get("outcome") == "reserved":
                # #2175 persisted this reservation before claiming a launch.
                # Reusing its ID cannot start a second runtime.
                report = self.backend.repair(self.state, target, target["attempts"][-1])
                if report.get("outcome") == "running":
                    self.later(target)
                else:
                    self.repaired(target, report)
            elif report.get("outcome") == "running":
                self.later(target)
            else:
                self.repaired(target, report)
            return
        if stage == "post_merge_wait":
            self.post_merge(target)
            return
        if self.knowledge:
            self.step_knowledge(target)
            return
        reason = self.backend.local(self.state)
        if reason:
            self.pause(target, reason)
            return
        pr = self.backend.pr(target["number"])
        if pr.get("merged"):
            self.pause(target, "PR merged outside this progression")
            return
        head = pr["head"]["sha"]
        if target["ci_head"] != head or target["ci_since"] is None:
            target.update(ci_head=head, ci_since=self.clock())
        ci = self.backend.ci(target["number"], head)
        target["ci"] = ci
        if ci["status"] == "pending":
            if self.clock() - target["ci_since"] >= CI_TIMEOUT:
                self.pause(target, "ci_timeout", resume="ci_wait")
            else:
                target["stage"] = "ci_wait"
                self.later(target)
            return
        if ci["status"] == "failure":
            self.start_repair(target, "ci_failure", ci=ci)
            return
        if ci["status"] != "success":
            raise ValueError("unknown CI observation")
        # Current PR identity is read only after the CI succeeds. A new head
        # goes back to CI; it never triggers a new review or resolution record.
        current = self.backend.pr(target["number"])
        if current["head"]["sha"] != head:
            target["stage"] = "ci_wait"
            target["next_poll"] = 0
            self.save()
            return
        reason = self.backend.shape(target, current)
        if reason:
            self.start_repair(target, reason)
            return
        reason = self.backend.human(self.state, target)
        if reason:
            self.pause(target, reason)
            return
        if not self.backend.dependency(target):
            target["stage"] = "dependency_wait"
            self.later(target)
            return
        blocked = self.backend.local(self.state)
        if blocked:
            self.pause(target, blocked, resume="ci_wait")
            return
        target.update(stage="merge_pending", merge_head=head)
        self.save()
        try:
            self.backend.merge(target, head)
        except MergeHold:
            target.pop("merge_head")
            target["stage"] = "ci_wait"
            self.later(target)
            return
        except Exception as exc:
            target["merge_error"] = str(exc)
            self.save()
        self.confirm_merge(target)

    def step_knowledge(self, target: dict[str, Any]) -> None:
        reason = self.backend.local(self.state) or self.backend.human(
            self.state, target
        )
        if reason:
            self.pause(target, reason)
            return
        pr = self.backend.pr(target["number"])
        if reason := self.backend.shape(target, pr):
            self.pause(target, reason, resume="ci_wait")
            return
        head = pr["head"]["sha"]
        if head != target.get("evidence_head", target["head_sha"]):
            self.pause(
                target,
                "knowledge PR HEAD changed; confirm new evidence",
                resume="ci_wait",
            )
            return
        if target["ci_since"] is None:
            target.update(ci_head=head, ci_since=self.clock())
        ci = self.backend.knowledge_ci(pr)
        target["ci"] = ci
        if ci["status"] == "failure":
            self.pause(target, "; ".join(ci["reasons"]), resume="ci_wait")
            return
        if ci["status"] == "pending":
            self.wait_knowledge_ci(target, "; ".join(ci["reasons"]))
            return
        if ci["status"] != "success":
            raise ValueError("unknown knowledge CI observation")
        if not self.backend.dependency(target):
            target["stage"] = "dependency_wait"
            self.later(target)
            return

        def record_intent() -> None:
            target.update(
                stage="merge_pending", merge_head=head, merge_base=ci["base_sha"]
            )
            self.save()

        try:
            self.backend.merge_knowledge(self.state, target, ci, record_intent)
        except MergeHold as exc:
            if exc.retry:
                self.wait_knowledge_ci(target, str(exc))
            else:
                self.pause(target, str(exc), resume="ci_wait")
            return
        except RateWait:
            raise
        except Exception as exc:
            if target["stage"] != "merge_pending":
                raise
            target["merge_error"] = str(exc)
            self.save()
        self.confirm_merge(target)

    def wait_knowledge_ci(self, target: dict[str, Any], reason: str) -> None:
        target["ci"].update(status="pending", reasons=[reason])
        if self.clock() - target["ci_since"] >= CI_TIMEOUT:
            self.pause(target, f"knowledge CI timeout: {reason}", resume="ci_wait")
        else:
            target["stage"] = "ci_wait"
            self.later(target)

    def start_repair(
        self, target: dict[str, Any], reason: str, *, ci: dict[str, Any] | None = None
    ) -> None:
        # Human controls are checked even when the cheap gate short-circuited.
        blocked = self.backend.local(self.state) or self.backend.human(
            self.state, target
        )
        if blocked:
            self.pause(target, blocked, resume="ci_wait")
            return
        if len(target["attempts"]) >= REPAIR_LIMIT:
            self.pause(target, "repair_limit", resume="ci_wait")
            return
        details = {"reason": reason, "head": target["ci_head"]}
        if ci is not None:
            details.update(self.backend.failure(target, ci))
        attempt = {"id": uuid.uuid4().hex, "details": details}
        target["attempts"].append(attempt)
        target["stage"] = "repair_pending"
        self.save()
        report = self.backend.repair(self.state, target, attempt)
        if report.get("outcome") == "running":
            self.later(target)
        else:
            self.repaired(target, report)

    def repaired(self, target: dict[str, Any], report: dict[str, Any]) -> None:
        target["attempts"][-1]["report"] = report
        if report["outcome"] == "needs_parent":
            self.pause(target, report["parent_request"], resume="ci_wait")
        elif report["outcome"] == "repaired":
            target.update(stage="ci_wait", next_poll=0, ci_since=None)
            self.save()
        else:
            raise ValueError("unknown repair outcome")

    def confirm_merge(self, target: dict[str, Any]) -> None:
        pr = self.backend.pr(target["number"])
        if (
            pr.get("merged") is not True
            or not pr.get("merge_commit_sha")
            or pr["head"]["sha"] != target["merge_head"]
        ):
            self.pause(target, "recovery_required")
            return
        if self.knowledge:
            self.backend.confirm_knowledge(target, pr)
        target.update(
            stage="post_merge_wait",
            merge_commit=pr["merge_commit_sha"],
            merged_at=self.clock(),
            next_poll=0,
        )
        self.save()

    def post_merge(self, target: dict[str, Any]) -> None:
        if target.get("merged_at") is None:
            target["merged_at"] = self.clock()
            self.save()
        if self.knowledge:
            states = self.backend.issue_states(target["related_issues"])
            target["issue_states"] = states
            issues_closed = all(
                states[str(n)] == {"state": "closed", "state_reason": "completed"}
                for n in target["issues"]
            )
        elif not self.backend.issues_completed(target):
            self.pause(target, "issue_not_completed")
            return
        else:
            issues_closed = True
        ci = self.backend.ci(target["number"], target["merge_commit"], main=True)
        target["main_ci"] = ci
        if ci["status"] == "success" and issues_closed:
            if self.knowledge:
                pr = self.backend.pr(target["number"])
                if (
                    pr.get("merged") is not True
                    or pr.get("merge_commit_sha") != target["merge_commit"]
                    or pr["head"]["sha"] != target["merge_head"]
                ):
                    self.pause(target, "knowledge merge result changed")
                    return
                self.backend.confirm_knowledge(target, pr)
            target.update(stage="completed", issues_confirmed=bool(target["issues"]))
            self.save()
        elif ci["status"] == "failure":
            self.pause(target, "main_ci_failed")
        elif ci["status"] not in {"pending", "success"}:
            raise ValueError("unknown main CI observation")
        elif self.clock() - target["merged_at"] >= CI_TIMEOUT:
            self.pause(
                target, "main_ci_timeout" if issues_closed else "issue_not_completed"
            )
        else:
            self.later(target)
