"""Durable manual-parent notifications and receiver routing for progression events."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from board import notifiers, receivers
from board.access import ownership_fence
from board.storage import directory
from herdr_labels.coordination import owner_token
from herdr_labels.service import registry
from lib.identity import PARENT_ENGINES
from lib.ops_review.progression import read
from lib.review_settings import run_id
from parent.policy import digest, number
from parent.state import Store

KINDS = {"major", "parent_wait", "completed", "publication_warning"}
# Controller reasons can include local exceptions or agent-written text. Only
# fixed reasons and the fixed part of structured diagnostics cross this boundary.
PUBLIC_REASONS = {
    "auto_merge_disabled",
    "checkout_interrupted",
    "ci_timeout",
    "conflict",
    "creating_worker_changed",
    "draft",
    "issue_not_completed",
    "knowledge CI timeout",
    "knowledge PR HEAD changed; confirm new evidence",
    "knowledge merge result changed",
    "main_ci_cancelled_before_start",
    "main_ci_failed",
    "main_ci_timeout",
    "major findings require parent instruction",
    "mergeability_unknown",
    "needs-human",
    "observation_failed",
    "previous major findings require explicit continuation",
    "PR merged outside this progression",
    "PR is no longer open",
    "recovery_required",
    "repair_capacity_deferred",
    "repair_limit",
    "review result is saved; display publication failed",
    "review_capacity_deferred",
    "review_failed",
    "target_stopped",
}


def delivery(state: str, engine: str | None, at: float) -> dict[str, Any]:
    return {
        "state": state,
        "reason": None if state == "notified" else state,
        "engine": engine,
        "at": at,
    }


def record(state: dict[str, Any], event: dict[str, Any], at: float) -> dict[str, Any]:
    reason = event.get("reason")
    if reason is not None:
        code = reason.partition(":")[0] if isinstance(reason, str) else ""
        reason = code if code in PUBLIC_REASONS else "details_in_merge_status"
    engine = state["owner"].partition(":")[0]
    pr = event.get("pr")
    return {
        "progression_id": run_id(state["id"]),
        "event_id": number(event["id"]),
        "kind": event["kind"],
        "pr": None if pr is None else number(pr),
        "reason": reason,
        "parent_issue": number(state["request"]["parent_issue"]),
        "repo": state["request"]["repo"].casefold(),
        "owner": owner_token(state["owner"]),
        "generation": state["generation"],
        "created_at": at,
        "acknowledged": False,
        # A restart after this write cannot know whether the route accepted it.
        "delivery": delivery(
            "notification_uncertain", engine if engine in PARENT_ENGINES else None, at
        ),
    }


def deliver_bell(
    env: Mapping[str, str], store: Store, notice: dict[str, Any], owner: str
) -> str:
    engine = notice["delivery"]["engine"]
    if engine == "agy":
        return "awaiting_hook"
    send = {"claude": notifiers.claude, "codex": notifiers.codex}.get(engine)
    if send is None:
        return "route_unavailable"
    failure = "route_unavailable"
    try:
        # Board routes are scoped to their own registry. An alternate parent
        # state directory cannot borrow a matching identity from that registry.
        if store.directory.absolute() != directory(env).parent / "parent-sessions":
            return failure
        repo, parent = notice["repo"], notice["parent_issue"]
        generation = notice["generation"]
        with ownership_fence(env):
            row = registry(env).get(f"{repo}#{parent}", {})
            if (
                row.get("status") != "active"
                or row.get("session") != owner
                or row.get("generation") != generation
            ):
                return "owner_changed"
            assignment = receivers.active_assignment(env, repo, parent)
            if assignment is None:
                return failure
            identifier, event_id = notice["progression_id"], notice["event_id"]
            pr = "" if notice["pr"] is None else f" PR #{notice['pr']}"
            reason = f" ({notice['reason']})" if notice["reason"] else ""
            bell = receivers.ReceiverBell(
                repo=repo,
                parent=parent,
                assignment_id=assignment.id,
                engine=engine,
                owner_session=owner,
                owner_generation=generation,
                idempotency_key=digest(
                    ["merge-progression", identifier, event_id, owner, generation]
                ),
                bell=(
                    f"[merge-progression:{identifier}#{event_id}] parent #{parent}{pr}"
                    f" {notice['kind']}{reason}: parent --parent {parent}"
                    f" merge-status --run-id {identifier}"
                ),
            )
            failure = "notification_uncertain"
            result = send(env, bell)
            if result.accepted:
                return "notified"
            if result.reason == "route_unavailable":
                return "route_unavailable"
            if result.reason == "receiver_not_accepting":
                return "receiver_not_accepting"
    except Exception:
        # Route/queue diagnostics can contain private destinations. Persist
        # only the result that can be established at this delivery boundary.
        return failure
    return failure


def _resolved(notice: dict[str, Any], state: dict[str, Any]) -> bool:
    kind, pr = notice["kind"], notice["pr"]
    status: str = state["status"]
    acknowledgements = state.get("notification_acknowledgements", {})
    if (
        notice["acknowledged"]
        or notice["event_id"]
        <= acknowledgements.get("run" if pr is None else str(pr), 0)
        or status == "transferred"
    ):
        return True
    if kind == "major":
        review_stage: str = state["review"]["stage"]
        return review_stage != "parent_wait"
    if kind == "completed":
        return status != "completed"
    if kind == "parent_wait":
        stage: str | None = state["targets"].get(str(pr), {}).get("stage")
        return stage == "transferred" or (
            stage != "parent_wait" and status != "parent_wait"
        )
    return False


def notifications(
    store: Store, row: dict[str, Any], *, state: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Project current-owner notices and acknowledgements without writing state."""
    if row.get("status") != "active" or row.get("dispatch"):
        return []
    owner, generation = owner_token(row["session"]), row["generation"]
    states = {} if state is None else {state["id"]: state}
    result = []
    for notice in row.get("progression_notifications", {}).values():
        if notice.get("owner") != owner or notice.get("generation") != generation:
            continue  # Includes legacy dispatcher events without a manual owner.
        identifier = notice["progression_id"]
        if state is not None and identifier != state["id"]:
            continue
        if identifier not in states:
            try:
                states[identifier] = read(
                    store.directory / "progressions" / run_id(identifier) / "state.json"
                )
            except (OSError, ValueError):
                raise ValueError(
                    "progression notification state is unavailable"
                ) from None
        current = states[identifier]
        if (
            current["id"] != identifier
            or current["owner"] != row["session"]
            or current["generation"] != generation
            or current["request"]["repo"].casefold() != notice["repo"]
            or current["request"]["parent_issue"] != notice["parent_issue"]
        ):
            continue
        events, event_id = current["events"], notice["event_id"]
        if 1 <= event_id <= len(events) and all(
            events[event_id - 1][field] == notice[key]
            for field, key in (("id", "event_id"), ("kind", "kind"), ("pr", "pr"))
        ):
            result.append({**notice, "acknowledged": _resolved(notice, current)})
    return sorted(
        result,
        key=lambda item: (item["created_at"], item["progression_id"], item["event_id"]),
    )
