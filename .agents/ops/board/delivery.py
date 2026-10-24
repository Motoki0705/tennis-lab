"""Injectable, durable bell dispatch. No runtime transports are installed here."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from herdr_labels.coordination import locked
from lib.identity import session_identity

from .access import parent_state
from .errors import BoardError
from .names import for_parent, resolve_name, timestamp
from .records import REASONS, post_row, recipient_views
from .safety import public
from .storage import Store, directory


@dataclass(frozen=True)
class Notification:
    post_id: str
    assignment_id: int
    name: str
    repo: str
    parent: int
    engine: str
    idempotency_key: str
    sender_name: str
    sender_repo: str | None
    sender_parent: int | None
    kind: str

    @property
    def bell(self) -> str:
        sender = (
            "human"
            if self.sender_parent is None
            else f"{self.sender_name} ({self.sender_repo}#{self.sender_parent})"
        )
        return f"[board:{self.post_id}] {sender} {self.kind}: board read {self.post_id}"


@dataclass(frozen=True)
class NotificationResult:
    accepted: bool
    reason: str | None = None
    # True certifies that the route did not accept a bell. An ambiguous send
    # must never claim this, even when an exception looks transient.
    retryable: bool = False


Notifiers = Mapping[str, Callable[[Notification], NotificationResult]]


@dataclass
class FakeNotifier:
    """In-memory route acceptance, deduplication and known-unsent failures."""

    accepting: bool = True
    fail_once_for: set[int] = field(default_factory=set)
    calls: list[Notification] = field(default_factory=list)
    accepted: dict[str, Notification] = field(default_factory=dict)

    def __call__(self, notification: Notification) -> NotificationResult:
        self.calls.append(notification)
        if notification.idempotency_key in self.accepted:
            return NotificationResult(True)
        if not self.accepting:
            return NotificationResult(False, "receiver_not_accepting", True)
        if notification.assignment_id in self.fail_once_for:
            self.fail_once_for.remove(notification.assignment_id)
            return NotificationResult(False, "route_unavailable", True)
        self.accepted[notification.idempotency_key] = notification
        return NotificationResult(True)


def _result(value: Any) -> NotificationResult:
    if (
        isinstance(value, NotificationResult)
        and type(value.accepted) is bool
        and type(value.retryable) is bool
    ):
        if value.accepted and value.reason is None and not value.retryable:
            return value
        if (
            not value.accepted
            and value.reason in REASONS
            and value.reason not in {"recipient_unavailable", "recipient_retired"}
            and not (value.reason == "notification_uncertain" and value.retryable)
        ):
            return value
    return NotificationResult(False, "notification_uncertain")


def held_for_name(env: Mapping[str, str], name: str) -> list[dict[str, Any]]:
    """Read held recipient rows without clearing them or attempting delivery."""
    store = Store(directory(env))
    with parent_state(env) as parents, store.read() as connection:
        assignment = resolve_name(connection, name)
        ids = connection.execute(
            "SELECT post_id FROM deliveries WHERE assignment_id=? AND state='held' ORDER BY post_id",
            (assignment.id,),
        ).fetchall()
        result = [
            dict(post_id=post_id, **row)
            for (post_id,) in ids
            for row in recipient_views(connection, post_id)
            if row["assignment_id"] == assignment.id
        ]
        public(result, parents, env)
        return result


def dispatch(
    env: Mapping[str, str],
    notifiers: Notifiers,
    *,
    post_id: str | None = None,
    assignment_id: int | None = None,
    held_only: bool = False,
    retry_only: bool = False,
) -> list[dict[str, Any]]:
    """Trusted hook/API entry. Recheck the destination owner for every attempt.

    A durable uncertainty marker is committed BEFORE calling a route. A process
    crash, exception or malformed result therefore cannot cause automatic
    duplicate sends. Only a certified unsent failure may be retried. Never hold
    a SQLite transaction while calling a notifier. With retry_only, recheck the
    active sender under each parent lock, leave held rows alone and report
    attempts/skips from inside the notification lock.
    """
    if set(notifiers) - {"claude", "codex"}:
        raise BoardError("unknown board notification engine")
    store = Store(directory(env))
    filters: list[str] = []
    params: list[Any] = []
    if post_id is not None:
        filters.append("post_id=?")
        params.append(post_id)
    if assignment_id is not None:
        filters.append("assignment_id=?")
        params.append(assignment_id)
    if held_only:
        filters.append("state='held'")
    where = " WHERE " + " AND ".join(filters) if filters else ""
    with locked(store.directory / "notifications.lock"):
        reports: dict[tuple[str, int], dict[str, Any]] = {}
        with store.read() as connection:
            items = connection.execute(
                "SELECT post_id, assignment_id FROM deliveries"
                + where
                + " ORDER BY post_id, assignment_id",
                params,
            ).fetchall()
        for target_post, target_id in items:
            report = {"attempted": False, "outcome": "skipped", "skip_reason": None}
            reports[target_post, target_id] = report
            if target_id is None:
                # The human receives posts through Web polling, without a bell.
                report["skip_reason"] = "human_recipient"
                continue
            with parent_state(env) as parents:
                with store.transaction() as connection:
                    if retry_only:
                        # Initial CLI authorization predates notifications.lock.
                        # Fence the caller again before any delivery mutation;
                        # this parent lock also spans the notifier and its result.
                        try:
                            caller_identity = session_identity(env)
                        except ValueError:
                            raise BoardError(
                                "invalid notification retry identity"
                            ) from None
                        post = post_row(connection, target_post)
                        sender = parents.get(
                            f"{post['sender_repo']}#{post['sender_parent']}"
                        )
                        sender_assignment = for_parent(
                            connection, post["sender_repo"], post["sender_parent"]
                        )
                        if (
                            sender is None
                            or sender["status"] != "active"
                            or sender["session"] != caller_identity
                            or sender_assignment is None
                            or sender_assignment.retired_at is not None
                            or sender_assignment.id != post["sender_assignment_id"]
                        ):
                            raise BoardError(
                                "only the active post sender may retry its notifications"
                            )
                    row = connection.execute(
                        "SELECT state, retryable, idempotency_key, reason FROM deliveries WHERE post_id=? AND assignment_id=?",
                        (target_post, target_id),
                    ).fetchone()
                    if row[0] == "notified":
                        report["skip_reason"] = "already_notified"
                        continue
                    if row[0] == "delivery_failed" and not row[1]:
                        report["skip_reason"] = (
                            "notification_uncertain"
                            if row[3] == "notification_uncertain"
                            else "not_retryable"
                        )
                        continue
                    recipient = next(
                        r
                        for r in recipient_views(connection, target_post)
                        if r["assignment_id"] == target_id
                    )
                    if recipient["read_at"] is not None:
                        report["skip_reason"] = "already_read"
                        continue  # Explicit read/response already made the bell unnecessary.
                    if retry_only and row[0] == "held":
                        report["skip_reason"] = "held"
                        continue
                    report["outcome"] = "failed"
                    owner = parents.get(f"{recipient['repo']}#{recipient['parent']}")
                    retired = connection.execute(
                        "SELECT retired_at FROM name_assignments WHERE id=?",
                        (target_id,),
                    ).fetchone()[0]
                    if (
                        retired is not None
                        or owner is None
                        or owner["status"] == "ended"
                    ):
                        report["skip_reason"] = "recipient_retired"
                        connection.execute(
                            "UPDATE deliveries SET state='delivery_failed', reason='recipient_retired', retryable=0 WHERE post_id=? AND assignment_id=?",
                            (target_post, target_id),
                        )
                        continue
                    if owner["status"] != "active":
                        report["skip_reason"] = "recipient_unavailable"
                        connection.execute(
                            "UPDATE deliveries SET state='held', reason='recipient_unavailable', retryable=1 WHERE post_id=? AND assignment_id=?",
                            (target_post, target_id),
                        )
                        continue
                    engine = owner["session"].partition(":")[0]
                    if engine not in {"claude", "codex"}:
                        raise BoardError("invalid recipient engine identity")
                    notifier = notifiers.get(engine)
                    if notifier is None:
                        report["skip_reason"] = "notifier_unconfigured"
                        connection.execute(
                            "UPDATE deliveries SET state='delivery_failed', reason='notifier_unconfigured', retryable=1, engine=? WHERE post_id=? AND assignment_id=?",
                            (engine, target_post, target_id),
                        )
                        continue
                    post = post_row(connection, target_post)
                    notification = Notification(
                        target_post,
                        target_id,
                        recipient["name"],
                        recipient["repo"],
                        recipient["parent"],
                        engine,
                        row[2],
                        post["sender_name"],
                        post["sender_repo"],
                        post["sender_parent"],
                        post["kind"],
                    )
                    public(
                        [notification.bell, notification.name, notification.repo],
                        parents,
                        env,
                    )
                    connection.execute(
                        """UPDATE deliveries SET state='delivery_failed', reason='notification_uncertain', retryable=0,
                           engine=?, attempts=attempts+1, attempted_at=? WHERE post_id=? AND assignment_id=?""",
                        (engine, timestamp(datetime.now(UTC)), target_post, target_id),
                    )
                report["attempted"] = True
                try:
                    outcome = _result(notifier(notification))
                except Exception:
                    # Exception text may contain private runtime destinations.
                    outcome = NotificationResult(False, "notification_uncertain")
                report["outcome"] = "notified" if outcome.accepted else "failed"
                with store.transaction() as connection:
                    connection.execute(
                        "UPDATE deliveries SET state=?, reason=?, retryable=?, notified_at=? WHERE post_id=? AND assignment_id=?",
                        (
                            "notified" if outcome.accepted else "delivery_failed",
                            outcome.reason,
                            int(outcome.retryable),
                            timestamp(datetime.now(UTC)) if outcome.accepted else None,
                            target_post,
                            target_id,
                        ),
                    )
        with parent_state(env) as parents, store.read() as connection:
            result = [
                dict(post_id=p, **row, **(reports[p, a] if retry_only else {}))
                for p, a in items
                for row in recipient_views(connection, p)
                if row["assignment_id"] == a
            ]
            public(result, parents, env)
            return result


def dispatch_held(
    env: Mapping[str, str], name: str, notifiers: Notifiers
) -> list[dict[str, Any]]:
    with Store(directory(env)).read() as connection:
        assignment = resolve_name(connection, name)
    return dispatch(env, notifiers, assignment_id=assignment.id, held_only=True)
