"""Engine-specific bells. No fallback transport and no session launch."""

from __future__ import annotations

import json
import socket
import stat
import subprocess
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from herdr_labels.coordination import locked
from lib import codex_queue

from . import receivers
from .delivery import Notification, NotificationResult, Notifiers
from .errors import BoardError
from .safety import public

TRANSPORT_TIMEOUT = 5


def claude(env: Mapping[str, str], notification: Notification) -> NotificationResult:
    with locked(receivers.root(env) / "routes.lock"):
        try:
            record = receivers.resolve(env, notification)
            if not record["accepting"]:
                return NotificationResult(False, "receiver_not_accepting", True)
            path = Path(record["socket"])
            info = path.stat()
            if not path.is_absolute() or not stat.S_ISSOCK(info.st_mode):
                raise BoardError("Claude socket unavailable")
            token = record["token"]
            if not isinstance(token, str) or not token:
                raise BoardError("Claude credential unavailable")
        except (BoardError, OSError, ValueError, KeyError, TypeError):
            return NotificationResult(False, "route_unavailable", True)
        payload = [
            {"type": "auth", "token": token},
            {
                "type": "user",
                "message": {"role": "user", "content": notification.bell},
                "from": "tennis-board",
                "uuid": str(uuid.UUID(notification.idempotency_key[:32])),
                "msg_id": notification.idempotency_key,
            },
        ]
        data = (
            "\n".join(json.dumps(item, ensure_ascii=False) for item in payload) + "\n"
        ).encode()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(TRANSPORT_TIMEOUT)
            try:
                client.connect(str(path))
            except OSError:
                return NotificationResult(False, "route_unavailable", True)
            try:
                client.sendall(data)
            except OSError:
                # sendall cannot prove how much the peer received.
                return NotificationResult(False, "notification_uncertain")
        # There is no synchronous receipt. This never marks a post read.
        return NotificationResult(True)


def _command(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, check=False, **kwargs)


def _accepted(task: dict[str, Any]) -> NotificationResult:
    delivery = task["deliveries"][-1] if task["deliveries"] else {}
    if delivery.get("queue_id") or delivery.get("turn_id"):
        return NotificationResult(True)
    return NotificationResult(False, "notification_uncertain")


def codex(env: Mapping[str, str], notification: Notification) -> NotificationResult:
    with locked(receivers.root(env) / "routes.lock"):
        try:
            record = receivers.resolve(env, notification)
            if not Path(record["codex"]).is_file():
                raise BoardError("Codex CLI unavailable")
            task_dir = receivers.root(env) / "queue" / notification.idempotency_key
            task_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            task_path = task_dir / "task.json"
            task = receivers.read(task_path)
            if task is not None:
                existing = task
                codex_queue.reconcile(
                    existing, lambda: receivers.write(task_path, existing)
                )
                return _accepted(task)
            task = {
                **record,
                "post_id": notification.post_id,
                "name": notification.name,
                "prompt": notification.bell,
                "deliveries": [],
                "queue_timeout_seconds": TRANSPORT_TIMEOUT,
            }
            _, task["observed_boundary"] = codex_queue.history(task, None)
            if not task["observed_boundary"]["line_complete"]:
                raise BoardError("Codex rollout is incomplete")
        except (BoardError, OSError, ValueError, KeyError, TypeError):
            return NotificationResult(False, "route_unavailable", True)

        def persist() -> None:
            receivers.write(task_path, task)

        codex_queue.enqueue(
            task,
            f"[board-delivery:{notification.idempotency_key}]",
            persist=persist,
            command=_command,
        )
        codex_queue.reconcile(task, persist)
        return _accepted(task)


def configured(env: Mapping[str, str]) -> Notifiers:
    return {
        "claude": lambda bell: claude(env, bell),
        "codex": lambda bell: codex(env, bell),
    }


def receipts(env: Mapping[str, str]) -> dict[str, Any]:
    """Refresh phases on an explicit board event; never send or poll."""
    from .access import as_parent

    with as_parent(env) as (connection, caller, parents):
        permitted = {
            (post, assignment)
            for post, assignment in connection.execute(
                """SELECT r.post_id, r.assignment_id FROM recipients r JOIN posts p ON p.id=r.post_id
                   WHERE p.sender_assignment_id=? OR r.assignment_id=?""",
                (caller.id, caller.id),
            )
        }
    result = []
    with locked(receivers.root(env) / "routes.lock"):
        for path in sorted((receivers.root(env) / "queue").glob("*/task.json")):
            task = receivers.read(path)
            if (
                task is None
                or (task["post_id"], task["binding"]["assignment_id"]) not in permitted
            ):
                continue
            codex_queue.reconcile(task, persist=lambda: None)
            receivers.write(path, task)
            delivery = task["deliveries"][-1]
            result.append(
                {
                    "post_id": task["post_id"],
                    "name": task["name"],
                    "phase": delivery["phase"],
                    "history_blocked": bool(task.get("blocked_reason")),
                }
            )
    response = {"receipts": result}
    public(response, parents, env)
    return response
