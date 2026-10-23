"""Parent restoration callbacks, always outside the registry writer lock."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .access import as_parent
from .delivery import dispatch_held
from .display import snapshot, update
from .notifiers import configured
from .receivers import bind_receiver


def restored(env: Mapping[str, str], name: str) -> dict[str, Any]:
    receiver = bind_receiver(env)
    deliveries = dispatch_held(env, name, configured(env))
    with as_parent(env) as (_, caller, _):
        assignment_id = caller.id
    displays = update(env, [assignment_id])
    failed = any(row["delivery_state"] == "delivery_failed" for row in deliveries)
    return {
        "status": "error"
        if failed or any(row["status"] == "error" for row in displays)
        else "ok",
        "receiver": receiver,
        "deliveries": deliveries,
        "inbox_display": displays,
        "inbox": snapshot(env),
    }
