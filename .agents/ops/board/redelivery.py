"""Sender-authorized retries through the existing durable dispatcher."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from .access import as_parent
from .delivery import Notifiers, dispatch
from .errors import BoardError
from .records import post_row, recipient_views
from .safety import public


def dispatch_post(
    env: Mapping[str, str],
    notifiers: Notifiers,
    *,
    post_id: str,
    to: str | None = None,
) -> dict[str, Any]:
    with as_parent(env) as (connection, caller, parents):
        row = post_row(connection, post_id)
        if row["sender_assignment_id"] != caller.id:
            raise BoardError("only the post sender may retry its notifications")
        recipients = recipient_views(connection, post_id)
        assignment_id: int | None | Literal["all"] = "all"
        if to is not None:
            matches = [r for r in recipients if r["name"].casefold() == to.casefold()]
            if len(matches) != 1:
                raise BoardError("dispatch name must match a saved post recipient")
            assignment_id = matches[0]["assignment_id"]
        public(recipients, parents, env)
    # Release the parent lock before entering the dispatcher's notification lock.
    recipients = dispatch(
        env, notifiers, post_id=post_id, assignment_id=assignment_id, retry_only=True
    )
    outcomes = {r["outcome"] for r in recipients}
    status = (
        "failed"
        if "failed" in outcomes
        else "notified"
        if "notified" in outcomes
        else "no_targets"
    )
    return {"post_id": post_id, "status": status, "recipients": recipients}
