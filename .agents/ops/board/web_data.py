"""Read-only board projections and process-local cursors, without web imports."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from herdr_labels.service import registry

from . import activity
from .errors import BoardError
from .names import for_parent
from .records import post_view
from .safety import public
from .service import agents
from .storage import Store, directory

POST_ID = re.compile(r"[0-9a-f]{32}")
CURSOR = re.compile(r"([0-9a-f]{16})\.([0-9a-f]{16})\.([0-9]{1,20})")
ALL_MENTION = re.compile(r"(?<![\w@.+/\-])@all(?![\w@\-])", re.ASCII | re.IGNORECASE)


class InvalidQuery(ValueError):
    """A query does not match the public API contract."""


class MissingPost(LookupError):
    """A well-formed post ID has no saved post."""


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def matches(post: dict[str, Any], name: str, direction: str) -> bool:
    if not name:
        return True
    sent = post["sender"]["name"].casefold() == name
    received = any(row["name"].casefold() == name for row in post["recipients"])
    return (sent and direction != "received") or (received and direction != "sent")


class BoardReader:
    """Cache external roster sources once; subsequent reads touch only SQLite.

    Cursors track recipient/delivery updates as well as appended posts. Each
    post keeps its latest revision, so clients do not need a retained event log.
    A process restart or changed name/direction requires a full reset.
    """

    def __init__(self, env: Mapping[str, str]) -> None:
        self.env = dict(env)
        self.store = Store(directory(env))
        self.parents = registry(env)
        self.roster = agents(env)["agents"]
        if registry(env) != self.parents:
            raise BoardError("parent registry changed during web startup; retry")
        self.roster_observed_at = datetime.now(UTC).isoformat()
        self.epoch = secrets.token_hex(8)
        self.revision = 0
        self.versions: dict[str, int] = {}
        self.hashes: dict[str, str] = {}
        self.roster_hash = ""
        self.roster_revision = 0
        self.lock = Lock()

    def _snapshot(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self.store.read() as connection:
            posts = []
            for (post_id,) in connection.execute(
                "SELECT id FROM posts ORDER BY id DESC"
            ):
                post = post_view(connection, post_id)
                attempts = dict(
                    connection.execute(
                        "SELECT assignment_id, attempted_at FROM deliveries WHERE post_id=?",
                        (post_id,),
                    )
                )
                for recipient in post["recipients"]:
                    recipient["attempted_at"] = attempts[recipient["assignment_id"]]
                # The schema stores resolved recipients, not the --to all flag.
                # Only an explicit, validated @all in the body is identifiable.
                post["mentions_all"] = bool(ALL_MENTION.search(post["body"]))
                posts.append(post)
            roster = deepcopy(self.roster)
            for row in roster:
                assignment = for_parent(connection, row["repo"], row["parent"])
                if assignment is not None and assignment.retired_at is not None:
                    assignment = None
                state = (
                    activity.summary(connection, assignment.id)
                    if assignment is not None
                    else activity.Activity()
                )
                row.update(
                    name=assignment.name if assignment else None,
                    name_status="assigned" if assignment else "unassigned",
                    identifier=assignment.identifier if assignment else None,
                    unread_count=state.unread_count,
                    last_activity=state.last_post_at,
                )
        # Revalidate saved public text, including legacy/direct-SQL data. Never
        # serialize private source records or exception messages to the browser.
        public({"posts": posts, "agents": roster}, self.parents, self.env)
        return posts, roster

    def changes(
        self, *, cursor: str | None = None, name: str = "", direction: str = "both"
    ) -> dict[str, Any]:
        if direction not in {"both", "sent", "received"}:
            raise InvalidQuery
        if name and not re.fullmatch(r"[A-Za-z]+", name):
            raise InvalidQuery
        parsed = CURSOR.fullmatch(cursor) if cursor is not None else None
        if cursor is not None and parsed is None:
            raise InvalidQuery
        name = name.casefold()
        scope = fingerprint([name, direction])[:16]
        with self.lock:
            posts, roster = self._snapshot()
            hashes = {post["id"]: fingerprint(post) for post in posts}
            roster_hash = fingerprint(roster)
            changed = [
                post_id
                for post_id, value in hashes.items()
                if self.hashes.get(post_id) != value
            ]
            if changed or roster_hash != self.roster_hash:
                self.revision += 1
                self.versions.update(dict.fromkeys(changed, self.revision))
                if roster_hash != self.roster_hash:
                    self.roster_revision = self.revision
                self.hashes, self.roster_hash = hashes, roster_hash
            reset = parsed is None or parsed[1] != self.epoch or parsed[2] != scope
            after = 0 if reset or parsed is None else int(parsed[3])
            if after > self.revision:
                reset, after = True, 0
            result = {
                "cursor": f"{self.epoch}.{scope}.{self.revision}",
                "reset": reset,
                "posts": [
                    post
                    for post in posts
                    if (reset or self.versions[post["id"]] > after)
                    and matches(post, name, direction)
                ],
                "agents": roster if reset or self.roster_revision > after else None,
                "changed_threads": sorted(
                    {
                        post["thread_id"]
                        for post in posts
                        if reset or self.versions[post["id"]] > after
                    }
                ),
                "roster_observed_at": self.roster_observed_at,
            }
        return result

    def thread(self, post_id: str) -> dict[str, Any]:
        if POST_ID.fullmatch(post_id) is None:
            raise InvalidQuery
        posts, _ = self._snapshot()
        selected = next((post for post in posts if post["id"] == post_id), None)
        if selected is None:
            raise MissingPost
        return {
            "thread_id": selected["thread_id"],
            "posts": [
                post
                for post in reversed(posts)
                if post["thread_id"] == selected["thread_id"]
            ],
        }
