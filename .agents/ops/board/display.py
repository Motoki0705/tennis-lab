"""Event-only unread and renamed-name projections onto confirmed parent terminals."""

from __future__ import annotations

import copy
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from herdr_labels.client import Herdr
from herdr_labels.coordination import (
    Guard,
    locked,
    owner_token,
    read_record,
    state_root,
)
from herdr_labels.service import issue_label, parent_summary, registry
from lib.agy_session import AgyTui
from lib.identity import session_identity

from .access import _as_registered_parent, as_parent, parent_state
from .activity import summary
from .errors import BoardError
from .storage import Store, directory


@dataclass(frozen=True)
class Projection:
    assignment_id: int
    name: str
    repo: str
    parent: int
    retired: str | None
    owner: dict[str, Any] | None
    destination: dict[str, Any] | None
    count: int

    def inputs(self) -> tuple[Any, ...]:
        binding = (
            None
            if self.owner is None
            else tuple(
                self.owner.get(key)
                for key in ("session", "status", "generation", "display_summary")
            )
        )
        return (
            self.name,
            self.repo,
            self.parent,
            self.retired,
            binding,
            self.destination,
            self.count,
        )


def _capture(env: Mapping[str, str], assignment_id: int) -> Projection:
    with parent_state(env) as parents, Store(directory(env)).read() as connection:
        row = connection.execute(
            "SELECT name, repo, parent, retired_at FROM name_assignments WHERE id=?",
            (assignment_id,),
        ).fetchone()
        if row is None:
            raise BoardError("display assignment missing")
        name, repo, parent, retired = row
        owner = copy.deepcopy(parents.get(f"{repo}#{parent}"))
        destination = (
            None
            if retired or owner is None or owner["status"] != "active"
            else read_record(
                state_root(env)
                / "herdr-labels/owners"
                / f"{owner_token(owner['session'])}.json"
            )
        )
        return Projection(
            assignment_id,
            name,
            repo,
            parent,
            retired,
            owner,
            destination,
            summary(connection, assignment_id).unread_count,
        )


def _verify(env: Mapping[str, str], expected: Projection) -> None:
    if _capture(env, expected.assignment_id).inputs() != expected.inputs():
        raise BoardError("display assignment or projection inputs changed")


@contextmanager
def _confirmed_target(
    env: Mapping[str, str],
    identity: str,
    owner: dict[str, Any] | None,
) -> Iterator[tuple[Herdr, Guard] | None]:
    root = state_root(env) / "herdr-labels"
    if read_record(root / "owners" / f"{owner_token(identity)}.json") != owner:
        raise BoardError("confirmed display destination changed")
    if owner is None:
        yield None
        return
    servers = Herdr(env).call("session", "list", "--json")["sessions"]
    matches = [
        row
        for row in servers
        if row.get("running")
        and row.get("socket_path")
        and owner_token(str(Path(row["socket_path"]).resolve())) == owner["scope"]
    ]
    if len(matches) != 1:
        raise BoardError("confirmed display server unavailable")
    client = Herdr(env, matches[0]["name"])
    agents = client.call("agent", "list")["agents"]
    candidates = [a for a in agents if a.get("terminal_id") == owner["terminal_id"]]
    if len(candidates) != 1:
        raise BoardError("ambiguous confirmed display terminal")
    pane = client.pane(candidates[0]["pane_id"])
    if (
        pane["terminal_id"] != owner["terminal_id"]
        or pane.get("agent") != identity.partition(":")[0]
    ):
        raise BoardError("confirmed display occupant changed")
    guard = Guard(env, identity, client, pane)
    runtime = read_record(guard.path)
    if runtime is None and pane.get("tokens", {}).get("tl_owner") != owner_token(
        identity
    ):
        raise BoardError("confirmed display conversation is unverified")
    with guard.hold():
        guard.claim()
        pane = guard.pane
        runtime = read_record(guard.path)
        if runtime is None or runtime.get("owner") != owner_token(identity):
            raise BoardError("confirmed display conversation changed")
        native = pane.get("agent_session")
        if (
            pane["agent"] != "agy"
            and native is not None
            and (
                native.get("kind") != "id"
                or native.get("agent") != pane["agent"]
                or native.get("value") != identity.partition(":")[2]
            )
        ):
            raise BoardError("confirmed display native session changed")
        token = pane.get("tokens", {}).get("tl_owner")
        if token and token != owner_token(identity):
            raise BoardError("confirmed display owner changed")
        yield client, guard
        after = client.pane(pane["pane_id"])
        if (
            after["terminal_id"] != owner["terminal_id"]
            or after.get("agent") != identity.partition(":")[0]
            or (pane["agent"] != "agy" and after.get("agent_session") != native)
        ):
            raise BoardError("confirmed display native session or terminal changed")


def _publish(
    env: Mapping[str, str], identity: str, count: int, *, expected: Projection
) -> str:
    with _confirmed_target(env, identity, expected.destination) as destination:
        if destination is None:
            return "not_configured"
        client, guard = destination

        def check() -> None:
            _verify(env, expected)
            guard.check()

        check()
        pane = guard.pane
        desired = {
            "tl_inbox": str(count),
            "tl_inbox_label": f"未読:{count}" if count else "",
        }
        if all(pane.get("tokens", {}).get(k, "") == v for k, v in desired.items()):
            return "unchanged"
        before = {k: v for k, v in pane.items() if k != "tokens"}
        other_tokens = {
            k: v for k, v in pane.get("tokens", {}).items() if k not in desired
        }
        check()
        args = ["pane", "report-metadata", pane["pane_id"], "--source", "tennis-board"]
        for key, value in desired.items():
            args.extend(
                ["--token", f"{key}={value}"] if value else ["--clear-token", key]
            )
        client.call(*args)
        after = client.pane(pane["pane_id"])
        check()
        if (
            any(
                after.get(k) != before.get(k)
                for k in ("terminal_id", "agent", "display_agent")
            )
            or any(
                after.get("tokens", {}).get(k, "") != value
                for k, value in desired.items()
            )
            or {k: v for k, v in after.get("tokens", {}).items() if k not in desired}
            != other_tokens
        ):
            raise BoardError("unread display readback mismatch")
    return "updated"


def _publish_name(
    env: Mapping[str, str],
    owner: Mapping[str, Any],
    repo: str,
    parent: int,
    name: str,
    *,
    expected: Projection,
) -> str:
    identity = owner["session"]
    with _confirmed_target(env, identity, expected.destination) as destination:
        if destination is None:
            return "not_configured"
        client, guard = destination

        def check() -> None:
            _verify(env, expected)
            guard.check()

        check()
        # The parent record is the only source of the summary. A token left by
        # an older title-based label, or a summary not yet synchronized, is an
        # error rather than text to carry over into the renamed label.
        summary = parent_summary(env, f"{repo}#{parent}", owner)
        pane = guard.pane
        tokens = pane.get("tokens", {})
        if (
            tokens.get("tl_owner") != owner_token(identity)
            or tokens.get("tl_parent") != f"{repo}#{parent}"
            or tokens.get("tl_issue") != issue_label(parent)
            or tokens.get("tl_engine") != pane["agent"]
            or tokens.get("tl_topic") != summary
        ):
            raise BoardError(
                "confirmed parent label changed; synchronize parent display"
            )
        result = client.publish(
            pane,
            agent=pane["agent"],
            display=f"{name} · {summary}",
            tokens={
                "tl_name": name,
                **{
                    key: tokens[key]
                    for key in (
                        "tl_owner",
                        "tl_parent",
                        "tl_issue",
                        "tl_topic",
                        "tl_engine",
                    )
                },
            },
            validate=check,
        )
        return str(result["status"])


def update(
    env: Mapping[str, str], assignments: Sequence[int], *, renamed: bool = False
) -> list[dict[str, Any]]:
    results = []
    for assignment_id in sorted(set(assignments)):
        result: dict[str, Any] = {"assignment_id": assignment_id}
        try:
            # Serialize before taking the snapshot, and never await this lock
            # while holding the registry or a SQLite read transaction.
            with locked(directory(env) / f"display-{assignment_id}.lock"):
                expected = _capture(env, assignment_id)
                name, repo, parent = expected.name, expected.repo, expected.parent
                result.update(name=name, repo=repo, parent=parent)
                owner = expected.owner
                if expected.retired or not owner or owner["status"] != "active":
                    result["status"] = "recipient_unavailable"
                elif renamed:
                    result["status"] = _publish_name(
                        env, owner, repo, parent, name, expected=expected
                    )
                else:
                    result["status"] = _publish(
                        env, owner["session"], expected.count, expected=expected
                    )
                _verify(env, expected)
        except (
            BoardError,
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            subprocess.SubprocessError,
        ):
            # This is a separate display failure, never another bell route.
            result.update(status="error", reason="route_unavailable")
        results.append(result)
    return results


def posted(env: Mapping[str, str], result: dict[str, Any]) -> dict[str, Any]:
    post = result["post"]
    ids = [
        post["sender"]["assignment_id"],
        *(r["assignment_id"] for r in post["recipients"]),
    ]
    return {
        **result,
        "inbox_display": update(env, [value for value in ids if value is not None]),
    }


def read_by_caller(env: Mapping[str, str], result: dict[str, Any]) -> dict[str, Any]:
    with as_parent(env) as (_, caller, _):
        assignment_id = caller.id
    return {**result, "inbox_display": update(env, [assignment_id])}


def _snapshot(env: Mapping[str, str], identity: str) -> dict[str, Any] | None:
    owned = [
        row
        for row in registry(env).values()
        if row["session"] == identity and row["status"] == "active"
    ]
    if not owned:
        return None
    with _as_registered_parent(env, identity) as (connection, caller, _):
        count = summary(connection, caller.id).unread_count
        rows = connection.execute(
            """SELECT state, COUNT(*) FROM deliveries WHERE assignment_id=?
               AND state IN ('held','delivery_failed') GROUP BY state""",
            (caller.id,),
        ).fetchall()
        problems = dict(rows)
        return {
            "name": caller.name,
            "repo": caller.repo,
            "parent": caller.parent,
            "unread_count": count,
            "held_count": problems.get("held", 0),
            "delivery_failed_count": problems.get("delivery_failed", 0),
        }


def snapshot(env: Mapping[str, str]) -> dict[str, Any] | None:
    """Counts only, for hooks and for restored parent sessions."""
    return _snapshot(env, session_identity(env))


def agy_unread(env: Mapping[str, str], tui: AgyTui, conversation: str) -> int:
    """For a native hook, use the lock and a previously verified registration."""
    tui.require_conversation(conversation)
    state = _snapshot(env, f"agy:{conversation}")
    tui.require_conversation(conversation)
    return int(state["unread_count"]) if state else 0


def context(env: Mapping[str, str]) -> str:
    state = snapshot(env)
    if state is None:
        return ""
    return (
        f"[board] {state['name']}（#{state['parent']}）宛て未読:{state['unread_count']}、"
        f"配送保留:{state['held_count']}、配送失敗:{state['delivery_failed_count']}。"
        "board inbox --unread で確認。question/request は作業の区切りで応答またはackする。"
        "投稿は人間の合意・承認・レビュー指示・merge許可を代行しない。"
    )
