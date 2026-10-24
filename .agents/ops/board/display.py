"""Event-only unread and renamed-name projections onto confirmed parent terminals."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from herdr_labels.client import Herdr
from herdr_labels.coordination import Guard, owner_token, read_record, state_root

from .access import as_parent, parent_state
from .activity import summary
from .errors import BoardError
from .storage import Store, directory


@contextmanager
def _confirmed_target(
    env: Mapping[str, str], identity: str
) -> Iterator[tuple[Herdr, Guard] | None]:
    root = state_root(env) / "herdr-labels"
    owner = read_record(root / "owners" / f"{owner_token(identity)}.json")
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
        if native is not None and (
            native.get("kind") != "id"
            or native.get("agent") != pane["agent"]
            or native.get("value") != identity.partition(":")[2]
        ):
            raise BoardError("confirmed display native session changed")
        token = pane.get("tokens", {}).get("tl_owner")
        if token and token != owner_token(identity):
            raise BoardError("confirmed display owner changed")
        yield client, guard


def _publish(env: Mapping[str, str], identity: str, count: int) -> str:
    with _confirmed_target(env, identity) as destination:
        if destination is None:
            return "not_configured"
        client, guard = destination
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
        guard.check()
        args = ["pane", "report-metadata", pane["pane_id"], "--source", "tennis-board"]
        for key, value in desired.items():
            args.extend(
                ["--token", f"{key}={value}"] if value else ["--clear-token", key]
            )
        client.call(*args)
        after = client.pane(pane["pane_id"])
        guard.check()
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
    env: Mapping[str, str], identity: str, repo: str, parent: int, name: str
) -> str:
    with _confirmed_target(env, identity) as destination:
        if destination is None:
            return "not_configured"
        client, guard = destination
        pane = guard.pane
        tokens = pane.get("tokens", {})
        if (
            tokens.get("tl_owner") != owner_token(identity)
            or tokens.get("tl_parent") != f"{repo}#{parent}"
            or tokens.get("tl_engine") != pane["agent"]
        ):
            raise BoardError(
                "confirmed parent label changed; synchronize parent display"
            )
        result = client.publish(
            pane,
            agent=pane["agent"],
            display=f"{name} · #{parent} · 親",
            tokens={
                "tl_name": name,
                **{key: tokens[key] for key in ("tl_owner", "tl_parent", "tl_engine")},
            },
            validate=guard.check,
        )
        return str(result["status"])


def update(
    env: Mapping[str, str], assignments: Sequence[int], *, renamed: bool = False
) -> list[dict[str, Any]]:
    results = []
    for assignment_id in sorted(set(assignments)):
        result: dict[str, Any] = {"assignment_id": assignment_id}
        try:
            with (
                parent_state(env) as parents,
                Store(directory(env)).read() as connection,
            ):
                row = connection.execute(
                    "SELECT name, repo, parent, retired_at FROM name_assignments WHERE id=?",
                    (assignment_id,),
                ).fetchone()
                if row is None:
                    raise BoardError("display assignment missing")
                name, repo, parent, retired = row
                result.update(name=name, repo=repo, parent=parent)
                owner = parents.get(f"{repo}#{parent}")
                if retired or not owner or owner["status"] != "active":
                    result["status"] = "recipient_unavailable"
                elif renamed:
                    result["status"] = _publish_name(
                        env, owner["session"], repo, parent, name
                    )
                else:
                    count = summary(connection, assignment_id).unread_count
                    result["status"] = _publish(env, owner["session"], count)
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
    return {**result, "inbox_display": update(env, ids)}


def read_by_caller(env: Mapping[str, str], result: dict[str, Any]) -> dict[str, Any]:
    with as_parent(env) as (_, caller, _):
        assignment_id = caller.id
    return {**result, "inbox_display": update(env, [assignment_id])}


def snapshot(env: Mapping[str, str]) -> dict[str, Any] | None:
    """Counts only; shared by Claude hooks and restored Codex/Claude parents."""
    from herdr_labels.service import registry
    from lib.identity import session_identity

    identity = session_identity(env)
    owned = [
        row
        for row in registry(env).values()
        if row["session"] == identity and row["status"] == "active"
    ]
    if not owned:
        return None
    with as_parent(env) as (connection, caller, _):
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
