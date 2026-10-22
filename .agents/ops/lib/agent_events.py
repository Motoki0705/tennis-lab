"""Validate CLI event streams and retain runtime evidence separately from pins."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class EvidenceError(ValueError):
    """Runtime configuration evidence is missing or disagrees with the selection."""


def parse_json(text: str | bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate key in runtime JSON")
            value[key] = item
        return value

    def invalid_constant(value: str) -> Any:
        raise ValueError("non-finite value in runtime JSON")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def unavailable(reason: str) -> dict[str, Any]:
    return {"value": None, "status": "unavailable", "reason": reason}


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value
    ):
        raise ValueError(
            "runtime returned a missing/invalid session or turn identifier"
        )
    return value


def codex_context(home: Path, session: str) -> dict[str, Any]:
    identifier(session)
    paths = [
        p
        for directory in (home / "sessions", home / "archived_sessions")
        if directory.exists()
        for p in directory.rglob(f"*{session}.jsonl")
    ]
    if len(paths) != 1:
        raise ValueError("Codex transcript missing or ambiguous")
    meta, context = None, None
    for line in paths[0].open(encoding="utf-8"):
        record = parse_json(line)
        if not isinstance(record, dict):
            raise ValueError("invalid Codex transcript record")
        if record.get("type") == "session_meta":
            meta = record.get("payload")
        if record.get("type") == "turn_context":
            context = record.get("payload")
    if (
        not isinstance(meta, dict)
        or meta.get("id") != session
        or not isinstance(context, dict)
    ):
        raise ValueError("Codex transcript has no matching session/turn context")
    return {
        "path": str(paths[0]),
        "session_id": meta["id"],
        "turn_id": identifier(context.get("turn_id")),
        "cwd": context.get("cwd"),
        "model": context.get("model"),
        "effort": context.get("effort"),
        "service_tier": context.get("service_tier"),
    }


class Events:
    def __init__(
        self, agent: str, selected: dict[str, str], expected_session: str | None
    ) -> None:
        self.agent, self.selected = agent, selected
        self.expected_session = expected_session
        self.session: str | None = None
        self.started = False
        self.completed = False
        self.failed = False
        self.errors: list[Any] = []
        self.final: str | None = None
        self.turn: str | None = None
        self.usage: dict[str, Any] = unavailable("runtime has not reported usage")
        self.effective = {
            k: unavailable("runtime has not reported this setting") for k in selected
        }

    def session_id(self, value: Any) -> None:
        value = identifier(value)
        if (self.session is not None and value != self.session) or (
            self.expected_session is not None and value != self.expected_session
        ):
            raise ValueError("runtime session differs from the requested session")
        self.session = value

    def observed(self, key: str, value: Any, source: str) -> None:
        if not isinstance(value, str) or not value:
            raise EvidenceError(f"runtime did not report effective {key}")
        self.effective[key] = {"value": value, "status": "observed", "source": source}
        if value != self.selected[key]:
            raise EvidenceError(f"effective {key} differs from selected setting")

    def token_usage(self, value: Any, source: str) -> None:
        if value is None:
            self.usage = unavailable(f"{source} does not report usage")
            return
        if not isinstance(value, dict) or any(
            type(value.get(k)) is not int or value[k] < 0
            for k in ("input_tokens", "output_tokens")
        ):
            raise ValueError("runtime usage contains missing/invalid token counts")
        self.usage = {"value": value, "status": "observed", "source": source}

    def feed(self, line: bytes) -> None:
        event = parse_json(line)
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError("CLI output is not a structured event")
        kind = event["type"]
        if self.agent == "codex":
            if kind == "thread.started":
                if self.started:
                    raise ValueError("duplicate runtime session start")
                self.session_id(event.get("thread_id"))
                self.started = True
            elif kind == "turn.started" and event.get("turn_id"):
                self.turn = identifier(event["turn_id"])
            elif kind == "item.completed":
                item = event.get("item")
                if not isinstance(item, dict):
                    raise ValueError("invalid completed item")
                if item.get("type") == "agent_message":
                    self.final = item.get("text")
            elif kind == "turn.completed":
                if self.completed:
                    raise ValueError("duplicate completed turn")
                self.completed = True
                self.token_usage(event.get("usage"), "turn.completed")
            elif kind in {"turn.failed", "error"}:
                self.failed = True
                self.errors.append(event)
        else:
            if kind == "system" and event.get("subtype") == "init":
                if self.started:
                    raise ValueError("duplicate runtime session start")
                self.session_id(event.get("session_id"))
                self.started = True
                self.observed("model", event.get("model"), "system.init")
                self.effective["effort"] = unavailable(
                    "Claude stream-json does not expose effective effort"
                )
            elif kind == "result":
                if self.completed:
                    raise ValueError("duplicate result event")
                self.session_id(event.get("session_id"))
                self.completed = True
                self.failed = (
                    event.get("is_error") is not False
                    or event.get("subtype") != "success"
                )
                self.final = event.get("result")
                if self.failed:
                    self.errors.append(event)
                self.token_usage(event.get("usage"), "result")
                self.usage.update(
                    model_usage=event.get("modelUsage"),
                    cost_usd=event.get("total_cost_usd"),
                )

    def evidence(self, home: Path, cwd: Path, previous_turn: str | None) -> None:
        if self.agent != "codex" or not self.session:
            return
        context = codex_context(home, self.session)
        if context["cwd"] != str(cwd):
            raise ValueError(
                "Codex runtime worktree differs from the selected worktree"
            )
        if context["turn_id"] == previous_turn:
            raise ValueError("Codex transcript contains no new turn context")
        if self.turn is not None and self.turn != context["turn_id"]:
            raise ValueError("Codex event/transcript turn IDs disagree")
        self.turn = context["turn_id"]
        for key in ("model", "effort"):
            self.observed(key, context[key], context["path"])
        if context["service_tier"] is not None:
            self.observed("service_tier", context["service_tier"], context["path"])
        else:
            self.effective["service_tier"] = unavailable(
                "Codex turn_context does not expose effective service_tier"
            )

    def validate(self, out: Path) -> None:
        if self.failed:
            raise ValueError("runtime reported an unsuccessful turn")
        if not self.started or not self.completed:
            raise ValueError("CLI stream is missing session/turn completion")
        if not isinstance(self.final, str) or not self.final.strip():
            raise ValueError("agent produced an empty final message")
        if self.agent == "codex":
            if not out.is_file() or not out.read_text(encoding="utf-8").strip():
                raise ValueError("agent produced an empty final message")
            if out.read_text(encoding="utf-8").strip() != self.final.strip():
                raise ValueError("final file and completed agent message disagree")
        else:
            out.write_text(self.final, encoding="utf-8")

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session,
            "requested_session_id": self.expected_session,
            "turn_id": self.turn,
            "turn_id_unavailable_reason": "Claude stream-json does not expose a turn ID"
            if self.agent == "claude"
            else None,
            "effective": self.effective,
            "usage": self.usage,
            "runtime_errors": self.errors,
        }
