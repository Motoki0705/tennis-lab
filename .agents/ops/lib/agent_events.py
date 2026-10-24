"""Validate CLI event streams and retain runtime evidence separately from pins."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_output import OutputSchema


class EvidenceError(ValueError):
    """Runtime configuration evidence is missing or disagrees with the selection."""


def _invalid_constant(value: str) -> Any:
    raise ValueError("non-finite value in runtime JSON")


def parse_json(text: str | bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate key in runtime JSON")
            value[key] = item
        return value

    return json.loads(text, object_pairs_hook=pairs, parse_constant=_invalid_constant)


class _JsonPairs(list[tuple[str, Any]]):
    """Keep JSON objects distinct from arrays until duplicate keys are checked."""


def parse_cli_event(text: str | bytes, agent: str) -> Any:
    """Disambiguate Codex's flattened web-search IDs at the CLI boundary only."""
    if agent != "codex":
        return parse_json(text)
    raw = json.loads(
        text, object_pairs_hook=_JsonPairs, parse_constant=_invalid_constant
    )
    kinds = [v for k, v in raw if k == "type"] if isinstance(raw, _JsonPairs) else []
    item_event = len(kinds) == 1 and kinds[0] in (
        "item.started",
        "item.updated",
        "item.completed",
    )

    def restore(value: Any, path: tuple[str, ...]) -> Any:
        if isinstance(value, _JsonPairs):
            ids = [v for k, v in value if k == "id"]
            types = [v for k, v in value if k == "type"]
            web_search_ids = (
                item_event
                and path == ("item",)
                and types == ["web_search"]
                and len(ids) == 2
                and isinstance(ids[0], str)
                and re.fullmatch(r"item_[0-9]+", ids[0]) is not None
                and isinstance(ids[1], str)
                and bool(ids[1])
                and not any(k == "web_search_id" for k, _ in value)
            )
            result: dict[str, Any] = {}
            for key, item in value:
                if key in result:
                    if web_search_ids and key == "id":
                        key = "web_search_id"
                    else:
                        raise ValueError("duplicate key in runtime JSON")
                result[key] = restore(item, (*path, key))
            return result
        if isinstance(value, list):
            return [restore(item, (*path, str(i))) for i, item in enumerate(value)]
        return value

    return restore(raw, ())


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


def _codex_transcript(
    home: Path, session: str
) -> tuple[Path, bytes, list[dict[str, Any]]]:
    identifier(session)
    paths = [
        p
        for directory in (home / "sessions", home / "archived_sessions")
        if directory.exists()
        for p in directory.rglob(f"*{session}.jsonl")
    ]
    if len(paths) != 1:
        raise ValueError("Codex transcript missing or ambiguous")
    data = paths[0].read_bytes()
    records = []
    for line in data.splitlines():
        record = parse_json(line)
        if not isinstance(record, dict):
            raise ValueError("invalid Codex transcript record")
        records.append(record)
    return paths[0], data, records


def _codex_context(
    path: Path, records: list[dict[str, Any]], session: str
) -> dict[str, Any]:
    meta, context = None, None
    for record in records:
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
        "path": str(path),
        "session_id": meta["id"],
        "turn_id": identifier(context.get("turn_id")),
        "cwd": context.get("cwd"),
        "model": context.get("model"),
        "effort": context.get("effort"),
        "service_tier": context.get("service_tier"),
    }


def codex_context(home: Path, session: str) -> dict[str, Any]:
    path, _, records = _codex_transcript(home, session)
    return _codex_context(path, records, session)


def codex_checkpoint(home: Path, session: str) -> dict[str, Any]:
    """Pin the transcript prefix before launching a continuation."""
    path, data, records = _codex_transcript(home, session)
    context = _codex_context(path, records, session)
    return {
        "path": str(path),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "turn_id": context["turn_id"],
    }


def interrupted_codex_turn(
    home: Path, session: str, before: dict[str, Any] | None, prompt: str
) -> str:
    """Bind one appended turn to the saved input instead of adopting the latest turn."""
    path, data, records = _codex_transcript(home, session)
    _codex_context(path, records, session)
    if before is not None:
        size = before["size"]
        if (
            type(size) is not int
            or not 0 < size <= len(data)
            or data[size - 1 : size] != b"\n"
            or hashlib.sha256(data[:size]).hexdigest() != before["sha256"]
        ):
            raise ValueError(
                "Codex transcript prefix differs from the launch checkpoint"
            )
        records = [parse_json(line) for line in data[size:].splitlines()]
    turns = {
        identifier(record["payload"].get("turn_id"))
        for record in records
        if record.get("type") == "turn_context"
        and isinstance(record.get("payload"), dict)
    }
    if len(turns) != 1 or (before is not None and before["turn_id"] in turns):
        raise ValueError("interrupted Codex run has no unique new turn")
    prompts = []
    for record in records:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if record.get("type") == "event_msg" and payload.get("type") == "user_message":
            prompts.append(payload.get("message"))
        if (
            record.get("type") == "response_item"
            and payload.get("type") == "message"
            and payload.get("role") == "user"
            and isinstance(payload.get("content"), list)
        ):
            prompts.append(
                "".join(
                    item["text"]
                    for item in payload["content"]
                    if isinstance(item, dict)
                    and item.get("type") == "input_text"
                    and isinstance(item.get("text"), str)
                )
            )
    if not prompt.strip() or not any(
        isinstance(value, str) and value.strip() == prompt.strip() for value in prompts
    ):
        raise ValueError("interrupted Codex turn does not match the saved prompt")
    return turns.pop()


class Events:
    def __init__(
        self,
        agent: str,
        selected: dict[str, str],
        expected_session: str | None,
        output_schema: OutputSchema | None = None,
    ) -> None:
        self.agent, self.selected = agent, selected
        self.expected_session = expected_session
        self.session: str | None = None
        self.started = False
        self.completed = False
        self.failed = False
        self.errors: list[Any] = []
        self.final: str | None = None
        self.output_schema = output_schema
        self.structured_output: Any = None
        self.has_structured_output = False
        self.invalid_structured_output = False
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
        event = parse_cli_event(line, self.agent)
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
                self.has_structured_output = "structured_output" in event
                self.structured_output = event.get("structured_output")
                self.invalid_structured_output = (
                    self.output_schema is not None
                    and event.get("subtype") == "error_max_structured_output_retries"
                )
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
        if self.output_schema is not None and self.agent == "claude":
            if not self.has_structured_output:
                raise ValueError("result event is missing structured_output")
            self.output_schema.validate(self.structured_output)
            self.final = (
                json.dumps(
                    self.structured_output,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )
                + "\n"
            )
        if not isinstance(self.final, str) or not self.final.strip():
            raise ValueError("agent produced an empty final message")
        if self.agent == "codex":
            if not out.is_file() or not out.read_text(encoding="utf-8").strip():
                raise ValueError("agent produced an empty final message")
            if out.read_text(encoding="utf-8").strip() != self.final.strip():
                raise ValueError("final file and completed agent message disagree")
            if self.output_schema is not None:
                self.output_schema.validate(parse_json(self.final))
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
