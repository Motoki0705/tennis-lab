"""Reconcile native launches with Codex rollouts, independently of agent JSON."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "lib"))
from agent_events import codex_context, identifier, parse_json  # noqa: E402


class NativeEvidenceError(ValueError):
    """Native execution evidence is absent, ambiguous, or inconsistent."""


@dataclass(frozen=True)
class Transcript:
    path: str
    sha256: str


@dataclass(frozen=True)
class NativeAgent:
    agent_id: str
    task_name: str
    spawn_call_id: str
    turn_id: str
    started_at: str
    completed_at: str | None
    status: str
    result: str | None
    model: str
    effort: str
    sandbox_policy: dict[str, Any]
    transcript: Transcript


@dataclass(frozen=True)
class RejectedSpawn:
    task_name: str
    spawn_call_id: str
    error: str


@dataclass(frozen=True)
class NativeRun:
    parent_session_id: str
    parent_turn_id: str
    transcript: Transcript
    agents: tuple[NativeAgent, ...]
    rejections: tuple[RejectedSpawn, ...]


def _object(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NativeEvidenceError("expected an evidence object")
    return value


def _text(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NativeEvidenceError("missing/non-text evidence field")
    return value


def _timestamp(row: dict[str, Any]) -> datetime:
    value = datetime.fromisoformat(_text(row["timestamp"]).replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise NativeEvidenceError("evidence timestamp has no timezone")
    return value


def _read(path: Path) -> tuple[Transcript, list[dict[str, Any]]]:
    data = path.read_bytes()
    if not data or not data.endswith(b"\n"):
        raise NativeEvidenceError("empty or truncated transcript")
    rows = [_object(parse_json(line)) for line in data.splitlines()]
    for row in rows:
        _text(row["type"])
        _object(row["payload"])
    return Transcript(str(path), hashlib.sha256(data).hexdigest()), rows


def _meta(rows: list[dict[str, Any]], thread: str, root_session: str) -> dict[str, Any]:
    metas = [row["payload"] for row in rows if row["type"] == "session_meta"]
    if (
        len(metas) != 1
        or rows[0]["type"] != "session_meta"
        or metas[0].get("id") != thread
        or metas[0].get("session_id") != root_session
    ):
        raise NativeEvidenceError("missing, duplicate, or mismatched session metadata")
    return _object(metas[0])


def _turn(rows: list[dict[str, Any]], turn_id: str) -> list[dict[str, Any]]:
    starts = [
        i
        for i, row in enumerate(rows)
        if row["type"] == "event_msg" and row["payload"].get("type") == "task_started"
    ]
    matches = [i for i in starts if rows[i]["payload"].get("turn_id") == turn_id]
    if len(matches) != 1:
        raise NativeEvidenceError("missing or duplicate turn start")
    start = matches[0]
    end = next((i for i in starts if i > start), len(rows))
    selected = rows[start:end]
    for row in selected:
        payload = row["payload"]
        lifecycle = row["type"] == "event_msg" and payload.get("type") in {
            "task_started",
            "task_complete",
            "turn_aborted",
        }
        if (row["type"] == "turn_context" or lifecycle) and payload.get(
            "turn_id"
        ) != turn_id:
            raise NativeEvidenceError("missing or foreign turn mixed into transcript")
        attribution = payload.get("internal_chat_message_metadata_passthrough", {})
        if _object(attribution).get("turn_id", turn_id) != turn_id:
            raise NativeEvidenceError("tool record belongs to a different turn")
    return selected


def _context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    contexts = [row["payload"] for row in rows if row["type"] == "turn_context"]
    if len(contexts) != 1:
        raise NativeEvidenceError("missing or ambiguous turn context")
    return _object(contexts[0])


def _launches(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], tuple[RejectedSpawn, ...]]:
    calls: dict[str, dict[str, Any]] = {}
    outputs: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = row["payload"]
        kind = payload.get("type") if row["type"] == "response_item" else None
        if kind not in {"function_call", "function_call_output"}:
            continue
        call_id = identifier(payload.get("call_id"))
        target = calls if kind == "function_call" else outputs
        if call_id in target:
            raise NativeEvidenceError("duplicate tool call/output")
        target[call_id] = row
    if outputs.keys() - calls.keys():
        raise NativeEvidenceError("tool output has no matching call")
    launches: dict[str, dict[str, Any]] = {}
    rejected = []
    for call_id, row in calls.items():
        call = row["payload"]
        if call.get("name") == "followup_task":
            raise NativeEvidenceError(
                "native followup_task is not covered by spawn evidence"
            )
        if call.get("name") != "spawn_agent":
            continue
        if call.get("namespace") != "collaboration" or call_id not in outputs:
            raise NativeEvidenceError("unsupported or unpaired native spawn")
        args = _object(parse_json(call["arguments"]))
        requested = _text(args["task_name"])
        if args.get("fork_turns") != "none":
            raise NativeEvidenceError("native evidence requires fork_turns=none")
        response = outputs[call_id]
        if _timestamp(response) < _timestamp(row):
            raise NativeEvidenceError("spawn output precedes its call")
        output = _text(response["payload"]["output"])
        # This is the observed native rejection, not malformed success JSON.
        if output == "collab spawn failed: agent thread limit reached":
            rejected.append(RejectedSpawn(requested, call_id, output))
            continue
        name = _text(_object(parse_json(output))["task_name"])
        if name != f"/root/{requested}" or name in launches:
            raise NativeEvidenceError("mismatched or duplicate native task name")
        launches[name] = {"call_id": call_id, "call": row, "response": response}
    return launches, tuple(rejected)


def _child_paths(home: Path, parent: str) -> list[tuple[Path, dict[str, Any]]]:
    children = []
    for directory in (home / "sessions", home / "archived_sessions"):
        if not directory.exists():
            continue
        for path in directory.rglob("*.jsonl"):
            with path.open("rb") as stream:
                header = _object(parse_json(stream.readline()))
            if header.get("type") != "session_meta":
                raise NativeEvidenceError("transcript has no metadata header")
            meta = _object(header["payload"])
            source = meta.get("source")
            subagent = source.get("subagent") if isinstance(source, dict) else None
            spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
            spawn = spawn if isinstance(spawn, dict) else {}
            if parent in (meta.get("parent_thread_id"), spawn.get("parent_thread_id")):
                children.append((path, meta))
    return children


def _child(
    path: Path,
    meta: dict[str, Any],
    parent: str,
    parent_turn: str,
    context: dict[str, Any],
    launches: dict[str, dict[str, Any]],
) -> NativeAgent | None:
    agent_id = identifier(meta.get("id"))
    reference, rows = _read(path)
    if _meta(rows, agent_id, parent) != meta:
        raise NativeEvidenceError("child metadata changed during inspection")
    if not path.name.endswith(f"{agent_id}.jsonl"):
        raise NativeEvidenceError("child session ID and transcript filename disagree")
    spawn = _object(_object(_object(meta["source"])["subagent"])["thread_spawn"])
    if (
        meta.get("parent_thread_id") != parent
        or spawn.get("parent_thread_id") != parent
        or type(spawn.get("depth")) is not int
        or spawn.get("depth") != 1
        or spawn.get("agent_path") != meta.get("agent_path")
        or meta.get("cwd") != context["cwd"]
    ):
        raise NativeEvidenceError("child parent/depth/path/worktree mismatch")
    starts = [
        row
        for row in rows
        if row["type"] == "event_msg" and row["payload"].get("type") == "task_started"
    ]
    if not starts:
        raise NativeEvidenceError("child has no turn start")
    start = starts[0]
    attribution = _object(start["payload"]["turn_attribution"])
    if identifier(attribution.get("parent_turn_id")) != parent_turn:
        return None  # Another runner turn of the same parent session.
    turn_id = identifier(start["payload"].get("turn_id"))
    if (
        attribution.get("root_turn_id") != parent_turn
        or start["payload"].get("root_turn_id") != parent_turn
        or attribution.get("turn_id") != turn_id
    ):
        raise NativeEvidenceError("child root turn mismatch")
    name = _text(meta["agent_path"])
    if name not in launches:
        raise NativeEvidenceError("child transcript has no matching spawn")
    launch = launches[name]
    selected = _turn(rows, turn_id)
    child_context = _context(selected)
    for key in ("cwd", "model", "effort", "sandbox_policy", "approval_policy"):
        if child_context.get(key) != context[key]:
            raise NativeEvidenceError(f"child inherited {key} mismatch")
    if _timestamp(start) < _timestamp(launch["call"]):
        raise NativeEvidenceError("child started before its spawn call")
    completions = [
        row
        for row in selected
        if row["type"] == "event_msg" and row["payload"].get("type") == "task_complete"
    ]
    aborted = any(
        row["type"] == "event_msg" and row["payload"].get("type") == "turn_aborted"
        for row in selected
    )
    if len(completions) > 1 or (completions and aborted):
        raise NativeEvidenceError("duplicate/conflicting child completion")
    completed_at, result = None, None
    if completions:
        finish = completions[0]
        if _timestamp(finish) < _timestamp(start):
            raise NativeEvidenceError("child completion precedes its start")
        if finish["payload"].get("root_turn_id") != parent_turn:
            raise NativeEvidenceError("child completion root turn mismatch")
        result = _text(finish["payload"]["last_agent_message"])
        messages = [
            row["payload"]
            for row in selected[: selected.index(finish)]
            if row["type"] == "response_item"
            and row["payload"].get("type") == "message"
            and row["payload"].get("role") == "assistant"
        ]
        if not messages or result != "".join(
            _text(part["text"])
            for part in messages[-1]["content"]
            if _object(part).get("type") == "output_text"
        ):
            raise NativeEvidenceError("child completion and final message disagree")
        completed_at = _text(finish["timestamp"])
    return NativeAgent(
        agent_id=agent_id,
        task_name=_text(meta["agent_path"]),
        spawn_call_id=launch["call_id"],
        turn_id=turn_id,
        started_at=_text(start["timestamp"]),
        completed_at=completed_at,
        status="completed" if completions else "aborted" if aborted else "incomplete",
        result=result,
        model=_text(child_context["model"]),
        effort=_text(child_context["effort"]),
        sandbox_policy=_object(child_context["sandbox_policy"]),
        transcript=reference,
    )


def inspect_run(run_dir: Path) -> NativeRun:
    """Read one agent-runtime directory; incomplete children stay explicit.

    The recorded CODEX_HOME is authoritative. No CLI, default home, agent final
    JSON, or timestamp-based search substitutes for missing execution records.
    """
    try:
        result = _object(parse_json((run_dir / "result.json").read_bytes()))
        if type(result["schema_version"]) is not int or result["schema_version"] != 1:
            raise NativeEvidenceError("unsupported runner result version")
        if result["result"] not in {"running", "completed", "failed"}:
            raise NativeEvidenceError("invalid runner result state")
        contract = _object(result["contract"])
        if contract["settings"]["agent"] != "codex" or contract["mode"] != "read-only":
            raise NativeEvidenceError("native evidence requires read-only Codex")
        runtime = _object(result["runtime"])
        parent = identifier(runtime.get("session_id"))
        turn_id = identifier(runtime.get("turn_id"))
        home = Path(_text(contract["runtime_home"]))
        if not home.is_absolute() or not Path(_text(contract["cwd"])).is_absolute():
            raise NativeEvidenceError("runner home/worktree must be absolute")
        path = Path(codex_context(home, parent)["path"])
        reference, rows = _read(path)
        meta = _meta(rows, parent, parent)
        if meta.get("source") != "exec" or meta.get("cwd") != contract["cwd"]:
            raise NativeEvidenceError("parent session/worktree mismatch")
        selected = _turn(rows, turn_id)
        context = _context(selected)
        if selected[0]["payload"].get("root_turn_id") != turn_id:
            raise NativeEvidenceError("parent root turn mismatch")
        completions = [
            row
            for row in selected
            if row["type"] == "event_msg"
            and row["payload"].get("type") == "task_complete"
        ]
        if len(completions) > 1 or (
            result["result"] == "completed" and len(completions) != 1
        ):
            raise NativeEvidenceError("parent runner/transcript completion mismatch")
        if completions and completions[0]["payload"].get("root_turn_id") != turn_id:
            raise NativeEvidenceError("parent completion root turn mismatch")
        if context.get("cwd") != contract["cwd"] or context.get("sandbox_policy") != {
            "type": "read-only"
        }:
            raise NativeEvidenceError("parent worktree/sandbox mismatch")
        for key in ("model", "effort"):
            if _text(context.get(key)) != _text(contract["settings"]["selected"][key]):
                raise NativeEvidenceError(f"parent effective {key} mismatch")
        _text(context["approval_policy"])
        launches, rejected = _launches(selected)
        agents: dict[str, NativeAgent] = {}
        for child_path, child_meta in _child_paths(home, parent):
            agent = _child(child_path, child_meta, parent, turn_id, context, launches)
            if agent is None:
                continue
            name = agent.task_name
            if name in agents or any(
                a.agent_id == agent.agent_id for a in agents.values()
            ):
                raise NativeEvidenceError("ambiguous child transcript/name/ID")
            agents[name] = agent
        if launches.keys() != agents.keys():
            raise NativeEvidenceError(
                "spawned child transcript missing or parent turn mismatch"
            )
        return NativeRun(
            parent,
            turn_id,
            reference,
            tuple(agents[name] for name in launches),
            rejected,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise NativeEvidenceError(f"native execution evidence: {exc}") from exc


def reconcile(run_dir: Path, task_agents: Mapping[str, str]) -> dict[str, NativeAgent]:
    """Require a bijection from ledger task IDs to completed native session IDs."""
    evidence = inspect_run(run_dir)
    if any(not isinstance(task, str) or not task.strip() for task in task_agents):
        raise NativeEvidenceError("invalid ledger task ID")
    ids = list(task_agents.values())
    if any(not isinstance(value, str) or not value for value in ids) or len(
        set(ids)
    ) != len(ids):
        raise NativeEvidenceError("missing or duplicate ledger native ID")
    agents = {agent.agent_id: agent for agent in evidence.agents}
    if set(ids) != agents.keys():
        raise NativeEvidenceError(
            "ledger task/native ID mismatch or missing assignment"
        )
    if any(agent.status != "completed" for agent in agents.values()):
        raise NativeEvidenceError("native child execution is not completed")
    return {task: agents[agent_id] for task, agent_id in task_agents.items()}
