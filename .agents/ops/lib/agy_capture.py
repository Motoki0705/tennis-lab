"""Offline command observations from agy streams; never used to grant permission."""

from __future__ import annotations

import re
import shlex
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_events import parse_json
from agy_permissions import command_allowed as grammar_allows


def command_form(command: str, *, public: bool = False) -> str:
    """Describe argv shapes, preserving flag spelling and shell syntax.

    This intentionally is not a shell safety parser. Permission is evaluated on
    the original command independently; raw examples must accompany each form.
    """
    # Unknown executables, subcommands, flags and their values can all contain
    # private text. Only render known grammar; census descriptions stay private.
    if public and not grammar_allows(command):
        return "<unlisted command>"
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        lexer.commenters = ""
        words = list(lexer)
    except ValueError:
        return "<unparseable shell>"
    if not words:
        return "<empty>"
    separators = {"|", "||", "&&", ";", ">", ">>", "<", "(", ")", "&"}
    if any(word in separators for word in words):
        parts: list[str] = []
        clause: list[str] = []
        for word in words:
            if word in separators:
                if clause:
                    parts.append(_argv_form(clause, public=public))
                    clause = []
                parts.append(word)
            else:
                clause.append(word)
        if clause:
            parts.append(_argv_form(clause, public=public))
        return " ".join(parts)
    return _argv_form(words, public=public)


def _argv_form(words: list[str], *, public: bool = False) -> str:
    program = words[0]
    result = [program]
    index = 1
    if program == "git" and index < len(words) and words[index] == "--no-pager":
        result.append(words[index])
        index += 1
    subcommands = 2 if program == "gh" else 1 if program == "git" else 0
    result.extend(words[index : index + subcommands])
    index += subcommands
    search_operand = program in {"rg", "grep"} or result[:2] == ["git", "grep"]
    value_flags = {
        "-n",
        "-A",
        "-B",
        "-C",
        "-g",
        "--glob",
        "--json",
        "--repo",
        "-R",
        "--limit",
        "--state",
        "-maxdepth",
        "-mindepth",
        "-type",
        "-name",
    }
    # -n is a switch for search programs, and a value for head/tail/git log.
    if search_operand:
        value_flags.remove("-n")
    if Path(program).name in {"python", "python3", "sh", "bash"}:
        value_flags.add("-c")
    value_after: str | None = None
    for word in words[index:]:
        if word in {"|", "||", "&&", ";", ">", ">>", "<", "(", ")", "&"}:
            result.append(word)
        elif value_after:
            if value_after == "-c":
                result.append("<CODE>")
            elif value_after == "--json" and public:
                result.append("<FIELDS>")
            elif value_after in {"--json", "--state", "-type"}:
                result.append(word)
            elif value_after in {"--repo", "-R"}:
                result.append("<OWNER/REPO>")
            elif word.isdecimal():
                result.append("<N>")
            else:
                result.append("<VALUE>")
            value_after = None
        elif word.startswith("-"):
            if "=" in word:
                flag, value = word.split("=", 1)
                result.append(
                    flag
                    + "="
                    + (
                        value
                        if flag in {"--json", "--state"} and not public
                        else "<VALUE>"
                    )
                )
            elif re.fullmatch(r"-[nABC][0-9]+", word):
                result.append(word[:2] + "<N>")
            else:
                result.append(word)
                if word in value_flags:
                    value_after = word
            if word == "--files":
                search_operand = False
        elif search_operand:
            result.append("<PATTERN>")
            search_operand = False
        elif word.isdecimal():
            result.append("<N>")
        elif word in {".", ".."}:
            result.append(word)
        elif "/" in word or "." in word or word.startswith("~"):
            shape = "ABS_PATH" if word.startswith("/") else "PATH"
            result.append("<" + shape + ">")
        else:
            result.append("<ARG>")
    return " ".join(result)


def capture_stream(raw: str, log: str = "", *, strict: bool = True) -> dict[str, Any]:
    """Keep a step once, including ACTIVE-only steps terminated by soft denial."""
    steps: dict[tuple[str, int], dict[str, Any]] = {}
    result: dict[str, Any] | None = None
    init: dict[str, Any] | None = None
    conversation: str | None = None
    subagent_steps: list[dict[str, Any]] = []
    errors: list[str] = []
    result_seen = False
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = parse_json(line)
            kind = event.get("event")
            if kind == "init":
                init = event["init"]
                conversation = event["conversation_id"]
                if not isinstance(init, dict) or not isinstance(conversation, str):
                    raise ValueError("invalid init event")
            elif kind == "result":
                if result_seen:
                    raise ValueError("multiple result events in a single-turn stream")
                result_seen = True
                if not isinstance(event["result"], dict):
                    raise ValueError("result event must contain an object")
                result = event["result"]
            elif kind == "step_update":
                step = event["step_update"]
                if "subagent_info" in step:
                    subagent_steps.append(step)
                info = step.get("tool_info")
                if info is None:
                    continue
                cid = step.get("conversation_id", conversation)
                if not isinstance(cid, str):
                    raise ValueError(f"missing conversation at line {line_number}")
                parameters = info.get("parameters", {})
                if (
                    type(step["step_index"]) is not int
                    or step["step_index"] < 0
                    or not isinstance(step["state"], str)
                    or not isinstance(info["name"], str)
                    or not isinstance(parameters, dict)
                ):
                    raise ValueError("invalid tool update")
                key = (cid, step["step_index"])
                entry = steps.setdefault(key, {"conversation_id": cid, "step": key[1]})
                entry.update(
                    tool=info["name"],
                    state=step["state"],
                    parameters={
                        **entry.get("parameters", {}),
                        **parameters,
                    },
                )
                for field in ("error", "output"):
                    if field in info:
                        entry[field] = info[field]
                if (
                    entry["tool"] in {"run_command", "read_file"}
                    and entry["state"] == "DONE"
                    and isinstance(entry.get("output"), str)
                ):
                    # Later updates must not hide a completed execution.
                    entry["execution_observed"] = True
            else:
                raise ValueError(f"unknown stream event at line {line_number}: {kind}")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            if strict:
                raise ValueError(f"invalid stream event at line {line_number}") from exc
            errors.append(f"invalid stream event at line {line_number}")
    soft_denied = {
        int(match)
        for match in re.findall(
            r'Print mode: soft-denying tool confirmation "RunCommand" at step (\d+)',
            log,
        )
    }
    commands: list[dict[str, Any]] = []
    reads: list[dict[str, Any]] = []
    for entry in steps.values():
        if entry["tool"] not in {"run_command", "read_file"}:
            continue
        reading = entry["tool"] == "read_file"
        parameter = "Path" if reading else "CommandLine"
        target = entry["parameters"].get(parameter)
        if not isinstance(target, str):
            if strict:
                raise ValueError("tool observation lacks its target")
            errors.append("tool observation lacks its target")
            continue
        error = entry.get("error", {})
        if not reading and entry["step"] in soft_denied:
            outcome = "soft_denied"
        elif "denied by pre-tool hook" in str(error).lower():
            outcome = "hook_denied"
        elif "permission" in str(error).lower() and "den" in str(error).lower():
            outcome = "denied"
        elif error:
            outcome = "tool_error"
        elif entry["state"] == "DONE" and isinstance(entry.get("output"), str):
            outcome = "completed"
        else:
            outcome = "unknown"
        observations = reads if reading else commands
        observations.append(
            {
                **entry,
                "outcome": outcome,
                **(
                    {"path": target, "form": "read_file <PATH>"}
                    if reading
                    else {
                        "command": target,
                        "form": command_form(target),
                        "grammar_allowed": grammar_allows(target),
                        "command_exit_code": (
                            int(match.group(1))
                            if (
                                match := re.search(
                                    r"Command exited with code (-?\d+)",
                                    str(entry.get("output", "")),
                                )
                            )
                            else None
                        ),
                    }
                ),
            }
        )
    seen_denials = {
        entry["step"] for entry in commands if entry["outcome"] == "soft_denied"
    }
    return {
        "errors": errors,
        "init": init,
        "result": result,
        "commands": commands,
        "reads": reads,
        "tool_counts": dict(Counter(entry["tool"] for entry in steps.values())),
        "subagent_steps": subagent_steps,
        "uncaptured_soft_denial_steps": sorted(soft_denied - seen_denials),
    }


def summarize(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate descriptions, while retaining every raw example and its policy."""
    groups: dict[str, dict[str, Any]] = {}
    for run in runs:
        for command in run["commands"]:
            form = command["form"]
            group = groups.setdefault(
                form,
                {
                    "form": form,
                    "count": 0,
                    "runs": set(),
                    "outcomes": Counter(),
                    "examples": {},
                },
            )
            group["count"] += 1
            group["runs"].add(run["run_id"])
            group["outcomes"][command["outcome"]] += 1
            example = group["examples"].setdefault(
                command["command"],
                {
                    "command": command["command"],
                    "count": 0,
                    "grammar_allowed": command["grammar_allowed"],
                },
            )
            example["count"] += 1
    return [
        {
            **group,
            "runs": sorted(group["runs"]),
            "outcomes": dict(group["outcomes"]),
            "examples": list(group["examples"].values()),
        }
        for _, group in sorted(groups.items())
    ]


def restore_hook_commands(
    capture: dict[str, Any],
    raw_hooks: str | list[dict[str, Any]],
    *,
    require_hook: bool = False,
) -> None:
    """Reconcile command/read observations with the independent pre-tool input.

    Preserve the display value and provenance. Arbitrary disagreements and hook
    steps absent from the stream fail loudly rather than being overwritten.
    """
    fields = {"run_command": "command", "read_file": "path"}
    hooks: dict[tuple[str, int], dict[str, Any]] = {}
    records = (
        [parse_json(line) for line in raw_hooks.splitlines()]
        if isinstance(raw_hooks, str)
        else raw_hooks
    )
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid hook tool record")
        # Only legacy census command records omit the tool name.
        tool = record.get("tool", "run_command")
        index = record.get("step")
        if (
            not isinstance(tool, str)
            or tool not in fields
            or type(index) is not int
            or index < 0
            or not isinstance(record.get(fields[tool]), str)
        ):
            raise ValueError("invalid hook tool record")
        if record.get("decision") not in {"allow", "deny"}:
            raise ValueError("invalid hook decision")
        key = (tool, index)
        if key in hooks and hooks[key] != record:
            raise ValueError("conflicting hook records for one tool step")
        hooks[key] = record
    seen: set[tuple[str, int]] = set()
    restored = 0
    for entry in [*capture["commands"], *capture.get("reads", [])]:
        key = (entry["tool"], entry["step"])
        field = fields[entry["tool"]]
        source = field + "_capture_source"
        if key not in hooks:
            entry[source] = "stream"
            if require_hook:
                raise ValueError("tool step has no hook decision")
            continue
        if key in seen:
            raise ValueError("hook step is ambiguous across conversations")
        seen.add(key)
        record = hooks[key]
        shown, full = entry[field], record[field]
        if shown != full:
            if field == "path":
                raise ValueError("stream and hook paths disagree")
            suffix = "…" if shown.endswith("…") else "..."
            if (
                not shown.endswith(suffix)
                or not full.startswith(shown[: -len(suffix)])
                or len(full) <= len(shown)
            ):
                raise ValueError("stream and hook commands disagree without truncation")
            entry["stream_command"] = shown
            entry["command"] = full
            entry["parameters"]["CommandLine"] = full
            entry["form"] = command_form(full)
            entry["grammar_allowed"] = grammar_allows(full)
            entry[source] = "pre_tool_hook"
            restored += 1
        else:
            if "stream_command" in entry:
                entry[source] = "pre_tool_hook"
                restored += 1
            else:
                entry[source] = "stream_and_hook"
        entry["hook_decision"] = record["decision"]
        if record["decision"] == "deny" and (
            entry["outcome"] == "completed"
            or entry.get("execution_observed") is True
            or entry.get("command_exit_code") is not None
        ):
            raise ValueError("a hook-denied tool was observed executing")
    if hooks.keys() - seen:
        raise ValueError("hook tool step is missing from the stream")
    capture["restored_stream_commands"] = restored


def _public_tool_summary(
    observations: list[dict[str, Any]],
    hooks: list[dict[str, Any]],
    field: str,
    form: Callable[[str], str],
) -> dict[str, Any]:
    attempts = {(entry["step"], entry[field]): entry for entry in observations}
    hook_denied = set()
    for hook in hooks:
        target = hook.get(field)
        if not isinstance(target, str):
            continue
        key = (hook["step"], target)
        attempts.setdefault(key, {field: target, "outcome": "unknown"})
        if hook["decision"] == "deny":
            hook_denied.add(key)
    denied = hook_denied | {
        key
        for key, entry in attempts.items()
        if entry["outcome"] in {"soft_denied", "hook_denied", "denied"}
    }
    forms = Counter(form(entry[field]) for entry in attempts.values())
    return {
        "attempted_count": len(attempts),
        "denied_count": len(denied),
        "hook_denied_count": len(hook_denied),
        "forms": [
            {"form": shape, "count": count} for shape, count in sorted(forms.items())
        ],
    }


def public_command_summary(capture: dict[str, Any]) -> dict[str, Any]:
    """Publish counts and redacted grammar shapes, never raw examples or errors."""
    hooks = capture.get("hook_decisions", [])
    result = _public_tool_summary(
        capture.get("commands", []),
        [entry for entry in hooks if entry.get("tool", "run_command") == "run_command"],
        "command",
        lambda command: command_form(command, public=True),
    )
    read_hooks = [entry for entry in hooks if entry.get("tool") == "read_file"]
    if capture.get("reads") or read_hooks:
        reads = _public_tool_summary(
            capture.get("reads", []), read_hooks, "path", lambda _: "read_file <PATH>"
        )
        result["read_file"] = {
            key: value for key, value in reads.items() if key != "hook_denied_count"
        }
    return result


def reconcile_commands(
    capture: dict[str, Any], trajectory: list[dict[str, Any]]
) -> dict[str, Any]:
    """Compare both directions, including denied attempts and exact command text.

    The caller reads only its own conversation records and supplies one entry
    per command step. Results contain raw commands and remain private.
    """

    def counts(commands: list[dict[str, Any]]) -> Counter[tuple[str, int, str]]:
        return Counter(
            (entry["conversation_id"], entry["step"], entry["command"])
            for entry in commands
        )

    observed, recorded = counts(capture["commands"]), counts(trajectory)
    missing = list((recorded - observed).elements())
    extra = list((observed - recorded).elements())
    return {
        "capture_count": sum(observed.values()),
        "trajectory_count": sum(recorded.values()),
        "missing": missing,
        "extra": extra,
        "matches": not missing and not extra,
    }
