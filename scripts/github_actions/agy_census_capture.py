"""Offline command observations from agy streams; never used to grant permission."""

from __future__ import annotations

import importlib
import json
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".agents/ops/lib"))
permissions_api: Any = importlib.import_module("agy_permissions")


def grammar_allows(command: str) -> bool:
    """Evaluate the production regex grammar against the original bytes as text."""
    rules: dict[str, list[str]] = permissions_api.command_permissions()

    def matches(rule: str) -> bool:
        return re.search(rule[len("command(regex:") : -1], command) is not None

    return not any(matches(rule) for rule in rules["deny"]) and any(
        matches(rule) for rule in rules["allow"]
    )


def command_form(command: str) -> str:
    """Describe argv shapes, preserving flag spelling and shell syntax.

    This intentionally is not a shell safety parser. Permission is evaluated on
    the original command independently; raw examples must accompany each form.
    """
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
                    parts.append(_argv_form(clause))
                    clause = []
                parts.append(word)
            else:
                clause.append(word)
        if clause:
            parts.append(_argv_form(clause))
        return " ".join(parts)
    return _argv_form(words)


def _argv_form(words: list[str]) -> str:
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
                    flag + "=" + (value if flag in {"--json", "--state"} else "<VALUE>")
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


def capture_stream(raw: str, log: str = "") -> dict[str, Any]:
    """Keep a step once, including ACTIVE-only steps terminated by soft denial."""
    steps: dict[tuple[str, int], dict[str, Any]] = {}
    result: dict[str, Any] | None = None
    init: dict[str, Any] | None = None
    conversation: str | None = None
    subagent_steps: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        kind = event.get("event")
        if kind == "init":
            init = event["init"]
            conversation = event["conversation_id"]
        elif kind == "result":
            if result is not None:
                raise ValueError("multiple result events in a single-turn stream")
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
            key = (cid, step["step_index"])
            entry = steps.setdefault(key, {"conversation_id": cid, "step": key[1]})
            entry.update(
                tool=info["name"],
                state=step["state"],
                parameters={
                    **entry.get("parameters", {}),
                    **info.get("parameters", {}),
                },
            )
            for field in ("error", "output"):
                if field in info:
                    entry[field] = info[field]
        else:
            raise ValueError(f"unknown stream event at line {line_number}: {kind}")
    soft_denied = {
        int(match)
        for match in re.findall(
            r'Print mode: soft-denying tool confirmation "RunCommand" at step (\d+)',
            log,
        )
    }
    commands = []
    for entry in steps.values():
        if entry["tool"] != "run_command":
            continue
        command = entry["parameters"].get("CommandLine")
        if not isinstance(command, str):
            raise ValueError("run_command observation lacks its full CommandLine")
        error = entry.get("error", {})
        if entry["step"] in soft_denied:
            outcome = "soft_denied"
        elif "denied by pre-tool hook" in str(error).lower():
            outcome = "hook_denied"
        elif "permission" in str(error).lower() and "den" in str(error).lower():
            outcome = "denied"
        elif error:
            outcome = "tool_error"
        elif entry["state"] == "DONE" and "output" in entry:
            outcome = "completed"
        else:
            outcome = "unknown"
        commands.append(
            {
                **entry,
                "command": command,
                "form": command_form(command),
                "grammar_allowed": grammar_allows(command),
                "outcome": outcome,
                "command_exit_code": (
                    int(match.group(1))
                    if (
                        match := re.search(
                            r"Command exited with code (-?\d+)", entry.get("output", "")
                        )
                    )
                    else None
                ),
            }
        )
    seen_denials = {
        entry["step"] for entry in commands if entry["outcome"] == "soft_denied"
    }
    return {
        "init": init,
        "result": result,
        "commands": commands,
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
