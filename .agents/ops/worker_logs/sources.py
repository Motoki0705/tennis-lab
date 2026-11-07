"""Freeze producer inputs and terminal runner evidence without changing the runner."""

from __future__ import annotations

import copy
import math
import os
import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import journal, schema
from .journal import PublicationError, require
from .safety import Guard

TEMPLATE = Path(__file__).resolve().parents[1] / "parent/worker_prompt.md"
IDENTITIES = (
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_PARENT_SESSION_ID",
    "CLAUDE_CODE_AGENT_ID",
    "CODEX_SESSION_ID",
    "CODEX_THREAD_ID",
    "AGENT_RUNTIME_RUN_DIR",
    "AGENT_RUNTIME_TASK_ID",
)


def original(text: Any, field: str = "worker instruction") -> str:
    if not isinstance(text, str) or not text.strip():
        raise PublicationError("empty_worker_instruction")
    return text


def read_original(path: Path | str) -> str:
    try:
        return original(Path(path).read_bytes().decode("utf-8"))
    except (OSError, UnicodeError):
        raise PublicationError("instruction_unavailable") from None


def public_key(phase: dict[str, Any]) -> str:
    if "public_key" not in phase:
        require(
            not any(t.get("log", {}).get("journal") for t in phase.get("turns", [])),
            "public_key_missing",
        )
        phase["public_key"] = uuid.uuid4().hex
    key = phase["public_key"]
    if not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{32}", key):
        raise PublicationError("invalid_public_key")
    return key


def location(
    state: Path, repo: str, parent: int, child: int, key: str, sequence: int
) -> Path:
    require(bool(re.fullmatch(r"[a-f0-9]{32}", key)), "invalid_public_key")
    require(
        schema.integer(parent) and schema.integer(child) and schema.integer(sequence),
        "invalid_log_reference",
    )
    return (
        state
        / "worker-logs"
        / schema.sha(repo.casefold().encode())
        / str(parent)
        / str(child)
        / key
        / str(sequence)
        / "journal.json"
    )


def standard_reference(raw: bytes) -> dict[str, str]:
    root = TEMPLATE.parents[3]
    name = ".agents/ops/parent/worker_prompt.md"

    def git(*args: str) -> bytes:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, check=False, timeout=30
        )
        require(result.returncode == 0, "standard_unverifiable")
        return result.stdout

    commit = git("rev-parse", "HEAD").decode().strip()
    require(git("show", f"{commit}:{name}") == raw, "standard_unverifiable")
    blob = git("rev-parse", f"{commit}:{name}").decode().strip()
    return {"commit": commit, "path": name, "blob": blob}


def freeze(session: Any, child: int, prompt: str, standard: bytes) -> Path:
    task = session.row["tasks"][str(child)]
    phase = task["preflight"]
    key = public_key(phase)
    turn = phase["turns"][-1]
    sequence = len(phase["turns"])
    target = location(
        session.store.directory, session.gh.repo, session.parent, child, key, sequence
    )
    # Older turns have no trustworthy instruction snapshot. Do not parse prompts.
    for old in phase["turns"][:-1]:
        old.setdefault("log", {"state": "legacy", "reason": "pre_capture_turn"})
    turn["log"] = {"journal": str(target), "state": "captured"}
    inputs = turn["instruction_source"]
    decision = None
    if turn["snapshot"]["decision_id"]:
        saved = next(
            d
            for d in phase["decisions"]
            if d["decision_id"] == turn["snapshot"]["decision_id"]
        )
        source_sequence = next(
            i + 1
            for i, t in enumerate(phase["turns"])
            if t["snapshot"]["turn_id"] == saved["turn_id"]
        )
        decision = {"action": saved["action"], "source_sequence": source_sequence}
    private = {str(session.row["session"]).partition(":")[2], phase["lineage_id"]}
    for item in phase["turns"]:
        private.update(
            v
            for v in (
                item.get("delegation_id"),
                item["snapshot"]["turn_id"],
                item["snapshot"]["decision_id"],
                item.get("source"),
            )
            if v
        )
    private.update(
        v
        for k, v in os.environ.items()
        if v
        and (
            k in IDENTITIES
            or re.search(r"TOKEN|PASSWORD|SECRET|API_KEY", k, re.IGNORECASE)
        )
    )
    source = {
        "repo": session.gh.repo,
        "parent": session.parent,
        "issue": child,
        "public_key": key,
        "sequence": sequence,
        "phase": turn["snapshot"]["phase"],
        "kind": "resume" if turn["snapshot"]["previous_turn_id"] else "initial",
        "decision": decision,
        "instruction": inputs["text"],
        "instruction_kind": inputs["kind"],
        "additional": inputs["additional"],
        "issue_sha256": inputs["issue_sha256"],
        "turn_id": turn["snapshot"]["turn_id"],
        "lineage_id": phase["lineage_id"],
        "prompt_sha256": schema.sha(prompt.encode("utf-8")),
        "worktree": task["worktree"],
        "worker_branch": task["branch"],
        "head_at_start": turn["head_at_start"],
        "captured_at": datetime.now(UTC).isoformat(),
        "private": sorted(private),
        "run_directory": turn["source"],
        "task_id": task["delegation_id"],
    }
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=task["worktree"],
        capture_output=True,
        check=False,
        timeout=30,
    )
    require(result.returncode == 0, "repository_unavailable")
    common = Path(result.stdout.decode().strip())
    require(common.name == ".git", "invalid_repository_layout")
    source["repository_root"] = str(common.parent)
    with journal.locked(target):
        if target.exists():
            existing = journal.read(target)
            require(
                existing["source"]["prompt_sha256"] == source["prompt_sha256"],
                "source_changed",
            )
            return target
        row: dict[str, Any] = {
            "version": 1,
            "source": source,
            "state": "captured",
            "stage": "awaiting_final",
            "error": None,
        }
        try:
            source["standard"] = standard_reference(standard)
        except (ValueError, OSError, subprocess.SubprocessError):
            source["standard_raw"] = standard.decode("utf-8")
            row.update(state="blocked", error="standard_unverifiable")
        journal.atomic(target, row)
    return target


def bind(path: Path, directory: Path, task_id: str, *, resume: bool) -> None:
    with journal.locked(path):
        row = journal.read(path)
        source = row["source"]
        require(source["run_directory"] in (None, str(directory)), "runtime_changed")
        source.update(
            run_directory=str(directory),
            task_id=task_id,
            kind="resume" if resume else "initial",
        )
        journal.atomic(path, row)


def guard_for(
    source: dict[str, Any], record: dict[str, Any], commit: str | None = None
) -> Guard:
    private = set(source["private"])
    private.update(
        str(v)
        for v in (
            record.get("run_id"),
            record.get("run_dir"),
            record.get("session_root"),
        )
        if v
    )
    runtime = record.get("runtime", {})
    private.update(
        str(runtime[k])
        for k in ("session_id", "turn_id", "requested_session_id")
        if runtime.get(k)
    )
    return Guard(
        private,
        [source["worktree"]],
        repo=source["repo"],
        commit=commit or source["head_at_start"],
    )


def receive(path: Path, proof: dict[str, Any]) -> dict[str, Any]:
    """Called only with status/cleanup evidence, never just an exit code."""
    with journal.locked(path):
        row = journal.read(path)
        if row["state"] in {"merged", "pending", "ready"}:
            return journal.view(row)
        if row["state"] == "blocked" and row.get("error") == "standard_unverifiable":
            return journal.view(row)
        source = row["source"]
        try:
            record, summary = proof["record"], proof["summary"]
            require(
                record.get("cleanup_complete") is True
                and all(
                    summary.get(k) is False
                    for k in ("runner_alive", "agent_alive", "agent_group_alive")
                ),
                "runner_not_ended",
            )
            if (
                record.get("result") != "completed"
                or record.get("turn_end_reason") != "completed"
            ):
                journal.fail(path, row, "worker_failed", state="worker_failed")
                return journal.view(row)
            contract = record["contract"]
            require(
                contract["task_id"] == source["task_id"]
                and contract["cwd"] == source["worktree"]
                and contract["branch"] == source["worker_branch"]
                and contract["mode"] == "write",
                "runner_contract_changed",
            )
            require(
                record.get("prompt_sha256") == source["prompt_sha256"],
                "sent_prompt_changed",
            )
            directory = Path(source["run_directory"])
            require(
                schema.sha((directory / "prompt.md").read_bytes())
                == source["prompt_sha256"],
                "sent_prompt_changed",
            )
            final_path = directory / "final.md"
            if not final_path.is_file() or final_path.is_symlink():
                journal.fail(path, row, "missing_final", state="missing_final")
                return journal.view(row)
            raw = final_path.read_bytes()
            original(raw.decode("utf-8"))
            receipt = {
                "final": raw.decode("utf-8"),
                "final_sha256": schema.sha(raw),
                "record": copy.deepcopy(record),
            }
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=source["worktree"],
                capture_output=True,
                check=False,
                timeout=30,
            )
            require(head.returncode == 0, "worker_head_unavailable")
            receipt["head_at_finish"] = head.stdout.decode().strip()
            if row.get("receipt") is not None:
                require(row["receipt"] == receipt, "received_final_changed")
            row["receipt"] = receipt
            # Persist the exact first received final before sanitization/Git work.
            journal.atomic(path, row)
            files = render(source, receipt)
            row.update(
                files={k: v.decode("utf-8") for k, v in files.items()},
                state="ready",
                stage="receipt_verified",
                error=None,
            )
            journal.atomic(path, row)
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            AttributeError,
            UnicodeError,
            subprocess.SubprocessError,
        ) as exc:
            code = (
                str(exc)
                if isinstance(exc, PublicationError)
                else "receipt_unverifiable"
            )
            journal.fail(path, row, code)
        return journal.view(row)


def render(source: dict[str, Any], receipt: dict[str, Any]) -> dict[str, bytes]:
    require(
        schema.sha(receipt["final"].encode("utf-8")) == receipt["final_sha256"],
        "received_final_changed",
    )
    record = receipt["record"]
    guard = guard_for(source, record)
    instruction, ni = guard.sanitize(source["instruction"])
    final, nf = guard_for(source, record, receipt.get("head_at_finish")).sanitize(
        receipt["final"]
    )
    additional = [guard.sanitize(s) for s in source["additional"]]
    settings = record["contract"]["settings"]
    require(
        settings["agent"] == "codex"
        and settings.get("profile") is None
        and settings["selected"]["effort"] == "max",
        "runner_settings_changed",
    )
    selected = {k: settings["selected"][k] for k in ("model", "effort")}
    effective = {}
    for key in selected:
        observed = record.get("runtime", {}).get("effective", {}).get(key, {})
        require(
            isinstance(observed, dict)
            and observed.get("status") in {"observed", "unavailable"},
            "runner_settings_changed",
        )
        if observed.get("status") == "observed":
            require(observed.get("value") == selected[key], "runner_settings_changed")
            effective[key] = {"status": "observed", "value": observed["value"]}
        else:
            require(observed.get("value") is None, "runner_settings_changed")
            effective[key] = {"status": "unavailable", "value": None}
    repo = source["repo"]
    prs = sorted(
        {
            int(n)
            for n in re.findall(
                re.escape(f"https://github.com/{repo}/pull/") + r"([1-9]\d*)\b", final
            )
        }
    )
    commits = {
        s
        for s in re.findall(
            re.escape(f"https://github.com/{repo}/commit/") + r"([a-f0-9]{40})\b", final
        )
    }
    commits.add(source["head_at_start"])
    if receipt.get("head_at_finish"):
        commits.add(receipt["head_at_finish"])
    files = {
        "instruction.md": instruction.encode("utf-8"),
        "final.md": final.encode("utf-8"),
    }
    meta = {
        "schema_version": 1,
        "repo": repo,
        "parent": source["parent"],
        "issue": source["issue"],
        "delegation_key": "d-" + source["public_key"],
        "turn": {
            "sequence": source["sequence"],
            "kind": source["kind"],
            "phase": source["phase"],
            "previous_sequence": source["sequence"] - 1 or None,
        },
        "decision": source["decision"],
        "sources": {
            "standard_instruction": source["standard"],
            "issue": {
                "url": f"https://github.com/{repo}/issues/{source['issue']}",
                "body_sha256": source["issue_sha256"],
            },
        },
        "instruction": {
            "kind": source["instruction_kind"],
            "additional_instructions": [s for s, _ in additional],
        },
        "worker": {"agent": "codex", "selected": selected, "effective": effective},
        "runtime": {"result": "completed", "turn_end_reason": "completed"},
        "artifacts": {"prs": prs, "commits": sorted(commits)},
        "files": {k: schema.sha(v) for k, v in files.items()},
        "sanitization": {
            "version": 1,
            "instruction": ni,
            "final": nf,
            "additional": sum(n for _, n in additional),
        },
        "started_at": timestamp(record.get("started_at")),
        "finished_at": timestamp(record.get("finished_at")),
        "captured_at": source["captured_at"],
    }
    files["meta.json"] = schema.json_bytes(meta)
    schema.validate(files, guard)
    return files


def timestamp(value: Any) -> str:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        "runner_time_unavailable",
    )
    require(math.isfinite(value), "runner_time_unavailable")
    return datetime.fromtimestamp(value, UTC).isoformat()
