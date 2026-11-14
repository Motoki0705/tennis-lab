"""Shared manual/managed hooks and the parent's explicit recovery commands."""

from __future__ import annotations

import copy
import re
import subprocess
from pathlib import Path
from typing import Any

from parent.state import RegistryConflict

from . import journal, schema, sources
from .journal import PublicationError, require
from .publication import Publisher


def capture(session: Any, child: int, issue_body: str, prepared: Any) -> None:
    turn = session.row["tasks"][str(child)]["preflight"]["turns"][-1]
    turn["instruction_source"]["issue_sha256"] = schema.sha(issue_body.encode("utf-8"))
    try:
        with session.unlocked() as external:
            sources.freeze(external, child, prepared.text, prepared.standard)
    except RegistryConflict:
        raise
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        AttributeError,
        subprocess.SubprocessError,
    ):
        turn["log"] = {"state": "blocked", "reason": "input_capture_failed"}


def publish(
    path: Path, *, dry_run: bool = False, retry: bool = False
) -> dict[str, Any]:
    try:
        row = journal.read(path)
        result = Publisher(Path(row["source"]["repository_root"]), path).publish(
            dry_run=dry_run, retry=retry
        )
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        AttributeError,
        subprocess.SubprocessError,
    ):
        result = {
            "state": "blocked",
            "error": "journal_unavailable",
            "saved_to_main": False,
        }
    from decision_logs.observations import worker_log

    worker_log(path, result, retry=retry)
    return result


def finish(
    path: Path, proof: dict[str, Any], *, publish_now: bool = True
) -> dict[str, Any]:
    try:
        result = sources.receive(path, proof)
        from decision_logs.observations import worker_log

        worker_log(path, result, entrypoint="worker_logs.receive")
        if publish_now and result["state"] in {"ready", "pending", "merged"}:
            return publish(path)
        return result
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        AttributeError,
        subprocess.SubprocessError,
    ):
        return {
            "state": "blocked",
            "error": "receipt_unavailable",
            "saved_to_main": False,
        }


def reject_unstarted(path: str | None) -> dict[str, Any]:
    """Retain a captured instruction without waiting for an impossible final."""
    if path is not None:
        try:
            target = Path(path)
            with journal.locked(target):
                row = journal.read(target)
                row["stage"] = "not_started"
                journal.fail(target, row, "runner_not_started", state="worker_failed")
                result = journal.view(row)
            from decision_logs.observations import worker_log

            worker_log(target, result, entrypoint="worker_logs.reject-unstarted")
            return result
        except (ValueError, OSError, KeyError, TypeError):
            pass
    return {"state": "blocked", "error": "journal_unavailable", "saved_to_main": False}


def finish_manual(path: str | None, directory: Path, runner: Path) -> dict[str, Any]:
    if path is None:
        return {
            "state": "blocked",
            "error": "input_capture_failed",
            "saved_to_main": False,
        }
    from dispatcher.runtime import Runtime

    try:
        proof = Runtime(runner).inspect(directory)
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        AttributeError,
        subprocess.SubprocessError,
    ):
        target = Path(path)
        try:
            with journal.locked(target):
                row = journal.read(target)
                journal.fail(target, row, "runner_evidence_unavailable")
                return journal.view(row)
        except (ValueError, OSError):
            return {
                "state": "blocked",
                "error": "runner_evidence_unavailable",
                "saved_to_main": False,
            }
    return finish(Path(path), proof)


def _command(
    gh: Any,
    store: Any,
    parent: int,
    child: int,
    identity: str | None,
    *,
    publish_logs: bool,
    sequence: int | None = None,
    key: str | None = None,
    dry_run: bool = False,
    retry: bool = False,
) -> dict[str, Any]:
    data = store.read_data()
    owner = data.get("parents", {}).get(f"{gh.repo.casefold()}#{parent}")
    require(isinstance(owner, dict), "log_owner_unavailable")
    if publish_logs:
        # Historical publication does not require the issue to remain open, or
        # re-authorize an implementation. The saved capture is the authority.
        require(owner.get("session") == identity, "log_owner_mismatch")
    if key is not None:
        require(bool(re.fullmatch(r"d-[a-f0-9]{32}", key)), "invalid_public_key")
    root = (
        store.directory
        / "worker-logs"
        / schema.sha(gh.repo.casefold().encode())
        / str(parent)
        / str(child)
    )
    paths = sorted(
        root.glob(
            f"{key[2:] if key else '*'}/{sequence if sequence else '*'}/journal.json"
        )
    )
    results = []
    for path in paths:
        row = journal.read(path)
        require(
            row["source"]["repo"].casefold() == gh.repo.casefold()
            and row["source"]["parent"] == parent
            and row["source"]["issue"] == child,
            "journal_identity_changed",
        )
        if publish_logs:
            if row.get("stage") == "not_started":
                results.append(journal.view(row))
                continue
            if row["state"] == "captured" or retry and "receipt" not in row:
                # Reconcile the original recorded run. Never launch/resume a worker.
                task = copy.deepcopy(owner.get("tasks", {}).get(str(child), {}))
                phase = task.get("preflight", {})
                turn = next(
                    (
                        t
                        for t in phase.get("turns", [])
                        if t["snapshot"]["turn_id"] == row["source"]["turn_id"]
                    ),
                    None,
                )
                require(
                    turn is not None
                    and turn.get("source") == row["source"]["run_directory"],
                    "runtime_reference_unavailable",
                )
                from dispatcher.runtime import Runtime
                from parent.manual_workers import RUNNER

                jobs = (owner.get("dispatch") or {}).get("jobs", {})
                runner = next(
                    (
                        Path(j["runner"])
                        for j in jobs.values()
                        if j.get("preflight", {}).get("turn_id")
                        == row["source"]["turn_id"]
                    ),
                    RUNNER,
                )
                if not dry_run:
                    result = finish(
                        path,
                        Runtime(runner).inspect(Path(row["source"]["run_directory"])),
                        publish_now=False,
                    )
                    if result["state"] not in {"ready", "pending", "merged"}:
                        results.append(result)
                        continue
            results.append(publish(path, dry_run=dry_run, retry=retry))
        else:
            result = journal.view(row)
            from decision_logs.observations import worker_log

            worker_log(path, result, entrypoint="worker_logs.status")
            results.append(result)
    phase = owner.get("tasks", {}).get(str(child), {}).get("preflight", {})
    legacy = [
        i + 1
        for i, turn in enumerate(phase.get("turns", []))
        if "instruction_source" not in turn
        or turn.get("log", {}).get("state") == "legacy"
    ]
    unavailable = [
        {"sequence": i + 1, "state": "blocked", "error": "input_capture_failed"}
        for i, turn in enumerate(phase.get("turns", []))
        if "instruction_source" in turn and not turn.get("log", {}).get("journal")
    ]
    if publish_logs:
        require(bool(paths), "no_captured_turn")
    return {
        "child": child,
        "logs": results,
        "unpublished_legacy_turns": legacy,
        "unavailable_logs": unavailable,
    }


def command(
    gh: Any, store: Any, parent: int, child: int, identity: str | None, **kwargs: Any
) -> dict[str, Any]:
    try:
        return _command(gh, store, parent, child, identity, **kwargs)
    except PublicationError as exc:
        code = str(exc)
        raise PublicationError(
            code if re.fullmatch(r"[a-z_]+", code) else "log_operation_failed"
        ) from None
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        AttributeError,
        subprocess.SubprocessError,
    ):
        raise PublicationError("log_operation_failed") from None
