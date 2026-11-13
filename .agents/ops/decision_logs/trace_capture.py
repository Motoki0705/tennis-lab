"""Private-to-public fact adapters. Generation never calls this module.

The termination receipt adapter is the sole boundary for the pending parent
publication API. It accepts an already-read #1961-shaped receipt; lifecycle CLI
wiring and the eventual receipt file location deliberately remain outside it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from worker_logs.journal import decode, require
from worker_logs.schema import digest, json_bytes, sha, validate
from worker_logs.schema import path as worker_path

from .capture import Operation, missing, observed, source


def worker_fact(
    execution: str,
    final: dict[str, Any] | None,
    proof: dict[str, Any],
    *,
    publication: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "kind": "worker_turn",
        "execution": {"state": execution, "evidence": [proof]},
        "final": {
            "state": "present"
            if final and final["status"] == "observed"
            else "absent"
            if final and final.get("reason") == "missing_final" and execution == "ended"
            else "unknown",
            "evidence": [final["value"]]
            if final and final["status"] == "observed"
            else [proof],
        },
        "publication": publication
        or {
            "state": "unknown",
            "stage": None,
            "error": None,
            "pr_url": None,
            "pr_head": None,
            "main_revision": None,
            "evidence": [proof],
        },
    }


def worker_log_snapshot(
    row: dict[str, Any], value: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any], list[dict[str, Any]]]:
    """A final can exist even when receipt validation or publication failed."""
    original = row["source"]
    proof = source("worker_logs.observation", "fixed worker journal snapshot", row)
    final = None
    execution = "unknown"
    evidence: list[dict[str, Any]] = []
    receipt = row.get("receipt")
    if isinstance(receipt, dict):
        text = receipt.get("final")
        if isinstance(text, str) and text.strip():
            require(
                sha(text.encode()) == receipt.get("final_sha256"),
                "received_final_changed",
            )
            final = observed(
                source(
                    "worker_logs.receive",
                    "fixed final receipt",
                    {"content_sha256": receipt["final_sha256"]},
                )
            )
        if receipt.get("record", {}).get("result") in {"completed", "failed"}:
            execution = "ended"
    elif original.get("run_directory"):
        directory = Path(original["run_directory"])
        runtime_path, final_path = directory / "result.json", directory / "final.md"
        require(
            not directory.is_symlink()
            and not runtime_path.is_symlink()
            and not final_path.is_symlink(),
            "private_storage_symlink",
        )
        runtime = decode(runtime_path.read_bytes()) if runtime_path.is_file() else None
        if runtime is not None:
            if original.get("task_id") is not None:
                require(
                    runtime.get("contract", {}).get("task_id") == original["task_id"],
                    "runtime_changed",
                )
            if original.get("prompt_sha256") is not None:
                require(
                    runtime.get("prompt_sha256") == original["prompt_sha256"],
                    "sent_prompt_changed",
                )
            if runtime.get("result") in {"completed", "failed"}:
                execution = (
                    "not_started" if runtime.get("agent_spawned") is False else "ended"
                )
        raw = final_path.read_bytes() if final_path.is_file() else b""
        if raw.strip():
            final = observed(
                source("worker_logs.runtime", "final.md", {"content_sha256": sha(raw)})
            )
        elif execution == "ended":
            final = missing("missing_final")
    if row.get("stage") == "not_started":
        execution = "not_started"
    raw_files = row.get("files")
    has_files = isinstance(raw_files, dict) and bool(raw_files)
    if has_files:
        assert isinstance(raw_files, dict)
        files = {k: v.encode() for k, v in raw_files.items()}
        meta = validate(files)
        require(
            meta["repo"].casefold() == original["repo"].casefold()
            and meta["parent"] == original["parent"]
            and meta["issue"] == original["issue"]
            and meta["delegation_key"] == "d-" + original["public_key"]
            and meta["turn"]["sequence"] == original["sequence"],
            "record_source",
        )
        execution = "ended"
        if digest(row.get("head"), 40):
            base = f"https://github.com/{meta['repo']}/blob/{row['head']}/{worker_path(meta)}"

            def public(name: str) -> dict[str, Any]:
                return {
                    "kind": "public",
                    "url": base + "/" + name,
                    "revision": row["head"],
                    "content_sha256": sha(files[name]),
                }

            evidence = [public("meta.json")]
            final = observed(public("final.md"))
        elif final is None:
            final = observed(
                source(
                    "worker_logs.receive",
                    "validated final bytes",
                    {"content_sha256": sha(files["final.md"])},
                )
            )
    number = row.get("pr_number")
    pr_known = type(number) is int and number > 0
    saved = bool(
        value.get("saved_to_main")
        and has_files
        and digest(row.get("main_observed"), 40)
    )
    public_state = (
        "main"
        if saved
        else "public_pr"
        if pr_known
        and has_files
        and digest(row.get("head"), 40)
        and row["state"] == "pending"
        else "pending"
        if row["state"]
        in {"ready", "pending", "blocked", "worker_failed", "missing_final"}
        else "unknown"
    )
    publication = {
        "state": public_state,
        "stage": row.get("stage"),
        "error": row.get("error"),
        "pr_url": f"https://github.com/{original['repo']}/pull/{number}"
        if pr_known
        else None,
        "pr_head": row.get("head") if digest(row.get("head"), 40) else None,
        "main_revision": row.get("main_observed")
        if digest(row.get("main_observed"), 40)
        else None,
        "evidence": evidence or [proof],
    }
    return (
        final,
        worker_fact(execution, final, proof, publication=publication),
        evidence,
    )


def capture_termination(
    operation: Operation,
    receipt: dict[str, Any] | Path,
    *,
    evidence: list[dict[str, Any]] | None = None,
    acceptance: dict[str, Any] | None = None,
) -> str:
    """Capture an already-read lifecycle receipt, without publishing or CLI IO.

    Assumed private receipt fields are exactly the parent's agreed #1961 shape.
    remaining decision_log uses unpublished_count and batch_key. Public actor
    and resume text must pass the existing projection guard, as do all facts.
    """
    if isinstance(receipt, Path):
        require(not receipt.is_symlink(), "private_storage_symlink")
        value: dict[str, Any] = decode(receipt.read_bytes())
    else:
        value = receipt
    expected = {
        "repo",
        "parent",
        "generation",
        "event",
        "observed_at",
        "main_revision",
        "remaining",
        "publication_actor",
        "resume_command",
    }
    require(set(value) == expected, "record_schema")
    scope = operation.row["scope"]
    require(
        value["repo"].casefold() == scope["repo"].casefold()
        and value["parent"] == scope["parent"]
        and scope["issue"] is None
        and scope["delegation"] is None,
        "record_source",
    )
    proof = source(
        "parent.termination",
        "fixed lifecycle publication reconciliation receipt",
        value,
    )
    observation = {
        "version": 1,
        "kind": "termination",
        **{k: copy.deepcopy(value[k]) for k in expected - {"repo", "parent"}},
        "remaining_status": "observed",
        "acceptance": acceptance or {"state": "unverified", "evidence": [proof]},
        "evidence": evidence or [proof],
    }
    with operation.transaction():
        return operation.result(
            value,
            artifact="lifecycle publication reconciliation receipt",
            status="partial" if value["remaining"] else "success",
            summary="Parent lifecycle and publication reconciliation observed; acceptance is recorded separately",
            observation=observation,
            evidence=evidence,
            key="termination:" + sha(json_bytes(observation)),
            at=value["observed_at"],
        )
