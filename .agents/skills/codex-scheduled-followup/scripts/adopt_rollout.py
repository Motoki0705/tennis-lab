"""Rebind a helper task only after its pending nonce completed in a verified continuation.

This does not write Codex state, send a message, restart a runtime, or touch timers.
The operator supplies the exact continuation path; no newest-file discovery occurs.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any


def load_helper(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("followup_adoption_helper", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the explicitly selected helper")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def reconciled(helper: ModuleType, task: dict[str, Any]) -> dict[str, Any]:
    candidate = copy.deepcopy(task)
    namespace = vars(helper)
    original_save = namespace["save"]
    try:
        namespace["save"] = lambda *_args: None
        helper.reconcile(Path("."), candidate)
    finally:
        namespace["save"] = original_save
    if candidate.get("blocked_reason"):
        raise ValueError(candidate["blocked_reason"])
    return candidate


def preview(
    helper: ModuleType, original: dict[str, Any], rollout: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    version = getattr(helper, "STATE_VERSION", 1)
    if original.get("version") != version:
        raise ValueError("Use the helper version that owns this task state")
    task = reconciled(helper, original)
    if not task["deliveries"] or task["deliveries"][-1]["phase"] not in {
        "queued", "started"
    }:
        raise ValueError("Only a pending received delivery can be adopted")
    parent = Path(task["rollout"]).resolve()
    rollout = rollout.resolve(strict=True)
    if rollout == parent:
        raise ValueError("The continuation must be a different rollout file")
    with rollout.open("rb") as stream:
        first = stream.readline()
        stat = os.fstat(stream.fileno())
        if not first.endswith(b"\n"):
            raise ValueError("Incomplete continuation metadata")
        record = json.loads(first)
        meta = record.get("payload", {})
        if record.get("type") != "session_meta" or meta.get("id") != task["thread_id"]:
            raise ValueError("Continuation thread does not match")
        if meta.get("cwd") != task["cwd"]:
            raise ValueError("Continuation cwd does not match")
        base = meta.get("history_base", {})
        if not isinstance(base, dict) or base.get("thread_id") != task["thread_id"]:
            raise ValueError("Missing same-thread history_base")
        offset = base.get("end_byte_offset")
        ordinal = base.get("end_ordinal_exclusive")
        if type(offset) is not int or type(ordinal) is not int or offset <= 0 or ordinal <= 0:
            raise ValueError("Invalid history_base byte/record boundary")
        if offset > task["observed_boundary"]["offset"]:
            raise ValueError("History base extends beyond the verified parent prefix")
        with parent.open("rb") as parent_stream:
            prefix = parent_stream.read(offset)
        if len(prefix) != offset or not prefix.endswith(b"\n") or prefix.count(b"\n") != ordinal:
            raise ValueError("History base does not match the parent record boundary")
        anchor = {
            "identity": [stat.st_dev, stat.st_ino],
            "offset": len(first),
            "line_complete": True,
        }
        if "window_sha256" in task["observed_boundary"]:
            anchor["window_sha256"] = helper.window_digest(stream, len(first))
        else:
            anchor["sha256"] = hashlib.sha256(first).hexdigest()
    candidate = copy.deepcopy(task)
    candidate["rollout"] = str(rollout)
    candidate["boundary"] = copy.deepcopy(anchor)
    candidate["observed_boundary"] = copy.deepcopy(anchor)
    delivery = candidate["deliveries"][-1]
    delivery["boundary"] = copy.deepcopy(anchor)
    candidate = reconciled(helper, candidate)
    delivery = candidate["deliveries"][-1]
    if delivery["phase"] != "completed" or not delivery.get("turn_id"):
        raise ValueError("Continuation does not prove the pending nonce completed")
    completed = datetime.fromisoformat(delivery["completed_at"].replace("Z", "+00:00"))
    prepared = datetime.fromisoformat(delivery["prepared_at"].replace("Z", "+00:00"))
    if completed.tzinfo is None or prepared.tzinfo is None or completed < prepared:
        raise ValueError("Completion predates the delivery or lacks a timezone")
    proof = {
        "recorded_at": helper.now(),
        "old_rollout": str(parent),
        "new_rollout": str(rollout),
        "thread_id": task["thread_id"],
        "cwd": task["cwd"],
        "history_base": base,
        "parent_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        "continuation_metadata_sha256": hashlib.sha256(first).hexdigest(),
        "old_boundary": task["boundary"],
        "old_observed_boundary": task["observed_boundary"],
        "old_delivery_boundary": task["deliveries"][-1]["boundary"],
        "old_delivery_phase": task["deliveries"][-1]["phase"],
        "marker": delivery["marker"],
        "turn_id": delivery["turn_id"],
        "completed_at": delivery["completed_at"],
    }
    for previous in candidate["deliveries"]:
        previous.setdefault("rollout", str(parent))
    delivery["rollout"] = str(rollout)
    candidate.setdefault("rollout_adoptions", []).append(proof)
    return candidate, proof


def adopt(
    helper: ModuleType, task_dir: Path, rollout: Path, *, apply: bool
) -> dict[str, Any]:
    with (task_dir / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = json.loads((task_dir / "task.json").read_text())
        candidate, proof = preview(helper, original, rollout)
        if apply:
            # Keep the original helper state and boundaries, even if interrupted
            # between backing it up and atomically publishing the new state.
            backup = task_dir / ("before-rollout-adoption-" + uuid.uuid4().hex + ".json")
            fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as handle:
                json.dump(original, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            candidate = reconciled(helper, candidate)
            helper.save(task_dir, candidate)
        return {"applied": apply, **proof}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--helper", type=Path, default=Path(__file__).with_name("cli_heartbeat.py"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = adopt(load_helper(args.helper), args.task_dir, args.rollout, apply=args.apply)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
