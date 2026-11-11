"""Finite event-driven acceptance: first #1899 model past600 updates plus Drive readback."""

from __future__ import annotations

import ctypes
import json
import math
import os
import select
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import torch
from control import (
    DRIVE,
    RELATIVE,
    ROOT,
    now,
    read,
    snapshot_store,
    store,
    validate_log,
    verify_plan_sources,
    write,
)
from train import named_optimizer, same


def audit():
    plan = read(ROOT / "plan.json")
    verify_plan_sources(plan)
    stage = plan["jobs"][0]["stages"][0]
    directory = Path(stage["output"])
    if not (directory / "train.jsonl").exists():
        return None
    # A concurrent final line may be incomplete; only parse committed newline-terminated rows.
    raw = (directory / "train.jsonl").read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.endswith(b"\n")]
    if len(rows) < 601:
        return None
    for row in rows[:601]:
        for key in ("loss", "gradient_norm", "learning_rate", "encoder_learning_rate"):
            if not math.isfinite(row[key]):
                raise ValueError("Nonfinite production row: " + key)
    if not rows[599]["encoder_frozen"] or rows[599]["encoder_learning_rate"] != 0:
        raise ValueError("CNN did not remain frozen through600 updates")
    if rows[600]["encoder_frozen"] or not math.isclose(
        rows[600]["encoder_learning_rate"],
        rows[600]["learning_rate"] * 0.1,
        rel_tol=1e-12,
    ):
        raise ValueError("CNN did not unfreeze at601 with the specified LR ratio")
    receipt_dir = DRIVE / RELATIVE / "persistence" / stage["name"]
    receipts = list(receipt_dir.glob("checkpoint-00000500-*.json"))
    if not receipts:
        return None
    if len(receipts) != 1:
        raise ValueError("Ambiguous first500 generation")
    receipt = read(receipts[0])
    with tempfile.TemporaryDirectory(prefix="query-controls-first500-") as temporary:
        destination = Path(temporary) / "restored"
        restored = snapshot_store(destination, RELATIVE / stage["name"]).restore(
            receipt["generation"], manifest_sha256=receipt["manifest_sha256"]
        )
        latest = read(destination / "latest.json")
        saved = torch.load(
            destination / latest["checkpoint"], map_location="cpu", weights_only=True
        )
        if saved["updates_completed"] != 500 or saved["sample_position"] != 10500:
            raise ValueError("First durable state has the wrong update/sample position")
        parent = torch.load(
            stage["parent"]["checkpoint"], map_location="cpu", weights_only=True
        )
        old = named_optimizer(parent["optimizer"], parent["optimizer_parameter_names"])
        current = named_optimizer(
            saved["optimizer"], saved["optimizer_parameter_names"]
        )
        cnn = 0
        query = 0
        for name, state in current.items():
            if name.startswith("encoder."):
                if not same(state, old[name]):
                    raise ValueError("Frozen named CNN AdamW state changed: " + name)
                if not same(
                    saved["model_artifact"]["state_dict"][name],
                    parent["model_artifact"]["state_dict"][name],
                ):
                    raise ValueError("Frozen CNN weight changed: " + name)
                cnn += 1
            else:
                if state is None or int(state["step"]) != 500:
                    raise ValueError("New query AdamW state missing or not500: " + name)
                query += 1
            if state is not None and any(
                not bool(torch.isfinite(value).all())
                for value in state.values()
                if isinstance(value, torch.Tensor)
            ):
                raise ValueError("Nonfinite optimizer state: " + name)
        if not cnn or not query or len(saved["rng"]["cuda"]) != 1:
            raise ValueError("Incomplete model/optimizer/RNG snapshot")
        if len(validate_log(stage, destination)) != 500:
            raise ValueError("Drive log prefix is incomplete")
        if (
            raw[: saved["train_log"]["bytes"]]
            != (destination / "train.jsonl").read_bytes()
        ):
            raise ValueError("Drive log differs from the live committed prefix")
        continuity = read(destination / "continuity.json")
        if not continuity["named_states_exact"] or continuity[
            "parent_step_histogram"
        ] != {"10000": cnn}:
            raise ValueError("Pretraining inheritance audit failed")
    service = read(ROOT / "queue-service-state.json")
    job = read(ROOT / "job-status/queries.json")
    processes = {}
    for role, pid in (
        ("service", service["service_pid"]),
        ("queue", service["worker_pid"]),
        ("trainer", job["pid"]),
    ):
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        status = Path(f"/proc/{pid}/status").read_text()
        if "State:\tZ" in status:
            raise ValueError("A claimed active process is a zombie")
        processes[role] = dict(
            pid=pid,
            command=cmd,
            ppid=int(
                next(
                    line.split()[1]
                    for line in status.splitlines()
                    if line.startswith("PPid:")
                )
            ),
        )
    if processes["service"]["ppid"] != 1:
        raise ValueError("Queue service is still owned by the finite launch process")
    evidence = dict(
        schema="ball_query_controls_startup_acceptance.v1",
        complete=True,
        accepted_at=now(),
        updates_observed=len(rows),
        rows=[rows[i] for i in (0, 499, 599, 600)],
        receipt=receipt,
        restore=restored,
        optimizer_states=dict(frozen_cnn=cnn, fresh_query=query),
        snapshot_position=10500,
        processes=processes,
        queued_successors=["cnn", "length"],
        held_models=[12, 24],
        initialization_audit=read(ROOT / "initialization-audit.json")["results"],
    )
    write(ROOT / "startup-acceptance.json", evidence)
    store().publish_file(ROOT / "startup-acceptance.json")
    return evidence


def wait():
    libc = ctypes.CDLL(None, use_errno=True)
    descriptor = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    watched = set()
    try:
        while True:
            for path in [ROOT, ROOT / "queries-s1-T32", ROOT / "notifications"]:
                if path.exists() and str(path) not in watched:
                    if (
                        libc.inotify_add_watch(
                            descriptor, os.fsencode(path), 0x2 | 0x8 | 0x80 | 0x100
                        )
                        < 0
                    ):
                        raise OSError(ctypes.get_errno(), "inotify_add_watch")
                    watched.add(str(path))
            if (ROOT / "FAILED.json").exists():
                raise RuntimeError(json.dumps(read(ROOT / "FAILED.json")))
            # Also catch prerequisite failure; it stops the common queue before a new model can start.
            from control import OLD_ROOT

            if (OLD_ROOT / "FAILED.json").exists():
                raise RuntimeError(json.dumps(read(OLD_ROOT / "FAILED.json")))
            evidence = audit()
            if evidence is not None:
                return evidence
            select.select([descriptor], [], [])
            os.read(descriptor, 65536)
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    print(json.dumps(wait()), flush=True)
