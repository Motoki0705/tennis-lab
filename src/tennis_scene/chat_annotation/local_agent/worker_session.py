"""Session occupancy, completion and continuation handoffs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .common import (
    atomic_write_json,
    utc_now,
)
from .configuration import paths
from .worker_context import Ctx, dump, summarize


def rollout_path(ctx: Ctx) -> Path | None:
    thread = os.environ.get("CODEX_THREAD_ID")
    events = ctx.dir / "events.jsonl"
    if not thread and events.exists():
        with events.open(encoding="utf-8") as handle:
            first = handle.readline()
        try:
            thread = json.loads(first).get("thread_id")
        except json.JSONDecodeError:
            thread = None
    if not thread:
        return None
    home = paths().codex_home
    for day in sorted((home / "sessions").glob("*/*/*"), reverse=True)[:3]:
        hits = list(day.glob(f"*{thread}*.jsonl"))
        if hits:
            return hits[0]
    return None


def context_fraction(ctx: Ctx) -> dict[str, Any]:
    path = rollout_path(ctx)
    if path is None:
        return {"available": False}
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 400_000))
        tail = handle.read().decode("utf-8", errors="ignore").splitlines()
    for line in reversed(tail):
        if '"token_count"' not in line:
            continue
        try:
            payload = json.loads(line).get("payload", {})
        except json.JSONDecodeError:
            continue
        info = payload.get("info") or {}
        last = info.get("last_token_usage") or {}
        window = info.get("model_context_window")
        if last and window:
            return {
                "available": True,
                "fraction": round(last["total_tokens"] / window, 4),
                "last_request_tokens": last["total_tokens"],
                "window": window,
                "rate_limits": payload.get("rate_limits"),
            }
    return {"available": False, "rollout": str(path)}


def cmd_context(ctx: Ctx, args: argparse.Namespace) -> int:
    value = context_fraction(ctx)
    value["stop_new_visual_work_at"] = ctx.task["context_stop_fraction"]
    dump(value)
    return 0


def cmd_finish(ctx: Ctx, args: argparse.Namespace) -> int:
    annotation = ctx.annotation()
    summary = summarize(annotation, ctx.manifest)
    problems = []
    if summary["error_count"]:
        problems.append("validation errors remain")
    all_reviewed = summary["reviewed"] == summary["frames"]
    if args.outcome == "completed" and summary["validation_status"] != "completed":
        problems.append(
            "outcome completed requires validation_status completed (no notes/issues/unresolved)"
        )
    if args.outcome == "partial" and not all_reviewed:
        problems.append(
            "outcome partial requires every frame reviewed; use needs_continuation"
        )
    if (
        args.outcome == "needs_continuation"
        and summary["validation_status"] == "completed"
    ):
        problems.append("annotation is already complete; no continuation is necessary")
    if args.outcome in ("completed", "partial") and annotation.status != args.outcome:
        problems.append(
            f"annotation status field is {annotation.status!r}; set it to {args.outcome!r} with apply"
        )
    notes = ctx.dir / "NOTES.md"
    if not notes.exists() or notes.stat().st_size < 40:
        problems.append("write a short NOTES.md handoff first")
    if problems:
        dump({"finished": False, "problems": problems, **summary})
        return 1
    started = ctx.task.get("launched_at")
    result = {
        "schema": "local_annotation_result.v1",
        "task_id": ctx.task["task_id"],
        "attempt": ctx.task["attempt"],
        "clip_id": ctx.task["clip_id"],
        "target": ctx.target,
        "outcome": args.outcome,
        "summary": args.summary,
        "annotation": str(ctx.annotation_path),
        "annotation_sha256": hashlib.sha256(
            ctx.annotation_path.read_bytes()
        ).hexdigest(),
        "validation": summary,
        "context": context_fraction(ctx),
        "launched_at": started,
        "finished_at": utc_now(),
    }
    atomic_write_json(ctx.dir / "result.json", result)
    print(
        f"OUTCOME: {args.outcome}\nRESULT: {ctx.dir / 'result.json'}\nSUMMARY: {args.summary}"
    )
    return 0
