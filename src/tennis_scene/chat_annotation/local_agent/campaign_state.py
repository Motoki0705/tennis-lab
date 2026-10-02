"""Campaign state (tasks, attempts, quota pause) shared by dispatcher, intake and status.

state.json is the single registry. Every read-modify-write holds an exclusive flock on
state.lock, so the dispatcher loop and orchestrator commands (intake/revise) never race.
"""

from __future__ import annotations

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .common import atomic_write_json, load_manifest, manifest_index, utc_now
from .configuration import ControlConfig, json_object, paths

# Alternate professional broadcasts and court-level / amateur channels so that an early
# stop still leaves a diverse set. Unknown channels are appended alphabetically.
CHANNEL_ORDER = [
    "rolandgarros",
    "tenniswithshane",
    "usopen",
    "amateurtennis",
    "tennistv",
    "marksansait",
    "wimbledon",
    "tennistroll",
    "australianopen",
    "winstondu",
    "karuesell",
]

# Statuses: pending -> running -> review | continue | failed ; review -> adopted | held ;
# continue -> running ; held -> continue (with parent review notes) ; skipped (external).
ELIGIBLE = ("continue", "pending")


@contextmanager
def locked_state() -> Iterator[dict[str, Any]]:
    (paths().campaign_dir / "state.lock").touch(exist_ok=True)
    with (paths().campaign_dir / "state.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        state = json_object(paths().state)
        yield state
        state["updated_at"] = utc_now()
        atomic_write_json(paths().state, state)


def read_state() -> dict[str, Any]:
    return json_object(paths().state)


def read_control() -> dict[str, Any]:
    result: dict[str, Any] = ControlConfig.model_validate(
        json_object(paths().control)
    ).model_dump(mode="json")
    return result


def log_event(kind: str, message: str) -> None:
    paths().logs.mkdir(parents=True, exist_ok=True)
    with paths().events.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {kind} {message}\n")


def channel_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    selection_path = paths().annotation_root / "selection.json"
    if not selection_path.exists():
        return mapping
    selection = json_object(selection_path)
    for channel in selection["channels"]:
        for video in channel["videos"]:
            mapping[video["id"]] = channel["slug"]
    return mapping


def clip_order(clip_ids: list[str], manifests: dict[str, Path]) -> list[str]:
    """Channel round-robin; inside a channel, source round-robin; inside a source, time order."""
    channels = channel_map()
    if not clip_ids:
        return []
    by_channel: dict[str, dict[str, list[tuple[int, str]]]] = {}
    for clip_id in clip_ids:
        manifest = load_manifest(manifests[clip_id])
        source = manifest.source.source_id
        channel = channels.get(source, "zz_unknown")
        by_channel.setdefault(channel, {}).setdefault(source, []).append(
            (manifest.target_range.start, clip_id)
        )
    ordered_channels = [c for c in CHANNEL_ORDER if c in by_channel] + sorted(
        c for c in by_channel if c not in CHANNEL_ORDER
    )
    per_channel: dict[str, list[str]] = {}
    for channel in ordered_channels:
        sources = by_channel[channel]
        queues = {s: [c for _, c in sorted(v)] for s, v in sorted(sources.items())}
        sequence: list[str] = []
        depth = max(len(q) for q in queues.values())
        for k in range(depth):
            for source in sorted(queues):
                if k < len(queues[source]):
                    sequence.append(queues[source][k])
        per_channel[channel] = sequence
    order: list[str] = []
    depth = max(len(v) for v in per_channel.values())
    for k in range(depth):
        for channel in ordered_channels:
            if k < len(per_channel[channel]):
                order.append(per_channel[channel][k])
    return order


def processed_path(target: str, clip_id: str) -> Path:
    return paths().annotated / "processed" / target / f"{clip_id}.json"


def initial_state(targets: list[str]) -> dict[str, Any]:
    """Phase 1 universe: every (clip, campaign target) without a processed annotation."""
    manifests = manifest_index()
    missing_clips = [
        clip_id
        for clip_id in manifests
        if any(not processed_path(t, clip_id).exists() for t in targets)
    ]
    order = clip_order(missing_clips, manifests)
    tasks: dict[str, Any] = {}
    first_target: dict[str, str] = {}
    for position, clip_id in enumerate(order):
        first_target[clip_id] = targets[position % len(targets)]
        for target in targets:
            if processed_path(target, clip_id).exists():
                continue
            tasks[f"{clip_id}__{target}"] = {
                "clip_id": clip_id,
                "target": target,
                "manifest": str(manifests[clip_id]),
                "status": "pending",
                "attempts": [],
                "phase": 1,
            }
    return {
        "schema": "codex_cli_campaign_state.v1",
        "targets": list(targets),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "order": order,
        "first_target": first_target,
        "tasks": tasks,
        "quota": {"paused_until": None, "reason": None, "last_rate_limits": None},
    }


def next_candidates(state: dict[str, Any], control: dict[str, Any]) -> list[str]:
    running_clips = {
        t["clip_id"] for t in state["tasks"].values() if t["status"] == "running"
    }
    position = {clip_id: i for i, clip_id in enumerate(state["order"])}
    pilot = set(control.get("pilot_tasks") or [])
    candidates = []
    for task_id, task in state["tasks"].items():
        if task["status"] not in ELIGIBLE or task["clip_id"] in running_clips:
            continue
        if control["mode"] == "pilot" and task_id not in pilot:
            continue
        first = state["first_target"].get(task["clip_id"], "ball")
        phase = task.get("phase", 1)
        key = (
            0 if task["status"] == "continue" else 1,
            phase,  # phase 2 (re-annotation) only after every phase-1 task
            task["rank"]
            if phase == 2
            else position.get(task["clip_id"], 10**9),  # phase 2: worst first
            0 if task["target"] == first else 1,
        )
        candidates.append((key, task_id))
    return [task_id for _, task_id in sorted(candidates)]
