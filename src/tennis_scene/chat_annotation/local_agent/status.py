"""Compact campaign status: counts, running workers (elapsed, context), quota, recent events."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from .campaign_state import read_control, read_state
from .configuration import paths
from .dispatcher import rollout_rate_limits, scan_events


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    state = read_state()
    control = read_control()
    counts = Counter(t["status"] for t in state["tasks"].values())
    now = datetime.now(UTC)
    running = []
    for task_id, task in state["tasks"].items():
        if task["status"] != "running":
            continue
        record = task["attempts"][-1]
        attempt_dir = Path(record["dir"])
        events = scan_events(attempt_dir / "events.jsonl")
        limits, fraction = rollout_rate_limits(events["thread_id"])
        annotation = next(attempt_dir.glob("annotation_*.json"), None)
        reviewed = None
        if annotation:
            data = json.loads(annotation.read_text(encoding="utf-8"))
            reviewed = (
                f"{sum(r['reviewed'] for r in data['frames'])}/{len(data['frames'])}"
            )
        elapsed = (
            now - datetime.fromisoformat(record["launched_at"])
        ).total_seconds() / 60
        running.append(
            {
                "task": task_id,
                "attempt": record["n"],
                "min": round(elapsed, 1),
                "ctx": round(fraction, 3) if fraction else None,
                "reviewed": reviewed,
                "weekly_used_pct": ((limits or {}).get("primary") or {}).get(
                    "used_percent"
                ),
            }
        )
    done = [
        r
        for t in state["tasks"].values()
        for r in t["attempts"]
        if r.get("kind") in ("done", "continuation")
    ]
    minutes = [
        (
            datetime.fromisoformat(r["ended_at"])
            - datetime.fromisoformat(r["launched_at"])
        ).total_seconds()
        / 60
        for r in done
        if r.get("ended_at")
    ]
    tail = (
        paths().events.read_text(encoding="utf-8").splitlines()[-12:]
        if paths().events.exists()
        else []
    )
    print(
        json.dumps(
            {
                "mode": control["mode"],
                "max_parallel": control["max_parallel"],
                "model": control["model"],
                "effort": control["effort"],
                "counts": dict(counts),
                "running": running,
                "finished_attempts": len(done),
                "mean_minutes_per_finished_attempt": round(
                    sum(minutes) / len(minutes), 1
                )
                if minutes
                else None,
                "quota": state["quota"],
                "recent_events": tail,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
