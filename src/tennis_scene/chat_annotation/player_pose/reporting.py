"""Durable campaign counts that distinguish review approval from publication."""

from collections import Counter
from pathlib import Path
from typing import Any

from .storage import clip_root, read_json, write_json


def campaign_metrics(
    campaign: Path, config: dict[str, Any], plan: dict[str, Any]
) -> dict[str, Any]:
    states: Counter[str] = Counter()
    added: Counter[str] = Counter()
    detections = crops = 0
    seconds = {"tracking": 0.0, "pose": 0.0, "review": 0.0}
    manifest = read_json(Path(config["dataset"]) / "manifest.json")
    for entry, published in zip(plan["clips"], manifest["clips"], strict=True):
        root = clip_root(campaign, entry["index"])
        state = "tracking_pending"
        if not entry["selected"]:
            state = "skipped"
        elif published["pose_status"] == "approved":
            state = "reused" if entry["action"] == "reuse" else "published"
        elif entry["action"] == "reuse":
            state = "reuse_pending"
        elif (root / "review.json").exists():
            state = (
                "pose_failed"
                if published["pose_status"] == "pose_failed"
                else "pose_pending"
            )
        elif (root / "review_status.json").exists():
            state = read_json(root / "review_status.json")["status"]
        elif (root / "generation.json").exists():
            state = "review_pending"
        elif (root / "failure.json").exists():
            state = "tracking_failed"
        states[state] += 1
        if entry["expansion"] == "added":
            added[state] += 1
        for filename, stage, count in (
            ("generation.json", "tracking", "all_detections"),
            ("pose.json", "pose", "pose_crops"),
        ):
            path = root / filename
            if path.exists():
                record = read_json(path)
                seconds[stage] += record.get("seconds", 0.0)
                if stage == "tracking":
                    detections += record.get(count, 0)
                else:
                    crops += record.get(count, 0)
        for path in (root / "review").glob("attempt-*/exit.json"):
            record = read_json(path)
            seconds["review"] += record.get("seconds", 0.0)
    return {
        "clips": len(plan["clips"]),
        "expansion": plan["expansion"],
        "states": dict(states),
        "added_clip_states": dict(added),
        "new_all_detections": detections,
        "new_pose_crops": crops,
        "seconds": seconds,
    }


def save_metrics(campaign: Path, config: dict[str, Any], plan: dict[str, Any]) -> None:
    write_json(campaign / "metrics.json", campaign_metrics(campaign, config, plan))
