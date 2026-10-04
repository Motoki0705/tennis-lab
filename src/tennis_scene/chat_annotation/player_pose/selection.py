from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.tasks.ball_detection.data.store import (
    POINT_KIND_CODES,
    SHARDS_DIR,
    BallFrameStore,
    shard_name,
)

from .storage import digest, read_json, write_json


def presence_counts(store: BallFrameStore, threshold: float) -> list[dict[str, Any]]:
    """One count per frame. Unresolved and estimated balls protect low visibility."""
    if not 0 <= threshold <= 1:
        raise ValueError("Presence threshold must lie in [0,1]")
    counts = store.frames["inst_count"]
    instance_frames: NDArray[np.int64] = np.repeat(np.arange(len(store)), counts)
    present: NDArray[np.bool_] = np.zeros(len(store), dtype=bool)
    unknown: NDArray[np.bool_] = np.zeros(len(store), dtype=bool)
    observed: NDArray[np.bool_] = np.zeros(len(store), dtype=bool)
    kinds = store.instances["point_kind"]
    present[instance_frames[kinds != POINT_KIND_CODES["out_of_frame"]]] = True
    unknown[instance_frames[kinds == POINT_KIND_CODES["unresolved"]]] = True
    observed[instance_frames[kinds == POINT_KIND_CODES["observed"]]] = True
    result = []
    for clip in store.clips:
        rows = store.clip_rows(clip)
        target = store.frames["is_target"][rows]
        reviewed = target & store.frames["annotated"][rows]
        denominator = int(reviewed.sum())
        numerator = int((present[rows] & reviewed).sum())
        ratio = numerator / denominator if denominator else None
        # Incomplete review is explicit; it never silently becomes a low-presence negative.
        if int(target.sum()) != denominator:
            selected, reason = True, "unreviewed_target_frames_retained"
        elif denominator == 0:
            selected, reason = False, "no_target_frames"
        else:
            selected = Fraction(numerator, denominator) > Fraction(str(threshold))
            reason = (
                "presence_above_threshold"
                if selected
                else "presence_at_or_below_threshold"
            )
        shard = store.directory / SHARDS_DIR / shard_name(clip.index)
        stat = shard.stat()
        result.append(
            {
                "index": clip.index,
                "clip_id": clip.clip_id,
                "source": clip.source,
                "split": clip.split,
                "camera_id": clip.camera_id,
                "frame_count": clip.frame_count,
                "width": clip.width,
                "height": clip.height,
                "fps": clip.fps,
                "target_frames": int(target.sum()),
                "reviewed_target_frames": denominator,
                "present_frames": numerator,
                "presence_rate": ratio,
                "observed_frames": int((observed[rows] & reviewed).sum()),
                "unresolved_frames": int((unknown[rows] & reviewed).sum()),
                "selected": selected,
                "reason": reason,
                "shard_stat": [stat.st_size, stat.st_mtime_ns],
            }
        )
    return result


def initialize(campaign: Path, config: dict[str, Any]) -> dict[str, Any]:
    from .expansion import plan_expansion

    config = dict(config)
    config.setdefault("generation_mode", "review_then_pose.v1")
    if config["generation_mode"] != "review_then_pose.v1":
        raise ValueError(
            "New campaigns require explicit review_then_pose.v1 generation"
        )
    if campaign.exists():
        raise FileExistsError(campaign)
    dataset = Path(config["dataset"])
    store_path = Path(config["store"])
    if (
        not dataset.is_absolute()
        or not campaign.is_absolute()
        or not store_path.is_absolute()
    ):
        raise ValueError("Campaign, dataset and store paths must be absolute")
    if dataset.resolve().is_relative_to(
        store_path.resolve()
    ) or campaign.resolve().is_relative_to(store_path.resolve()):
        raise ValueError("Pose outputs must not overwrite the ball store")
    if dataset.exists():
        raise FileExistsError(dataset)
    store = BallFrameStore(Path(config["store"]))
    clips = presence_counts(store, float(config["presence_threshold"]))
    expansion = plan_expansion(config, clips, store)
    config["store_hashes"] = {
        name: digest(store.directory / name) for name in ("metadata.json", "index.npz")
    }
    config["asset_hashes"] = {
        name: digest(Path(value))
        for name, value in config["assets"].items()
        if name != "dino_repository"
    }
    config["asset_stats"] = {
        name: [
            Path(config["assets"][name]).stat().st_size,
            Path(config["assets"][name]).stat().st_mtime_ns,
        ]
        for name in config["asset_hashes"]
    }
    project = Path(config["project_root"])
    folders = [
        project / "src/tennis_scene/chat_annotation/player_pose",
        project / "src/tasks/person_tracking",
        project / "src/tasks/player_association",
        project / "src/submodules",
    ]
    config["code_hashes"] = {
        str(p): digest(p) for folder in folders for p in sorted(folder.rglob("*.py"))
    }
    instruction = project / "src/tennis_scene/chat_annotation/player_pose/WORKER.md"
    if instruction.exists():
        config["code_hashes"][str(instruction)] = digest(instruction)
    plan = {
        "schema": "ball_store_player_pose_plan.v2",
        "generation_mode": config["generation_mode"],
        "expansion": expansion,
        "clips": clips,
        "selected_clips": sum(c["selected"] for c in clips),
        "selected_frames": sum(c["frame_count"] for c in clips if c["selected"]),
        "presence_definition": "reviewed target frames with any observed/interpolated/occlusion_estimated/unresolved ball; out_of_frame excluded",
        "court_policy": "disabled; all human tracks reviewed by GPT",
        "threshold": config["presence_threshold"],
    }
    campaign.mkdir(parents=True)
    write_json(campaign / "config.json", config)
    write_json(campaign / "plan.json", plan)
    write_json(
        campaign / "identity.json",
        {
            "config_sha256": digest(campaign / "config.json"),
            "plan_sha256": digest(campaign / "plan.json"),
        },
    )
    write_json(campaign / "generation_status.json", {"status": "pending"})
    write_json(campaign / "review_status.json", {"status": "pending"})
    write_json(campaign / "pose_status.json", {"status": "pending"})
    write_json(
        dataset / "manifest.json",
        {
            "schema": "ball_detection_player_poses.v1",
            "coordinate_system": "stored_jpeg_pixels",
            "ball_store": {
                "directory": config["store"],
                "hashes": config["store_hashes"],
            },
            "campaign": str(campaign),
            "court_policy": "disabled",
            "status": "partial" if plan["selected_clips"] else "complete",
            "clips": [
                {
                    **c,
                    "pose_status": "reuse_pending"
                    if c["action"] == "reuse"
                    else "tracking_pending"
                    if c["selected"]
                    else "skipped",
                }
                for c in clips
            ],
        },
    )
    return plan


def load_campaign(
    campaign: Path,
) -> tuple[dict[str, Any], dict[str, Any], BallFrameStore]:
    config = read_json(campaign / "config.json")
    for name in ("config", "plan"):
        if (
            digest(campaign / f"{name}.json")
            != read_json(campaign / "identity.json")[f"{name}_sha256"]
        ):
            raise ValueError(f"Campaign {name} changed")
    for filename, expected in config["code_hashes"].items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"Campaign code/instructions changed: {filename}")
    for name, expected in config["store_hashes"].items():
        if digest(Path(config["store"]) / name) != expected:
            raise ValueError(f"Ball store changed: {name}")
    return (
        config,
        read_json(campaign / "plan.json"),
        BallFrameStore(Path(config["store"])),
    )
