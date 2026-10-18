"""Pose inference on immutable, approved, real player observations only."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_detection.data.store import SHARDS_DIR, shard_name
from src.tasks.person_tracking.features import PoseModel, infer_poses

from .generation import _cached_chunk, _save_chunk
from .reviews import load_selection
from .selection import load_campaign
from .storage import (
    clip_root,
    digest,
    lock,
    read_json,
    verify_record,
    write_json,
    write_npz,
)


def pose_model(config: dict[str, Any]) -> Any:
    from src.submodules.configuration import ViTPoseHeadConfig
    from src.submodules.models.vitpose.pose2d import ViTPosePose2D

    return ViTPosePose2D(
        Path(config["assets"]["vitpose"]),
        device="cuda",
        flip_test=True,
        batch_size=4,
        precision="float32",
        head_config=ViTPoseHeadConfig(
            in_channels=1280,
            out_channels=17,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            final_conv_kernel=1,
            num_conv_layers=0,
            num_conv_kernels=(),
        ),
    )


def infer_selected_frame(
    frame: int,
    image: np.ndarray,
    selection: dict[str, Any],
    model: PoseModel,
) -> np.ndarray:
    """No model call is made for missing, duplicate or excluded observations."""
    observed = selection["observed"][frame]
    rows = selection["detection_rows"][frame, observed]
    if (rows < 0).any() or len(np.unique(rows)) != len(rows):
        raise ValueError("Pose crops must name unique real detection rows")
    result: np.ndarray = np.zeros((len(observed), 17, 3), np.float32)
    if observed.any():
        result[observed] = infer_poses(
            frame, image, selection["boxes_xyxy"][frame, observed], model, rows=rows
        )
    return result


def generate_selected_pose(campaign: Path, index: int) -> None:
    if (
        not os.environ.get("TENNIS_RUN_ID")
        or os.environ.get("TENNIS_GPU_RESOURCE") != "all"
    ):
        raise RuntimeError(
            "Selected-player pose must run in an exclusive shared training-queue job"
        )
    config, _, store = load_campaign(campaign)
    root = clip_root(campaign, index)
    with lock(root / "pose.lock", blocking=False):
        selection = load_selection(campaign, index)
        if (root / "pose.json").exists():
            verify_record(root / "pose.json")
            return
        shard = store.directory / SHARDS_DIR / shard_name(index)
        if digest(shard) != read_json(root / "input.json")["shard_sha256"]:
            raise ValueError("Reviewed JPEG bytes changed before pose inference")
        if (
            digest(Path(config["assets"]["vitpose"]))
            != config["asset_hashes"]["vitpose"]
        ):
            raise ValueError("Planned ViTPose weight changed")
        import torch

        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(
            float(config["cuda_memory_fraction"])
        )
        clip = store.clips[index]
        binding = digest(root / "review.json")
        output = np.zeros((*selection["observed"].shape, 17, 3), np.float32)
        model = None
        total_seconds = 0.0
        try:
            for start in range(0, clip.frame_count, config["chunk_frames"]):
                stop = min(start + config["chunk_frames"], clip.frame_count)
                path = root / "pose_chunks" / f"{start:06d}-{stop:06d}.npz"
                if _cached_chunk(path):
                    with np.load(path, allow_pickle=False) as data:
                        if str(data["review_sha256"]) != binding:
                            raise ValueError("Cached poses belong to another selection")
                        output[start:stop] = data["keypoints"]
                        total_seconds += read_json(path.with_suffix(".json"))["seconds"]
                    continue
                started = time.monotonic()
                for frame in range(start, stop):
                    write_json(
                        root / "pose_progress.json",
                        {
                            "stage": "selected_player_pose",
                            "frame": frame,
                            "frames": clip.frame_count,
                        },
                    )
                    if not selection["observed"][frame].any():
                        continue
                    if model is None:
                        model = pose_model(config)
                    output[frame] = infer_selected_frame(
                        frame,
                        store.read_bgr(store.row_of(clip, frame)),
                        selection,
                        model,
                    )
                seconds = time.monotonic() - started
                total_seconds += seconds
                _save_chunk(
                    path,
                    keypoints=output[start:stop],
                    review_sha256=np.asarray(binding),
                    seconds=seconds,
                )
        finally:
            if model is not None:
                model.unload()
        if not np.isfinite(output).all() or output[~selection["observed"]].any():
            raise ValueError("Selected pose output lost the missing-observation mask")
        write_npz(root / "poses.npz", **selection, keypoints=output)
        write_json(
            root / "pose.json",
            {
                "status": "complete",
                "stage": "selected_player_pose",
                "generation_mode": config["generation_mode"],
                "pose_crops": int(selection["observed"].sum()),
                "seconds": total_seconds,
                "vitpose_sha256": config["asset_hashes"]["vitpose"],
                "synthetic_poses_used": False,
                "files": {
                    name: digest(root / name)
                    for name in ("poses.npz", "review.json", "selection.npz")
                },
            },
        )
