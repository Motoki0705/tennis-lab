"""Identify suspicious source frames from the saved baseline, with no model inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.player_association.geometry.footpoints import (
    FootpointConfig,
    ground_footpoints,
)
from src.tasks.player_association.pseudo_annotation.frozen_run import (
    FrozenClip,
    FrozenRun,
)
from src.tasks.player_association.pseudo_annotation.job import read_json
from src.tasks.player_association.pseudo_annotation.sampling import (
    ASSOCIATION,
    CONFIDENT,
    NEAR_REJECTED,
    SINGLE_VIEW,
    STOPPED,
    STRATA,
    TRACKING,
    SamplingConfig,
    near_court,
    reconstruction_suspicions,
    sample_strata,
    tracking_suspicions,
)
from src.utils.checksum import dual_sha256


def clip_frames(frozen: FrozenClip, cfg: SamplingConfig) -> dict[str, Any]:
    clip = frozen.clip
    total = clip.num_frames
    masks = {name: np.zeros(total, dtype=bool) for name in STRATA}
    detections = np.zeros(total, dtype=bool)
    local_cameras = frozen.local_cameras()
    assignments = frozen.assignments()
    view_counts = np.zeros((4, total), dtype=np.int64)
    for camera in clip.camera_ids:
        offsets, _, _ = frozen.detections(camera)
        detections |= np.diff(offsets) > 0
        raw = frozen.raw(camera)
        selection = frozen.selection(camera)
        observed, selected = raw["observed"], selection["selected"]
        if (
            raw["camera_id"] != camera
            or selection["camera_id"] != camera
            or observed.shape != (len(raw["track_ids"]), total)
            or not np.array_equal(raw["track_ids"], selection["raw_track_ids"])
            or selected.shape != observed.shape
            or (selected & ~observed).any()
        ):
            raise ValueError(
                f"Tracking and selection timelines disagree: {clip.clip_id}/{camera}"
            )
        if camera in local_cameras:
            points, valid = ground_footpoints(
                raw["boxes_xyxy"],
                observed,
                local_cameras[camera],
                clip.height,
                FootpointConfig(cfg.footpoint_bottom_border_px),
            )
            near = near_court(points, valid, cfg)
            masks[NEAR_REJECTED] |= (near & ~selected).any(0)
            masks[TRACKING] |= tracking_suspicions(
                observed, points, valid, near, clip.fps, cfg
            )

        if assignments is not None:
            groups = selection["tracks.track_ids"]
            kept = selection["tracks.observed"]
            by_group = assignments.get(camera, {})
            if not set(by_group).issubset(groups.tolist()) or kept.shape != (
                len(groups),
                total,
            ):
                raise ValueError(
                    f"Association and selection tracks disagree: {clip.clip_id}/{camera}"
                )
            ids = np.full(kept.shape, -1, dtype=np.int64)
            for row, group in enumerate(groups.tolist()):
                if group in by_group:
                    ids[row] = by_group[group]
            if (ids < -1).any() or (ids >= len(view_counts)).any():
                raise ValueError(
                    f"Unexpected doubles player ID: {clip.clip_id}/{camera}"
                )
            masks[ASSOCIATION] |= (kept & (ids < 0)).any(0)
            for person in range(len(view_counts)):
                view_counts[person] += (kept & (ids == person)).any(0)

    outcome = frozen.outcome
    stop_reason = None if outcome["status"] == "generated" else outcome["reason"]
    masks[STOPPED][:] = stop_reason in cfg.stopped_reasons
    if assignments is not None:
        skeleton = frozen.read(
            "player_triangulation",
            "player_skeletons",
            1,
            "skeleton.valid",
            "skeleton.reasons",
            "skeleton.reprojection_px",
        )
        single, large, supported = reconstruction_suspicions(
            view_counts,
            skeleton["skeleton.valid"],
            skeleton["skeleton.reasons"],
            skeleton["skeleton.reprojection_px"],
            cfg,
        )
        masks[SINGLE_VIEW] = single
        masks[ASSOCIATION] |= large
        masks[CONFIDENT] = supported

    candidates, sampled = sample_strata(clip.clip_id, masks, detections, cfg)
    return {
        "num_frames": total,
        "status": outcome["status"],
        "stop_reason": stop_reason,
        "calibrated_cameras": sorted(local_cameras),
        "has_side": "court_side" in frozen.store.references,
        "has_association": assignments is not None,
        "has_tentative_association": bool(
            outcome.get("diagnostics", {}).get("association")
        ),
        "candidates": candidates,
        "sampled": sampled,
        "sampled_frames": sorted(
            {frame for frames in sampled.values() for frame in frames}
        ),
    }


def extract(frozen: FrozenRun, config_path: Path) -> dict[str, Any]:
    cfg = SamplingConfig.load(config_path)
    clips = {
        clip_id: clip_frames(frozen.clip(clip_id), cfg)
        for clip_id in sorted(frozen.dataset.clips)
    }
    return {
        "schema_version": 1,
        "dataset_id": frozen.dataset.dataset_id,
        "input_snapshot_sha256": frozen.sha256,
        "config_sha256": dual_sha256(config_path),
        "sampling_config": cfg.model_dump(),
        "clips": clips,
        "summary": {
            "clips": len(clips),
            "source_frames": sum(clip["num_frames"] for clip in clips.values()),
            "candidates_by_stratum": {
                name: sum(len(clip["candidates"][name]) for clip in clips.values())
                for name in STRATA
            },
            "sampled_by_stratum": {
                name: sum(len(clip["sampled"][name]) for clip in clips.values())
                for name in STRATA
            },
            "candidate_frames": sum(
                len({f for frames in clip["candidates"].values() for f in frames})
                for clip in clips.values()
            ),
            "sampled_frames": sum(
                len(clip["sampled_frames"]) for clip in clips.values()
            ),
            "sampled_clips": sum(
                bool(clip["sampled_frames"]) for clip in clips.values()
            ),
        },
    }


def load_plan(path: Path, frozen: FrozenRun) -> dict[str, Any]:
    plan: dict[str, Any] = read_json(path)
    if (
        plan["schema_version"] != 1
        or plan["input_snapshot_sha256"] != frozen.sha256
        or plan["dataset_id"] != frozen.dataset.dataset_id
    ):
        raise ValueError("Selection plan does not match the frozen input")
    return plan
