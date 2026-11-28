"""Check the measured units, identity matching and aggregation on CPU fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from parity import Scene, compare_scenes
from report import (
    GIB,
    MANAGED,
    aggregate_timing,
    appearance_wall_seconds,
    forward_batches,
)


def toy_scene(xs: tuple[float, float]) -> Scene:
    pose = np.zeros((2, 1, 30, 2, 2), np.float32)
    pose[..., 0] = np.array(xs)[:, None, None, None]
    return Scene(
        np.arange(2, dtype=np.int32),
        (1000, 500),
        pose,
        np.ones((2, 1, 30, 2), np.float32),
        np.zeros((2, 30, 2, 3), np.float32),
        np.zeros((2, 30, 2), bool),
        np.zeros((2, 30), bool),
    )


def test_compare_uses_pose_to_match_swapped_ids_and_ignores_invalid_points() -> None:
    reference = toy_scene((0.1, 0.5))
    candidate = toy_scene((0.505, 0.1))
    reference.joint_mask[0, 0] = True
    reference.joint_mask[1, 0, 0] = reference.joint_mask[0, 1, 0] = True
    candidate.joint_mask[1, 0] = True
    candidate.joint_mask[0, 0, 0] = candidate.joint_mask[1, 2, 0] = True
    candidate.joints[1, 0, 1, 0] = 1
    candidate.joints[0, 0, 0, 1] = 4
    candidate.joints[0, 1, 0, 1] = np.nan  # unused by the comparison
    reference.smpl_mask[0, 0] = reference.smpl_mask[1, 1] = True
    candidate.smpl_mask[1, 0] = candidate.smpl_mask[0, 2] = True

    row, distances = compare_scenes(reference, candidate)
    assert distances.tolist() == [0.0, 1.0, 4.0]
    assert row["matched_player_ids"] == 2 and row["same_numeric_id"] == 0
    assert [
        (match["baseline_id"], match["condition_id"])
        for match in row["player_id_mapping"]
    ] == [(0, 1), (1, 0)]
    assert row["joint_position_l2"] == {
        "common_valid_joint_points": 3,
        "median_m": 1.0,
        "p95_m": pytest.approx(3.7),
        "max_m": 4.0,
    }
    for key, both, neither in (
        ("joint_point_mask", 3, 115),
        ("joint_frame_mask", 2, 56),
        ("smpl_mask", 1, 57),
    ):
        mask = row[key]
        assert (
            mask["both_valid"],
            mask["reference_only"],
            mask["candidate_only"],
            mask["both_invalid"],
        ) == (both, 1, 1, neither)
        assert mask["agreement_fraction"] == pytest.approx(
            (both + neither) / (both + neither + 2)
        )


def test_no_supported_id_pair_has_no_distance_or_mask_agreement() -> None:
    reference = toy_scene((0.1, 0.5))
    candidate = toy_scene((0.1, 0.5))
    candidate.pose_confidence[:, :, 29] = 0  # 29 shared frames, below the rule
    row, distances = compare_scenes(reference, candidate)
    assert row["matched_player_ids"] == 0
    assert row["joint_point_mask"]["agreement_fraction"] is None
    assert row["joint_position_l2"]["p95_m"] is None
    assert not len(distances)


def test_vitpose_forward_fill_counts_crops_before_flip() -> None:
    def counts(items: int, succeeded: int) -> dict[str, int]:
        return {"items": items, "succeeded": succeeded, "failed": 0}

    progress = {
        "batch_shapes": {
            "DINO": {"[1, 3, 750, 1333]": counts(1, 2)},
            "ViTPose": {
                "[16, 3, 256, 192]": counts(8, 3),
                "[6, 3, 256, 192]": counts(3, 1),
            },
            "CLIP-ReID": {"[8, 3, 256, 128]": counts(8, 2)},
            "HMR2": {"[16, 3, 256, 256]": counts(16, 1)},
            "GVHMR-network": {"[2, 117, 1024]": counts(2, 1)},
        }
    }
    result = forward_batches(progress)
    assert result["vitpose"] == {
        "batches": 4,
        "items": 27,
        "total_capacity": 32,
        "fill_fraction": 27 / 32,
    }
    assert result["gvhmr"]["fill_fraction"] == 0.5


def test_model_wall_unions_processing_and_lifecycle_intervals(tmp_path: Path) -> None:
    spans = [
        ("CLIP-ReID", 0, 10),
        ("appearance", 6, 12),
        ("dino", 10, 20),
        ("clip", 15, 17),
    ]
    events = [
        {
            "id": index,
            "start_ns": start * 10**9,
            "end_ns": end * 10**9,
            "attributes": {"model": model},
        }
        for index, (model, start, end) in enumerate(spans)
    ]
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"events": events}, separators=(",", ":")) + "\n")
    assert appearance_wall_seconds(trace, len('{"events":[')) == 14


def test_serial_aggregate_sums_time_and_loads_but_takes_maximum_memory() -> None:
    def process(
        cli: int, observed: int, active: float, mean: int, vram: int, rss: int
    ) -> dict:
        return {
            "cli_wall_seconds": cli,
            "recorder_wall_seconds": cli - 2,
            "video_pass": {
                "wall_seconds": observed,
                "gpu_observed_wall_seconds": observed,
                "gpu_active_fraction": active,
                "gpu_mean_utilization_percent": mean,
            },
            "resources": {
                "nvml": {"cuda:0": {"used_peak_bytes": vram * GIB}},
                "ram_process_lifetime_peak_bytes": rss * GIB,
            },
            "lifecycle": {
                name: {
                    "load_count": 1,
                    "unload_count": 1,
                    "load_seconds": 2,
                    "unload_seconds": 0.5,
                }
                for name in MANAGED
            },
            "stage_wall_seconds": {"inference": 3},
            "model_wall_seconds": {"dino": 2},
            "batch_dispatches": {},
            "forward_batches": {},
        }

    result = aggregate_timing(
        [process(20, 8, 0.25, 20, 2, 2), process(40, 16, 0.5, 40, 1, 4)]
    )
    assert (
        result["cli_wall_seconds"],
        result["recorder_wall_seconds"],
        result["outside_recorder_wall_seconds"],
    ) == (60, 56, 4)
    assert (
        result["peak_vram_mib"],
        result["peak_rss_gib"],
        result["managed_model_loads"],
    ) == (2048, 4, 12)
    assert result["gpu_active_fraction"] == pytest.approx(5 / 12)
    assert result["gpu_mean_utilization_percent"] == pytest.approx(100 / 3)
    assert result["model_wall_seconds"]["dino"] == 4
