"""Physical-time and support-boundary checks for motion diagnostics."""

import json

import numpy as np
import pytest

from src.tasks.slcs.evaluation.motion import summarize_motion
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def _arrays() -> dict[str, np.ndarray]:
    time = np.arange(7)[None, :] / np.array([10, 20])[:, None]
    xyz = np.stack((3 * time, 2 * time ** 2, time ** 3), axis=-1)
    return {
        "video_ids": np.array(["v", "v"]), "clip_ids": np.array(["v/clip1", "v/clip2"]),
        "camera_ids": np.array(["c1", "c2"]), "frame_idx": np.tile(np.arange(7), (2, 1)),
        "padding_mask": np.zeros((2, 7), dtype=bool), "ball_mask": np.ones((2, 7), dtype=bool),
        "player_mask": np.ones((2, 2, 7), dtype=bool),
        "pred_ball_position": xyz.copy(), "target_ball_position": xyz.copy(),
        "pred_player_position": np.repeat(xyz[:, None], 2, axis=1),
        "target_player_position": np.repeat(xyz[:, None], 2, axis=1),
    }


FPS = {("v", "v/clip1"): 10.0, ("v", "v/clip2"): 20.0}


def test_mixed_fps_and_normalization() -> None:
    arrays = _arrays()
    # Polynomial third derivative is exactly 6, independent of sampling rate.
    report = summarize_motion(arrays, FPS, position_representation="meters")
    ball = report["rows"][0]["entities"]["ball"]
    assert ball["jerk"]["target_norm"]["mean"] == pytest.approx(6)
    assert ball["jerk"]["target_norm"]["count"] == 8
    assert ball["acceleration"]["error_norm"]["max"] == 0
    for key in arrays:
        if key.endswith("position"):
            arrays[key] /= np.array(COURT_COORD_SCALE_XYZ)
    normalized = summarize_motion(arrays, FPS, position_representation="normalized_court")
    assert normalized["rows"][0]["entities"]["ball"]["jerk"]["target_norm"]["mean"] == pytest.approx(6)
    json.dumps(normalized, allow_nan=False)


def test_constant_velocity_and_acceleration_error() -> None:
    arrays = _arrays()
    time = np.arange(7)[None] / np.array([10, 20])[:, None]
    arrays["target_ball_position"][:] = 0
    arrays["target_ball_position"][..., 0] = 3 * time
    arrays["pred_ball_position"] = arrays["target_ball_position"].copy()
    arrays["pred_ball_position"][..., 1] = 2 * time ** 2
    ball = summarize_motion(arrays, FPS, position_representation="meters")["rows"][0]["entities"]["ball"]
    assert ball["velocity"]["target_norm"]["mean"] == pytest.approx(3)
    assert ball["acceleration"]["error_norm"]["mean"] == pytest.approx(4)
    assert ball["jerk"]["pred_norm"]["max"] == pytest.approx(0, abs=1e-10)


def test_mask_gaps_padding_and_frame_jumps() -> None:
    arrays = _arrays()
    arrays["ball_mask"][0, 2] = False  # remaining runs: length 2 and length 4
    arrays["padding_mask"][0, 6] = True  # remaining length 4 becomes 3
    arrays["frame_idx"][1, 3:] += 10  # runs of length 3 and 4
    report = summarize_motion(arrays, FPS, position_representation="meters")
    ball = report["rows"][0]["entities"]["ball"]
    assert ball["velocity"]["pred_norm"]["count"] == 8
    assert ball["acceleration"]["pred_norm"]["count"] == 4
    assert ball["jerk"]["pred_norm"]["count"] == 1
    assert report["rows"][0]["entities"]["player_slot_0"]["velocity"]["pred_norm"]["count"] == 10


def test_stationary_and_empty_support() -> None:
    arrays = _arrays()
    arrays["target_ball_position"][:] = 0
    arrays["player_mask"][:] = False
    report = summarize_motion(arrays, FPS, position_representation="meters")
    entities = report["rows"][0]["entities"]
    assert entities["ball"]["position_variation"]["std_norm_ratio"] is None
    assert entities["player_slot_0"]["jerk"]["error_norm"] == {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "10"])
def test_bad_fps(bad: object) -> None:
    with pytest.raises(ValueError, match="FPS"):
        summarize_motion(_arrays(), {**FPS, ("v", "v/clip1"): bad}, position_representation="meters")  # type: ignore[dict-item]


def test_exact_fps_mapping() -> None:
    for mapping in ({}, {**FPS, ("v", "extra"): 30}):
        with pytest.raises(ValueError, match="exactly"):
            summarize_motion(_arrays(), mapping, position_representation="meters")


@pytest.mark.parametrize("key,value", [("frame_idx", np.zeros((2, 7))), ("ball_mask", np.ones((2, 7))),
                                      ("pred_ball_position", np.zeros((2, 6, 3))),
                                      ("target_ball_position", np.full((2, 7, 3), np.nan))])
def test_invalid_arrays(key: str, value: np.ndarray) -> None:
    arrays = _arrays()
    arrays[key] = value
    with pytest.raises(ValueError):
        summarize_motion(arrays, FPS, position_representation="meters")


def test_short_windows_have_no_high_order_samples() -> None:
    arrays = _arrays()
    for key in arrays:
        if key.endswith("position"):
            arrays[key] = arrays[key][..., :1, :]
        elif arrays[key].ndim > 1:
            arrays[key] = arrays[key][..., :1]
    report = summarize_motion(arrays, FPS, position_representation="meters")
    assert report["rows"][0]["entities"]["ball"]["jerk"]["pred_norm"]["count"] == 0


def test_camera_groups_and_player_slots_stay_separate() -> None:
    arrays = _arrays()
    arrays["video_ids"] = np.array(["v1", "v2"])
    arrays["camera_ids"][:] = "c1"
    arrays["pred_player_position"][:, 1, :, 0] += 5
    fps = {("v1", "v/clip1"): 10., ("v2", "v/clip2"): 20.}
    report = summarize_motion(arrays, fps, position_representation="meters")
    cameras = [row for row in report["rows"] if row["group_type"] == "camera"]
    assert [(row["video_id"], row["num_windows"]) for row in cameras] == [("v1", 1), ("v2", 1)]
    slots = report["rows"][0]["entities"]
    assert slots["player_slot_0"]["position_error_m"]["mean"] == 0
    assert slots["player_slot_1"]["position_error_m"]["mean"] == pytest.approx(5)
