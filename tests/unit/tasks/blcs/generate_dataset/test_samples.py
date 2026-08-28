"""Unit tests for BLCS sample strata and timeline adaptation."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from src.tasks.blcs.generate_dataset.samples import (
    cell_region,
    resolve_start_cell,
    subsample_blcs_scene,
)


def _scene() -> dict[str, object]:
    num_frames = 6
    num_balls = 2
    present: NDArray[np.bool_] = np.zeros((num_frames, num_balls), dtype=np.bool_)
    present[:3, 0] = True
    present[2:, 1] = True
    camera = {
        "ball_uv": np.arange(num_frames * num_balls * 2, dtype=np.float32).reshape(
            num_frames, num_balls, 2
        ),
        "ball_vis": np.array(
            [
                [True, False],
                [True, False],
                [True, True],
                [False, True],
                [False, False],
                [False, True],
            ]
        ),
        "court_kp_uv": np.zeros((20, 2), dtype=np.float32),
        "court_kp_vis": np.ones(20, dtype=np.bool_),
    }
    trajectory: NDArray[np.float32] = np.zeros(
        (num_frames, num_balls, 3), dtype=np.float32
    )
    return {
        "meta": {
            "num_frames": num_frames,
            "fps_out": 30,
            "shots": [
                {
                    "track_id": 0,
                    "shots": [
                        {
                            "shot_index": 0,
                            "t_start": 0,
                            "t_net": -1,
                            "t_bounce1": 4,
                            "t_bounce2": 9,
                            "t_bounce3": -1,
                            "t_return": 5,
                        }
                    ],
                }
            ],
            "track_instances": [
                {"track_id": 0, "birth_frame": 0, "death_frame": 3},
                {"track_id": 1, "birth_frame": 2, "death_frame": 6},
            ],
        },
        "ball_pos_world": trajectory.copy(),
        "ball_pos_norm": trajectory.copy(),
        "ball_vel_world": trajectory.copy(),
        "ball_vel_norm": trajectory.copy(),
        "ball_present": present,
        "num_balls": num_balls,
        "num_cameras": 1,
        "cameras": [camera],
    }


def test_cell_regions_preserve_deuce_ad_and_baseline_semantics() -> None:
    assert {cell_region(cell) for cell in (0, 2, 4, 6)} == {"deuce_side"}
    assert {cell_region(cell) for cell in (1, 3, 5, 7)} == {"ad_side"}
    assert cell_region(8) == "behind_baseline"


def test_start_cell_prefers_first_non_serve_hit() -> None:
    shots = (
        {"shot_type": "serve", "from_cell": 8, "to_cell": 0},
        {"shot_type": "groundstroke", "from_cell": 7, "to_cell": 1},
    )

    assert resolve_start_cell({"initial_from_cell": 8}, shots) == (
        7,
        "first_non_serve_from_cell",
    )


def test_start_cell_uses_serve_target_when_no_return_exists() -> None:
    shots = ({"shot_type": "serve", "from_cell": 8, "to_cell": 1},)

    assert resolve_start_cell({"initial_from_cell": 8}, shots) == (
        1,
        "serve_to_cell",
    )


def test_subsample_blcs_scene_remaps_events_presence_and_visibility() -> None:
    scene = _scene()
    original_meta = deepcopy(scene["meta"])
    original_uv = np.asarray(scene["cameras"][0]["ball_uv"]).copy()  # type: ignore[index]
    indices = np.array([0, 2, 5], dtype=np.int64)

    sampled = subsample_blcs_scene(scene, indices=indices, playback_fps=10)  # type: ignore[arg-type]

    assert np.asarray(sampled["ball_pos_world"]).shape == (3, 2, 3)
    assert np.asarray(sampled["ball_present"]).tolist() == [
        [True, False],
        [True, True],
        [False, True],
    ]
    sampled_camera = sampled["cameras"][0]
    sampled_uv = np.asarray(sampled_camera["ball_uv"])
    assert np.isnan(sampled_uv[0, 1]).all()
    assert np.isnan(sampled_uv[2, 0]).all()
    assert np.isfinite(sampled_uv[1]).all()
    assert np.asarray(sampled_camera["court_kp_uv"]).shape == (20, 2)
    assert sampled["meta"]["num_frames"] == 3
    assert sampled["meta"]["fps_out"] == 10
    remapped_shot = sampled["meta"]["shots"][0]["shots"][0]
    assert remapped_shot["t_bounce1"] == 2
    assert remapped_shot["t_bounce2"] == -1
    assert remapped_shot["t_return"] == 2
    assert sampled["meta"]["track_instances"] == [
        {"track_id": 0, "birth_frame": 0, "death_frame": 2},
        {"track_id": 1, "birth_frame": 1, "death_frame": 3},
    ]
    assert scene["meta"] == original_meta
    np.testing.assert_array_equal(scene["cameras"][0]["ball_uv"], original_uv)  # type: ignore[index]
