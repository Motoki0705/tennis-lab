"""Unit tests for PLCS sample-timeline adaptation."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from src.tasks.plcs.generate_dataset.io.scene_loader import AttrDict
from src.tasks.plcs.generate_dataset.samples import subsample_plcs_scene


def _scene() -> AttrDict:
    num_frames = 6
    num_people = 2
    present: NDArray[np.bool_] = np.zeros((num_frames, num_people), dtype=np.bool_)
    present[:3, 0] = True
    present[2:, 1] = True
    camera = AttrDict(
        human_kp_uv=np.arange(num_frames * num_people * 17 * 2).reshape(
            num_frames, num_people, 17, 2
        ),
        human_kp_vis=np.ones((num_frames, num_people, 17), dtype=np.bool_),
        court_kp_uv=np.zeros((num_frames, 20, 2), dtype=np.float32),
        court_kp_vis=np.ones((num_frames, 20), dtype=np.bool_),
        params={"camera": 0},
    )
    return AttrDict(
        meta={
            "num_frames": num_frames,
            "fps": 120.0,
            "track_instances": [
                {"track_id": 0, "birth_frame": 0, "death_frame": 3},
                {"track_id": 1, "birth_frame": 2, "death_frame": 6},
            ],
        },
        position=np.arange(num_frames * num_people * 3).reshape(
            num_frames, num_people, 3
        ),
        rotation=np.zeros((num_frames, num_people, 2), dtype=np.float32),
        canonical_pose_3d=np.zeros((num_frames, num_people, 17, 3), dtype=np.float32),
        human_kp_3d=np.zeros((num_frames, num_people, 17, 3), dtype=np.float32),
        person_present=present,
        num_persons=num_people,
        num_cameras=1,
        cameras=[camera],
    )


def test_subsample_plcs_scene_slices_every_temporal_field_without_mutation() -> None:
    scene = _scene()
    original_meta = deepcopy(scene.meta)
    original_position = scene.position.copy()
    indices = np.array([0, 2, 5], dtype=np.int64)

    sampled = subsample_plcs_scene(scene, indices=indices, playback_fps=12)

    assert sampled.position.shape == (3, 2, 3)
    assert sampled.canonical_pose_3d.shape == (3, 2, 17, 3)
    assert sampled.human_kp_3d.shape == (3, 2, 17, 3)
    assert sampled.person_present.tolist() == [
        [True, False],
        [True, True],
        [False, True],
    ]
    assert sampled.cameras[0].human_kp_uv.shape == (3, 2, 17, 2)
    assert sampled.cameras[0].court_kp_uv.shape == (3, 20, 2)
    assert sampled.meta["num_frames"] == 3
    assert sampled.meta["fps"] == 12.0
    assert sampled.meta["track_instances"] == [
        {"track_id": 0, "birth_frame": 0, "death_frame": 2},
        {"track_id": 1, "birth_frame": 1, "death_frame": 3},
    ]
    assert scene.meta == original_meta
    np.testing.assert_array_equal(scene.position, original_position)


def test_subsample_plcs_scene_requires_endpoint_inclusive_indices() -> None:
    scene = _scene()

    with np.testing.assert_raises_regex(ValueError, "endpoint-inclusive"):
        subsample_plcs_scene(
            scene,
            indices=np.array([1, 3, 5], dtype=np.int64),
            playback_fps=12,
        )
