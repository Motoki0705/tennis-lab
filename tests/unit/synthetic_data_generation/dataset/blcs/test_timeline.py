"""Tests for deterministic multi-court BLCS planning and chunk continuity."""

from __future__ import annotations

from collections import Counter

import numpy as np

from src.synthetic_data_generation.dataset.blcs.timeline import build_blcs_plans
from src.tasks.base.generate_dataset.camera_profiles import CameraProfileConfig


def test_plans_cover_all_frames_and_balance_all_courts(
    two_court_layout,
    default_camera_profile,
    blcs_assets,
    blcs_trajectory_factory,
) -> None:
    sources = tuple(
        blcs_trajectory_factory(f"trajectory-{index}") for index in range(4)
    )

    plans = build_blcs_plans(
        sources,
        dataset_scene_id="B00",
        layout=two_court_layout,
        camera_config=default_camera_profile,
        assets=blcs_assets,
        seed=695,
        chunk_size_frames=2,
    )

    assert [plan.global_frame_offset for plan in plans] == [0, 5, 10, 15]
    assert all(len(plan.camera_rig.cameras) == 6 for plan in plans)
    assert all(
        tuple(frame for chunk in plan.chunks for frame in chunk.frame_indices)
        == tuple(range(5))
        for plan in plans
    )
    counts = Counter(plan.target_court.court_instance_id for plan in plans)
    assert counts == {"court-0": 2, "court-1": 2}
    for plan in plans:
        court = two_court_layout.court(plan.target_court.court_instance_id)
        np.testing.assert_allclose(
            plan.positions_scene,
            court.scene_from_court.apply(plan.source.positions_court_m),
        )
        assert len(plan.composition.frames) == plan.source.frame_count


def test_same_seed_plan_is_deterministic(
    two_court_layout,
    default_camera_profile,
    blcs_assets,
    blcs_trajectory_factory,
) -> None:
    sources = tuple(
        blcs_trajectory_factory(f"trajectory-{index}") for index in range(2)
    )
    first = build_blcs_plans(
        sources,
        dataset_scene_id="B00",
        layout=two_court_layout,
        camera_config=default_camera_profile,
        assets=blcs_assets,
        seed=17,
        chunk_size_frames=3,
    )
    second = build_blcs_plans(
        sources,
        dataset_scene_id="B00",
        layout=two_court_layout,
        camera_config=default_camera_profile,
        assets=blcs_assets,
        seed=17,
        chunk_size_frames=3,
    )

    assert [plan.to_dict() for plan in first] == [plan.to_dict() for plan in second]
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left.camera_uv, right.camera_uv)


def test_broadcast_authority_produces_exactly_two_config_owned_cameras(
    two_court_layout,
    blcs_assets,
    blcs_trajectory_factory,
) -> None:
    profile = CameraProfileConfig.from_mapping(
        {
            "profile": "broadcast",
            "image_size": [32, 24],
            "expected_camera_count": 2,
            "slots": [
                {
                    "slot_id": f"broadcast-{side}",
                    "position_x_m": [0.0, 0.0],
                    "position_y_m": [y, y],
                    "height_m": [8.0, 8.0],
                    "look_at_x_m": [0.0, 0.0],
                    "look_at_y_m": [0.0, 0.0],
                    "look_at_height_m": [0.5, 0.5],
                    "hfov_degrees": [40.0, 40.0],
                }
                for side, y in (("near", -24.0), ("far", 24.0))
            ],
        }
    )

    plans = build_blcs_plans(
        (blcs_trajectory_factory("trajectory-0"),),
        dataset_scene_id="B00",
        layout=two_court_layout,
        camera_config=profile,
        assets=blcs_assets,
        seed=3,
        chunk_size_frames=2,
    )

    assert plans[0].camera_rig.profile == "broadcast"
    assert len(plans[0].camera_rig.cameras) == 2
