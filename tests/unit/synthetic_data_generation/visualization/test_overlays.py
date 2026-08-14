"""Pure rendering tests for canonical dataset overlays."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.runtime import (
    LogicalRenderSample,
    RenderSampleKey,
)
from src.synthetic_data_generation.visualization.overlays import (
    new_ball_history,
    render_blcs_overlay,
    render_court_overlay,
    render_plcs_overlay,
)
from src.synthetic_data_generation.visualization.sources import (
    BLCSSourceFrame,
    CourtSourceFrame,
    PLCSSourceFrame,
)


def _render(*, visible_pixel_count: int = 42) -> LogicalRenderSample:
    instance_ids: NDArray[np.int32] = np.zeros((96, 128), dtype=np.int32)
    instance_ids.reshape(-1)[:visible_pixel_count] = 1
    return LogicalRenderSample(
        key=RenderSampleKey(0, "camera-0"),
        rgb=np.zeros((96, 128, 3), dtype=np.float32),
        alpha=np.ones((96, 128, 1), dtype=np.float32),
        depth=np.ones((96, 128, 1), dtype=np.float32),
        instance_ids=instance_ids,
    )


def test_court_overlay_distinguishes_renderer_visible_points() -> None:
    classes = []
    names = (
        "doubles_left",
        "doubles_right",
        "singles_left",
        "singles_right",
        "service_left",
        "service_right",
        "service_t",
    )
    for class_id, name in enumerate(names):
        classes.append(
            {
                "class_id": class_id,
                "class_name": name,
                "renderer_visible": True,
                "points": [
                    {
                        "physical_index": class_id * 2,
                        "uv": [20.0 + class_id * 4, 60.0],
                        "camera_depth_m": 1.0,
                        "scene_xyz_m": [0.0, 0.0, 0.0],
                        "in_front": True,
                        "in_frame": True,
                        "renderer_visible": True,
                    },
                    {
                        "physical_index": class_id * 2 + 1,
                        "uv": [20.0 + class_id * 4, 80.0],
                        "camera_depth_m": 1.0,
                        "scene_xyz_m": [0.0, 0.0, 0.0],
                        "in_front": True,
                        "in_frame": True,
                        "renderer_visible": class_id != 0,
                    },
                ],
            }
        )
    frame = CourtSourceFrame(
        rgb=np.zeros((96, 128, 3), dtype=np.float32),
        sample_id="sample-0",
        view_id="view-0",
        trajectory_frame_index=0,
        projection={
            "courts": [
                {
                    "court_instance_id": "court-0",
                    "coverage_mode": "full",
                    "classes": classes,
                }
            ]
        },
    )

    output = render_court_overlay(frame, trajectory_id="orbit-0")

    assert output.shape == (96, 128, 3)
    assert output.dtype == np.uint8
    assert np.count_nonzero(output) > 0


def test_blcs_overlay_tracks_identity_presence_and_history() -> None:
    frame = BLCSSourceFrame(
        render=_render(),
        source_frame_index=0,
        global_frame_index=10,
        metadata={
            "objects": [
                {
                    "object_id": "ball-0",
                    "instance_id": 1,
                    "present": True,
                    "geometric_visible": True,
                    "rendered_visible": True,
                }
            ],
            "semantic_arrays": {
                "ball_uv": [[64.0, 48.0]],
                "present": [True],
                "geometric_visible": [True],
                "rendered_visible": [True],
                "instance_ids": [1],
            },
        },
    )
    history = new_ball_history(("ball-0",), history_frames=4)

    output = render_blcs_overlay(
        frame,
        logical_scene_id="trajectory-0",
        camera_id="camera-0",
        object_ids=("ball-0",),
        court_kp=np.zeros((20, 2), dtype=np.float32),
        court_vis=np.zeros((20,), dtype=np.bool_),
        history=history,
        history_frames=4,
    )

    assert tuple(history["ball-0"]) == ((64, 48),)
    assert output.dtype == np.uint8
    assert np.count_nonzero(output) > 0


def test_plcs_overlay_draws_projected_coco17_skeleton() -> None:
    human_kp: NDArray[np.float32] = np.zeros((1, 17, 2), dtype=np.float32)
    human_kp[0, 5] = (0.4, 0.4)
    human_kp[0, 6] = (0.6, 0.4)
    human_vis: NDArray[np.bool_] = np.zeros((1, 17), dtype=np.bool_)
    human_vis[0, 5:7] = True
    frame = PLCSSourceFrame(
        render=_render(),
        frame_index=0,
        label={
            "objects": [
                {
                    "object_id": "person-0",
                    "instance_id": 1,
                    "present": True,
                    "visible_pixel_count": 42,
                }
            ]
        },
        human_kp=human_kp,
        human_vis=human_vis,
        court_kp=np.zeros((20, 2), dtype=np.float32),
        court_vis=np.zeros((20,), dtype=np.bool_),
        present=np.ones((1,), dtype=np.bool_),
    )

    output = render_plcs_overlay(
        frame,
        logical_scene_id="logical-0",
        camera_id="camera-0",
        object_ids=("person-0",),
    )

    assert output.dtype == np.uint8
    assert np.count_nonzero(output) > 0


def test_blcs_overlay_rejects_visibility_not_observed_in_streamed_instances() -> None:
    frame = BLCSSourceFrame(
        render=_render(visible_pixel_count=0),
        source_frame_index=0,
        global_frame_index=0,
        metadata={
            "objects": [
                {
                    "object_id": "ball-0",
                    "instance_id": 1,
                    "present": True,
                    "geometric_visible": True,
                    "rendered_visible": True,
                }
            ],
            "semantic_arrays": {
                "ball_uv": [[64.0, 48.0]],
                "present": [True],
                "geometric_visible": [True],
                "rendered_visible": [True],
                "instance_ids": [1],
            },
        },
    )
    history = new_ball_history(("ball-0",), history_frames=4)

    with pytest.raises(ValueError, match="rendered_visible claims disagree"):
        render_blcs_overlay(
            frame,
            logical_scene_id="trajectory-0",
            camera_id="camera-0",
            object_ids=("ball-0",),
            court_kp=np.zeros((20, 2), dtype=np.float32),
            court_vis=np.zeros((20,), dtype=np.bool_),
            history=history,
            history_frames=4,
        )

    assert not history["ball-0"]


def test_plcs_overlay_rejects_pixel_count_not_observed_in_streamed_instances() -> None:
    frame = PLCSSourceFrame(
        render=_render(visible_pixel_count=3),
        frame_index=0,
        label={
            "objects": [
                {
                    "object_id": "person-0",
                    "instance_id": 1,
                    "present": True,
                    "visible_pixel_count": 4,
                }
            ]
        },
        human_kp=np.zeros((1, 17, 2), dtype=np.float32),
        human_vis=np.zeros((1, 17), dtype=np.bool_),
        court_kp=np.zeros((20, 2), dtype=np.float32),
        court_vis=np.zeros((20,), dtype=np.bool_),
        present=np.ones((1,), dtype=np.bool_),
    )

    with pytest.raises(ValueError, match="visible_pixel_count disagrees"):
        render_plcs_overlay(
            frame,
            logical_scene_id="logical-0",
            camera_id="camera-0",
            object_ids=("person-0",),
        )


def test_present_but_occluded_objects_remain_valid() -> None:
    blcs = BLCSSourceFrame(
        render=_render(visible_pixel_count=0),
        source_frame_index=0,
        global_frame_index=0,
        metadata={
            "objects": [
                {
                    "object_id": "ball-0",
                    "instance_id": 1,
                    "present": True,
                    "geometric_visible": True,
                    "rendered_visible": False,
                }
            ],
            "semantic_arrays": {
                "ball_uv": [[64.0, 48.0]],
                "present": [True],
                "geometric_visible": [True],
                "rendered_visible": [False],
                "instance_ids": [1],
            },
        },
    )
    plcs = PLCSSourceFrame(
        render=_render(visible_pixel_count=0),
        frame_index=0,
        label={
            "objects": [
                {
                    "object_id": "person-0",
                    "instance_id": 1,
                    "present": True,
                    "visible_pixel_count": 0,
                }
            ]
        },
        human_kp=np.zeros((1, 17, 2), dtype=np.float32),
        human_vis=np.zeros((1, 17), dtype=np.bool_),
        court_kp=np.zeros((20, 2), dtype=np.float32),
        court_vis=np.zeros((20,), dtype=np.bool_),
        present=np.ones((1,), dtype=np.bool_),
    )

    blcs_output = render_blcs_overlay(
        blcs,
        logical_scene_id="trajectory-0",
        camera_id="camera-0",
        object_ids=("ball-0",),
        court_kp=np.zeros((20, 2), dtype=np.float32),
        court_vis=np.zeros((20,), dtype=np.bool_),
        history=new_ball_history(("ball-0",), history_frames=4),
        history_frames=4,
    )
    plcs_output = render_plcs_overlay(
        plcs,
        logical_scene_id="logical-0",
        camera_id="camera-0",
        object_ids=("person-0",),
    )

    assert blcs_output.dtype == np.uint8
    assert plcs_output.dtype == np.uint8
