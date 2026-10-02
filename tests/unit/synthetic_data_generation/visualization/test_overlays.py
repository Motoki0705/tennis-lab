"""Pure rendering tests for canonical dataset overlays."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.runtime import (
    LogicalRenderSample,
    RenderSampleKey,
)
from src.synthetic_data_generation.visualization.overlays import render_court_overlay
from src.synthetic_data_generation.visualization.sources import CourtSourceFrame


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
