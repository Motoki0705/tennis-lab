"""Reject incorrect clip axes and preserve the four external point statuses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.pipeline.imports.external_ball_positions import (
    REPORTED_VISIBILITY,
    STATUSES,
    read_external_ball_positions,
)


@pytest.fixture
def external_clip(tmp_path: Path) -> tuple[ClipManifest, Path, dict[str, Any]]:
    manifest = ClipManifest(
        tmp_path,
        "fixture",
        "video_000/clip_065",
        "video_000",
        "clip_065",
        59.94005994005994,
        4,
        20,
        10,
        ("cam1", "cam0"),
        ("cam1.mp4", "cam0.mp4"),
        (),
    )
    shape = {"width": 20, "height": 10, "fps": manifest.fps, "frame_count": 4}
    first = [
        {"frame": i, "x": x, "y": y, "status": status, "visible_in_frame": visibility}
        for i, ((x, y), status, visibility) in enumerate(
            zip(
                [(0, 0), (20, 3), (4, -0.1), (19.99, 9.99)],
                STATUSES,
                [True, "verified", "out_of_frame", "no"],
                strict=True,
            )
        )
    ]
    second = [
        {
            "frame": i,
            "x": 2,
            "y": y,
            "status": "estimated",
            "visible_in_frame": visibility,
        }
        for i, (y, visibility) in enumerate(
            zip([10, 3, 4, 5], ["yes", "visible", False, "unverified"], strict=True)
        )
    ]
    document = {
        "metadata": {"total_cameras": 2, "frames_per_camera": 4, "fps": manifest.fps},
        "cameras": {
            "clip_065_cam0": {
                **shape,
                "input_file": "clip_065_cam0.mp4",
                "frames": second,
            },
            "clip_065_cam1": {
                **shape,
                "input_file": "clip_065_cam1(1).mp4",
                "frames": first,
            },
        },
    }
    return manifest, tmp_path / "clip_065_ball_positions.json", document


def test_external_positional_acceptance_uses_camera_order_and_exact_image_bounds(
    external_clip,
) -> None:
    manifest, path, doc = external_clip
    path.write_text(json.dumps(doc))
    positions = read_external_ball_positions(path, manifest)
    assert positions.camera_ids == ("cam1", "cam0")
    np.testing.assert_array_equal(
        positions.usable, [[True, False, False, True], [False, True, True, True]]
    )
    assert [STATUSES[code] for code in positions.status[0]] == list(STATUSES)
    assert [
        [REPORTED_VISIBILITY[code] for code in row]
        for row in positions.reported_visibility
    ] == [
        ["visible", "visible", "out_of_frame", "not_visible"],
        ["visible", "visible", "not_visible", "unverified"],
    ]
    assert positions.uv_px[0, 1].tolist() == [20, 3]


@pytest.mark.parametrize(
    ("keys", "new", "message"),
    [
        (("metadata", "fps"), 60, "FPS"),
        (("metadata", "total_cameras"), 3, "total_cameras"),
        (("cameras",), {"clip_066_cam1": {}}, "clip or camera IDs"),
        (
            ("cameras", "clip_065_cam1", "input_file"),
            "clip_065_cam0.mp4",
            "input filename",
        ),
        (("cameras", "clip_065_cam0", "width"), 21, "width"),
        (("cameras", "clip_065_cam1", "height"), 9, "height"),
        (("cameras", "clip_065_cam0", "fps"), 60, "fps"),
        (("cameras", "clip_065_cam1", "frame_count"), 3, "frame_count"),
        (("cameras", "clip_065_cam1", "frames"), [], "frame array length"),
        (("cameras", "clip_065_cam1", "frames", 2, "frame"), 3, "frame index 2"),
        (("cameras", "clip_065_cam1", "frames", 1, "x"), None, "finite pixel"),
        (
            ("cameras", "clip_065_cam1", "frames", 1, "visible_in_frame"),
            "maybe",
            "visible_in_frame",
        ),
    ],
)
def test_external_positions_reject_mismatched_clip_axes(
    external_clip, keys, new, message
) -> None:
    manifest, path, doc = external_clip
    target = doc
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = new
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match=message):
        read_external_ball_positions(path, manifest)
