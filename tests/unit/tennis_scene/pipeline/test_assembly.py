"""Final scene joins must reject ambiguous source axes before assigning arrays."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.tasks.base.data import ReferenceViewSelection, StableCameraIdTable
from src.tasks.base.generate_dataset import (
    build_court_view_record,
    resolve_court_keypoint_contract,
)
from src.tennis_scene.pipeline.assembly import assemble_automatic_scene
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    BallReconstructionResult,
)
from src.tennis_scene.pipeline.components.camera_geometry import CameraGeometryResult
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.observation_types import GroupedObservations
from src.utils.geometry.triangulation import PinholeCamera, TriangulatedPoints
from src.utils.video import VideoInfo


def inputs() -> dict[str, Any]:
    return dict(
        video_paths=tuple(Path(f"{c}.mp4") for c in ("west", "blocked", "east", "high")),
        camera_ids=("west", "blocked", "east", "high"), info=VideoInfo(30., 640, 480, 3),
        court=CourtKPResult(np.zeros((4, 3, 14, 2), np.float32), np.zeros((4, 3, 14), np.float32), np.arange(3, dtype=np.int32)),
        active_indices=(0, 2, 3), geometry=None, grouped_people=None, skeleton=None,
        players=None, ball=None, metadata={},
    )


def geometry() -> CameraGeometryResult:
    ids = ("west", "east", "high")
    cameras = tuple(PinholeCamera(c, np.eye(3), np.eye(3), np.array([float(v), 10., -5.])) for v, c in enumerate(ids))
    contract = resolve_court_keypoint_contract("camera_view_v2")
    selection = ReferenceViewSelection.create(stable_camera_id_table=StableCameraIdTable.from_complete_scene_camera_ids(ids),
        selected_views=tuple(build_court_view_record(camera_id=c.camera_id, camera_center_court_m=c.center.tolist(), contract=contract) for c in cameras),
        reference_camera_id="west")
    return CameraGeometryResult(ids, "west", (False,) * 3, cameras, selection, {})


@pytest.mark.parametrize("mismatch", ["camera_count", "duplicate_camera", "video_count", "frame_order", "frame_count"])
def test_assembly_rejects_source_camera_or_frame_mismatch(mismatch: str) -> None:
    args = inputs()
    if mismatch == "camera_count":
        args["camera_ids"] = ("west", "east", "high")
    elif mismatch == "duplicate_camera":
        args["camera_ids"] = ("west", "blocked", "east", "west")
    elif mismatch == "video_count":
        args["video_paths"] = args["video_paths"][:3]
    elif mismatch == "frame_order":
        args["court"].frame_indices[:] = [0, 2, 1]
    else:
        args["info"] = replace(args["info"], frame_count=4)
    with pytest.raises(ValueError, match="camera|frame"):
        assemble_automatic_scene(**args)


@pytest.mark.parametrize("active", [(-1, 2, 3), (0, 2, 4), (0, 2, 2)])
def test_assembly_rejects_invalid_active_camera_indices(active: tuple[int, ...]) -> None:
    args = inputs()
    args["active_indices"] = active
    with pytest.raises(ValueError, match="camera indices"):
        assemble_automatic_scene(**args)


@pytest.mark.parametrize("mismatch", ["geometry_ids", "pinhole_order", "observation_views", "observation_frames", "ball_views", "ball_frames"])
def test_assembly_rejects_compacted_axis_mismatch(mismatch: str) -> None:
    args = inputs()
    if mismatch in {"geometry_ids", "pinhole_order"}:
        model = geometry()
        args["geometry"] = replace(model, camera_ids=model.camera_ids[::-1]) if mismatch == "geometry_ids" else replace(model, cameras=model.cameras[::-1])
    elif mismatch.startswith("observation"):
        views, frames = (1, 3) if mismatch == "observation_views" else (3, 1)
        args["grouped_people"] = GroupedObservations(np.array([17], np.int64),
            np.zeros((1, views, frames, 17, 2), np.float32), np.zeros((1, views, frames, 17), np.float32),
            np.zeros((1, views, frames, 17), bool), np.full((1, views, frames), -1, np.int64))
    else:
        views, frames = (1, 3) if mismatch == "ball_views" else (3, 1)
        args["ball"] = BallReconstructionResult(np.zeros((views, frames, 2), np.float32), np.zeros((views, frames), bool),
            TriangulatedPoints(np.zeros((3, 3), np.float32), np.zeros(3, bool), np.ones(3, np.uint8), np.zeros((3, 3), bool), np.zeros((3, 3), np.float32)), "missing")
    with pytest.raises(ValueError, match="camera|timeline"):
        assemble_automatic_scene(**args)
