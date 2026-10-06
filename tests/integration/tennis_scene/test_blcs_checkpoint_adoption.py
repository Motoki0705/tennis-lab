"""Actual adopted checkpoint with saved physical observations and local CourtKP."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tasks.blcs.generate_dataset.io.dataset_io import load_scene
from src.tennis_scene.pipeline.components.ball_reconstruction import (
    single_ball_observations,
)
from src.tennis_scene.pipeline.components.blcs import (
    BallReconstructionInput,
    BLCSReconstructionModule,
)
from src.tennis_scene.pipeline.components.camera_alignment import CameraAlignmentOutput
from src.tennis_scene.pipeline.components.camera_geometry import CameraGeometryResult
from src.tennis_scene.pipeline.components.court_kp import CourtKPResult
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.observation_types import ObjectObservations
from src.utils.configuration import PathResolver, RuntimePathRoots
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.schema.court import CAMERA_VIEW_HALF_TURN_INDEX


@pytest.mark.local_data
@pytest.mark.cuda
@pytest.mark.skipif(os.environ.get("TENNIS_GPU_RESOURCE") != "all", reason="run under the shared GPU queue with resource=all")
def test_adopted_checkpoint_reconstructs_saved_observations() -> None:
    root = Path(os.environ["TRAINING_QUEUE_DIR"]).resolve().parent
    checkpoint = root / "ckpt/blcs/axial-base-kp14-t128-v3-6-e200-s42-epoch189.ckpt"
    scene_dir = root / "data/blcs/single_object/scenes/scene_000686"
    if not checkpoint.is_file() or not scene_dir.is_dir():
        pytest.skip("requires the adopted local checkpoint and dataset")
    scene = load_scene(scene_dir, court_keypoint_contract=resolve_court_keypoint_contract("physical_v1"))
    selected = scene["cameras"][:4]
    frames = len(scene["ball_pos_world"])
    ids = tuple(row["camera_id"] for row in selected)
    params = selected[0]["params"]
    width, height = int(params["w"]), int(params["h"])
    fps = float(scene["meta"]["fps_out"])
    source = ClipSource("synthetic-scene-000686", tuple(SourceVideo(c, scene_dir / f"{c}.mp4", "synthetic-no-rgb", frames, fps, width, height) for c in ids))
    cameras = []
    for row in selected:
        p = row["params"]
        rotation = np.asarray(p["R"], np.float64)
        center = np.asarray(p["C"], np.float64)
        k = np.array([[p["f"], 0., p["cx"]], [0., p["f"], p["cy"]], [0., 0., 1.]], np.float64)
        cameras.append(PinholeCamera(row["camera_id"], k, rotation, -rotation @ center))
    turns = tuple(bool(c.center[1] > 0) for c in cameras)
    reference = next(c.camera_id for c, turned in zip(cameras, turns, strict=True) if not turned)
    geometry = CameraGeometryResult(ids, reference, turns, tuple(cameras), cast(Any, None), {})
    uv = np.stack([row["ball_uv"] * [width, height] for row in selected]).astype(np.float32)
    observed = np.stack([row["ball_vis"] for row in selected]).astype(bool)
    raw = ObjectObservations(ids, source.size, fps, uv[:, :, None, None], observed[:, :, None, None].astype(np.float32),
        observed[:, :, None], np.zeros((len(ids), 1), np.int64))
    court = np.stack([row["court_kp_uv"][:14] for row in selected]).astype(np.float32)
    visible = np.stack([row["court_kp_vis"][:14] for row in selected]).astype(np.float32)
    permutation = np.asarray(CAMERA_VIEW_HALF_TURN_INDEX)
    for index, turned in enumerate(turns):
        if turned:
            court[index], visible[index] = court[index, permutation], visible[index, permutation]
    # Saved physical UV uses W/H; emulate the detector's camera-local W-1/H-1 grid.
    court *= np.array([width / (width - 1), height / (height - 1)], np.float32)
    court_result = CourtKPResult(np.repeat(court[:, None], frames, axis=1), np.repeat(visible[:, None], frames, axis=1),
        np.arange(frames, dtype=np.int32), {"output_keypoint_contract": "camera_view_v2"})
    roots = RuntimePathRoots.from_mapping({"project_root": str(root), "data_root": "data", "checkpoint_root": "ckpt",
        "artifact_root": "outputs", "output_root": "outputs", "cache_root": ".cache", "external_asset_root": "third_party"}, repository_root=root)
    module = BLCSReconstructionModule(ids, checkpoint=checkpoint, resolver=PathResolver(roots), device="cuda",
        window_size=128, reprojection_px=20., min_frames=5)
    output = module.process(BallReconstructionInput(source, CameraAlignmentOutput(geometry), single_ball_observations(raw, threshold=0.), court_result, ids))
    assert output.ball is not None
    track = output.ball.trajectory
    assert track.positions.shape == (frames, 3) and np.isfinite(track.positions).all()
    assert track.valid.any()
    assert not (track.valid & (observed.sum(0) < 2)).any()
    assert (track.positions[~track.valid] == 0).all()
    assert (track.inliers.sum(0)[track.valid] >= 2).all()
    module.unload()
