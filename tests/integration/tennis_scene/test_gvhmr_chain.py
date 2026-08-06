"""Integration test for GVHMR component-to-adapter typed execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

import src.submodules.models as submodule_models
import src.tennis_scene.pipeline.model_io.gvhmr as gvhmr_io
from src.tennis_scene.pipeline.components.gvhmr import GVHMRModule
from src.tennis_scene.pipeline.model_io.gvhmr import GVHMRChainAdapter
from src.utils.video import VideoInfo
from tests.unit.tennis_scene.pipeline.config_factories import make_gvhmr_config


class _Predictor:
    def __init__(self, result: object) -> None:
        self.result = result
        self.requests: list[object] = []
        self.is_loaded = False

    def load(self) -> None:
        self.is_loaded = True

    def unload(self) -> None:
        self.is_loaded = False

    def predict(self, request: object) -> object:
        self.requests.append(request)
        return self.result


class _Vertices:
    def __init__(self) -> None:
        self.requests: list[dict[str, torch.Tensor]] = []

    def reconstruct(self, request: dict[str, torch.Tensor]) -> torch.Tensor:
        self.requests.append(request)
        return torch.ones((2, 8, 3), dtype=torch.float32)


def _smpl_parameters() -> dict[str, torch.Tensor]:
    return {
        "body_pose": torch.zeros((2, 63), dtype=torch.float32),
        "betas": torch.zeros((2, 10), dtype=torch.float32),
        "global_orient": torch.zeros((2, 3), dtype=torch.float32),
        "transl": torch.zeros((2, 3), dtype=torch.float32),
    }


def test_component_executes_resolved_chain_without_raw_output_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.touch()
    monkeypatch.setattr(
        gvhmr_io,
        "probe_video_info",
        lambda _: VideoInfo(fps=30.0, width=1280, height=720, frame_count=2),
    )

    tracker = _Predictor(
        submodule_models.TrackResult(
            tracks={3: torch.tensor([[10.0, 20.0, 30.0, 60.0]] * 2)},
            num_frames=2,
        )
    )
    keypoints = torch.zeros((2, 17, 3), dtype=torch.float32)
    keypoints[..., 2] = 1.0
    pose = _Predictor(submodule_models.Pose2DResult(keypoints=keypoints))
    feature = _Predictor(
        submodule_models.ImageFeatureResult(
            features=torch.ones((2, 1024), dtype=torch.float32)
        )
    )
    mesh = _Predictor(
        submodule_models.GvhmrResult(
            smpl_params_incam=_smpl_parameters(),
            smpl_params_global=_smpl_parameters(),
            K_fullimg=torch.eye(3).expand(2, -1, -1),
        )
    )
    vertices = _Vertices()
    adapter = GVHMRChainAdapter(
        tracker=cast(Any, tracker),
        pose_model=cast(Any, pose),
        feature_model=cast(Any, feature),
        mesh_model=cast(Any, mesh),
        vertex_reconstructor=cast(Any, vertices),
    )
    module = GVHMRModule(make_gvhmr_config(tmp_path, num_tracks=1), adapter)

    result = module.process(video_path)

    np.testing.assert_array_equal(result.track_ids, np.array([3], dtype=np.int32))
    assert result.smpl_body_pose.shape == (1, 2, 63)
    assert result.smpl_vertices_local is not None
    assert result.smpl_vertices_local.shape == (1, 2, 8, 3)
    assert isinstance(mesh.requests[0], submodule_models.GvhmrRequest)
    assert len(vertices.requests) == 1
