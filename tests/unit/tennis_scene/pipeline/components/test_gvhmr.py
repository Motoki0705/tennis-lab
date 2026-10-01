"""GVHMR consumes chosen 2D observations; detector/tracker/pose are never invoked."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from src.tennis_scene.pipeline.body_types import BodyRecoveryRequest
from src.tennis_scene.pipeline.components.body_view_selection import (
    BodySelection,
    BodyViewSelectionOutput,
)
from src.tennis_scene.pipeline.components.gvhmr import GVHMRInput, GVHMRModule
from tests.unit.tennis_scene.pipeline.config_factories import make_people_config


def selected(tmp_path: Path) -> GVHMRInput:
    request = BodyRecoveryRequest(tmp_path / 'cam1.mp4', np.array([0, 2, 4], np.int64),
        np.zeros((3, 17, 3), np.float32), np.ones((3, 3), np.float32) * 100,
        (640, 480), np.eye(3, dtype=np.float64))
    return GVHMRInput(BodyViewSelectionOutput((BodySelection(7, 'cam1', (request,), (3,)),)))


def test_only_feature_and_mesh_models_run_on_declared_source_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    class Features:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def predict(self, request: Any) -> Any:
            calls.append('features')
            assert request.frame_indices.tolist() == [0, 2, 4]
            assert request.video_path == tmp_path / 'cam1.mp4'
            return SimpleNamespace(features=torch.ones((3, 1024)))
        def unload(self) -> None:
            calls.append('feature_unload')
    class Mesh:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        def predict(self, request: Any) -> Any:
            calls.append('mesh')
            assert request.kp2d.shape == (3, 17, 3)
            assert request.K_fullimg.shape == (3, 3)
            return SimpleNamespace(smpl_params_incam={name: torch.zeros((3, width)) for name, width in
                [('body_pose', 63), ('global_orient', 3), ('betas', 10), ('transl', 3)]})
        def unload(self) -> None:
            calls.append('mesh_unload')
    monkeypatch.setattr('src.tennis_scene.pipeline.components.gvhmr.Hmr2FeatureExtractor', Features)
    monkeypatch.setattr('src.tennis_scene.pipeline.components.gvhmr.GvhmrMeshRecovery', Mesh)
    result = GVHMRModule(make_people_config(tmp_path)).process(selected(tmp_path))
    assert result.bodies[0].person_id == 7 and result.bodies[0].camera_id == 'cam1'
    assert result.bodies[0].segments[0].source_frames.tolist() == [0, 2, 4]
    assert calls == ['features', 'mesh', 'feature_unload', 'mesh_unload']


def test_invalid_body_input_fails_before_image_feature_extraction(tmp_path: Path) -> None:
    request = selected(tmp_path).selection.selections[0].requests[0]
    with pytest.raises(ValueError, match='ordered source frames'):
        replace(request, source_frames=np.array([0, 2, 2], np.int64))
    with pytest.raises(ValueError, match='shape mismatch'):
        replace(request, keypoints=np.zeros((3, 16, 3), np.float32))


def test_empty_body_selection_needs_no_weights(tmp_path: Path) -> None:
    result = GVHMRModule(make_people_config(tmp_path)).process(GVHMRInput(BodyViewSelectionOutput(())))
    assert not result.bodies
