"""Tests for the tennis_scene GVHMR component (serialization / config)."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

import src.submodules.models as submodule_models
from src.tennis_scene.pipeline.components.gvhmr import (
    GVHMRModule,
    GVHMRResult,
)
from tests.unit.tennis_scene.pipeline.config_factories import make_gvhmr_config


def make_result(with_vertices: bool = True) -> GVHMRResult:
    rng = np.random.default_rng(0)
    P, T = 2, 4
    return GVHMRResult(
        smpl_body_pose=rng.normal(size=(P, T, 63)).astype(np.float32),
        smpl_global_orient=rng.normal(size=(P, T, 3)).astype(np.float32),
        smpl_betas=rng.normal(size=(P, 10)).astype(np.float32),
        smpl_vertices_local=(
            rng.normal(size=(P, T, 16, 3)).astype(np.float32) if with_vertices else None
        ),
        human_kp_2d=rng.normal(size=(P, T, 17, 2)).astype(np.float32),
        human_kp_vis=rng.random(size=(P, T, 17)).astype(np.float32),
        bbx_xys=rng.normal(size=(P, T, 3)).astype(np.float32),
        track_ids=np.array([3, 7], dtype=np.int32),
    )


class TestGVHMRResultRoundTrip:
    def test_save_load_roundtrip(self, tmp_path: Path):
        result = make_result()
        path = tmp_path / "gvhmr_result.json"
        result.save(path)
        loaded = GVHMRResult.load(path)

        np.testing.assert_allclose(
            loaded.smpl_body_pose, result.smpl_body_pose, rtol=1e-6
        )
        np.testing.assert_allclose(loaded.smpl_betas, result.smpl_betas, rtol=1e-6)
        assert loaded.smpl_vertices_local is not None
        assert result.smpl_vertices_local is not None
        np.testing.assert_allclose(
            loaded.smpl_vertices_local, result.smpl_vertices_local, rtol=1e-6
        )
        np.testing.assert_allclose(loaded.human_kp_2d, result.human_kp_2d, rtol=1e-6)
        np.testing.assert_allclose(loaded.bbx_xys, result.bbx_xys, rtol=1e-6)
        assert loaded.track_ids is not None
        np.testing.assert_array_equal(loaded.track_ids, result.track_ids)

    def test_optional_fields_survive_roundtrip(self, tmp_path: Path):
        result = make_result(with_vertices=False)
        result.track_ids = None
        path = tmp_path / "gvhmr_result.json"
        result.save(path)
        loaded = GVHMRResult.load(path)
        assert loaded.smpl_vertices_local is None
        assert loaded.track_ids is None


class TestGVHMRConfig:
    def test_explicit_assets_and_model_choices_are_preserved(
        self, tmp_path: Path
    ) -> None:
        config = make_gvhmr_config(tmp_path)
        assert config.yolo_checkpoint == (tmp_path / "ckpt/yolo.pt").resolve()
        assert config.dino_checkpoint == (tmp_path / "ckpt/dino.pth").resolve()
        assert config.detector == "dino"
        assert config.vitpose_checkpoint == (tmp_path / "ckpt/vitpose.pth").resolve()
        assert config.hmr2_checkpoint == (tmp_path / "ckpt/hmr2.ckpt").resolve()
        assert config.track_selection == "auto"
        assert config.runtime.allow_device_fallback is False

    def test_rejects_unknown_detector(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="detector must be"):
            make_gvhmr_config(tmp_path, detector="unknown")


def test_load_selects_dino_detector_and_existing_tracker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    constructed: dict[str, Any] = {}

    class FakeModel:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def load(self) -> None:
            return None

    class FakeDinoTracker(FakeModel):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            constructed["dino_tracker"] = self

    def fail_yolo(**kwargs: Any) -> None:
        del kwargs
        raise AssertionError("YOLO tracker must not be constructed in DINO mode")

    monkeypatch.setattr(submodule_models, "DinoPersonTracker", FakeDinoTracker)
    monkeypatch.setattr(submodule_models, "YoloPersonTracker", fail_yolo)
    monkeypatch.setattr(submodule_models, "ViTPosePose2D", FakeModel)
    monkeypatch.setattr(submodule_models, "Hmr2FeatureExtractor", FakeModel)
    monkeypatch.setattr(submodule_models, "GvhmrMeshRecovery", FakeModel)
    monkeypatch.setattr(submodule_models, "SmplVertexReconstructor", FakeModel)

    module = GVHMRModule(
        make_gvhmr_config(
            tmp_path,
            detector="dino",
            dino_confidence=0.42,
        )
    )
    module.load()

    tracker = constructed["dino_tracker"]
    assert tracker.kwargs == {
        "checkpoint": (tmp_path / "ckpt/dino.pth").resolve(),
        "repository": (tmp_path / "third_party/DINO").resolve(),
        "device": "cpu",
        "allow_device_fallback": False,
        "confidence": 0.42,
        "short_side": 800,
        "max_long_side": 1333,
    }


class _FakeTrackResult:
    track_ids = [11, 22]

    def __init__(self) -> None:
        self.requests: list[int] = []

    def bbx_xys(self, track_id: int, *, base_enlarge: float) -> torch.Tensor:
        assert base_enlarge == 1.2
        self.requests.append(track_id)
        return torch.full((5, 3), float(track_id), dtype=torch.float32)


class _FakeTracker:
    def __init__(self, result: _FakeTrackResult) -> None:
        self.result = result
        self.request: Any | None = None

    def predict(self, request: Any) -> _FakeTrackResult:
        self.request = request
        return self.result


def test_process_runs_direct_chain_and_saves_result(tmp_path: Path) -> None:
    output_path = tmp_path / "gvhmr_result.json"
    config = make_gvhmr_config(
        tmp_path,
        track_selection="auto",
        num_tracks=2,
        save_result=True,
        output_path=output_path,
    )
    module = GVHMRModule(config)
    track_result = _FakeTrackResult()
    tracker = _FakeTracker(track_result)
    module._tracker = cast(Any, tracker)
    module._mesh_model = cast(Any, object())

    def fake_run_track(
        video_path: str | Path,
        track_id: int,
        bbx_xys: torch.Tensor,
    ) -> dict[str, Any]:
        del video_path
        num_frames = int(bbx_xys.shape[0])
        return {
            "track_id": track_id,
            "smpl_body_pose": np.full((num_frames, 63), track_id, dtype=np.float32),
            "smpl_global_orient": np.full((num_frames, 3), track_id, dtype=np.float32),
            "smpl_betas": np.full((10,), track_id, dtype=np.float32),
            "smpl_vertices_local": np.full(
                (num_frames, 4, 3), track_id, dtype=np.float32
            ),
            "human_kp_2d": np.full((num_frames, 17, 2), track_id, dtype=np.float32),
            "human_kp_vis": np.ones((num_frames, 17), dtype=np.float32),
            "bbx_xys": bbx_xys.numpy().astype(np.float32),
        }

    module._run_track = fake_run_track  # type: ignore[method-assign]

    result = module.process(Path("video.mp4"), max_frames=3)

    assert tracker.request is not None
    assert str(tracker.request.video_path) == "video.mp4"
    assert tracker.request.num_tracks == 2
    assert tracker.request.interactive is False
    assert track_result.requests == [11, 22]
    assert result.smpl_body_pose.shape == (2, 3, 63)
    np.testing.assert_array_equal(result.track_ids, np.array([11, 22], dtype=np.int32))
    assert output_path.exists()
    loaded = GVHMRResult.load(output_path)
    np.testing.assert_allclose(loaded.smpl_body_pose, result.smpl_body_pose)
