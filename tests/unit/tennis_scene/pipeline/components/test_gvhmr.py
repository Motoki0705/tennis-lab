"""Tests for GVHMR stage configuration and typed-chain delegation."""

from pathlib import Path

import numpy as np
import pytest

import src.tennis_scene.pipeline.components.gvhmr as gvhmr_component
from src.tennis_scene.pipeline.components.gvhmr import GVHMRModule
from src.tennis_scene.pipeline.model_io.gvhmr import (
    GVHMRChainRequest,
    GVHMRResult,
)
from tests.unit.tennis_scene.pipeline.config_factories import make_gvhmr_config


def _make_result() -> GVHMRResult:
    return GVHMRResult(
        smpl_body_pose=np.zeros((2, 3, 63), dtype=np.float32),
        smpl_global_orient=np.zeros((2, 3, 3), dtype=np.float32),
        smpl_betas=np.zeros((2, 10), dtype=np.float32),
        smpl_vertices_local=np.zeros((2, 3, 4, 3), dtype=np.float32),
        human_kp_2d=np.zeros((2, 3, 17, 2), dtype=np.float32),
        human_kp_vis=np.ones((2, 3, 17), dtype=np.float32),
        bbx_xys=np.ones((2, 3, 3), dtype=np.float32),
        track_ids=np.array([11, 22], dtype=np.int32),
    )


class _FakeChain:
    def __init__(self, result: GVHMRResult) -> None:
        self.result = result
        self.is_loaded = False
        self.requests: list[GVHMRChainRequest] = []

    def load(self) -> None:
        self.is_loaded = True

    def unload(self) -> None:
        self.is_loaded = False

    def predict(self, request: GVHMRChainRequest) -> GVHMRResult:
        self.requests.append(request)
        return self.result


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

    def test_rejects_unknown_detector(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="detector must be"):
            make_gvhmr_config(tmp_path, detector="unknown")


def test_component_does_not_reexport_model_io_result() -> None:
    assert not hasattr(gvhmr_component, "GVHMRResult")


def test_execute_requires_resolved_chain(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a resolved chain"):
        GVHMRModule(make_gvhmr_config(tmp_path), None)


def test_load_source_forbids_inference_chain(tmp_path: Path) -> None:
    config = make_gvhmr_config(
        tmp_path,
        source="load",
        load_path=tmp_path / "result.json",
    )
    with pytest.raises(ValueError, match="forbids an inference chain"):
        GVHMRModule(config, _FakeChain(_make_result()))


def test_process_delegates_to_resolved_chain_and_saves_result(tmp_path: Path) -> None:
    output_path = tmp_path / "gvhmr_result.json"
    config = make_gvhmr_config(
        tmp_path,
        track_selection="auto",
        num_tracks=2,
        save_result=True,
        output_path=output_path,
    )
    expected = _make_result()
    chain = _FakeChain(expected)
    module = GVHMRModule(config, chain)

    result = module.process(Path("video.mp4"), max_frames=3)

    assert result is expected
    assert len(chain.requests) == 1
    request = chain.requests[0]
    assert request.video_path == Path("video.mp4")
    assert request.max_frames == 3
    assert request.num_tracks == 2
    assert request.interactive is False
    assert request.bbox_enlarge == 1.2
    assert output_path.is_file()
    loaded = GVHMRResult.load(output_path)
    np.testing.assert_allclose(loaded.smpl_body_pose, expected.smpl_body_pose)


def test_load_preloads_only_the_resolved_chain(tmp_path: Path) -> None:
    chain = _FakeChain(_make_result())
    module = GVHMRModule(make_gvhmr_config(tmp_path), chain)

    module.load()

    assert module.is_loaded
