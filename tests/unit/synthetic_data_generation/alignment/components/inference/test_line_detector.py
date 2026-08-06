"""Tests for path-driven line-model loading and output-pixel coordinates."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from src.synthetic_data_generation.alignment.components.evidence.ground_line_raster import (
    GroundLineMapSettings,
)
from src.synthetic_data_generation.alignment.components.ground.projection import (
    ProjectedLinePixels,
)
from src.synthetic_data_generation.alignment.components.inference import (
    line_detector,
)
from src.synthetic_data_generation.scene_contract import SceneCamera
from src.tasks.base.model_io import bind_model_io
from src.tasks.court_detection.inference import CourtLinePredictor
from src.tasks.court_detection.model_io import (
    CourtLineModelIO,
    CourtLinePrediction,
    CourtModelSpec,
)
from src.tasks.court_detection.models.hierarchical_model import CourtHierarchicalModel
from src.utils.configuration import PathResolver, RuntimePathRoots


def _resolver(root: Path) -> PathResolver:
    absolute_root = root.resolve()
    return PathResolver(
        RuntimePathRoots(
            project_root=absolute_root,
            data_root=absolute_root / "data",
            checkpoint_root=absolute_root,
            artifact_root=absolute_root / "artifacts",
            output_root=absolute_root / "outputs",
            cache_root=absolute_root / ".cache",
            external_asset_root=absolute_root,
        )
    )


class _StaticLineModel(CourtHierarchicalModel):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.in_channels = 3
        self.num_classes = 1

    def forward(
        self,
        image: torch.Tensor,
        feature_1: torch.Tensor | None = None,
        feature_2: torch.Tensor | None = None,
        feature_3: torch.Tensor | None = None,
        feature_4: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del feature_1, feature_2, feature_3, feature_4
        return image.new_zeros(image.shape[0], 1, image.shape[-2], image.shape[-1])


def _line_predictor(*, short_side: int) -> CourtLinePredictor:
    adapter = CourtLineModelIO(
        CourtModelSpec(
            task="line",
            in_channels=3,
            output_channels=1,
            short_side=short_side,
        ),
        bce_weight=1.0,
        dice_weight=1.0,
        pos_weight=1.0,
    )
    return CourtLinePredictor(
        bind_model_io(_StaticLineModel(), adapter),
        torch.device("cpu"),
    )


def _patch_line_checkpoint_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, Any], CourtLinePredictor]:
    embedded = OmegaConf.create(
        {
            "model": {
                "encoder": {
                    "repository_path": "/content/dinov3",
                    "checkpoint_path": "backbone.pth",
                }
            }
        }
    )
    checkpoint = tmp_path / "line.ckpt"
    backbone_repository = tmp_path / "dinov3"
    backbone = backbone_repository / "backbone.pth"
    checkpoint.write_bytes(b"line")
    backbone_repository.mkdir()
    backbone.write_bytes(b"backbone")
    monkeypatch.setattr(
        line_detector.torch,
        "load",
        lambda *_args, **_kwargs: {
            "hyper_parameters": {"config": embedded},
        },
    )
    captured: dict[str, Any] = {}
    predictor = _line_predictor(short_side=256)

    def fake_load_from_checkpoint(
        checkpoint_path: Path,
        **kwargs: Any,
    ) -> CourtLinePredictor:
        captured["checkpoint"] = checkpoint_path
        captured.update(kwargs)
        return predictor

    monkeypatch.setattr(
        line_detector.CourtLinePredictor,
        "load_from_checkpoint",
        fake_load_from_checkpoint,
    )
    return checkpoint, backbone_repository, backbone, captured, predictor


def test_load_detector_uses_real_predictor_adapter_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint, backbone_repository, backbone, captured, predictor = (
        _patch_line_checkpoint_loading(monkeypatch, tmp_path)
    )

    detector = line_detector.load_line_detector(
        checkpoint,
        backbone_repository=backbone_repository,
        backbone_checkpoint=backbone,
        device="cpu",
        expected_short_side=256,
        resolver=_resolver(tmp_path),
    )

    assert detector.embedded_backbone_path == "backbone.pth"
    config = captured["config"]
    assert config["model"]["encoder"]["repository_path"] == str(backbone_repository)
    assert config["model"]["encoder"]["checkpoint_path"] == str(backbone)
    assert detector.predictor is predictor
    assert detector.predictor.adapter.spec.short_side == 256
    assert not hasattr(detector.predictor, "short_side")


def test_load_detector_rejects_adapter_spec_short_side_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint, backbone_repository, backbone, _, _ = (
        _patch_line_checkpoint_loading(monkeypatch, tmp_path)
    )

    with pytest.raises(ValueError, match="expected 512, loaded 256"):
        line_detector.load_line_detector(
            checkpoint,
            backbone_repository=backbone_repository,
            backbone_checkpoint=backbone,
            device="cpu",
            expected_short_side=512,
            resolver=_resolver(tmp_path),
        )


def test_line_pixels_map_output_corners_to_original_corners() -> None:
    probability = np.asarray(
        [
            [0.9, 0.0, 0.8],
            [0.0, 0.0, 0.0],
            [0.7, 0.0, 0.6],
        ],
        dtype=np.float32,
    )

    pixels, scores = line_detector.line_pixels_in_original_image(
        probability,
        original_width=959,
        original_height=539,
        probability_threshold=0.5,
    )

    np.testing.assert_allclose(
        pixels,
        np.asarray(
            [
                [0.0, 0.0],
                [958.0, 0.0],
                [0.0, 538.0],
                [958.0, 538.0],
            ]
        ),
    )
    np.testing.assert_allclose(scores, (0.9, 0.8, 0.7, 0.6))


def test_infer_line_projection_consumes_typed_prediction(monkeypatch: Any) -> None:
    probability = torch.tensor(
        [[0.1, 0.8, 0.2], [0.9, 0.3, 0.7]], dtype=torch.float32
    )

    class FakePredictor:
        def predict(self, _image: np.ndarray) -> CourtLinePrediction:
            return CourtLinePrediction(
                probability=probability,
                logits=torch.zeros_like(probability),
            )

    camera = SceneCamera(
        camera_id="camera_0",
        source_camera_id="source_0",
        image_uri="frame.png",
        source_frame_index=0,
        group_id=0,
        width=6,
        height=4,
        intrinsics=(1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0),
        camera_to_scene=(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ),
    )
    projected = ProjectedLinePixels(
        points_scene=np.empty((0, 3), dtype=np.float64),
        points_uv=np.empty((0, 2), dtype=np.float64),
        probabilities=np.empty(0, dtype=np.float32),
        camera_ranges=np.empty(0, dtype=np.float64),
        proximity_weights=np.empty(0, dtype=np.float64),
        input_count=3,
        invalid_parallel_count=0,
        invalid_behind_count=0,
        invalid_range_count=0,
        invalid_bounds_count=0,
    )
    captured: dict[str, np.ndarray] = {}

    def fake_project(
        _camera: SceneCamera,
        pixels_xy: np.ndarray,
        probabilities: np.ndarray,
        **_kwargs: Any,
    ) -> ProjectedLinePixels:
        captured["pixels"] = pixels_xy
        captured["probabilities"] = probabilities
        return projected

    monkeypatch.setattr(line_detector, "project_line_pixels_to_ground", fake_project)

    observation = line_detector.infer_line_projection(
        np.zeros((4, 6, 3), dtype=np.uint8),
        camera,
        detector=line_detector.LineDetector(
            predictor=cast(Any, FakePredictor()),
            embedded_backbone_path="backbone.pth",
        ),
        plane=cast(Any, object()),
        bounds=(-1.0, 1.0, -1.0, 1.0),
        settings=GroundLineMapSettings(probability_threshold=0.5),
    )

    assert observation.projection is projected
    assert observation.output_width == 3
    assert observation.output_height == 2
    assert observation.selected_line_pixel_count == 3
    np.testing.assert_allclose(captured["probabilities"], (0.8, 0.9, 0.7))
