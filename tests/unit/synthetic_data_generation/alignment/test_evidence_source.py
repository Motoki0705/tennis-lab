"""Tests for fail-closed measured alignment evidence preflight."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.alignment.evidence_source import (
    MeasuredAlignmentEvidenceSource,
    _resolve_primary_scale,
)
from src.synthetic_data_generation.alignment.settings import (
    AlignmentEvidenceSettings,
    CorrespondenceSettings,
    CourtCandidateFitSettings,
    CourtLineArchitectureSettings,
    CourtLineModelSettings,
    GroundPlaneSettings,
    LineProjectionSettings,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera


@dataclass
class _Detector:
    error: Exception | None = None
    preflight_calls: int = 0

    def preflight(self) -> None:
        self.preflight_calls += 1
        if self.error is not None:
            raise self.error

    def predict_probability(
        self,
        image_rgb: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        return np.ones(image_rgb.shape[:2], dtype=np.float32)


def test_measured_source_preflight_checks_real_images_and_detector(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path, camera_count=4)
    detector = _Detector()
    source = MeasuredAlignmentEvidenceSource(_settings(tmp_path), detector)

    source.preflight(scene)

    assert detector.preflight_calls == 1


def test_measured_source_preflight_fails_before_detector_when_partitions_unavailable(
    tmp_path: Path,
) -> None:
    scene = _scene(tmp_path, camera_count=2)
    detector = _Detector()
    source = MeasuredAlignmentEvidenceSource(_settings(tmp_path), detector)

    with pytest.raises(ValueError, match="independent alignment camera partitions"):
        source.preflight(scene)
    assert detector.preflight_calls == 0


def test_measured_source_has_no_detector_fallback(tmp_path: Path) -> None:
    scene = _scene(tmp_path, camera_count=4)
    detector = _Detector(error=RuntimeError("trained detector unavailable"))
    source = MeasuredAlignmentEvidenceSource(_settings(tmp_path), detector)

    with pytest.raises(RuntimeError, match="trained detector unavailable"):
        source.preflight(scene)
    assert detector.preflight_calls == 1


def test_primary_hypothesis_is_the_metric_scale_authority() -> None:
    scale, maximum_deviation = _resolve_primary_scale(
        np.asarray((0.076, 0.058), dtype=np.float64),
        maximum_relative_deviation=0.3,
    )

    assert scale == pytest.approx(0.076)
    assert maximum_deviation == pytest.approx(abs(0.058 / 0.076 - 1.0))


def _scene(tmp_path: Path, *, camera_count: int) -> StandardSceneExport:
    export = tmp_path / "export"
    images = export / "images"
    model = export / "model"
    images.mkdir(parents=True)
    model.mkdir()
    cameras: list[SceneCamera] = []
    for index in range(camera_count):
        image_path = images / f"camera-{index}.png"
        Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(image_path)
        cameras.append(
            SceneCamera(
                camera_id=f"camera-{index}",
                source_frame_index=index,
                width=12,
                height=8,
                intrinsics=(8.0, 0.0, 6.0, 0.0, 8.0, 4.0, 0.0, 0.0, 1.0),
                camera_to_scene=RigidTransform.identity(),
                image_path=str(image_path.resolve()),
            )
        )
    points = np.asarray(
        [
            [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return StandardSceneExport(
        scene_id="scene-a",
        export_root=export,
        scene_path=export / "scene.json",
        cameras=tuple(cameras),
        points_scene=points,
        scene_from_sfm=tuple(float(value) for value in np.eye(4).ravel()),
        sfm_from_scene=tuple(float(value) for value in np.eye(4).ravel()),
        checkpoint_path=model / "checkpoint.pt",
        runtime_config_path=model / "config.json",
    )


def _settings(tmp_path: Path) -> AlignmentEvidenceSettings:
    architecture = CourtLineArchitectureSettings(
        backbone_name="dinov3_vitb16",
        backbone_strict=True,
        backbone_train_mode="frozen",
        backbone_last_n_blocks=0,
        backbone_out_indices=(2, 5, 8, 11),
        backbone_layer_mode="uniform",
        lora_enabled=True,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        lora_target_modules=("qkv", "proj", "fc1", "fc2"),
        decoder_channels=256,
        decoder_reassemble_factors=(4.0, 2.0, 1.0, 0.5),
        line_bce_weight=1.0,
        line_dice_weight=1.0,
        line_positive_weight=8.0,
    )
    return AlignmentEvidenceSettings(
        seed=42,
        fit_fraction=2.0 / 3.0,
        holdout_fraction=1.0 / 3.0,
        minimum_fit_cameras=2,
        minimum_holdout_cameras=1,
        maximum_cameras=3,
        line_model=CourtLineModelSettings(
            checkpoint_path=(tmp_path / "line.ckpt").resolve(),
            backbone_repository_path=(tmp_path / "dinov3").resolve(),
            backbone_checkpoint_path=(tmp_path / "backbone.pth").resolve(),
            device="cpu",
            expected_short_side=256,
            probability_threshold=0.5,
            maximum_selected_pixels_per_camera=100,
            architecture=architecture,
        ),
        ground_plane=GroundPlaneSettings(
            footprint_quantile=0.0,
            footprint_margin=1.0,
            minimum_camera_height=0.1,
            maximum_camera_height=2.0,
            histogram_bin_width=0.1,
            candidate_half_width=0.2,
            ransac_threshold=0.1,
            refine_threshold=0.1,
            ransac_iterations=10,
            ransac_sample_limit=4,
            refine_iterations=1,
            minimum_candidate_points=3,
            minimum_support_points=3,
            minimum_normal_up_cosine=0.9,
            minimum_positive_camera_fraction=0.5,
            support_bounds_quantile=0.0,
        ),
        projection=LineProjectionSettings(
            minimum_ray_plane_cosine=0.01,
            maximum_ray_distance=10.0,
            bounds_margin=1.0,
            minimum_projected_points_per_camera=3,
        ),
        candidate_fit=CourtCandidateFitSettings(
            candidate_count=1,
            samples_per_metre=2.0,
            minimum_nht_scene_units_per_metre=0.05,
            maximum_nht_scene_units_per_metre=0.1,
            orientation_minimum_radians=-1.0,
            orientation_maximum_radians=1.0,
            score_distance_metres=0.25,
            minimum_template_score=0.1,
            family_orientation_tolerance_radians=0.1,
            family_scale_relative_tolerance=0.1,
            minimum_center_separation_metres=10.0,
            separation_penalty=10.0,
            optimizer_maximum_iterations=2,
            optimizer_population_size=2,
            optimizer_tolerance=1.0e-4,
            maximum_fit_points=100,
            common_scale_relative_tolerance=0.1,
        ),
        correspondences=CorrespondenceSettings(
            maximum_match_distance_metres=0.25,
            maximum_correspondences_per_camera=20,
            minimum_correspondences_per_camera=3,
        ),
    )
