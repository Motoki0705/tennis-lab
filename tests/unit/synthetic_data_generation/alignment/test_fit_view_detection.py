"""Tests for fit-only court detection and immutable artifact publication."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.fit_view_detection import (
    CourtKeypointPrediction,
    FitViewDetectionSettings,
    infer_fit_view_court_detections,
    load_fit_view_court_detections,
    load_provider_rgb_image,
    publish_fit_view_court_detections,
)
from src.synthetic_data_generation.provider.bundle import (
    BundleFile,
    ExporterProvenance,
    LoadedSceneProviderBundle,
    ProviderImage,
    ProviderNormalization,
    SceneProviderBundle,
    sha256_file,
)
from src.synthetic_data_generation.scene_contract import ArtifactRef, SceneCamera
from src.tasks.court_detection.evaluation.contracts import (
    HomographyEvaluationCriteria,
)
from src.tasks.court_detection.evaluation.image_evidence import COURT_LINE_EDGES
from src.tasks.court_detection.geometry import court_template_xy, project_points


class _FixedPredictor:
    def __init__(self, keypoints_xy: NDArray[np.float32]) -> None:
        self.keypoints_xy = keypoints_xy
        self.call_count = 0

    def predict_rgb(
        self,
        image_rgb: NDArray[np.uint8],
    ) -> CourtKeypointPrediction:
        assert image_rgb.shape == (180, 320, 3)
        self.call_count += 1
        return CourtKeypointPrediction(
            keypoints_xy=self.keypoints_xy,
            peak_scores=np.full(14, 0.9, dtype=np.float32),
        )


def _camera(camera_id: str, group_id: int) -> SceneCamera:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(group_id)
    return SceneCamera(
        camera_id=camera_id,
        source_camera_id="colmap-1",
        image_uri=f"images/{camera_id}.png",
        source_frame_index=group_id * 32,
        group_id=group_id,
        width=320,
        height=180,
        intrinsics=(250.0, 0.0, 160.0, 0.0, 250.0, 90.0, 0.0, 0.0, 1.0),
        camera_to_scene=tuple(float(value) for value in pose.ravel()),
    )


def _bundle_file(path: Path, root: Path) -> BundleFile:
    return BundleFile(
        relative_path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _make_bundle(
    root: Path,
) -> tuple[LoadedSceneProviderBundle, NDArray[np.float32]]:
    width, height = 320, 180
    homography = np.asarray(
        [
            [0.040, 0.002, 0.50],
            [0.002, -0.025, 0.48],
            [0.002, -0.012, 1.00],
        ],
        dtype=np.float32,
    )
    projected = project_points(court_template_xy(), homography)
    keypoints_xy = projected * np.asarray(
        [width - 1, height - 1],
        dtype=np.float32,
    )
    image_bgr: NDArray[np.uint8] = np.zeros((height, width, 3), dtype=np.uint8)
    for first, second in COURT_LINE_EDGES:
        cv2.line(
            image_bgr,
            tuple(np.rint(keypoints_xy[first]).astype(int)),
            tuple(np.rint(keypoints_xy[second]).astype(int)),
            (255, 255, 255),
            3,
        )

    image_dir = root / "images"
    image_dir.mkdir(parents=True)
    cameras = tuple(_camera(f"frame_{group:06d}", group) for group in range(4))
    images: list[ProviderImage] = []
    for camera in cameras:
        path = root / camera.image_uri
        assert cv2.imwrite(str(path), image_bgr)
        images.append(
            ProviderImage(
                camera_id=camera.camera_id,
                source_image_name=f"{camera.camera_id}.jpg",
                file=_bundle_file(path, root),
            )
        )

    point_cloud_path = root / "points_scene.npy"
    np.save(point_cloud_path, np.zeros((4, 3), dtype=np.float64))
    normalization_matrix = np.eye(4, dtype=np.float64)
    normalization = ProviderNormalization(
        scene_from_source_world=tuple(
            float(value) for value in normalization_matrix.ravel()
        ),
        sha256=hashlib.sha256(normalization_matrix.tobytes()).hexdigest(),
    )
    manifest = SceneProviderBundle.create(
        bundle_id="test-bundle-v1",
        provider_backend="test@1",
        source_artifacts=(
            ArtifactRef(
                artifact_id="checkpoint",
                uri="artifact://test/checkpoint",
                sha256="a" * 64,
                size_bytes=1,
            ),
        ),
        cameras=cameras,
        images=images,
        point_cloud=_bundle_file(point_cloud_path, root),
        point_cloud_shape=(4, 3),
        normalization=normalization,
        exporter=ExporterProvenance(
            git_revision="deadbeef",
            git_dirty=True,
            code_sha256="b" * 64,
            command="python -m test",
            python_version="3.11",
            numpy_version="2.3",
            opencv_version="4.13",
            geometry_python_version="3.12",
            geometry_numpy_version="2.4",
            geometry_pycolmap_version="4.1",
        ),
    )
    return LoadedSceneProviderBundle(root=root, manifest=manifest), keypoints_xy


def _settings() -> FitViewDetectionSettings:
    return FitViewDetectionSettings(
        artifact_id="fit-courts-v1",
        holdout_group_ids=(1, 3),
        min_peak_score=0.3,
        min_confident_keypoints=12,
        homography=HomographyEvaluationCriteria(
            min_court_area_ratio=0.001,
            min_line_edge_support=0.5,
            require_ground_view=False,
        ),
    )


def test_inference_never_reads_or_predicts_holdout_images(tmp_path: Path) -> None:
    bundle, keypoints_xy = _make_bundle(tmp_path)
    predictor = _FixedPredictor(keypoints_xy)
    read_camera_ids: list[str] = []

    def guarded_loader(path: Path) -> NDArray[np.uint8]:
        camera_id = path.stem
        if camera_id in {"frame_000001", "frame_000003"}:
            raise AssertionError(f"Holdout image was read: {camera_id}")
        read_camera_ids.append(camera_id)
        return load_provider_rgb_image(path)

    artifact = infer_fit_view_court_detections(
        bundle,
        predictor,
        settings=_settings(),
        detector={"checkpoint_sha256": "c" * 64},
        provenance={"git_revision": "deadbeef"},
        created_at_utc="2026-07-25T00:00:00+00:00",
        image_loader=guarded_loader,
    )

    assert read_camera_ids == ["frame_000000", "frame_000002"]
    assert predictor.call_count == 2
    assert artifact["split"]["holdout_inference_status"] == "not_run"
    assert artifact["split"]["holdout_camera_ids"] == [
        "frame_000001",
        "frame_000003",
    ]
    assert artifact["summary"]["accepted_count"] == 2


def test_artifact_round_trip_refuses_overwrite_and_detects_tampering(
    tmp_path: Path,
) -> None:
    bundle, keypoints_xy = _make_bundle(tmp_path / "provider")
    artifact = infer_fit_view_court_detections(
        bundle,
        _FixedPredictor(keypoints_xy),
        settings=_settings(),
        detector={"checkpoint_sha256": "c" * 64},
        provenance={"git_revision": "deadbeef"},
        created_at_utc="2026-07-25T00:00:00+00:00",
    )
    path = publish_fit_view_court_detections(
        artifact,
        output_dir=tmp_path / "artifacts",
    )

    loaded = load_fit_view_court_detections(path, bundle=bundle)
    assert loaded == artifact
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_fit_view_court_detections(
            artifact,
            output_dir=tmp_path / "artifacts",
        )

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["summary"]["accepted_count"] = 0
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_fit_view_court_detections(tampered_path)


def test_settings_reject_missing_or_duplicate_holdout_policy() -> None:
    settings = _settings()
    with pytest.raises(ValueError, match="non-empty and unique"):
        replace(settings, holdout_group_ids=())
    with pytest.raises(ValueError, match="non-empty and unique"):
        replace(settings, holdout_group_ids=(1, 1))
    with pytest.raises(ValueError, match="path-safe"):
        replace(settings, artifact_id="../escape")
