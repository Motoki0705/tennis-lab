"""Integration coverage for complete deterministic publication bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

import src.synthetic_data_generation.visualization.publication.bundle as bundle_module
from src.synthetic_data_generation.alignment.contracts import (
    AlignmentEvidence,
    AlignmentResult,
    AlignmentTrace,
    AlignmentTraceCandidateState,
    AlignmentTracePhase,
    AlignmentTraceStep,
    GroundPlaneFrame,
)
from src.synthetic_data_generation.alignment.heatmaps import (
    AlignmentLineHeatmaps,
    AlignmentLineHeatmapView,
)
from src.synthetic_data_generation.dataset.blcs.contracts import BLCS_DATASET_SCHEMA
from src.synthetic_data_generation.dataset.plcs.assembler import PLCS_DATASET_SCHEMA
from src.synthetic_data_generation.dataset.runtime import (
    LogicalRenderSample,
    RenderSampleKey,
)
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)
from src.synthetic_data_generation.visualization.publication import (
    generate_publication_bundle,
    validate_publication_bundle,
    validate_publication_bundle_structure_only,
)
from src.synthetic_data_generation.visualization.publication.alignment import (
    ALIGNMENT_AGREEMENT_METRIC_SCHEMA,
    AlignmentPublicationData,
)
from src.synthetic_data_generation.visualization.publication.cameras import (
    PublicationCameraCollection,
)
from src.synthetic_data_generation.visualization.publication.contracts import (
    REQUIRED_PUBLICATION_ARTIFACTS,
    PublicationArtifactName,
    PublicationArtifactRecord,
    PublicationBundleResult,
    PublicationDrawingSettings,
    PublicationRequest,
)
from src.synthetic_data_generation.visualization.publication.figures import (
    overview_panel_bounds,
)
from src.synthetic_data_generation.visualization.sources import (
    BLCSSourceFrame,
    CourtSourceFrame,
    PLCSSourceFrame,
)

_SCENE_ID = "scene-0"
_LOGICAL_SCENE_ID = "logical-0"
_CAMERA_IDS = ("camera-0", "camera-1")
_CAPTURED_CAMERA_IDS = tuple(f"captured-{index:03d}" for index in range(491))
_FRAME_INDICES = (0, 2)
_DATASET_SIZE = (240, 135)
_ALIGNMENT_SIZE = (320, 240)
_FIGURE_SIZE = (320, 240)
_OVERVIEW_SIZE = (1200, 700)
_GIF_DURATION_MS = 40


def _logical_render(frame_index: int) -> LogicalRenderSample:
    rgb: NDArray[np.float32] = np.zeros((180, 320, 3), dtype=np.float32)
    rgb[..., 1] = 0.15 + frame_index * 0.1
    instance_ids: NDArray[np.int32] = np.zeros((180, 320), dtype=np.int32)
    instance_ids.reshape(-1)[:24] = 1
    return LogicalRenderSample(
        key=RenderSampleKey(frame_index, "camera-0"),
        rgb=rgb,
        alpha=np.ones((180, 320, 1), dtype=np.float32),
        depth=np.ones((180, 320, 1), dtype=np.float32),
        instance_ids=instance_ids,
    )


def _court_projection() -> Mapping[str, object]:
    names = (
        "doubles_left",
        "doubles_right",
        "singles_left",
        "singles_right",
        "service_left",
        "service_right",
        "service_t",
    )
    return {
        "courts": [
            {
                "court_instance_id": "court-0",
                "coverage_mode": "full",
                "classes": [
                    {
                        "class_id": class_id,
                        "class_name": name,
                        "renderer_visible": True,
                        "points": [
                            {
                                "uv": [20.0 + class_id * 12.0, 80.0],
                                "in_frame": True,
                                "renderer_visible": True,
                            },
                            {
                                "uv": [20.0 + class_id * 12.0, 145.0],
                                "in_frame": True,
                                "renderer_visible": True,
                            },
                        ],
                    }
                    for class_id, name in enumerate(names)
                ],
            }
        ]
    }


class _CourtSource:
    dataset_schema = "canonical_court_dataset_v1"
    dataset_scene_id = _SCENE_ID
    width = 320
    height = 180
    frame_count = 3
    frame_order = tuple(
        {
            "sample_id": f"sample-{index}",
            "view_id": "view-0",
            "trajectory_frame_index": index,
        }
        for index in range(frame_count)
    )

    def frames(self) -> Iterator[CourtSourceFrame]:
        for index in range(self.frame_count):
            rgb: NDArray[np.float32] = np.zeros((180, 320, 3), dtype=np.float32)
            rgb[..., 0] = 0.12 * index
            yield CourtSourceFrame(
                rgb=rgb,
                sample_id=f"sample-{index}",
                view_id="view-0",
                trajectory_frame_index=index,
                projection=_court_projection(),
            )


class _FailingCourtSource(_CourtSource):
    def frames(self) -> Iterator[CourtSourceFrame]:
        frames = super().frames()
        yield next(frames)
        raise ValueError("synthetic corrupt court frame")


class _BLCSSource:
    dataset_schema = BLCS_DATASET_SCHEMA
    dataset_scene_id = _SCENE_ID
    width = 320
    height = 180
    frame_count = 3
    source_fps = 60.0
    object_ids = ("ball-0",)
    court_kp: NDArray[np.float32] = np.zeros((20, 2), dtype=np.float32)
    court_vis: NDArray[np.bool_] = np.zeros((20,), dtype=np.bool_)
    frame_order = tuple(
        {"source_frame_index": index, "global_frame_index": 20 + index}
        for index in range(frame_count)
    )

    def frames(self) -> Iterator[BLCSSourceFrame]:
        for index in range(self.frame_count):
            yield BLCSSourceFrame(
                render=_logical_render(index),
                source_frame_index=index,
                global_frame_index=20 + index,
                metadata={
                    "objects": [
                        {
                            "object_id": "ball-0",
                            "instance_id": 1,
                            "present": True,
                            "geometric_visible": True,
                            "rendered_visible": True,
                        }
                    ],
                    "semantic_arrays": {
                        "ball_uv": [[100.0 + index * 12.0, 95.0]],
                        "present": [True],
                        "geometric_visible": [True],
                        "rendered_visible": [True],
                        "instance_ids": [1],
                    },
                },
            )


class _PLCSSource:
    dataset_schema = PLCS_DATASET_SCHEMA
    dataset_scene_id = _SCENE_ID
    width = 320
    height = 180
    frame_count = 3
    object_ids = ("person-0",)
    frame_order = tuple({"frame_index": index} for index in range(frame_count))

    def frames(self) -> Iterator[PLCSSourceFrame]:
        for index in range(self.frame_count):
            keypoints: NDArray[np.float32] = np.zeros((1, 17, 2), dtype=np.float32)
            keypoints[0, 5] = (0.4 + index * 0.02, 0.4)
            keypoints[0, 6] = (0.6 + index * 0.02, 0.4)
            visible: NDArray[np.bool_] = np.zeros((1, 17), dtype=np.bool_)
            visible[0, 5:7] = True
            yield PLCSSourceFrame(
                render=_logical_render(index),
                frame_index=index,
                label={
                    "objects": [
                        {
                            "object_id": "person-0",
                            "instance_id": 1,
                            "present": True,
                            "visible_pixel_count": 24,
                        }
                    ]
                },
                human_kp=keypoints,
                human_vis=visible,
                court_kp=np.zeros((20, 2), dtype=np.float32),
                court_vis=np.zeros((20,), dtype=np.bool_),
                present=np.ones((1,), dtype=np.bool_),
            )


def _alignment_data() -> AlignmentPublicationData:
    identity = RigidTransform.identity()
    court = CourtInstance(
        court_instance_id="court-0",
        candidate_id="candidate-0",
        scene_from_court=identity,
        court_from_scene=identity,
        fit_status="accepted",
        fit_metrics={},
        holdout_status="accepted",
        holdout_metrics={},
    )
    layout = MultiCourtLayout(
        courts=(court,),
        complex_bounds_scene=(-8.0, -14.0, -1.0, 8.0, 14.0, 8.0),
        primary_court_instance_id="court-0",
    )
    phases = tuple(AlignmentTracePhase)
    trace = AlignmentTrace(
        final_candidate_ids=("candidate-0",),
        steps=tuple(
            AlignmentTraceStep(
                step_index=index,
                phase=phase,
                candidates=(
                    AlignmentTraceCandidateState(
                        candidate_id="candidate-0",
                        center_uv_metres=(index * 0.05, -index * 0.05),
                        orientation_radians=index * 0.01,
                        nht_scene_units_per_metre=1.0,
                        template_score=0.7 + index * 0.05,
                        orientation_band_index=0,
                        center_tile_index=0,
                        residual_point_count_before_suppression=20,
                        residual_point_count_after_suppression=10,
                    ),
                ),
                score_sum=0.7 + index * 0.05,
            )
            for index, phase in enumerate(phases)
        ),
    )
    plane = GroundPlaneFrame(
        origin_metric_scene=(0.0, 0.0, 0.0),
        basis_u_metric_scene=(1.0, 0.0, 0.0),
        basis_v_metric_scene=(0.0, 1.0, 0.0),
        normal_metric_scene=(0.0, 0.0, 1.0),
        bounds_uv_metres=(-15.0, 15.0, -20.0, 20.0),
    )
    projected = np.asarray([(-5.0, -10.0), (0.0, 0.0), (5.0, 10.0)], dtype=np.float64)
    probabilities = np.asarray((0.8, 0.9, 0.7), dtype=np.float32)
    heatmaps = AlignmentLineHeatmaps(
        bounds_uv=plane.bounds_uv_metres,
        grid_spacing=1.0,
        proximity_scale=5.0,
        proximity_power=2.0,
        views=(
            AlignmentLineHeatmapView(
                camera_id="camera-0",
                probability=np.full((8, 8), 0.75, dtype=np.float32),
                points_uv=projected,
                projected_probabilities=probabilities,
                proximity_weights=np.ones(3, dtype=np.float64),
                included_in_aggregate=True,
            ),
        ),
    )
    metrics: Mapping[str, object] = {
        "schema": ALIGNMENT_AGREEMENT_METRIC_SCHEMA,
        "court_line_sample_count": 128,
        "projected_evidence_point_count": 3,
        "court_line_mean_probability": 0.8,
        "court_line_probability_q50": 0.8,
        "court_line_coverage_fraction_at_0_5": 1.0,
        "projected_evidence_nearest_court_mean_metres": 0.05,
        "projected_evidence_nearest_court_q95_metres": 0.1,
        "ground_plane_binding_max_abs_error_metres": 0.0,
    }
    return AlignmentPublicationData(
        result=cast(AlignmentResult, SimpleNamespace(layout=layout)),
        evidence=cast(
            AlignmentEvidence,
            SimpleNamespace(
                alignment_trace=trace,
                ground_plane_frame=plane,
            ),
        ),
        heatmaps=heatmaps,
        court_segments_uv=(
            np.asarray(((-5.0, -10.0), (5.0, -10.0)), dtype=np.float64),
            np.asarray(((-5.0, 10.0), (5.0, 10.0)), dtype=np.float64),
        ),
        court_line_samples_uv=projected,
        projected_evidence_uv=projected,
        projected_probabilities=probabilities,
        metrics=metrics,
    )


def _camera(camera_id: str, *, x: float, y: float) -> SceneCamera:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (x, y, 7.0)
    return SceneCamera(
        camera_id=camera_id,
        source_frame_index=int(x + 2.0),
        width=128,
        height=96,
        intrinsics=(90.0, 0.0, 64.0, 0.0, 90.0, 48.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(matrix),
        image_path=f"images/{camera_id}.png",
    )


def _camera_collection(
    owner: str,
    *,
    y_offset: float,
) -> PublicationCameraCollection:
    camera_ids = _CAPTURED_CAMERA_IDS if owner == "reconstruction" else _CAMERA_IDS
    cameras = tuple(
        _camera(
            camera_id,
            x=float(index) * (0.05 if owner == "reconstruction" else 2.0),
            y=(
                y_offset + float(np.sin(index / 20.0))
                if owner == "reconstruction"
                else y_offset + index
            ),
        )
        for index, camera_id in enumerate(camera_ids)
    )
    return PublicationCameraCollection(
        owner=owner,
        schema=f"{owner}_fixture_v1",
        scene_id=_SCENE_ID,
        logical_scene_id=None if owner == "reconstruction" else _LOGICAL_SCENE_ID,
        camera_ids=camera_ids,
        cameras=cameras,
        camera_to_metric_scene=np.stack(
            [camera.camera_to_scene.matrix() for camera in cameras]
        ),
    )


def _loaded_inputs(*, court_source: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        alignment=_alignment_data(),
        court_source=_CourtSource() if court_source is None else court_source,
        blcs_source=_BLCSSource(),
        plcs_source=_PLCSSource(),
        captured_cameras=_camera_collection("reconstruction", y_offset=-5.0),
        blcs_cameras=_camera_collection("blcs", y_offset=0.0),
        plcs_cameras=_camera_collection("plcs", y_offset=0.0),
    )


def _request(scene_root: Path, output: Path) -> PublicationRequest:
    return PublicationRequest(
        scene_id=_SCENE_ID,
        scene_root=scene_root,
        output_bundle=output,
        artifact_names=REQUIRED_PUBLICATION_ARTIFACTS,
        court_trajectory_id="trajectory-0",
        court_frame_indices=_FRAME_INDICES,
        blcs_logical_scene_id=_LOGICAL_SCENE_ID,
        blcs_camera_id="camera-0",
        blcs_frame_indices=_FRAME_INDICES,
        blcs_camera_ids=_CAMERA_IDS,
        plcs_logical_scene_id=_LOGICAL_SCENE_ID,
        plcs_camera_id="camera-0",
        plcs_frame_indices=_FRAME_INDICES,
        plcs_camera_ids=_CAMERA_IDS,
        captured_camera_ids=_CAPTURED_CAMERA_IDS,
        drawing=PublicationDrawingSettings(
            dataset_size=_DATASET_SIZE,
            alignment_size=_ALIGNMENT_SIZE,
            figure_size=_FIGURE_SIZE,
            overview_size=_OVERVIEW_SIZE,
            gif_duration_ms=_GIF_DURATION_MS,
            frustum_depth_metres=1.5,
            line_width=1.0,
            font_size=8,
            history_frames=2,
            maximum_rendered_captured_cameras=24,
            coincident_centre_tolerance_metres=1.0e-6,
            coincident_forward_angle_tolerance_degrees=1.0e-6,
            maximum_artifact_bytes=5_000_000,
            maximum_bundle_bytes=20_000_000,
        ),
    )


def _scene_root(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "scenes" / _SCENE_ID
    root.mkdir(parents=True)
    return root


def _record_by_name(
    result: PublicationBundleResult,
) -> dict[PublicationArtifactName, PublicationArtifactRecord]:
    manifest = result.manifest
    return {record.file_name: record for record in manifest.artifacts}


@pytest.fixture(scope="module")
def authoritative_bundle_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Path, PublicationRequest]]:
    tmp_path = tmp_path_factory.mktemp("authoritative-publication")
    scene_root = _scene_root(tmp_path)
    loaded = _loaded_inputs()
    patch = pytest.MonkeyPatch()
    patch.setattr(bundle_module, "_load_inputs", lambda _request: loaded)
    request = _request(scene_root, tmp_path / "publication-authoritative")
    result = generate_publication_bundle(request)
    try:
        yield result.bundle_path, request
    finally:
        patch.undo()


def _tamper_bundle(
    source_bundle: Path,
    destination: Path,
) -> tuple[Path, dict[str, object]]:
    bundle = shutil.copytree(source_bundle, destination)
    manifest_path = bundle / "manifest.json"
    payload = cast(
        dict[str, object], json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    return bundle, payload


def _write_manifest_with_recomputed_media_digests(
    bundle: Path,
    payload: dict[str, object],
) -> None:
    artifacts = cast(list[dict[str, object]], payload["artifacts"])
    for artifact in artifacts:
        path = bundle / cast(str, artifact["file_name"])
        artifact["byte_size"] = path.stat().st_size
        artifact["content_digest_blake2b_256"] = hashlib.blake2b(
            path.read_bytes(), digest_size=32
        ).hexdigest()
    asset_policy = cast(dict[str, object], payload["asset_policy"])
    asset_policy["artifact_bytes"] = sum(
        cast(int, artifact["byte_size"]) for artifact in artifacts
    )
    (bundle / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _artifact_payload(
    payload: dict[str, object],
    file_name: PublicationArtifactName,
) -> dict[str, object]:
    return next(
        artifact
        for artifact in cast(list[dict[str, object]], payload["artifacts"])
        if artifact["file_name"] == file_name.value
    )


def _resize_artifact_and_rebind_record(
    bundle: Path,
    payload: dict[str, object],
    artifact_name: PublicationArtifactName,
) -> None:
    artifact = _artifact_payload(payload, artifact_name)
    target_size = (cast(int, artifact["width"]) + 1, cast(int, artifact["height"]))
    path = bundle / artifact_name.value
    with Image.open(path) as image:
        image_format = image.format
        duration_ms = image.info.get("duration")
        frames = []
        for index in range(image.n_frames):
            image.seek(index)
            frames.append(
                image.convert("RGB").resize(
                    target_size,
                    resample=Image.Resampling.NEAREST,
                )
            )
    if image_format == "GIF":
        frames[0].save(
            path,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            disposal=2,
            optimize=False,
        )
    elif image_format == "PNG":
        frames[0].save(path, format="PNG", optimize=False, compress_level=9)
    else:
        raise AssertionError(f"Unexpected publication media format: {image_format}")
    artifact["width"] = target_size[0]
    artifact["height"] = target_size[1]
    dataset_domain = {
        PublicationArtifactName.DATASET_COURT: "court",
        PublicationArtifactName.DATASET_BLCS: "blcs",
        PublicationArtifactName.DATASET_PLCS: "plcs",
    }.get(artifact_name)
    if dataset_domain is not None:
        source_owners = cast(dict[str, dict[str, object]], payload["source_owners"])
        source_owners[dataset_domain]["output_size"] = list(target_size)


def test_authoritative_validator_rejects_missing_request_context(
    authoritative_bundle_fixture: tuple[Path, PublicationRequest],
) -> None:
    source_bundle, _request = authoritative_bundle_fixture

    with pytest.raises(TypeError, match="requires expected_request"):
        validate_publication_bundle(
            source_bundle,
            expected_request=cast(PublicationRequest, None),
        )


@pytest.mark.parametrize(
    ("owner_name", "field", "replacement"),
    [
        ("court", "owner_path", "evil"),
        ("court", "domain", "blcs"),
        ("court", "schema", "evil-schema"),
        ("court", "source_fps", 99.0),
        ("court", "source_size", [1, 1]),
        ("court", "trajectory_id", "wrong-trajectory"),
        ("blcs", "owner_path", "evil"),
        ("blcs", "domain", "court"),
        ("blcs", "schema", "evil-schema"),
        ("blcs", "source_fps", 99.0),
        ("blcs", "source_size", [1, 1]),
        ("plcs", "owner_path", "evil"),
        ("plcs", "domain", "court"),
        ("plcs", "schema", "evil-schema"),
        ("plcs", "source_fps", 99.0),
        ("plcs", "source_size", [1, 1]),
        ("alignment", "owner_path", "evil"),
        ("alignment", "schema", "evil-schema"),
    ],
)
def test_authoritative_validator_rejects_source_owner_rebinding(
    authoritative_bundle_fixture: tuple[Path, PublicationRequest],
    tmp_path: Path,
    owner_name: str,
    field: str,
    replacement: object,
) -> None:
    source_bundle, request = authoritative_bundle_fixture
    bundle, payload = _tamper_bundle(
        source_bundle, tmp_path / f"{owner_name}-{field}"
    )
    source_owners = cast(dict[str, dict[str, object]], payload["source_owners"])
    source_owners[owner_name][field] = replacement
    _write_manifest_with_recomputed_media_digests(bundle, payload)

    validate_publication_bundle_structure_only(bundle)
    with pytest.raises(ValueError, match="validated request sources"):
        validate_publication_bundle(bundle, expected_request=request)


@pytest.mark.parametrize(
    ("domain", "artifact", "mapping_field", "replacement"),
    [
        (
            "court",
            PublicationArtifactName.DATASET_COURT,
            "sample_id",
            "wrong-sample",
        ),
        (
            "blcs",
            PublicationArtifactName.DATASET_BLCS,
            "global_frame_index",
            999_999,
        ),
        (
            "plcs",
            PublicationArtifactName.DATASET_PLCS,
            "frame_index",
            999_999,
        ),
    ],
)
def test_authoritative_validator_rejects_dataset_source_record_rebinding(
    authoritative_bundle_fixture: tuple[Path, PublicationRequest],
    tmp_path: Path,
    domain: str,
    artifact: PublicationArtifactName,
    mapping_field: str,
    replacement: object,
) -> None:
    source_bundle, request = authoritative_bundle_fixture
    bundle, payload = _tamper_bundle(source_bundle, tmp_path / domain)
    mapping = cast(list[dict[str, object]], _artifact_payload(payload, artifact)["mapping"])
    mapping[-1][mapping_field] = replacement
    _write_manifest_with_recomputed_media_digests(bundle, payload)

    validate_publication_bundle_structure_only(bundle)
    with pytest.raises(ValueError, match="validated dataset source"):
        validate_publication_bundle(bundle, expected_request=request)


@pytest.mark.parametrize(
    ("domain", "artifact"),
    [
        ("court", PublicationArtifactName.DATASET_COURT),
        ("blcs", PublicationArtifactName.DATASET_BLCS),
        ("plcs", PublicationArtifactName.DATASET_PLCS),
    ],
)
def test_authoritative_validator_rejects_self_consistent_selection_rebinding(
    authoritative_bundle_fixture: tuple[Path, PublicationRequest],
    tmp_path: Path,
    domain: str,
    artifact: PublicationArtifactName,
) -> None:
    source_bundle, request = authoritative_bundle_fixture
    bundle, payload = _tamper_bundle(source_bundle, tmp_path / domain)
    source_owners = cast(dict[str, dict[str, object]], payload["source_owners"])
    owner = source_owners[domain]
    owner["source_count"] = 2
    owner["selected_indices"] = [0, 1]
    mapping = cast(list[dict[str, object]], _artifact_payload(payload, artifact)["mapping"])
    mapping[-1]["source_index"] = 1
    _write_manifest_with_recomputed_media_digests(bundle, payload)

    validate_publication_bundle_structure_only(bundle)
    with pytest.raises(ValueError, match="validated request sources"):
        validate_publication_bundle(bundle, expected_request=request)


@pytest.mark.parametrize(
    ("domain", "artifact"),
    [
        ("blcs", PublicationArtifactName.DATASET_BLCS),
        ("plcs", PublicationArtifactName.DATASET_PLCS),
    ],
)
def test_authoritative_validator_rejects_gif_camera_rebinding(
    authoritative_bundle_fixture: tuple[Path, PublicationRequest],
    tmp_path: Path,
    domain: str,
    artifact: PublicationArtifactName,
) -> None:
    source_bundle, request = authoritative_bundle_fixture
    bundle, payload = _tamper_bundle(source_bundle, tmp_path / domain)
    source_owners = cast(dict[str, dict[str, object]], payload["source_owners"])
    source_owners[domain]["gif_camera_id"] = "camera-1"
    mapping = cast(list[dict[str, object]], _artifact_payload(payload, artifact)["mapping"])
    for record in mapping:
        record["camera_id"] = "camera-1"
    _write_manifest_with_recomputed_media_digests(bundle, payload)

    validate_publication_bundle_structure_only(bundle)
    with pytest.raises(ValueError, match="validated request sources"):
        validate_publication_bundle(bundle, expected_request=request)


@pytest.mark.parametrize(
    ("domain", "dataset_artifact", "camera_artifact"),
    [
        (
            "blcs",
            PublicationArtifactName.DATASET_BLCS,
            PublicationArtifactName.BLCS_CAMERA_LAYOUT,
        ),
        (
            "plcs",
            PublicationArtifactName.DATASET_PLCS,
            PublicationArtifactName.PLCS_CAMERA_LAYOUT,
        ),
    ],
)
def test_authoritative_validator_rejects_logical_scene_rebinding(
    authoritative_bundle_fixture: tuple[Path, PublicationRequest],
    tmp_path: Path,
    domain: str,
    dataset_artifact: PublicationArtifactName,
    camera_artifact: PublicationArtifactName,
) -> None:
    source_bundle, request = authoritative_bundle_fixture
    bundle, payload = _tamper_bundle(source_bundle, tmp_path / domain)
    wrong_logical_scene = "logical-wrong"
    resolved_config = cast(dict[str, dict[str, object]], payload["resolved_config"])
    resolved_config[domain]["logical_scene_id"] = wrong_logical_scene
    source_owners = cast(dict[str, dict[str, object]], payload["source_owners"])
    source_owners[domain]["logical_scene_id"] = wrong_logical_scene

    dataset_mapping = cast(
        list[dict[str, object]],
        _artifact_payload(payload, dataset_artifact)["mapping"],
    )
    for record in dataset_mapping:
        record["logical_scene_id"] = wrong_logical_scene

    camera_mapping = cast(
        list[dict[str, object]],
        _artifact_payload(payload, camera_artifact)["mapping"],
    )
    for record in camera_mapping:
        record["logical_scene_id"] = wrong_logical_scene

    comparison = cast(
        list[dict[str, object]],
        _artifact_payload(
            payload, PublicationArtifactName.CAMERA_LAYOUT_COMPARISON
        )["mapping"],
    )
    comparison_start = 1 if domain == "blcs" else 1 + len(_CAMERA_IDS)
    comparison[comparison_start : comparison_start + len(_CAMERA_IDS)] = [
        dict(record) for record in camera_mapping[1:]
    ]
    _write_manifest_with_recomputed_media_digests(bundle, payload)

    validate_publication_bundle_structure_only(bundle)
    with pytest.raises(ValueError, match="resolved config differs"):
        validate_publication_bundle(bundle, expected_request=request)


@pytest.mark.parametrize(
    ("owner_name", "artifact", "field"),
    [
        (owner_name, artifact, field)
        for owner_name, artifact in (
            ("reconstruction", PublicationArtifactName.CAPTURED_CAMERA_TRAJECTORY),
            ("blcs", PublicationArtifactName.BLCS_CAMERA_LAYOUT),
            ("plcs", PublicationArtifactName.PLCS_CAMERA_LAYOUT),
        )
        for field in (
            "source_frame_index",
            "width",
            "height",
            "intrinsics",
            "image_path",
        )
    ],
)
def test_authoritative_validator_rejects_camera_source_record_rebinding(
    authoritative_bundle_fixture: tuple[Path, PublicationRequest],
    tmp_path: Path,
    owner_name: str,
    artifact: PublicationArtifactName,
    field: str,
) -> None:
    source_bundle, request = authoritative_bundle_fixture
    bundle, payload = _tamper_bundle(
        source_bundle, tmp_path / f"{owner_name}-{field}"
    )
    artifact_mapping = cast(
        list[dict[str, object]], _artifact_payload(payload, artifact)["mapping"]
    )
    pose = artifact_mapping[1]
    if field == "intrinsics":
        intrinsics = cast(list[float], pose[field])
        intrinsics[0] += 1.0
    elif field == "image_path":
        pose[field] = "images/wrong.png"
    else:
        pose[field] = cast(int, pose[field]) + 1

    if owner_name != "reconstruction":
        comparison = cast(
            list[dict[str, object]],
            _artifact_payload(
                payload, PublicationArtifactName.CAMERA_LAYOUT_COMPARISON
            )["mapping"],
        )
        comparison_index = 1 if owner_name == "blcs" else 1 + len(_CAMERA_IDS)
        comparison[comparison_index] = dict(pose)
    _write_manifest_with_recomputed_media_digests(bundle, payload)

    validate_publication_bundle_structure_only(bundle)
    with pytest.raises(ValueError, match="validated camera source"):
        validate_publication_bundle(bundle, expected_request=request)


def test_authoritative_validator_rejects_alignment_metric_rebinding(
    authoritative_bundle_fixture: tuple[Path, PublicationRequest],
    tmp_path: Path,
) -> None:
    source_bundle, request = authoritative_bundle_fixture
    bundle, payload = _tamper_bundle(source_bundle, tmp_path / "alignment-metrics")
    metrics = cast(dict[str, dict[str, object]], payload["metrics"])
    metrics["alignment"]["court_line_mean_probability"] = 0.01
    _write_manifest_with_recomputed_media_digests(bundle, payload)

    validate_publication_bundle_structure_only(bundle)
    with pytest.raises(ValueError, match="alignment metrics"):
        validate_publication_bundle(bundle, expected_request=request)


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_authoritative_validator_rejects_alignment_metric_key_rebinding(
    authoritative_bundle_fixture: tuple[Path, PublicationRequest],
    tmp_path: Path,
    mutation: str,
) -> None:
    source_bundle, request = authoritative_bundle_fixture
    bundle, payload = _tamper_bundle(
        source_bundle,
        tmp_path / f"alignment-metrics-{mutation}",
    )
    metrics = cast(dict[str, dict[str, object]], payload["metrics"])["alignment"]
    if mutation == "missing":
        metrics.pop("court_line_mean_probability")
    else:
        metrics["unrecorded_alignment_metric"] = 0.5
    _write_manifest_with_recomputed_media_digests(bundle, payload)

    validate_publication_bundle_structure_only(bundle)
    with pytest.raises(ValueError, match="alignment metrics"):
        validate_publication_bundle(bundle, expected_request=request)


def test_authoritative_validator_rejects_alignment_progression_rebinding(
    authoritative_bundle_fixture: tuple[Path, PublicationRequest],
    tmp_path: Path,
) -> None:
    source_bundle, request = authoritative_bundle_fixture
    bundle, payload = _tamper_bundle(source_bundle, tmp_path / "alignment-trace")
    mapping = cast(
        list[dict[str, object]],
        _artifact_payload(
            payload, PublicationArtifactName.ALIGNMENT_PROGRESSION
        )["mapping"],
    )
    mapping[1]["score_sum"] = 123.0
    mapping[1]["candidate_scores"] = [123.0]
    _write_manifest_with_recomputed_media_digests(bundle, payload)

    validate_publication_bundle_structure_only(bundle)
    with pytest.raises(ValueError, match="alignment progression"):
        validate_publication_bundle(bundle, expected_request=request)


@pytest.mark.parametrize(
    ("field", "rebound_value"),
    (
        ("step_index", 99),
        ("candidate_ids", ["candidate-rebound"]),
    ),
)
def test_authoritative_validator_rejects_alignment_progression_identity_rebinding(
    authoritative_bundle_fixture: tuple[Path, PublicationRequest],
    tmp_path: Path,
    field: str,
    rebound_value: object,
) -> None:
    source_bundle, request = authoritative_bundle_fixture
    bundle, payload = _tamper_bundle(
        source_bundle,
        tmp_path / f"alignment-trace-{field}",
    )
    mapping = cast(
        list[dict[str, object]],
        _artifact_payload(
            payload, PublicationArtifactName.ALIGNMENT_PROGRESSION
        )["mapping"],
    )
    mapping[1][field] = rebound_value
    _write_manifest_with_recomputed_media_digests(bundle, payload)

    validate_publication_bundle_structure_only(bundle)
    with pytest.raises(ValueError, match="alignment progression"):
        validate_publication_bundle(bundle, expected_request=request)


def test_authoritative_validator_rejects_media_dimensions_rebound_from_config(
    authoritative_bundle_fixture: tuple[Path, PublicationRequest],
    tmp_path: Path,
) -> None:
    source_bundle, request = authoritative_bundle_fixture
    bundle, payload = _tamper_bundle(source_bundle, tmp_path / "media-dimensions")
    overview_path = bundle / PublicationArtifactName.PUBLICATION_OVERVIEW.value
    with Image.open(overview_path) as image:
        resized = image.resize((_OVERVIEW_SIZE[0] + 1, _OVERVIEW_SIZE[1]))
        resized.save(overview_path, format="PNG", optimize=False, compress_level=9)
    overview = _artifact_payload(payload, PublicationArtifactName.PUBLICATION_OVERVIEW)
    overview["width"] = _OVERVIEW_SIZE[0] + 1
    _write_manifest_with_recomputed_media_digests(bundle, payload)

    validate_publication_bundle_structure_only(bundle)
    with pytest.raises(ValueError, match="resolved drawing dimensions"):
        validate_publication_bundle(bundle, expected_request=request)


@pytest.mark.parametrize(
    ("artifact_name", "drawing_field"),
    (
        (PublicationArtifactName.DATASET_COURT, "dataset_size"),
        (PublicationArtifactName.DATASET_BLCS, "dataset_size"),
        (PublicationArtifactName.DATASET_PLCS, "dataset_size"),
        (PublicationArtifactName.ALIGNMENT_PROGRESSION, "alignment_size"),
        (PublicationArtifactName.ALIGNMENT_HEATMAP_COURT, "figure_size"),
        (PublicationArtifactName.CAPTURED_CAMERA_TRAJECTORY, "figure_size"),
        (PublicationArtifactName.BLCS_CAMERA_LAYOUT, "figure_size"),
        (PublicationArtifactName.PLCS_CAMERA_LAYOUT, "figure_size"),
        (PublicationArtifactName.CAMERA_LAYOUT_COMPARISON, "figure_size"),
        (PublicationArtifactName.PUBLICATION_OVERVIEW, "overview_size"),
    ),
    ids=lambda value: value.value if isinstance(value, PublicationArtifactName) else value,
)
def test_authoritative_validator_binds_all_artifact_dimension_categories(
    authoritative_bundle_fixture: tuple[Path, PublicationRequest],
    tmp_path: Path,
    artifact_name: PublicationArtifactName,
    drawing_field: str,
) -> None:
    source_bundle, request = authoritative_bundle_fixture
    bundle, payload = _tamper_bundle(
        source_bundle,
        tmp_path / f"dimensions-{artifact_name.value}",
    )
    expected_size = cast(tuple[int, int], getattr(request.drawing, drawing_field))
    artifact = _artifact_payload(payload, artifact_name)
    assert (artifact["width"], artifact["height"]) == expected_size
    _resize_artifact_and_rebind_record(bundle, payload, artifact_name)
    _write_manifest_with_recomputed_media_digests(bundle, payload)

    validate_publication_bundle_structure_only(bundle)
    with pytest.raises(
        ValueError,
        match=rf"PublicationRequest\.drawing\.{drawing_field}",
    ):
        validate_publication_bundle(bundle, expected_request=request)


def test_complete_bundle_reopens_with_exact_mappings_and_is_byte_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_root = _scene_root(tmp_path)
    loaded = _loaded_inputs()
    monkeypatch.setattr(bundle_module, "_load_inputs", lambda _request: loaded)
    first_request = _request(scene_root, tmp_path / "outputs" / "publication-a")
    second_request = _request(scene_root, tmp_path / "outputs" / "publication-b")

    first = generate_publication_bundle(first_request)
    second = generate_publication_bundle(second_request)

    assert (
        validate_publication_bundle(
            first.bundle_path, expected_request=first_request
        ).to_dict()
        == first.manifest.to_dict()
    )
    assert (
        validate_publication_bundle(
            second.bundle_path, expected_request=second_request
        ).to_dict()
        == second.manifest.to_dict()
    )
    assert first.manifest.to_dict() == second.manifest.to_dict()
    expected_files = {item.value for item in REQUIRED_PUBLICATION_ARTIFACTS} | {
        "manifest.json"
    }
    assert {path.name for path in first.bundle_path.iterdir()} == expected_files
    for file_name in expected_files:
        assert (first.bundle_path / file_name).read_bytes() == (
            second.bundle_path / file_name
        ).read_bytes()

    records = _record_by_name(first)
    for artifact in REQUIRED_PUBLICATION_ARTIFACTS:
        record = records[artifact]
        with Image.open(first.bundle_path / artifact.value) as image:
            expected_format = "GIF" if artifact.value.endswith(".gif") else "PNG"
            assert image.format == expected_format
            assert image.size == (record.width, record.height)
            assert image.n_frames == record.frame_count
            for index in range(image.n_frames):
                image.seek(index)
                image.convert("RGB").load()
                if expected_format == "GIF":
                    assert image.info["duration"] == _GIF_DURATION_MS

    assert records[PublicationArtifactName.DATASET_COURT].mapping == (
        {
            "source_index": 0,
            "sample_id": "sample-0",
            "view_id": "view-0",
            "trajectory_frame_index": 0,
        },
        {
            "source_index": 2,
            "sample_id": "sample-2",
            "view_id": "view-0",
            "trajectory_frame_index": 2,
        },
    )
    assert records[PublicationArtifactName.DATASET_BLCS].mapping == (
        {
            "source_index": 0,
            "source_frame_index": 0,
            "global_frame_index": 20,
            "logical_scene_id": _LOGICAL_SCENE_ID,
            "camera_id": "camera-0",
        },
        {
            "source_index": 2,
            "source_frame_index": 2,
            "global_frame_index": 22,
            "logical_scene_id": _LOGICAL_SCENE_ID,
            "camera_id": "camera-0",
        },
    )
    assert records[PublicationArtifactName.DATASET_PLCS].mapping == (
        {
            "source_index": 0,
            "frame_index": 0,
            "logical_scene_id": _LOGICAL_SCENE_ID,
            "camera_id": "camera-0",
        },
        {
            "source_index": 2,
            "frame_index": 2,
            "logical_scene_id": _LOGICAL_SCENE_ID,
            "camera_id": "camera-0",
        },
    )
    assert tuple(
        item["phase"]
        for item in records[PublicationArtifactName.ALIGNMENT_PROGRESSION].mapping
    ) == tuple(phase.value for phase in AlignmentTracePhase)
    assert (
        tuple(
            item["camera_id"]
            for item in records[PublicationArtifactName.BLCS_CAMERA_LAYOUT].mapping[1:]
        )
        == _CAMERA_IDS
    )
    assert (
        tuple(
            item["camera_id"]
            for item in records[PublicationArtifactName.PLCS_CAMERA_LAYOUT].mapping[1:]
        )
        == _CAMERA_IDS
    )
    captured_mapping = records[
        PublicationArtifactName.CAPTURED_CAMERA_TRAJECTORY
    ].mapping
    captured_policy = captured_mapping[0]
    captured_poses = captured_mapping[1:]
    rendered_indices = cast(Sequence[int], captured_policy["rendered_camera_indices"])
    assert len(captured_poses) == 491
    reconstruction_owner = cast(
        Mapping[str, object], first.manifest.source_owners["reconstruction"]
    )
    assert len(cast(Sequence[str], reconstruction_owner["camera_ids"])) == 491
    assert tuple(item["camera_id"] for item in captured_poses) == _CAPTURED_CAMERA_IDS
    assert tuple(item["camera_index"] for item in captured_poses) == tuple(range(491))
    assert len(rendered_indices) == 24
    assert rendered_indices[0] == 0
    assert rendered_indices[-1] == 490
    assert tuple(cast(Sequence[str], captured_policy["rendered_camera_ids"])) == tuple(
        _CAPTURED_CAMERA_IDS[index] for index in rendered_indices
    )
    assert (
        captured_poses[0]["camera_to_metric_scene"]
        != captured_poses[-1]["camera_to_metric_scene"]
    )

    comparison_mapping = records[
        PublicationArtifactName.CAMERA_LAYOUT_COMPARISON
    ].mapping
    comparison_summary = comparison_mapping[0]
    assert tuple(item["camera_id"] for item in comparison_mapping[1:]) == (
        *_CAMERA_IDS,
        *_CAMERA_IDS,
    )
    comparison_metrics = cast(
        Mapping[str, object], comparison_summary["comparison_metrics"]
    )
    assert comparison_metrics["coincident_camera_count"] == 2
    assert comparison_metrics["coincident_camera_fraction"] == 1.0
    camera_metrics = cast(
        Mapping[str, Mapping[str, object]], first.manifest.metrics["cameras"]
    )
    assert camera_metrics["reconstruction"]["trajectory_segment_count"] == 490
    assert (
        camera_metrics["blcs"]
        .keys()
        .isdisjoint({"trajectory_segment_count", "trajectory_length_metres"})
    )

    expected_bounds = overview_panel_bounds(_OVERVIEW_SIZE)
    overview_mapping = records[PublicationArtifactName.PUBLICATION_OVERVIEW].mapping
    assert tuple(item["panel"] for item in overview_mapping) == tuple(
        label for label, _bounds in expected_bounds
    )
    assert tuple(
        tuple(cast(Sequence[int], item["bounds_pixels"])) for item in overview_mapping
    ) == tuple(bounds for _label, bounds in expected_bounds)
    for item in overview_mapping:
        left, top, right, bottom = cast(Sequence[int], item["bounds_pixels"])
        assert 0 <= left < right <= _OVERVIEW_SIZE[0]
        assert 0 <= top < bottom <= _OVERVIEW_SIZE[1]


def test_render_failure_removes_private_staging_and_never_publishes_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_root = _scene_root(tmp_path)
    loaded = _loaded_inputs(court_source=_FailingCourtSource())
    monkeypatch.setattr(bundle_module, "_load_inputs", lambda _request: loaded)
    request = _request(scene_root, tmp_path / "outputs" / "publication-failure")

    with pytest.raises(ValueError, match="synthetic corrupt court frame"):
        generate_publication_bundle(request)

    assert not request.output_bundle.exists()
    assert (
        list(
            request.output_bundle.parent.glob(
                f".{request.output_bundle.name}.*.staging"
            )
        )
        == []
    )
