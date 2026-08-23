"""Canonical Court semantic-manifest derivation and repeat validation."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.court.components.labels import (
    PHYSICAL_INDICES_BY_CLASS,
    SEMANTIC_CLASS_NAMES,
    attach_renderer_visibility,
    project_court_semantics_v2,
    project_court_semantics_v3,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    ResolvedTargetCourtV2,
    TargetCourtResolutionPolicy,
)
from src.synthetic_data_generation.dataset.court.schema import (
    COURT_SEMANTIC_CLASS_NAMES_V2,
    COURT_SEMANTIC_CLASS_NAMES_V3,
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.dataset.court.semantic_manifest import (
    build_court_semantic_manifest,
    require_equal_court_semantic_manifests,
    validate_court_semantic_manifest,
    validate_v2_published_court_geometry,
    validate_v3_published_court_geometry,
)
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
    SceneCamera,
)
from src.utils.schema.court import CAMERA_VIEW_HALF_TURN_INDEX


def test_semantic_manifest_is_recomputed_exactly_from_dataset_semantics() -> None:
    dataset = _dataset()
    manifest = build_court_semantic_manifest(dataset)

    assert validate_court_semantic_manifest(dataset, manifest) == manifest
    assert manifest["trajectory_groups"] == dataset["trajectory_groups"]
    assert manifest["samples"][0]["disposition"] == "accepted"  # type: ignore[index]

    mutated = copy.deepcopy(manifest)
    counts = mutated["counts"]
    assert isinstance(counts, dict)
    counts["accepted_sample_count"] = 2
    with pytest.raises(ValueError, match="semantic manifest disagrees"):
        validate_court_semantic_manifest(dataset, mutated)
    with pytest.raises(ValueError, match="not exactly equal"):
        require_equal_court_semantic_manifests(manifest, mutated)


def test_semantic_manifest_rejects_renderer_semantic_and_operational_mutations() -> (
    None
):
    dataset = _dataset()
    manifest = build_court_semantic_manifest(dataset)

    mutated_dataset = copy.deepcopy(dataset)
    sample = mutated_dataset["samples"]
    assert isinstance(sample, list)
    camera = sample[0]["camera"]
    assert isinstance(camera, dict)
    transform = camera["camera_to_scene"]
    assert isinstance(transform, list)
    transform[3] = 0.25
    with pytest.raises(ValueError, match="semantic manifest disagrees"):
        validate_court_semantic_manifest(mutated_dataset, manifest)

    mutated_manifest = copy.deepcopy(manifest)
    mutated_manifest["wall_seconds"] = 1.0
    with pytest.raises(ValueError, match="operational field"):
        validate_court_semantic_manifest(dataset, mutated_manifest)


def test_v2_semantic_manifest_binds_exact_schemas_target_and_singleton_geometry() -> (
    None
):
    dataset = _v2_dataset()
    manifest = build_court_semantic_manifest(dataset)

    assert manifest["schema"] == "court_renderer_semantic_manifest_v2"
    assert manifest["dataset_schema"] == "canonical_court_dataset_v2"
    assert manifest["sample_schema"] == "canonical_court_sample_v2"
    semantic_schema = manifest["semantic_schema"]
    assert isinstance(semantic_schema, dict)
    assert semantic_schema["class_names"] == list(COURT_SEMANTIC_CLASS_NAMES_V2)
    assert semantic_schema["physical_point_count_per_class"] == 1
    assert validate_court_semantic_manifest(dataset, manifest) == manifest

    mixed = copy.deepcopy(manifest)
    mixed["schema"] = "court_renderer_semantic_manifest_v1"
    with pytest.raises(ValueError, match="schemas are mixed"):
        validate_court_semantic_manifest(dataset, mixed)

    mutated_target = copy.deepcopy(dataset)
    samples = mutated_target["samples"]
    assert isinstance(samples, list)
    target = samples[0]["target_court"]
    assert isinstance(target, dict)
    binding = target["binding"]
    assert isinstance(binding, dict)
    scene_from_court = binding["scene_from_court"]
    assert isinstance(scene_from_court, list)
    scene_from_court[3] = 0.25
    with pytest.raises(ValueError, match="binding disagrees"):
        build_court_semantic_manifest(mutated_target)


def test_v2_false_ambiguity_and_near_far_mutations_are_rejected() -> None:
    dataset = _v2_dataset()

    false_ambiguity = copy.deepcopy(dataset)
    accepted = false_ambiguity["samples"]
    rejected = false_ambiguity["rejected_samples"]
    metrics = false_ambiguity["metrics"]
    assert isinstance(accepted, list)
    assert isinstance(rejected, list)
    assert isinstance(metrics, dict)
    record = copy.deepcopy(accepted[0])
    record["sample_index"] = 1
    record["sample_id"] = "sample-000001"
    camera = record["camera"]
    assert isinstance(camera, dict)
    camera["camera_id"] = "sample-000001"
    record["projection"] = None
    record["reasons"] = ["ambiguous_camera_relative_near_far:court-a"]
    rejected.append(record)
    metrics["proposal_count"] = 2
    metrics["rejected_frame_count"] = 1
    with pytest.raises(ValueError, match="ambiguity reason disagrees"):
        build_court_semantic_manifest(false_ambiguity)

    wrong_permutation = copy.deepcopy(dataset)
    samples = wrong_permutation["samples"]
    assert isinstance(samples, list)
    projection = samples[0]["projection"]
    assert isinstance(projection, dict)
    courts = projection["courts"]
    assert isinstance(courts, list)
    classes = courts[0]["classes"]
    assert isinstance(classes, list)
    first_point = classes[0]["points"][0]
    second_point = classes[1]["points"][0]
    first_point["physical_index"], second_point["physical_index"] = (
        second_point["physical_index"],
        first_point["physical_index"],
    )
    with pytest.raises(ValueError, match="rigid court geometry"):
        build_court_semantic_manifest(wrong_permutation)


def test_v3_semantic_manifest_recomputes_full_camera_view_geometry() -> None:
    dataset = _v3_dataset()
    manifest = build_court_semantic_manifest(dataset)

    assert manifest["schema"] == "court_renderer_semantic_manifest_v3"
    assert manifest["dataset_schema"] == "canonical_court_dataset_v3"
    assert manifest["sample_schema"] == "canonical_court_sample_v3"
    assert validate_v3_published_court_geometry(dataset)
    assert validate_court_semantic_manifest(dataset, manifest) == manifest

    samples = dataset["samples"]
    assert isinstance(samples, list)
    projection = samples[0]["projection"]
    assert isinstance(projection, dict)
    courts = projection["courts"]
    assert isinstance(courts, list)
    classes = courts[0]["classes"]
    assert isinstance(classes, list)
    assert tuple(
        semantic["points"][0]["physical_index"] for semantic in classes
    ) == CAMERA_VIEW_HALF_TURN_INDEX

    wrong_validator = copy.deepcopy(dataset)
    wrong_validator["schema"] = "canonical_court_dataset_v2"
    with pytest.raises(ValueError, match="V3 dataset"):
        validate_v3_published_court_geometry(wrong_validator)


def test_v3_semantic_manifest_accepts_finite_lateral_projected_u_reversal() -> None:
    dataset = _v3_dataset(camera_center=(-30.0, -2.0, 12.0))
    projection = _mapping_at_path(dataset, ("samples", 0, "projection"))
    courts = cast(list[dict[str, object]], projection["courts"])
    classes = cast(list[dict[str, object]], courts[0]["classes"])
    semantic_u = [
        cast(list[float], cast(list[dict[str, object]], value["points"])[0]["uv"])[
            0
        ]
        for value in classes
    ]

    assert semantic_u[0] < semantic_u[1]
    assert semantic_u[2] > semantic_u[3]
    manifest = build_court_semantic_manifest(dataset)
    assert validate_court_semantic_manifest(dataset, manifest) == manifest
    assert validate_v3_published_court_geometry(dataset)


def test_v3_semantic_manifest_rejects_nonfinite_projected_uv() -> None:
    dataset = _v3_dataset()
    point = _mapping_at_path(
        dataset,
        ("samples", 0, "projection", "courts", 0, "classes", 0, "points", 0),
    )
    point["uv"] = [float("nan"), 0.0]

    with pytest.raises(ValueError, match="finite"):
        build_court_semantic_manifest(dataset)


@pytest.mark.parametrize(
    ("field", "mutate", "message"),
    [
        ("uv", lambda value: [value[0] + 0.25, value[1]], "UV disagrees"),
        ("camera_depth_m", lambda value: value + 0.25, "depth disagrees"),
        ("in_front", lambda value: not value, "in_front disagrees"),
        ("in_frame", lambda value: not value, "in_frame disagrees"),
    ],
)
def test_v3_semantic_manifest_rejects_projected_geometry_mutations(
    field: str,
    mutate: object,
    message: str,
) -> None:
    dataset = _v3_dataset()
    point = _mapping_at_path(
        dataset,
        ("samples", 0, "projection", "courts", 0, "classes", 0, "points", 0),
    )
    transform = cast(Callable[[Any], Any], mutate)
    point[field] = transform(point[field])
    if field == "in_frame":
        court = _mapping_at_path(
            dataset,
            ("samples", 0, "projection", "courts", 0),
        )
        court["coverage_mode"] = "near_full"
        projection = _mapping_at_path(dataset, ("samples", 0, "projection"))
        projection["coverage_modes"] = ["near_full"]
        metrics = _mapping_at_path(dataset, ("metrics",))
        metrics["coverage_counts"] = {"near_full": 1}

    with pytest.raises(ValueError, match=message):
        build_court_semantic_manifest(dataset)


def test_v3_semantic_manifest_rejects_renderer_visibility_mutation() -> None:
    dataset = _v3_dataset()
    point = _mapping_at_path(
        dataset,
        ("samples", 0, "projection", "courts", 0, "classes", 0, "points", 0),
    )
    point["renderer_visible"] = False

    with pytest.raises(ValueError, match="class visibility is inconsistent"):
        build_court_semantic_manifest(dataset)


@pytest.mark.parametrize(
    ("container_path", "missing_key"),
    [
        (("samples", 0), "target_court"),
        (("samples", 0), "projection"),
        (("samples", 0, "projection"), "courts"),
        (("samples", 0, "projection", "courts", 0), "classes"),
        (
            ("samples", 0, "projection", "courts", 0, "classes", 0),
            "points",
        ),
        (
            (
                "samples",
                0,
                "projection",
                "courts",
                0,
                "classes",
                0,
                "points",
                0,
            ),
            "physical_index",
        ),
        (
            (
                "samples",
                0,
                "projection",
                "courts",
                0,
                "classes",
                0,
                "points",
                0,
            ),
            "scene_xyz_m",
        ),
    ],
)
def test_v2_persisted_geometry_rejects_missing_required_keys(
    container_path: tuple[str | int, ...],
    missing_key: str,
) -> None:
    dataset = _v2_dataset()
    container = _mapping_at_path(dataset, container_path)
    del container[missing_key]

    with pytest.raises((KeyError, TypeError, ValueError)):
        validate_v2_published_court_geometry(dataset)


def test_v2_persisted_geometry_rejects_conflicting_candidate_binding() -> None:
    dataset = _v2_dataset()
    samples = dataset["samples"]
    assert isinstance(samples, list)
    conflicting = copy.deepcopy(samples[0])
    assert isinstance(conflicting, dict)
    target = conflicting["target_court"]
    assert isinstance(target, dict)
    binding = target["binding"]
    assert isinstance(binding, dict)
    binding["candidate_id"] = "candidate-conflict"
    samples.append(conflicting)

    with pytest.raises(ValueError, match="candidate disagrees"):
        validate_v2_published_court_geometry(dataset)


def _mapping_at_path(
    value: dict[str, object],
    path: tuple[str | int, ...],
) -> dict[str, object]:
    current: object = value
    for part in path:
        if isinstance(part, int):
            assert isinstance(current, list)
            current = current[part]
        else:
            assert isinstance(current, dict)
            current = current[part]
    assert isinstance(current, dict)
    return current


def _dataset() -> dict[str, object]:
    projection = _projection()
    camera = {
        "camera_id": "sample-000000",
        "source_frame_index": 0,
        "width": 4,
        "height": 3,
        "intrinsics": [4.0, 0.0, 1.5, 0.0, 4.0, 1.0, 0.0, 0.0, 1.0],
        "camera_to_scene": [
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
        ],
        "image_path": "generated/sample-000000.png",
    }
    sample = {
        "sample_index": 0,
        "sample_id": "sample-000000",
        "trajectory_group_id": "group-a",
        "trajectory_id": "trajectory-a",
        "view_id": "view-a",
        "trajectory_frame_index": 0,
        "split": "train",
        "shard_id": "shard-000",
        "camera": camera,
        "projection": projection,
        "metadata": {},
    }
    return {
        "schema": "canonical_court_dataset_v1",
        "scene_id": "B00",
        "profile": "train",
        "seed": 695,
        "sampling_policy": {"mode": "uniform_arc_length"},
        "metadata_fields": [],
        "trajectory_groups": [
            {
                "trajectory": {
                    "trajectory_group_id": "group-a",
                    "trajectory_id": "trajectory-a",
                }
            }
        ],
        "samples": [sample],
        "rejected_samples": [],
        "metrics": {
            "proposal_count": 1,
            "accepted_frame_count": 1,
            "rejected_frame_count": 0,
            "trajectory_group_count": 1,
            "split_frame_counts": {"train": 1},
            "coverage_counts": {"full": 1},
            "renderer_visible_points_by_class": {
                name: 1 for name in SEMANTIC_CLASS_NAMES
            },
        },
    }


def _projection() -> dict[str, object]:
    classes = []
    for class_id, (class_name, physical_indices) in enumerate(
        zip(SEMANTIC_CLASS_NAMES, PHYSICAL_INDICES_BY_CLASS, strict=True)
    ):
        classes.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "renderer_visible": True,
                "points": [
                    {
                        "physical_index": physical_index,
                        "uv": [1.0 + point_index, 1.0],
                        "camera_depth_m": 5.0,
                        "scene_xyz_m": [float(physical_index), 0.0, 0.0],
                        "in_front": True,
                        "in_frame": True,
                        "renderer_visible": point_index == 0,
                    }
                    for point_index, physical_index in enumerate(physical_indices)
                ],
            }
        )
    return {
        "camera_id": "sample-000000",
        "resolution": [4, 3],
        "coverage_modes": ["full"],
        "visible_class_names": list(SEMANTIC_CLASS_NAMES),
        "visible_point_count": len(SEMANTIC_CLASS_NAMES),
        "courts": [
            {
                "court_instance_id": "court-a",
                "coverage_mode": "full",
                "classes": classes,
            }
        ],
    }


def _singleton_dataset(
    schema_version: CourtDatasetSchemaVersion,
    *,
    camera_center: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    transform = RigidTransform.identity()
    court = CourtInstance(
        court_instance_id="court-a",
        candidate_id="candidate-a",
        scene_from_court=transform,
        court_from_scene=transform,
        fit_status="accepted",
        fit_metrics={"rms_error_m": 0.01},
        holdout_status="accepted",
        holdout_metrics={"rms_error_m": 0.02},
    )
    layout = MultiCourtLayout(
        courts=(court,),
        complex_bounds_scene=(-20.0, -25.0, -1.0, 20.0, 25.0, 12.0),
        primary_court_instance_id=court.court_instance_id,
    )
    camera = _singleton_camera(
        center=(
            camera_center
            if camera_center is not None
            else (
                0.0,
                30.0 if schema_version is CourtDatasetSchemaVersion.V3 else -30.0,
                12.0,
            )
        )
    )
    projection = (
        project_court_semantics_v2(camera, layout)
        if schema_version is CourtDatasetSchemaVersion.V2
        else project_court_semantics_v3(camera, layout)
    )
    visible = attach_renderer_visibility(
        projection,
        alpha=np.ones((480, 640, 1), dtype=np.float32),
        depth=np.ones((480, 640, 1), dtype=np.float32),
    )
    target = ResolvedTargetCourtV2(
        binding=TargetCourtBinding(
            court_instance_id="court-a",
            candidate_id="candidate-a",
            scene_from_court=transform,
            selection_seed=695,
        ),
        resolution_policy=TargetCourtResolutionPolicy.TRAJECTORY_CENTER_COURT,
        camera_to_court_center_distance_m=float(
            np.linalg.norm(camera.camera_to_scene.matrix()[:3, 3])
        ),
    )
    sample = {
        "sample_index": 0,
        "sample_id": camera.camera_id,
        "trajectory_group_id": "group-a",
        "trajectory_id": "trajectory-a",
        "view_id": "view-a",
        "trajectory_frame_index": 0,
        "split": "train",
        "shard_id": "shard-000",
        "camera": camera.to_dict(),
        "projection": visible.to_dict(),
        "target_court": target.to_dict(),
        "metadata": {},
    }
    coverage = visible.courts[0].coverage_mode
    return {
        "schema": (
            "canonical_court_dataset_v2"
            if schema_version is CourtDatasetSchemaVersion.V2
            else "canonical_court_dataset_v3"
        ),
        "scene_id": "B00",
        "profile": schema_version.value,
        "seed": 695,
        "sampling_policy": {"mode": "uniform_arc_length"},
        "metadata_fields": [],
        "trajectory_groups": [
            {
                "trajectory": {
                    "trajectory_group_id": "group-a",
                    "trajectory_id": "trajectory-a",
                },
                "target_court_policy": {
                    "mode": "trajectory_center_court",
                    "centre_court_instance_id": "court-a",
                },
            }
        ],
        "samples": [sample],
        "rejected_samples": [],
        "metrics": {
            "proposal_count": 1,
            "accepted_frame_count": 1,
            "rejected_frame_count": 0,
            "trajectory_group_count": 1,
            "split_frame_counts": {"train": 1},
            "coverage_counts": {coverage: 1},
            "renderer_visible_points_by_class": {
                name: 1
                for name in (
                    COURT_SEMANTIC_CLASS_NAMES_V2
                    if schema_version is CourtDatasetSchemaVersion.V2
                    else COURT_SEMANTIC_CLASS_NAMES_V3
                )
            },
        },
    }


def _v2_dataset() -> dict[str, object]:
    return _singleton_dataset(CourtDatasetSchemaVersion.V2)


def _v3_dataset(
    *,
    camera_center: tuple[float, float, float] | None = None,
) -> dict[str, object]:
    return _singleton_dataset(
        CourtDatasetSchemaVersion.V3,
        camera_center=camera_center,
    )


def _singleton_camera(*, center: tuple[float, float, float]) -> SceneCamera:
    center_array = np.asarray(center, dtype=np.float64)
    target: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
    forward = target - center_array
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, down, forward))
    matrix[:3, 3] = center_array
    return SceneCamera(
        camera_id="sample-000000",
        source_frame_index=0,
        width=640,
        height=480,
        intrinsics=(500.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(matrix),
        image_path="generated/sample-000000.png",
    )
