"""Canonical Court semantic-manifest derivation and repeat validation."""

from __future__ import annotations

import copy
import json
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
    scene_from_court_from_published_points,
)
from src.synthetic_data_generation.dataset.court.contracts import (
    ResolvedTargetCourtV2,
    TargetCourtResolutionPolicy,
)
from src.synthetic_data_generation.dataset.court.schema import (
    COURT_SEMANTIC_CLASS_NAMES_V2,
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
from src.utils.schema.court import (
    CAMERA_VIEW_HALF_TURN_INDEX,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)


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
    assert (
        tuple(semantic["points"][0]["physical_index"] for semantic in classes)
        == CAMERA_VIEW_HALF_TURN_INDEX
    )

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
        cast(list[float], cast(list[dict[str, object]], value["points"])[0]["uv"])[0]
        for value in classes
    ]

    assert semantic_u[0] < semantic_u[1]
    assert semantic_u[2] > semantic_u[3]
    manifest = build_court_semantic_manifest(dataset)
    assert validate_court_semantic_manifest(dataset, manifest) == manifest
    assert validate_v3_published_court_geometry(dataset)


def test_v3_json_round_trip_uses_exact_oblique_binding_authority() -> None:
    dataset = _serialized_oblique_v3_dataset()
    binding = _mapping_at_path(dataset, ("samples", 0, "target_court", "binding"))
    serialized_transform = cast(list[float], binding["scene_from_court"])
    published_points = _published_points_by_index(dataset)
    recovered_transform = scene_from_court_from_published_points(published_points)
    camera = SceneCamera.from_dict(_mapping_at_path(dataset, ("samples", 0, "camera")))
    observed_uv = _published_uv_by_index(dataset)
    recovered_uv = _project_uv(camera, recovered_transform)

    assert not np.allclose(serialized_transform, np.eye(4).ravel())
    assert np.max(np.abs(recovered_uv - observed_uv)) > 1.0e-6
    geometry = validate_v3_published_court_geometry(dataset)
    assert geometry["court-a"].to_list() == serialized_transform
    manifest = build_court_semantic_manifest(dataset)
    assert validate_court_semantic_manifest(dataset, manifest) == manifest


def test_v3_json_round_trip_rejects_strict_uv_corruption() -> None:
    dataset = _serialized_oblique_v3_dataset()
    point = _mapping_at_path(
        dataset,
        ("samples", 0, "projection", "courts", 0, "classes", 0, "points", 0),
    )
    uv = cast(list[float], point["uv"])
    uv[0] += 2.0e-6

    with pytest.raises(ValueError, match="UV disagrees"):
        build_court_semantic_manifest(dataset)


@pytest.mark.parametrize("corruption", ["physical_point", "binding"])
def test_v3_json_round_trip_rejects_physical_binding_corruption(
    corruption: str,
) -> None:
    dataset = _serialized_oblique_v3_dataset()
    if corruption == "physical_point":
        court = _mapping_at_path(
            dataset,
            ("samples", 0, "projection", "courts", 0),
        )
        for semantic_class in cast(list[dict[str, object]], court["classes"]):
            point = cast(list[dict[str, object]], semantic_class["points"])[0]
            scene_xyz = cast(list[float], point["scene_xyz_m"])
            scene_xyz[0] += 2.0e-6
    else:
        binding = _mapping_at_path(
            dataset,
            ("samples", 0, "target_court", "binding"),
        )
        scene_from_court = cast(list[float], binding["scene_from_court"])
        scene_from_court[3] += 2.0e-6

    with pytest.raises(ValueError, match="binding disagrees"):
        build_court_semantic_manifest(dataset)


def test_v3_binding_authority_rejects_cross_court_corruption() -> None:
    dataset = _serialized_two_court_v3_geometry_dataset()
    assert set(validate_v3_published_court_geometry(dataset)) == {"court-a", "court-b"}
    first_binding = _mapping_at_path(
        dataset,
        ("samples", 0, "target_court", "binding"),
    )
    second_binding = _mapping_at_path(
        dataset,
        ("samples", 1, "target_court", "binding"),
    )
    first_binding["court_instance_id"], second_binding["court_instance_id"] = (
        second_binding["court_instance_id"],
        first_binding["court_instance_id"],
    )

    with pytest.raises(ValueError, match="binding disagrees"):
        validate_v3_published_court_geometry(dataset)


def test_v3_binding_authority_rejects_candidate_corruption() -> None:
    dataset = _serialized_two_court_v3_geometry_dataset()
    samples = cast(list[dict[str, object]], dataset["samples"])
    conflicting_binding = _mapping_at_path(
        samples[1],
        ("target_court", "binding"),
    )
    conflicting_binding["candidate_id"] = "candidate-a"

    with pytest.raises(ValueError, match="candidate IDs must be unique"):
        validate_v3_published_court_geometry(dataset)


@pytest.mark.parametrize("physical_index", range(14))
def test_v3_binding_authority_checks_every_point_in_repeated_samples(
    physical_index: int,
) -> None:
    dataset = _serialized_oblique_v3_dataset()
    _append_repeated_record(dataset, rejected=False)
    point = _published_point(
        dataset,
        sample_index=1,
        court_index=0,
        physical_index=physical_index,
    )
    scene_xyz = cast(list[float], point["scene_xyz_m"])
    scene_xyz[0] += 2.0e-6

    with pytest.raises(ValueError, match="geometry disagrees across published samples"):
        validate_v3_published_court_geometry(dataset)


def test_v3_binding_authority_covers_repeated_and_rejected_records() -> None:
    dataset = _serialized_oblique_v3_dataset()
    _append_repeated_record(dataset, rejected=False)
    rejected = _append_repeated_record(dataset, rejected=True)

    geometry = validate_v3_published_court_geometry(dataset)
    manifest = build_court_semantic_manifest(dataset)

    assert geometry["court-a"].to_list() == cast(
        list[float],
        _mapping_at_path(dataset, ("samples", 0, "target_court", "binding"))[
            "scene_from_court"
        ],
    )
    assert [
        entry["disposition"]
        for entry in cast(list[dict[str, object]], manifest["samples"])
    ] == ["accepted", "accepted", "rejected"]

    binding = _mapping_at_path(rejected, ("target_court", "binding"))
    scene_from_court = cast(list[float], binding["scene_from_court"])
    scene_from_court[3] += 5.0e-7
    with pytest.raises(ValueError, match="binding disagrees across published samples"):
        validate_v3_published_court_geometry(dataset)


def test_v3_binding_authority_accepts_only_proven_mid_plane_null_rejection() -> None:
    dataset = _serialized_oblique_v3_dataset()
    rejected = _append_repeated_record(dataset, rejected=True)
    rejected["projection"] = None
    rejected["reasons"] = ["ambiguous_camera_relative_near_far:court-a"]
    _set_record_camera_center_court(rejected, center=(-30.0, 0.0, 12.0))

    manifest = build_court_semantic_manifest(dataset)
    entries = cast(list[dict[str, object]], manifest["samples"])
    assert entries[1]["semantic_projection"] is None

    _set_record_camera_center_court(rejected, center=(-30.0, 2.0e-6, 12.0))
    with pytest.raises(ValueError, match="ambiguity reason disagrees"):
        build_court_semantic_manifest(dataset)


def test_v3_published_geometry_rejects_duplicate_physical_inventory() -> None:
    dataset = _serialized_oblique_v3_dataset()
    point = _published_point(
        dataset,
        sample_index=0,
        court_index=0,
        physical_index=13,
    )
    point["physical_index"] = 0

    with pytest.raises(ValueError, match="duplicate indices"):
        validate_v3_published_court_geometry(dataset)


def test_v2_json_round_trip_retains_recovered_svd_authority() -> None:
    dataset = _serialized_oblique_v3_dataset()
    dataset["schema"] = "canonical_court_dataset_v2"
    binding = _mapping_at_path(dataset, ("samples", 0, "target_court", "binding"))
    serialized_transform = cast(list[float], binding["scene_from_court"])
    recovered = scene_from_court_from_published_points(
        _published_points_by_index(dataset)
    )

    geometry = validate_v2_published_court_geometry(dataset)

    assert geometry["court-a"] == recovered
    assert geometry["court-a"].to_list() != serialized_transform
    assert build_court_semantic_manifest(dataset)["schema"] == (
        "court_renderer_semantic_manifest_v2"
    )


def test_published_geometry_validators_do_not_alias_v2_and_v3() -> None:
    v3_dataset = _serialized_oblique_v3_dataset()
    v2_dataset = copy.deepcopy(v3_dataset)
    v2_dataset["schema"] = "canonical_court_dataset_v2"

    with pytest.raises(ValueError, match="V2 dataset"):
        validate_v2_published_court_geometry(v3_dataset)
    with pytest.raises(ValueError, match="V3 dataset"):
        validate_v3_published_court_geometry(v2_dataset)


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
    scene_from_court: RigidTransform | None = None,
    camera: SceneCamera | None = None,
) -> dict[str, object]:
    transform = (
        RigidTransform.identity() if scene_from_court is None else scene_from_court
    )
    court = CourtInstance(
        court_instance_id="court-a",
        candidate_id="candidate-a",
        scene_from_court=transform,
        court_from_scene=transform.inverse(),
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
    if camera is None:
        center = (
            camera_center
            if camera_center is not None
            else (
                0.0,
                30.0 if schema_version is CourtDatasetSchemaVersion.V3 else -30.0,
                12.0,
            )
        )
        scene_camera = _singleton_camera(center=center)
    else:
        scene_camera = camera
    projection = (
        project_court_semantics_v2(scene_camera, layout)
        if schema_version is CourtDatasetSchemaVersion.V2
        else project_court_semantics_v3(scene_camera, layout)
    )
    visible = attach_renderer_visibility(
        projection,
        alpha=np.ones((scene_camera.height, scene_camera.width, 1), dtype=np.float32),
        depth=np.ones((scene_camera.height, scene_camera.width, 1), dtype=np.float32),
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
            np.linalg.norm(
                transform.inverse().apply(
                    scene_camera.camera_to_scene.matrix()[:3, 3].reshape(1, 3)
                )[0]
            )
        ),
    )
    sample = {
        "sample_index": 0,
        "sample_id": scene_camera.camera_id,
        "trajectory_group_id": "group-a",
        "trajectory_id": "trajectory-a",
        "view_id": "view-a",
        "trajectory_frame_index": 0,
        "split": "train",
        "shard_id": "shard-000",
        "camera": scene_camera.to_dict(),
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
                semantic_class.class_name: int(semantic_class.renderer_visible)
                for semantic_class in visible.courts[0].classes
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


def _serialized_oblique_v3_dataset() -> dict[str, object]:
    scene_from_court, camera = _oblique_v3_calibration()
    dataset = _singleton_dataset(
        CourtDatasetSchemaVersion.V3,
        scene_from_court=scene_from_court,
        camera=camera,
    )
    parsed: object = json.loads(json.dumps(dataset, allow_nan=False, sort_keys=True))
    assert isinstance(parsed, dict)
    return cast(dict[str, object], parsed)


def _serialized_two_court_v3_geometry_dataset() -> dict[str, object]:
    dataset = _serialized_oblique_v3_dataset()
    samples = cast(list[dict[str, object]], dataset["samples"])
    projection = _mapping_at_path(samples[0], ("projection",))
    courts = cast(list[dict[str, object]], projection["courts"])
    second_court = copy.deepcopy(courts[0])
    second_court["court_instance_id"] = "court-b"
    for semantic_class in cast(list[dict[str, object]], second_court["classes"]):
        point = cast(list[dict[str, object]], semantic_class["points"])[0]
        scene_xyz = cast(list[float], point["scene_xyz_m"])
        scene_xyz[0] += 40.0
    courts.append(second_court)

    second_sample = copy.deepcopy(samples[0])
    target_binding = _mapping_at_path(second_sample, ("target_court", "binding"))
    target_binding["court_instance_id"] = "court-b"
    target_binding["candidate_id"] = "candidate-b"
    scene_from_court = cast(list[float], target_binding["scene_from_court"])
    scene_from_court[3] += 40.0
    samples.append(second_sample)

    parsed: object = json.loads(json.dumps(dataset, allow_nan=False, sort_keys=True))
    assert isinstance(parsed, dict)
    return cast(dict[str, object], parsed)


def _oblique_v3_calibration() -> tuple[RigidTransform, SceneCamera]:
    angle = np.deg2rad(33.8)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    scene_from_court_matrix = np.asarray(
        (
            (cosine, -sine, 0.0, 12.3456789),
            (sine, cosine, 0.0, -7.6543211),
            (0.0, 0.0, 1.0, 2.3456789),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    scene_from_court = RigidTransform.from_matrix(scene_from_court_matrix)

    points_court = np.asarray(
        court_keypoints_3d(STANDARD_COURT_CONFIG)[:14].numpy(),
        dtype=np.float64,
    )
    camera_center_court = np.asarray((-30.0, -2.0, 12.0), dtype=np.float64)
    first_point_direction = points_court[0] - camera_center_court
    first_point_direction /= np.linalg.norm(first_point_direction)
    forward = np.cross(
        points_court[0] - camera_center_court,
        np.asarray((0.31, -0.77, 0.55), dtype=np.float64),
    )
    forward /= np.linalg.norm(forward)
    forward += 5.0e-5 * first_point_direction
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0), dtype=np.float64))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    camera_to_court = np.eye(4, dtype=np.float64)
    camera_to_court[:3, :3] = np.column_stack((right, down, forward))
    camera_to_court[:3, 3] = camera_center_court
    camera_to_scene = RigidTransform.from_matrix(
        scene_from_court.matrix() @ camera_to_court
    )
    return scene_from_court, SceneCamera(
        camera_id="sample-000000",
        source_frame_index=0,
        width=640,
        height=480,
        intrinsics=(500.0, 0.0, 319.5, 0.0, 500.0, 239.5, 0.0, 0.0, 1.0),
        camera_to_scene=camera_to_scene,
        image_path="generated/sample-000000.png",
    )


def _published_points_by_index(
    dataset: dict[str, object],
) -> dict[int, tuple[float, float, float]]:
    court = _mapping_at_path(dataset, ("samples", 0, "projection", "courts", 0))
    result: dict[int, tuple[float, float, float]] = {}
    for semantic_class in cast(list[dict[str, object]], court["classes"]):
        point = cast(list[dict[str, object]], semantic_class["points"])[0]
        physical_index = cast(int, point["physical_index"])
        scene_xyz = cast(list[float], point["scene_xyz_m"])
        result[physical_index] = (scene_xyz[0], scene_xyz[1], scene_xyz[2])
    return result


def _published_point(
    dataset: dict[str, object],
    *,
    sample_index: int,
    court_index: int,
    physical_index: int,
) -> dict[str, object]:
    court = _mapping_at_path(
        dataset,
        ("samples", sample_index, "projection", "courts", court_index),
    )
    for semantic_class in cast(list[dict[str, object]], court["classes"]):
        point = cast(list[dict[str, object]], semantic_class["points"])[0]
        if point["physical_index"] == physical_index:
            return point
    raise AssertionError(f"physical index {physical_index} is absent")


def _append_repeated_record(
    dataset: dict[str, object],
    *,
    rejected: bool,
) -> dict[str, object]:
    samples = cast(list[dict[str, object]], dataset["samples"])
    rejected_samples = cast(list[dict[str, object]], dataset["rejected_samples"])
    record = copy.deepcopy(samples[0])
    record_index = len(samples) + len(rejected_samples)
    sample_id = f"sample-{record_index:06d}"
    record["sample_index"] = record_index
    record["sample_id"] = sample_id
    camera = _mapping_at_path(record, ("camera",))
    camera["camera_id"] = sample_id
    camera["source_frame_index"] = record_index
    projection = _mapping_at_path(record, ("projection",))
    projection["camera_id"] = sample_id
    metrics = _mapping_at_path(dataset, ("metrics",))
    metrics["proposal_count"] = cast(int, metrics["proposal_count"]) + 1
    if rejected:
        record["reasons"] = ["renderer_visibility_rejected"]
        rejected_samples.append(record)
        metrics["rejected_frame_count"] = cast(int, metrics["rejected_frame_count"]) + 1
        return record

    samples.append(record)
    metrics["accepted_frame_count"] = cast(int, metrics["accepted_frame_count"]) + 1
    split_counts = cast(dict[str, int], metrics["split_frame_counts"])
    split_counts["train"] += 1
    coverage_counts = cast(dict[str, int], metrics["coverage_counts"])
    for court in cast(list[dict[str, object]], projection["courts"]):
        coverage = cast(str, court["coverage_mode"])
        coverage_counts[coverage] = coverage_counts.get(coverage, 0) + 1
    visible_counts = cast(dict[str, int], metrics["renderer_visible_points_by_class"])
    for court in cast(list[dict[str, object]], projection["courts"]):
        for semantic_class in cast(list[dict[str, object]], court["classes"]):
            class_name = cast(str, semantic_class["class_name"])
            points = cast(list[dict[str, object]], semantic_class["points"])
            visible_counts[class_name] += sum(
                point["renderer_visible"] is True for point in points
            )
    return record


def _set_record_camera_center_court(
    record: dict[str, object],
    *,
    center: tuple[float, float, float],
) -> None:
    binding = _mapping_at_path(record, ("target_court", "binding"))
    scene_from_court = RigidTransform(
        tuple(cast(list[float], binding["scene_from_court"]))
    )
    center_scene = scene_from_court.apply(np.asarray((center,), dtype=np.float64))[0]
    camera = _mapping_at_path(record, ("camera",))
    camera_to_scene = cast(list[float], camera["camera_to_scene"])
    camera_to_scene[3] = float(center_scene[0])
    camera_to_scene[7] = float(center_scene[1])
    camera_to_scene[11] = float(center_scene[2])
    target = _mapping_at_path(record, ("target_court",))
    target["camera_to_court_center_distance_m"] = float(np.linalg.norm(center))


def _published_uv_by_index(dataset: dict[str, object]) -> NDArray[np.float64]:
    court = _mapping_at_path(dataset, ("samples", 0, "projection", "courts", 0))
    result: NDArray[np.float64] = np.empty((14, 2), dtype=np.float64)
    for semantic_class in cast(list[dict[str, object]], court["classes"]):
        point = cast(list[dict[str, object]], semantic_class["points"])[0]
        physical_index = cast(int, point["physical_index"])
        result[physical_index] = cast(list[float], point["uv"])
    return result


def _project_uv(
    camera: SceneCamera,
    scene_from_court: RigidTransform,
) -> NDArray[np.float64]:
    points_court = np.asarray(
        court_keypoints_3d(STANDARD_COURT_CONFIG)[:14].numpy(),
        dtype=np.float64,
    )
    points_camera = camera.camera_to_scene.inverse().apply(
        scene_from_court.apply(points_court)
    )
    homogeneous = points_camera @ np.asarray(camera.intrinsics).reshape(3, 3).T
    return homogeneous[:, :2] / points_camera[:, 2, None]


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
