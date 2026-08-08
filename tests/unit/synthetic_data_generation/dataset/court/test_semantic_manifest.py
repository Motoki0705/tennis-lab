"""Canonical Court semantic-manifest derivation and repeat validation."""

from __future__ import annotations

import copy

import pytest

from src.synthetic_data_generation.dataset.court.components.labels import (
    PHYSICAL_INDICES_BY_CLASS,
    SEMANTIC_CLASS_NAMES,
)
from src.synthetic_data_generation.dataset.court.semantic_manifest import (
    build_court_semantic_manifest,
    require_equal_court_semantic_manifests,
    validate_court_semantic_manifest,
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


def test_semantic_manifest_rejects_renderer_semantic_and_operational_mutations() -> None:
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
