"""Canonical renderer-derived semantic manifest for Court datasets."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np

from src.synthetic_data_generation.dataset.court.components.labels import (
    AMBIGUOUS_CAMERA_RELATIVE_NEAR_FAR_REASON,
    CAMERA_RELATIVE_MID_PLANE_TOLERANCE_M,
    PHYSICAL_INDICES_BY_CLASS,
    PUBLISHED_COURT_GEOMETRY_ATOL_M,
    camera_center_court_y,
    coverage_mode_from_in_frame_point_count,
    scene_from_court_from_published_points,
)
from src.synthetic_data_generation.dataset.court.contracts import ResolvedTargetCourtV2
from src.synthetic_data_generation.dataset.court.schema import (
    COURT_SEMANTIC_MANIFEST_SCHEMA_V1,
    CourtDatasetSchemaVersion,
    CourtSchemaDefinition,
    court_schema_from_dataset_schema,
    court_schema_from_semantic_manifest_schema,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.utils.schema.court import (
    OPPOSITE_COURT_END_INDEX,
    STANDARD_COURT_CONFIG,
    court_keypoints_3d,
)

COURT_SEMANTIC_MANIFEST_SCHEMA = COURT_SEMANTIC_MANIFEST_SCHEMA_V1
COURT_SEMANTIC_MANIFEST_PATH = "diagnostics/semantic-manifest.json"

_OPERATIONAL_FIELDS = {
    "alpha",
    "alpha_preview",
    "cpu_seconds",
    "cuda_peak_bytes",
    "dense_reference_bytes",
    "depth",
    "directory",
    "generated_bytes",
    "image_path",
    "labels",
    "peak_rss_bytes",
    "published_bytes",
    "raw_pixels",
    "rgb",
    "rgb_preview",
    "wall_seconds",
}
_PUBLISHED_COURT_TRANSFORM_ATOL = 1.0e-8


def build_court_semantic_manifest(
    dataset: Mapping[str, object],
) -> dict[str, object]:
    """Derive the stable semantic projection of one validated ``dataset.json``.

    The result deliberately contains no filesystem location, runtime measurement,
    byte count, or rendered pixel payload.  It retains the renderer-derived label
    semantics needed to compare independent same-seed executions.
    """
    definition = court_schema_from_dataset_schema(dataset.get("schema"))
    groups = _mapping_sequence(
        dataset.get("trajectory_groups"), name="trajectory_groups"
    )
    accepted = _mapping_sequence(dataset.get("samples"), name="samples")
    rejected = _mapping_sequence(
        dataset.get("rejected_samples"), name="rejected_samples"
    )
    policy = _mapping(dataset.get("sampling_policy"), name="sampling_policy")
    metrics = _mapping(dataset.get("metrics"), name="metrics")
    metadata_fields = _string_sequence(
        dataset.get("metadata_fields"),
        name="metadata_fields",
    )
    published_court_geometry = (
        validate_v2_published_court_geometry(dataset)
        if definition.version is CourtDatasetSchemaVersion.V2
        else None
    )

    ordered_groups = sorted(groups, key=_trajectory_group_id)
    if len({_trajectory_group_id(group) for group in ordered_groups}) != len(
        ordered_groups
    ):
        raise ValueError("Court semantic manifest group IDs must be unique.")

    sample_entries: list[dict[str, object]] = []
    accepted_split_counts: Counter[str] = Counter()
    rejected_split_counts: Counter[str] = Counter()
    derived_visibility: Counter[str] = Counter()
    derived_coverage: Counter[str] = Counter()
    for record in accepted:
        entry, visibility, coverage = _sample_entry(
            record,
            disposition="accepted",
            metadata_fields=metadata_fields,
            definition=definition,
            published_court_geometry=published_court_geometry,
        )
        sample_entries.append(entry)
        accepted_split_counts[_text(entry["split"], name="sample.split")] += 1
        derived_visibility.update(visibility)
        derived_coverage.update(coverage)
    for record in rejected:
        entry, _visibility, _coverage = _sample_entry(
            record,
            disposition="rejected",
            metadata_fields=metadata_fields,
            definition=definition,
            published_court_geometry=published_court_geometry,
        )
        sample_entries.append(entry)
        rejected_split_counts[_text(entry["split"], name="sample.split")] += 1
    sample_entries.sort(
        key=lambda entry: (
            _integer(entry["sample_index"], name="sample_index", minimum=0),
            _text(entry["sample_id"], name="sample_id"),
        )
    )
    indices = [
        _integer(entry["sample_index"], name="sample_index", minimum=0)
        for entry in sample_entries
    ]
    if indices != list(range(len(sample_entries))):
        raise ValueError(
            "Court semantic manifest samples must cover proposal indices exactly."
        )
    sample_ids = [
        _text(entry["sample_id"], name="sample_id") for entry in sample_entries
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Court semantic manifest sample IDs must be unique.")

    visible_counts = _class_counts(
        metrics.get("renderer_visible_points_by_class"),
        name="renderer_visible_points_by_class",
        definition=definition,
    )
    if visible_counts != {
        name: derived_visibility[name] for name in definition.semantic_class_names
    }:
        raise ValueError(
            "Court renderer-visible class metrics disagree with sample semantics."
        )
    coverage_counts = _positive_count_mapping(
        metrics.get("coverage_counts"),
        name="coverage_counts",
    )
    if coverage_counts != dict(sorted(derived_coverage.items())):
        raise ValueError("Court coverage metrics disagree with sample semantics.")

    proposal_count = _integer_metric(metrics, "proposal_count", minimum=1)
    accepted_count = _integer_metric(metrics, "accepted_frame_count", minimum=1)
    rejected_count = _integer_metric(metrics, "rejected_frame_count", minimum=0)
    group_count = _integer_metric(metrics, "trajectory_group_count", minimum=1)
    if (
        proposal_count != len(sample_entries)
        or accepted_count != len(accepted)
        or rejected_count != len(rejected)
        or group_count != len(ordered_groups)
    ):
        raise ValueError(
            "Court semantic manifest counts disagree with dataset inventory."
        )
    metric_split_counts = _positive_count_mapping(
        metrics.get("split_frame_counts"),
        name="split_frame_counts",
    )
    if metric_split_counts != dict(sorted(accepted_split_counts.items())):
        raise ValueError("Court split metrics disagree with accepted semantic samples.")

    manifest: dict[str, object] = {
        "schema": definition.semantic_manifest_schema,
        "dataset_schema": definition.dataset_schema,
        "sample_schema": definition.sample_schema,
        "scene_id": _text(dataset.get("scene_id"), name="scene_id"),
        "profile": _text(dataset.get("profile"), name="profile"),
        "seed": _integer(dataset.get("seed"), name="seed", minimum=0),
        "sampling_policy": dict(policy),
        "metadata_fields": list(metadata_fields),
        "semantic_schema": {
            "class_names": list(definition.semantic_class_names),
            "physical_point_count_per_class": definition.points_per_class,
            "renderer_visibility_source": "validated_nht_alpha_and_depth",
        },
        "counts": {
            "proposal_count": proposal_count,
            "accepted_sample_count": accepted_count,
            "rejected_sample_count": rejected_count,
            "trajectory_group_count": group_count,
            "accepted_by_split": dict(sorted(accepted_split_counts.items())),
            "rejected_by_split": dict(sorted(rejected_split_counts.items())),
            "coverage_modes": coverage_counts,
            "renderer_visible_points_by_class": visible_counts,
        },
        "trajectory_groups": [dict(group) for group in ordered_groups],
        "samples": sample_entries,
    }
    _reject_operational_fields(manifest)
    return cast(dict[str, object], _canonicalize(manifest))


def validate_court_semantic_manifest(
    dataset: Mapping[str, object],
    semantic_manifest: object,
) -> dict[str, object]:
    """Recompute the canonical manifest and require exact parsed equality."""
    if not isinstance(semantic_manifest, Mapping):
        raise TypeError("Court semantic manifest must be a JSON object.")
    observed = dict(semantic_manifest)
    observed_definition = court_schema_from_semantic_manifest_schema(
        observed.get("schema")
    )
    dataset_definition = court_schema_from_dataset_schema(dataset["schema"])
    if observed_definition is not dataset_definition:
        raise ValueError("Court semantic manifest and dataset schemas are mixed.")
    _reject_operational_fields(observed)
    expected = build_court_semantic_manifest(dataset)
    if observed != expected:
        raise ValueError(
            "Court semantic manifest disagrees with the canonical dataset semantics."
        )
    return expected


def require_equal_court_semantic_manifests(
    first: object,
    second: object,
) -> None:
    """Fail when two independently parsed renderer-derived manifests differ."""
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        raise TypeError("Court repeat evidence must contain two semantic manifests.")
    _reject_operational_fields(first)
    _reject_operational_fields(second)
    if dict(first) != dict(second):
        raise ValueError("Same-seed Court semantic manifests are not exactly equal.")


def validate_v2_published_court_geometry(
    dataset: Mapping[str, object],
) -> dict[str, RigidTransform]:
    """Derive v2 court geometry from physical points and bind every target to it."""
    accepted = _mapping_sequence(dataset["samples"], name="samples")
    rejected = _mapping_sequence(
        dataset["rejected_samples"], name="rejected_samples"
    )
    records = (*accepted, *rejected)
    geometry_by_court: dict[str, RigidTransform] = {}
    expected_court_inventory: tuple[str, ...] | None = None
    points_court = np.asarray(
        court_keypoints_3d(STANDARD_COURT_CONFIG)[:14].numpy(),
        dtype=np.float64,
    )
    for record in records:
        projection = record.get("projection")
        if projection is None:
            continue
        raw_projection = _exact_mapping(
            projection,
            keys={
                "camera_id",
                "resolution",
                "coverage_modes",
                "visible_class_names",
                "visible_point_count",
                "courts",
            },
            name="sample.projection",
        )
        courts = _mapping_sequence(
            raw_projection["courts"], name="projection.courts"
        )
        court_inventory: list[str] = []
        for court in courts:
            raw_court = _exact_mapping(
                court,
                keys={"court_instance_id", "coverage_mode", "classes"},
                name="projection.court",
            )
            court_id = _text(
                raw_court["court_instance_id"], name="court_instance_id"
            )
            if court_id in court_inventory:
                raise ValueError(
                    "Court v2 projection contains duplicate court geometry."
                )
            court_inventory.append(court_id)
            points_by_index = _published_scene_points_by_physical_index(raw_court)
            points_scene = np.asarray(
                [points_by_index[index] for index in range(14)],
                dtype=np.float64,
            )
            if court_id not in geometry_by_court:
                geometry_by_court[court_id] = (
                    scene_from_court_from_published_points(points_by_index)
                )
            elif not np.allclose(
                geometry_by_court[court_id].apply(points_court),
                points_scene,
                atol=PUBLISHED_COURT_GEOMETRY_ATOL_M,
                rtol=0.0,
            ):
                raise ValueError(
                    "Court v2 physical geometry disagrees across published samples."
                )
        current_inventory = tuple(court_inventory)
        if expected_court_inventory is None:
            expected_court_inventory = current_inventory
        elif current_inventory != expected_court_inventory:
            raise ValueError(
                "Court v2 published court geometry inventory changed across samples."
            )
    if not geometry_by_court or expected_court_inventory is None:
        raise ValueError("Court v2 requires published physical geometry for every court.")

    candidate_by_court: dict[str, str] = {}
    for record in records:
        target = ResolvedTargetCourtV2.from_mapping(record.get("target_court"))
        binding = target.binding
        try:
            published_transform = geometry_by_court[binding.court_instance_id]
        except KeyError as error:
            raise ValueError(
                "Court v2 target binding references unpublished court geometry."
            ) from error
        if not np.allclose(
            binding.scene_from_court.matrix(),
            published_transform.matrix(),
            atol=_PUBLISHED_COURT_TRANSFORM_ATOL,
            rtol=0.0,
        ):
            raise ValueError(
                "Court v2 target binding disagrees with published physical geometry."
            )
        if binding.court_instance_id not in candidate_by_court:
            candidate_by_court[binding.court_instance_id] = binding.candidate_id
        elif candidate_by_court[binding.court_instance_id] != binding.candidate_id:
            raise ValueError(
                "Court v2 target candidate disagrees across published samples."
            )
    if set(candidate_by_court) != set(geometry_by_court):
        raise ValueError(
            "Court v2 requires one target candidate binding for every published court."
        )
    return dict(sorted(geometry_by_court.items()))


def _sample_entry(
    record: Mapping[str, object],
    *,
    disposition: str,
    metadata_fields: tuple[str, ...],
    definition: CourtSchemaDefinition,
    published_court_geometry: Mapping[str, RigidTransform] | None,
) -> tuple[dict[str, object], Counter[str], Counter[str]]:
    scene_camera = SceneCamera.from_dict(record.get("camera"))
    camera = _camera_semantics(record.get("camera"))
    sample_id = _text(record.get("sample_id"), name="sample_id")
    if camera["camera_id"] != sample_id:
        raise ValueError("Court semantic camera ID disagrees with sample ID.")
    reasons: list[str]
    if disposition == "accepted":
        reasons = []
    else:
        raw_reasons = record.get("reasons")
        reasons = list(_string_sequence(raw_reasons, name="rejection reasons"))
        if not reasons:
            raise ValueError("Rejected Court semantic samples require reasons.")
    raw_projection = record.get("projection")
    if raw_projection is None:
        if (
            definition.version is not CourtDatasetSchemaVersion.V2
            or disposition != "rejected"
            or published_court_geometry is None
        ):
            raise ValueError("Only an ambiguous v2 rejection may omit projection.")
        _validate_ambiguous_near_far_rejection(
            reasons,
            camera=scene_camera,
            published_court_geometry=published_court_geometry,
        )
        projection = None
        visibility: Counter[str] = Counter()
        coverage: Counter[str] = Counter()
    else:
        if any(_is_ambiguous_near_far_reason(reason) for reason in reasons):
            raise ValueError(
                "Court v2 ambiguity reasons require a null semantic projection."
            )
        projection, visibility, coverage = _projection_summary(
            raw_projection,
            accepted=disposition == "accepted",
            definition=definition,
            camera=scene_camera,
            published_court_geometry=published_court_geometry,
        )
        if projection["camera_id"] != sample_id:
            raise ValueError("Court semantic projection ID disagrees with sample ID.")
    metadata = _mapping(record.get("metadata"), name="sample.metadata")
    if tuple(metadata) != metadata_fields:
        raise ValueError("Court semantic sample metadata field order changed.")

    entry = {
        "sample_index": _integer(
            record.get("sample_index"), name="sample_index", minimum=0
        ),
        "sample_id": sample_id,
        "trajectory_group_id": _text(
            record.get("trajectory_group_id"), name="trajectory_group_id"
        ),
        "trajectory_id": _text(record.get("trajectory_id"), name="trajectory_id"),
        "view_id": _text(record.get("view_id"), name="view_id"),
        "trajectory_frame_index": _integer(
            record.get("trajectory_frame_index"),
            name="trajectory_frame_index",
            minimum=0,
        ),
        "split": _text(record.get("split"), name="split"),
        "shard_id": _text(record.get("shard_id"), name="shard_id"),
        "disposition": disposition,
        "rejection_reasons": reasons,
        "camera": camera,
        "semantic_projection": projection,
        "metadata": dict(metadata),
    }
    if definition.version is CourtDatasetSchemaVersion.V2:
        entry["target_court"] = ResolvedTargetCourtV2.from_mapping(
            record.get("target_court")
        ).to_dict()
    return entry, visibility, coverage


def _camera_semantics(value: object) -> dict[str, object]:
    camera = SceneCamera.from_dict(value)
    return {
        "camera_id": camera.camera_id,
        "source_frame_index": camera.source_frame_index,
        "width": camera.width,
        "height": camera.height,
        "intrinsics": list(camera.intrinsics),
        "camera_to_scene": camera.camera_to_scene.to_list(),
    }


def _projection_summary(
    value: object,
    *,
    accepted: bool,
    definition: CourtSchemaDefinition,
    camera: SceneCamera,
    published_court_geometry: Mapping[str, RigidTransform] | None,
) -> tuple[dict[str, object], Counter[str], Counter[str]]:
    projection = _exact_mapping(
        value,
        keys={
            "camera_id",
            "resolution",
            "coverage_modes",
            "visible_class_names",
            "visible_point_count",
            "courts",
        },
        name="sample.projection",
    )
    resolution = _integer_sequence(
        projection["resolution"], name="projection.resolution", size=2, minimum=2
    )
    courts = _mapping_sequence(projection["courts"], name="projection.courts")
    if not courts:
        raise ValueError("Court semantic projection must contain accepted courts.")
    court_ids: set[str] = set()
    visibility: Counter[str] = Counter()
    coverage: Counter[str] = Counter()
    summaries: list[dict[str, object]] = []
    for court in courts:
        raw_court = _exact_mapping(
            court,
            keys={"court_instance_id", "coverage_mode", "classes"},
            name="projection.court",
        )
        court_id = _text(raw_court["court_instance_id"], name="court_instance_id")
        if court_id in court_ids:
            raise ValueError("Court semantic projection contains duplicate court IDs.")
        court_ids.add(court_id)
        coverage_mode = _text(raw_court["coverage_mode"], name="coverage_mode")
        classes = _mapping_sequence(raw_court["classes"], name="projection.classes")
        if len(classes) != definition.semantic_class_count:
            raise ValueError(
                "Court semantic projection class count disagrees with its schema."
            )
        court_counts: dict[str, int] = {}
        in_frame_point_count = 0
        for class_id, semantic_class in enumerate(classes):
            raw_class = _exact_mapping(
                semantic_class,
                keys={"class_id", "class_name", "renderer_visible", "points"},
                name="projection.class",
            )
            class_name = definition.semantic_class_names[class_id]
            if (
                _integer(raw_class["class_id"], name="class_id", minimum=0) != class_id
                or raw_class["class_name"] != class_name
            ):
                raise ValueError("Court semantic class identity/order changed.")
            points = _mapping_sequence(raw_class["points"], name="projection.points")
            if len(points) != definition.points_per_class:
                raise ValueError(
                    "Court semantic class cardinality disagrees with its schema."
                )
            physical_indices: tuple[int, ...]
            if definition.version is CourtDatasetSchemaVersion.V1:
                physical_indices = PHYSICAL_INDICES_BY_CLASS[class_id]
            else:
                physical_indices = (
                    _integer(
                        points[0].get("physical_index"),
                        name="physical_index",
                        minimum=0,
                    ),
                )
            visible_count = 0
            for point, physical_index in zip(points, physical_indices, strict=True):
                raw_point = _exact_mapping(
                    point,
                    keys={
                        "physical_index",
                        "uv",
                        "camera_depth_m",
                        "scene_xyz_m",
                        "in_front",
                        "in_frame",
                        "renderer_visible",
                    },
                    name="projection.point",
                )
                if raw_point["physical_index"] != physical_index:
                    raise ValueError("Court semantic physical point identity changed.")
                _number_sequence(raw_point["uv"], name="point.uv", size=2)
                _number(raw_point["camera_depth_m"], name="point.camera_depth_m")
                _number_sequence(
                    raw_point["scene_xyz_m"], name="point.scene_xyz_m", size=3
                )
                _boolean(raw_point["in_front"], name="point.in_front")
                in_frame_point_count += int(
                    _boolean(raw_point["in_frame"], name="point.in_frame")
                )
                renderer_visible = raw_point["renderer_visible"]
                if renderer_visible is None:
                    if accepted:
                        raise ValueError(
                            "Accepted Court semantics require renderer visibility."
                        )
                else:
                    visible_count += int(
                        _boolean(renderer_visible, name="point.renderer_visible")
                    )
            class_visible = _boolean(
                raw_class["renderer_visible"], name="class.renderer_visible"
            )
            if class_visible != (visible_count > 0):
                raise ValueError("Court semantic class visibility is inconsistent.")
            court_counts[class_name] = visible_count
            visibility[class_name] += visible_count
        derived_coverage_mode = coverage_mode_from_in_frame_point_count(
            in_frame_point_count
        )
        if coverage_mode != derived_coverage_mode:
            raise ValueError(
                "Court semantic coverage mode disagrees with physical in-frame points."
            )
        coverage[derived_coverage_mode] += 1
        if definition.version is CourtDatasetSchemaVersion.V2:
            if published_court_geometry is None:
                raise ValueError("Court v2 published geometry is unavailable.")
            try:
                scene_from_court = published_court_geometry[court_id]
            except KeyError as error:
                raise ValueError(
                    "Court v2 projection references unpublished court geometry."
                ) from error
            observed_indices = tuple(
                _integer(
                    _mapping_sequence(
                        _mapping(classes[index], name="projection.class")["points"],
                        name="projection.points",
                    )[0].get("physical_index"),
                    name="physical_index",
                    minimum=0,
                )
                for index in range(len(classes))
            )
            expected_indices = _camera_relative_v2_indices(
                camera,
                scene_from_court,
            )
            if observed_indices != expected_indices:
                raise ValueError(
                    "Court v2 physical mapping disagrees with camera-relative near/far."
                )
        summaries.append(
            {
                "court_instance_id": court_id,
                "coverage_mode": derived_coverage_mode,
                "renderer_visible_points_by_class": court_counts,
            }
        )

    coverage_modes = _string_sequence(
        projection["coverage_modes"], name="projection.coverage_modes"
    )
    if tuple(coverage_modes) != tuple(
        summary["coverage_mode"] for summary in summaries
    ):
        raise ValueError("Court semantic projection coverage summary changed.")
    visible_class_names = _string_sequence(
        projection["visible_class_names"], name="projection.visible_class_names"
    )
    expected_visible_names = tuple(
        name for name in definition.semantic_class_names if visibility[name] > 0
    )
    if visible_class_names != expected_visible_names:
        raise ValueError("Court semantic visible class summary changed.")
    visible_point_count = _integer(
        projection["visible_point_count"], name="visible_point_count", minimum=0
    )
    if visible_point_count != sum(visibility.values()):
        raise ValueError("Court semantic visible point count changed.")
    return (
        {
            "camera_id": _text(projection["camera_id"], name="projection.camera_id"),
            "resolution": list(resolution),
            "coverage_modes": list(coverage_modes),
            "visible_class_names": list(visible_class_names),
            "visible_point_count": visible_point_count,
            "courts": summaries,
        },
        visibility,
        coverage,
    )


def _trajectory_group_id(group: Mapping[str, object]) -> str:
    trajectory = _mapping(group.get("trajectory"), name="group.trajectory")
    return _text(trajectory.get("trajectory_group_id"), name="trajectory_group_id")


def _class_counts(
    value: object,
    *,
    name: str,
    definition: CourtSchemaDefinition,
) -> dict[str, int]:
    raw = _mapping(value, name=name)
    if set(raw) != set(definition.semantic_class_names):
        raise ValueError(f"{name} semantic classes disagree with its schema.")
    return {
        class_name: _integer(raw[class_name], name=class_name, minimum=0)
        for class_name in definition.semantic_class_names
    }


def _camera_relative_v2_indices(
    camera: SceneCamera,
    scene_from_court: RigidTransform,
) -> tuple[int, ...]:
    """Recompute the camera end from authoritative published court geometry."""
    local_y = camera_center_court_y(
        camera,
        scene_from_court=scene_from_court,
    )
    if abs(local_y) <= CAMERA_RELATIVE_MID_PLANE_TOLERANCE_M:
        raise ValueError("Accepted Court v2 projection has ambiguous near/far.")
    if local_y < 0.0:
        return tuple(range(14))
    return tuple(int(value) for value in OPPOSITE_COURT_END_INDEX)


def _published_scene_points_by_physical_index(
    court: Mapping[str, object],
) -> dict[int, tuple[float, float, float]]:
    """Extract one exact named physical court from a serialized projection."""
    classes = _mapping_sequence(court.get("classes"), name="projection.classes")
    points_by_index: dict[int, tuple[float, float, float]] = {}
    for semantic_class in classes:
        points = _mapping_sequence(
            semantic_class.get("points"), name="projection.points"
        )
        for point in points:
            physical_index = _integer(
                point.get("physical_index"),
                name="physical_index",
                minimum=0,
            )
            if physical_index in points_by_index:
                raise ValueError(
                    "Court v2 published physical geometry contains duplicate indices."
                )
            scene_xyz = _number_sequence(
                point.get("scene_xyz_m"),
                name="point.scene_xyz_m",
                size=3,
            )
            points_by_index[physical_index] = (
                scene_xyz[0],
                scene_xyz[1],
                scene_xyz[2],
            )
    if set(points_by_index) != set(range(14)):
        raise ValueError(
            "Court v2 published physical geometry must contain indices 0..13 once."
        )
    return points_by_index


def _validate_ambiguous_near_far_rejection(
    reasons: Sequence[str],
    *,
    camera: SceneCamera,
    published_court_geometry: Mapping[str, RigidTransform],
) -> None:
    """Require one exact ambiguity reason and prove it from the published court."""
    if len(reasons) != 1 or not _is_ambiguous_near_far_reason(reasons[0]):
        raise ValueError(
            "A null Court v2 projection requires exactly one ambiguity reason."
        )
    court_id = reasons[0].removeprefix(
        f"{AMBIGUOUS_CAMERA_RELATIVE_NEAR_FAR_REASON}:"
    )
    try:
        scene_from_court = published_court_geometry[court_id]
    except KeyError as error:
        raise ValueError(
            "Court v2 ambiguity reason references unpublished court geometry."
        ) from error
    local_y = camera_center_court_y(
        camera,
        scene_from_court=scene_from_court,
    )
    if abs(local_y) > CAMERA_RELATIVE_MID_PLANE_TOLERANCE_M:
        raise ValueError(
            "Court v2 ambiguity reason disagrees with camera/court geometry."
        )


def _is_ambiguous_near_far_reason(reason: str) -> bool:
    prefix, separator, court_id = reason.partition(":")
    return (
        prefix == AMBIGUOUS_CAMERA_RELATIVE_NEAR_FAR_REASON
        and separator == ":"
        and bool(court_id)
        and court_id == court_id.strip()
    )


def _positive_count_mapping(value: object, *, name: str) -> dict[str, int]:
    raw = _mapping(value, name=name)
    result: dict[str, int] = {}
    for key, item in raw.items():
        key_text = _text(key, name=f"{name}.key")
        result[key_text] = _integer(item, name=f"{name}.{key_text}", minimum=1)
    return dict(sorted(result.items()))


def _integer_metric(metrics: Mapping[str, object], key: str, *, minimum: int) -> int:
    return _integer(metrics.get(key), name=key, minimum=minimum)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping.")
    return value


def _exact_mapping(
    value: object,
    *,
    keys: set[str],
    name: str,
) -> Mapping[str, object]:
    raw = _mapping(value, name=name)
    if set(raw) != keys:
        raise ValueError(f"{name} schema is invalid.")
    return raw


def _mapping_sequence(value: object, *, name: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of mappings.")
    return tuple(_mapping(item, name=name) for item in value)


def _string_sequence(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a string sequence.")
    return tuple(_text(item, name=name) for item in value)


def _integer_sequence(
    value: object,
    *,
    name: str,
    size: int,
    minimum: int,
) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an integer sequence.")
    result = tuple(_integer(item, name=name, minimum=minimum) for item in value)
    if len(result) != size:
        raise ValueError(f"{name} must contain exactly {size} values.")
    return result


def _number_sequence(value: object, *, name: str, size: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence.")
    result = tuple(_number(item, name=name) for item in value)
    if len(result) != size:
        raise ValueError(f"{name} must contain exactly {size} values.")
    return result


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{name} must be a non-empty trimmed string.")
    return value


def _integer(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{name} must be an integer >= {minimum}.")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean.")
    return value


def _reject_operational_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Court semantic manifest keys must be strings.")
            if (
                key in _OPERATIONAL_FIELDS
                or key == "path"
                or key.endswith("_path")
                or key.endswith("_bytes")
                or key.endswith("_seconds")
            ):
                raise ValueError(
                    f"Court semantic manifest contains operational field {key!r}."
                )
            _reject_operational_fields(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _reject_operational_fields(item)


def _canonicalize(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonicalize(item) for item in value]
    return value


__all__ = [
    "COURT_SEMANTIC_MANIFEST_PATH",
    "COURT_SEMANTIC_MANIFEST_SCHEMA",
    "build_court_semantic_manifest",
    "require_equal_court_semantic_manifests",
    "validate_court_semantic_manifest",
    "validate_v2_published_court_geometry",
]
