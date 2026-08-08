"""Fixed-path serialization and semantic validation for alignment outputs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.alignment.contracts import (
    ALIGNMENT_COORDINATE_CONVENTION,
    AlignmentEvidence,
    AlignmentEvidenceDiagnostics,
    AlignmentPartitions,
    AlignmentResult,
    CameraLineDiagnostics,
    CandidateEvidence,
    CandidateScaleDiagnostics,
    CorrespondenceSet,
    MeasuredCameraLines,
    MetricSceneAdapter,
)
from src.synthetic_data_generation.alignment.fitting import fit_alignment
from src.synthetic_data_generation.scene_contract import (
    CourtInstance,
    MultiCourtLayout,
    RigidTransform,
)

GROUND_LINE_MAP_FILE = "ground-line-map.npz"
COURT_GEOMETRY_FILE = "court-geometry.json"
ALIGNMENT_FILE = "alignment.json"
DIAGNOSTICS_DIRECTORY = "diagnostics"

_GROUND_LINE_KEYS = {
    "schema",
    "primary_candidate_id",
    "candidate_ids",
    "court_instance_ids",
    "fit_camera_ids",
    "holdout_camera_ids",
    "fit_candidate_index",
    "fit_camera_index",
    "fit_points_court",
    "fit_points_scene",
    "holdout_candidate_index",
    "holdout_camera_index",
    "holdout_points_court",
    "holdout_points_scene",
    "complex_points_scene",
    "nht_scene_from_metric_scene",
    "metric_scene_from_nht_scene",
    "nht_scene_units_per_metre",
    "diagnostic_selected_line_pixel_count",
    "diagnostic_projected_line_point_count",
    "diagnostic_candidate_scales",
    "diagnostic_template_scores",
    "diagnostic_maximum_relative_scale_deviation",
    "line_camera_index",
    "line_points_nht_scene",
}
_DIAGNOSTIC_FILES = {"candidate-metrics.json", "evidence.json", "summary.txt"}


class _SavezCompressed(Protocol):
    def __call__(self, file: Path, **arrays: NDArray[Any]) -> None: ...


_savez_compressed = cast(_SavezCompressed, np.savez_compressed)


def write_alignment_outputs(
    staging_path: Path,
    *,
    evidence: AlignmentEvidence,
    result: AlignmentResult,
) -> None:
    """Write only the declared fixed files beneath one provided staging path."""
    _require_staging_directory(staging_path)
    if any(staging_path.iterdir()):
        raise ValueError("Alignment staging must be empty before execution.")
    if result.partitions != evidence.partitions:
        raise ValueError("Alignment result partitions disagree with source evidence.")
    candidate_ids = tuple(candidate.candidate_id for candidate in result.candidates)
    evidence_candidate_ids = tuple(
        candidate.candidate_id for candidate in evidence.candidates
    )
    if candidate_ids != evidence_candidate_ids:
        raise ValueError(
            "Alignment result candidate order disagrees with source evidence."
        )

    archive = _evidence_archive(evidence)
    _savez_compressed(staging_path / GROUND_LINE_MAP_FILE, **archive)
    _write_json(staging_path / COURT_GEOMETRY_FILE, _court_geometry_payload(result))
    _write_json(staging_path / ALIGNMENT_FILE, result.to_dict())
    diagnostics = staging_path / DIAGNOSTICS_DIRECTORY
    diagnostics.mkdir(parents=False, exist_ok=False)
    _write_json(diagnostics / "candidate-metrics.json", _metrics_payload(result))
    _write_json(diagnostics / "evidence.json", evidence.diagnostics.to_dict())
    (diagnostics / "summary.txt").write_text(_human_summary(result), encoding="utf-8")


def validate_alignment_outputs(output_path: Path) -> AlignmentResult:
    """Validate the complete fixed inventory and cross-file semantic agreement."""
    _require_output_directory(output_path)
    expected = {
        GROUND_LINE_MAP_FILE,
        COURT_GEOMETRY_FILE,
        ALIGNMENT_FILE,
        DIAGNOSTICS_DIRECTORY,
    }
    actual = {path.name for path in output_path.iterdir()}
    if actual != expected:
        raise ValueError(
            "Alignment output inventory mismatch; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}."
        )
    diagnostics = output_path / DIAGNOSTICS_DIRECTORY
    if not diagnostics.is_dir() or diagnostics.is_symlink():
        raise ValueError("Alignment diagnostics must be an ordinary directory.")
    diagnostic_actual = {path.name for path in diagnostics.iterdir()}
    if diagnostic_actual != _DIAGNOSTIC_FILES:
        raise ValueError(
            "Alignment diagnostics inventory does not match the fixed schema."
        )
    if any(not path.is_file() or path.is_symlink() for path in diagnostics.iterdir()):
        raise ValueError("Alignment diagnostics must contain ordinary files only.")

    result = load_alignment_result(output_path / ALIGNMENT_FILE)
    evidence = _load_evidence_archive(output_path / GROUND_LINE_MAP_FILE)
    try:
        recomputed = fit_alignment(evidence, policy=result.policy)
    except ValueError as error:
        raise ValueError(
            "Ground-line evidence does not produce an accepted alignment."
        ) from error
    if recomputed.to_dict() != result.to_dict():
        raise ValueError(
            "Ground-line evidence disagrees with the fitted alignment result."
        )

    geometry = _load_json_object(output_path / COURT_GEOMETRY_FILE)
    if geometry != _court_geometry_payload(result):
        raise ValueError(
            "court-geometry.json disagrees with the final alignment result."
        )
    metrics = _load_json_object(diagnostics / "candidate-metrics.json")
    if metrics != _metrics_payload(result):
        raise ValueError("Alignment diagnostics disagree with the final result.")
    evidence_diagnostics = _load_json_object(diagnostics / "evidence.json")
    if evidence_diagnostics != evidence.diagnostics.to_dict():
        raise ValueError(
            "Measured evidence diagnostics disagree with the ground-line archive."
        )
    if (diagnostics / "summary.txt").read_text(encoding="utf-8") != _human_summary(
        result
    ):
        raise ValueError("Human alignment summary disagrees with the final result.")
    return result


def load_alignment_result(path: Path) -> AlignmentResult:
    """Load exactly one canonical ``alignment.json`` and fail closed on semantics."""
    if path.name != ALIGNMENT_FILE:
        raise ValueError(f"Expected the fixed alignment file name {ALIGNMENT_FILE!r}.")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Alignment result is not an ordinary file: {path}")
    return AlignmentResult.from_dict(_load_json_object(path))


def load_accepted_layout(alignment_directory: Path) -> MultiCourtLayout:
    """Validate the full fixed inventory and return the downstream court authority."""
    if alignment_directory.name != "alignment":
        raise ValueError("Expected the fixed alignment owner directory.")
    result = validate_alignment_outputs(alignment_directory)
    if any(
        not candidate.accepted
        for candidate in result.candidates
        if candidate.court_instance_id
        in {court.court_instance_id for court in result.layout.courts}
    ):
        raise ValueError("A rejected court candidate is present in MultiCourtLayout.")
    return result.layout


def validate_projection_equivalence(
    court: CourtInstance,
    *,
    camera_to_court: RigidTransform,
    intrinsics: Sequence[float],
    points_court: NDArray[np.floating[Any]],
    atol: float = 1.0e-7,
) -> float:
    """Require court-local and transformed scene-space projections to agree."""
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    if intrinsic.size != 9:
        raise ValueError("intrinsics must contain exactly nine values.")
    intrinsic = intrinsic.reshape(3, 3)
    if (
        not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0.0
        or intrinsic[1, 1] <= 0.0
    ):
        raise ValueError("intrinsics must be finite with positive focal lengths.")
    if not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), atol=1.0e-9, rtol=0.0):
        raise ValueError("intrinsics must have homogeneous bottom row [0, 0, 1].")
    if not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("atol must be positive and finite.")
    court_points = np.asarray(points_court, dtype=np.float64)
    if (
        court_points.ndim != 2
        or court_points.shape[1] != 3
        or not np.isfinite(court_points).all()
    ):
        raise ValueError("points_court must be a finite (N, 3) array.")
    camera_points_local = camera_to_court.inverse().apply(court_points)
    scene_from_camera = RigidTransform.from_matrix(
        court.scene_from_court.matrix() @ camera_to_court.matrix()
    )
    scene_points = court.scene_from_court.apply(court_points)
    camera_points_scene_route = scene_from_camera.inverse().apply(scene_points)
    local_pixels = _project(camera_points_local, intrinsic)
    scene_pixels = _project(camera_points_scene_route, intrinsic)
    error = (
        float(np.max(np.abs(local_pixels - scene_pixels))) if len(local_pixels) else 0.0
    )
    if error > atol:
        raise ValueError(
            f"Court-local and scene-space projection differ by {error:.3g}, above {atol:.3g}."
        )
    return error


def validate_court_transform_binding(
    layout: MultiCourtLayout,
    *,
    court_instance_id: str,
    candidate_id: str,
    transforms: Mapping[str, RigidTransform],
    atol: float = 1.0e-7,
) -> CourtInstance:
    """Reject camera/trajectory metadata that do not share one target-court transform."""
    if not transforms:
        raise ValueError("At least one named transform binding is required.")
    if any(not isinstance(name, str) or not name.strip() for name in transforms):
        raise TypeError("Transform binding names must be non-empty strings.")
    if any(
        not isinstance(transform, RigidTransform) for transform in transforms.values()
    ):
        raise TypeError("Transform bindings must contain only RigidTransform values.")
    if not np.isfinite(atol) or atol <= 0.0:
        raise ValueError("atol must be positive and finite.")
    court = layout.court(court_instance_id)
    if court.candidate_id != candidate_id:
        raise ValueError("candidate_id disagrees with the selected target court.")
    expected = court.scene_from_court.matrix()
    mismatches = [
        name
        for name, transform in transforms.items()
        if not np.allclose(transform.matrix(), expected, atol=atol, rtol=0.0)
    ]
    if mismatches:
        raise ValueError(
            f"Target-court transform mismatch for bindings: {sorted(mismatches)}."
        )
    return court


def _evidence_archive(evidence: AlignmentEvidence) -> dict[str, NDArray[Any]]:
    all_camera_ids = (
        evidence.partitions.fit_camera_ids + evidence.partitions.holdout_camera_ids
    )
    camera_indices = {
        camera_id: index for index, camera_id in enumerate(all_camera_ids)
    }
    fit_points_court: list[NDArray[np.float64]] = []
    fit_points_scene: list[NDArray[np.float64]] = []
    fit_candidate_index: list[NDArray[np.int32]] = []
    fit_camera_index: list[NDArray[np.int32]] = []
    holdout_points_court: list[NDArray[np.float64]] = []
    holdout_points_scene: list[NDArray[np.float64]] = []
    holdout_candidate_index: list[NDArray[np.int32]] = []
    holdout_camera_index: list[NDArray[np.int32]] = []
    for index, candidate in enumerate(evidence.candidates):
        fit_count = len(candidate.fit.points_court)
        holdout_count = len(candidate.holdout.points_court)
        fit_points_court.append(candidate.fit.points_court)
        fit_points_scene.append(candidate.fit.points_scene)
        fit_candidate_index.append(np.full(fit_count, index, dtype=np.int32))
        fit_camera_index.append(
            np.asarray(
                [camera_indices[item] for item in candidate.fit.camera_ids],
                dtype=np.int32,
            )
        )
        holdout_points_court.append(candidate.holdout.points_court)
        holdout_points_scene.append(candidate.holdout.points_scene)
        holdout_candidate_index.append(np.full(holdout_count, index, dtype=np.int32))
        holdout_camera_index.append(
            np.asarray(
                [camera_indices[item] for item in candidate.holdout.camera_ids],
                dtype=np.int32,
            )
        )
    line_camera_index = np.concatenate(
        [
            np.full(len(item.points_nht_scene), index, dtype=np.int32)
            for index, item in enumerate(evidence.measured_camera_lines)
        ]
    )
    line_points_nht_scene = np.concatenate(
        [item.points_nht_scene for item in evidence.measured_camera_lines]
    )
    return {
        "schema": np.asarray("semantic_ground_line_correspondences_v2"),
        "primary_candidate_id": np.asarray(evidence.primary_candidate_id or ""),
        "candidate_ids": np.asarray(
            [candidate.candidate_id for candidate in evidence.candidates], dtype=np.str_
        ),
        "court_instance_ids": np.asarray(
            [candidate.court_instance_id for candidate in evidence.candidates],
            dtype=np.str_,
        ),
        "fit_camera_ids": np.asarray(evidence.partitions.fit_camera_ids, dtype=np.str_),
        "holdout_camera_ids": np.asarray(
            evidence.partitions.holdout_camera_ids, dtype=np.str_
        ),
        "fit_candidate_index": np.concatenate(fit_candidate_index),
        "fit_camera_index": np.concatenate(fit_camera_index),
        "fit_points_court": np.concatenate(fit_points_court),
        "fit_points_scene": np.concatenate(fit_points_scene),
        "holdout_candidate_index": np.concatenate(holdout_candidate_index),
        "holdout_camera_index": np.concatenate(holdout_camera_index),
        "holdout_points_court": np.concatenate(holdout_points_court),
        "holdout_points_scene": np.concatenate(holdout_points_scene),
        "complex_points_scene": evidence.complex_points_scene,
        "line_camera_index": line_camera_index,
        "line_points_nht_scene": line_points_nht_scene,
        "nht_scene_from_metric_scene": evidence.metric_adapter.nht_matrix(),
        "metric_scene_from_nht_scene": evidence.metric_adapter.metric_matrix(),
        "nht_scene_units_per_metre": np.asarray(
            evidence.metric_adapter.nht_scene_units_per_metre,
            dtype=np.float64,
        ),
        "diagnostic_selected_line_pixel_count": np.asarray(
            [item.selected_line_pixel_count for item in evidence.diagnostics.cameras],
            dtype=np.int64,
        ),
        "diagnostic_projected_line_point_count": np.asarray(
            [item.projected_line_point_count for item in evidence.diagnostics.cameras],
            dtype=np.int64,
        ),
        "diagnostic_candidate_scales": np.asarray(
            [
                item.nht_scene_units_per_metre
                for item in evidence.diagnostics.candidate_scales
            ],
            dtype=np.float64,
        ),
        "diagnostic_template_scores": np.asarray(
            [item.template_score for item in evidence.diagnostics.candidate_scales],
            dtype=np.float64,
        ),
        "diagnostic_maximum_relative_scale_deviation": np.asarray(
            evidence.diagnostics.maximum_relative_scale_deviation,
            dtype=np.float64,
        ),
    }


def _load_evidence_archive(path: Path) -> AlignmentEvidence:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Ground-line evidence is not an ordinary file: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != _GROUND_LINE_KEYS:
            raise ValueError("Ground-line archive keys do not match the strict schema.")
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    schema = arrays["schema"]
    if (
        schema.ndim != 0
        or schema.dtype.kind != "U"
        or str(schema.item()) != "semantic_ground_line_correspondences_v2"
    ):
        raise ValueError("Unsupported ground-line correspondence schema.")
    primary = arrays["primary_candidate_id"]
    if primary.ndim != 0 or primary.dtype.kind != "U":
        raise ValueError("primary_candidate_id must be a Unicode scalar.")
    primary_candidate_id = str(primary.item()) or None
    string_names = (
        "candidate_ids",
        "court_instance_ids",
        "fit_camera_ids",
        "holdout_camera_ids",
    )
    for name in string_names:
        if (
            arrays[name].ndim != 1
            or arrays[name].dtype.kind != "U"
            or len(arrays[name]) == 0
        ):
            raise ValueError(f"Ground-line {name} must be a non-empty Unicode vector.")
        values = arrays[name].tolist()
        if len(values) != len(set(values)):
            raise ValueError(f"Ground-line {name} values must be unique.")
    if set(arrays["fit_camera_ids"].tolist()).intersection(
        arrays["holdout_camera_ids"].tolist()
    ):
        raise ValueError("Ground-line fit and holdout camera IDs overlap.")
    if len(arrays["candidate_ids"]) != len(arrays["court_instance_ids"]):
        raise ValueError("Ground-line candidate and court ID counts differ.")
    all_camera_count = len(arrays["fit_camera_ids"]) + len(arrays["holdout_camera_ids"])
    candidate_count = len(arrays["candidate_ids"])
    _validate_adapter_and_diagnostics_arrays(
        arrays,
        camera_count=all_camera_count,
        candidate_count=candidate_count,
    )
    _validate_measured_line_arrays(arrays, camera_count=all_camera_count)
    _validate_archive_partition(
        arrays,
        prefix="fit",
        candidate_count=candidate_count,
        camera_count=all_camera_count,
        allowed_camera_indices=set(range(len(arrays["fit_camera_ids"]))),
    )
    _validate_archive_partition(
        arrays,
        prefix="holdout",
        candidate_count=candidate_count,
        camera_count=all_camera_count,
        allowed_camera_indices=set(
            range(len(arrays["fit_camera_ids"]), all_camera_count)
        ),
    )
    complex_points = arrays["complex_points_scene"]
    if (
        complex_points.dtype != np.float64
        or complex_points.ndim != 2
        or complex_points.shape[1] != 3
    ):
        raise ValueError("complex_points_scene must be a float64 (N, 3) array.")
    if (
        len(complex_points) < 2
        or not np.isfinite(complex_points).all()
        or np.any(np.ptp(complex_points, axis=0) <= 0.0)
    ):
        raise ValueError("complex_points_scene must define finite positive 3-D bounds.")
    candidate_ids = cast(list[str], arrays["candidate_ids"].tolist())
    court_instance_ids = cast(list[str], arrays["court_instance_ids"].tolist())
    fit_camera_ids = cast(list[str], arrays["fit_camera_ids"].tolist())
    holdout_camera_ids = cast(list[str], arrays["holdout_camera_ids"].tolist())
    all_camera_ids = fit_camera_ids + holdout_camera_ids
    candidates: list[CandidateEvidence] = []
    for index, (court_instance_id, candidate_id) in enumerate(
        zip(court_instance_ids, candidate_ids, strict=True)
    ):
        fit_mask = arrays["fit_candidate_index"] == index
        holdout_mask = arrays["holdout_candidate_index"] == index
        candidates.append(
            CandidateEvidence(
                court_instance_id=court_instance_id,
                candidate_id=candidate_id,
                fit=CorrespondenceSet(
                    points_court=arrays["fit_points_court"][fit_mask],
                    points_scene=arrays["fit_points_scene"][fit_mask],
                    camera_ids=tuple(
                        all_camera_ids[int(item)]
                        for item in arrays["fit_camera_index"][fit_mask]
                    ),
                ),
                holdout=CorrespondenceSet(
                    points_court=arrays["holdout_points_court"][holdout_mask],
                    points_scene=arrays["holdout_points_scene"][holdout_mask],
                    camera_ids=tuple(
                        all_camera_ids[int(item)]
                        for item in arrays["holdout_camera_index"][holdout_mask]
                    ),
                ),
            )
        )
    adapter = MetricSceneAdapter(
        nht_scene_from_metric_scene=tuple(
            float(value) for value in arrays["nht_scene_from_metric_scene"].ravel()
        ),
        metric_scene_from_nht_scene=tuple(
            float(value) for value in arrays["metric_scene_from_nht_scene"].ravel()
        ),
        nht_scene_units_per_metre=float(arrays["nht_scene_units_per_metre"].item()),
    )
    diagnostics = AlignmentEvidenceDiagnostics(
        cameras=tuple(
            CameraLineDiagnostics(
                camera_id=camera_id,
                selected_line_pixel_count=int(
                    arrays["diagnostic_selected_line_pixel_count"][index]
                ),
                projected_line_point_count=int(
                    arrays["diagnostic_projected_line_point_count"][index]
                ),
            )
            for index, camera_id in enumerate(all_camera_ids)
        ),
        candidate_scales=tuple(
            CandidateScaleDiagnostics(
                candidate_id=candidate_id,
                nht_scene_units_per_metre=float(
                    arrays["diagnostic_candidate_scales"][index]
                ),
                template_score=float(arrays["diagnostic_template_scores"][index]),
            )
            for index, candidate_id in enumerate(candidate_ids)
        ),
        common_nht_scene_units_per_metre=adapter.nht_scene_units_per_metre,
        maximum_relative_scale_deviation=float(
            arrays["diagnostic_maximum_relative_scale_deviation"].item()
        ),
    )
    return AlignmentEvidence(
        partitions=AlignmentPartitions(
            fit_camera_ids=tuple(fit_camera_ids),
            holdout_camera_ids=tuple(holdout_camera_ids),
        ),
        candidates=tuple(candidates),
        measured_camera_lines=tuple(
            MeasuredCameraLines(
                camera_id=camera_id,
                points_nht_scene=arrays["line_points_nht_scene"][
                    arrays["line_camera_index"] == index
                ],
            )
            for index, camera_id in enumerate(all_camera_ids)
        ),
        complex_points_scene=arrays["complex_points_scene"],
        primary_candidate_id=primary_candidate_id,
        metric_adapter=adapter,
        diagnostics=diagnostics,
    )


def _validate_measured_line_arrays(
    arrays: Mapping[str, NDArray[Any]],
    *,
    camera_count: int,
) -> None:
    points = arrays["line_points_nht_scene"]
    camera_index = arrays["line_camera_index"]
    if (
        points.dtype != np.float64
        or points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
    ):
        raise ValueError(
            "Measured NHT line points must be a finite float64 (N, 3) array."
        )
    if (
        camera_index.dtype != np.int32
        or camera_index.shape != (len(points),)
        or np.any(camera_index < 0)
        or np.any(camera_index >= camera_count)
    ):
        raise ValueError("Measured line camera indices are invalid.")
    if not np.array_equal(
        np.unique(camera_index),
        np.arange(camera_count, dtype=np.int32),
    ):
        raise ValueError("Every declared camera must retain measured line points.")


def _validate_adapter_and_diagnostics_arrays(
    arrays: Mapping[str, NDArray[Any]],
    *,
    camera_count: int,
    candidate_count: int,
) -> None:
    for name in ("nht_scene_from_metric_scene", "metric_scene_from_nht_scene"):
        array = arrays[name]
        if (
            array.dtype != np.float64
            or array.shape != (4, 4)
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"{name} must be a finite float64 (4, 4) array.")
    scalar_names = (
        "nht_scene_units_per_metre",
        "diagnostic_maximum_relative_scale_deviation",
    )
    for name in scalar_names:
        array = arrays[name]
        if (
            array.dtype != np.float64
            or array.ndim != 0
            or not np.isfinite(array.item())
        ):
            raise ValueError(f"{name} must be a finite float64 scalar.")
    camera_names = (
        "diagnostic_selected_line_pixel_count",
        "diagnostic_projected_line_point_count",
    )
    for name in camera_names:
        array = arrays[name]
        if (
            array.dtype != np.int64
            or array.shape != (camera_count,)
            or np.any(array < 1)
        ):
            raise ValueError(
                f"{name} must contain one positive int64 value per camera."
            )
    candidate_names = ("diagnostic_candidate_scales", "diagnostic_template_scores")
    for name in candidate_names:
        array = arrays[name]
        if (
            array.dtype != np.float64
            or array.shape != (candidate_count,)
            or not np.isfinite(array).all()
            or np.any(array <= 0.0)
        ):
            raise ValueError(
                f"{name} must contain one positive float64 value per candidate."
            )


def _validate_archive_partition(
    arrays: Mapping[str, NDArray[Any]],
    *,
    prefix: str,
    candidate_count: int,
    camera_count: int,
    allowed_camera_indices: set[int],
) -> None:
    points_court = arrays[f"{prefix}_points_court"]
    points_scene = arrays[f"{prefix}_points_scene"]
    candidate_index = arrays[f"{prefix}_candidate_index"]
    camera_index = arrays[f"{prefix}_camera_index"]
    if points_court.dtype != np.float64 or points_scene.dtype != np.float64:
        raise ValueError(f"{prefix} correspondence points must have dtype float64.")
    if (
        points_court.ndim != 2
        or points_court.shape[1] != 3
        or points_scene.shape != points_court.shape
        or not np.isfinite(points_court).all()
        or not np.isfinite(points_scene).all()
    ):
        raise ValueError(f"{prefix} correspondence point arrays are invalid.")
    if candidate_index.dtype != np.int32 or camera_index.dtype != np.int32:
        raise ValueError(f"{prefix} index arrays must have dtype int32.")
    if candidate_index.shape != (len(points_court),) or camera_index.shape != (
        len(points_court),
    ):
        raise ValueError(f"{prefix} correspondence index shapes are invalid.")
    if np.any(candidate_index < 0) or np.any(candidate_index >= candidate_count):
        raise ValueError(f"{prefix} candidate indices are out of range.")
    if np.any(camera_index < 0) or np.any(camera_index >= camera_count):
        raise ValueError(f"{prefix} camera indices are out of range.")
    if not set(int(value) for value in camera_index).issubset(allowed_camera_indices):
        raise ValueError(f"{prefix} observations reference the wrong camera partition.")
    counts = np.bincount(candidate_index, minlength=candidate_count)
    if np.any(counts < 3):
        raise ValueError(
            f"Every candidate requires at least three {prefix} correspondences."
        )
    for index in range(candidate_count):
        observed = set(int(value) for value in camera_index[candidate_index == index])
        if not observed or not observed.issubset(allowed_camera_indices):
            raise ValueError(
                f"Every candidate must use measured cameras from the declared {prefix} "
                "partition."
            )


def _court_geometry_payload(result: AlignmentResult) -> dict[str, object]:
    return {
        "schema": "fitted_court_geometry_v2",
        "coordinate_convention": ALIGNMENT_COORDINATE_CONVENTION,
        "metric_scene_adapter": result.metric_adapter.to_dict(),
        "fit_camera_ids": list(result.partitions.fit_camera_ids),
        "candidates": [
            {
                "court_instance_id": candidate.court_instance_id,
                "candidate_id": candidate.candidate_id,
                "scene_from_court": candidate.scene_from_court.to_list(),
                "court_from_scene": candidate.court_from_scene.to_list(),
                "fit": candidate.fit.to_dict(),
            }
            for candidate in result.candidates
        ],
    }


def _metrics_payload(result: AlignmentResult) -> dict[str, object]:
    return {
        "schema": "alignment_candidate_metrics_v1",
        "accepted_court_instance_ids": [
            court.court_instance_id for court in result.layout.courts
        ],
        "candidates": [
            {
                "court_instance_id": candidate.court_instance_id,
                "candidate_id": candidate.candidate_id,
                "fit": candidate.fit.to_dict(),
                "holdout": candidate.holdout.to_dict(),
                "accepted": candidate.accepted,
            }
            for candidate in result.candidates
        ],
    }


def _human_summary(result: AlignmentResult) -> str:
    lines = [
        "Semantic multi-court alignment",
        f"fit cameras: {len(result.partitions.fit_camera_ids)}",
        f"holdout cameras: {len(result.partitions.holdout_camera_ids)}",
        f"evaluated courts: {len(result.candidates)}",
        f"accepted courts: {len(result.layout.courts)}",
        f"NHT scene units per metre: {result.metric_adapter.nht_scene_units_per_metre:.9g}",
    ]
    lines.extend(
        f"{candidate.court_instance_id} ({candidate.candidate_id}): "
        f"fit={candidate.fit.status.value}, holdout={candidate.holdout.status.value}"
        for candidate in result.candidates
    )
    return "\n".join(lines) + "\n"


def _project(
    points_camera: NDArray[np.float64], intrinsic: NDArray[np.float64]
) -> NDArray[np.float64]:
    if np.any(points_camera[:, 2] <= 0.0):
        raise ValueError(
            "Projection equivalence points must have positive camera-Z depth."
        )
    homogeneous = points_camera @ intrinsic.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.isfinite(pixels).all():
        raise ValueError("Projected pixel coordinates must be finite.")
    return pixels


def _write_json(path: Path, value: object) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Expected an ordinary JSON file: {path}")
    try:
        raw: object = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=_reject_constant
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON at {path}: {error}") from error
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise TypeError(f"JSON document must be an object with string keys: {path}")
    return cast(dict[str, object], raw)


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _require_staging_directory(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(
            f"Alignment staging must be an ordinary existing directory: {path}"
        )


def _require_output_directory(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(
            f"Alignment output must be an ordinary existing directory: {path}"
        )


__all__ = [
    "ALIGNMENT_FILE",
    "COURT_GEOMETRY_FILE",
    "DIAGNOSTICS_DIRECTORY",
    "GROUND_LINE_MAP_FILE",
    "load_accepted_layout",
    "load_alignment_result",
    "validate_alignment_outputs",
    "validate_court_transform_binding",
    "validate_projection_equivalence",
    "write_alignment_outputs",
]
