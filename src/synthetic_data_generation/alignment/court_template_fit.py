"""Fit distinct metric tennis-court instances to an aggregated ground-line map."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import differential_evolution

from src.synthetic_data_generation.alignment.ground_plane import GroundPlaneEstimate
from src.utils.schema.court import (
    HALF_DOUBLES_WIDTH,
    HALF_LENGTH,
    HALF_SINGLES_WIDTH,
    SERVICE_LINE_DISTANCE,
)

COURT_GEOMETRY_SCHEMA = "ground_court_geometry_v1"


@dataclass(frozen=True)
class CourtTemplateFitSettings:
    """Deterministic search settings for adjacent metric court instances."""

    seed: int = 20260725
    instance_count: int = 2
    blur_sigma_cells: float = 2.0
    samples_per_metre: float = 6.0
    min_scale_scene_per_metre: float = 0.055
    max_scale_scene_per_metre: float = 0.085
    orientation_min_radians: float = -math.pi / 2.0
    orientation_max_radians: float = math.pi / 2.0
    family_orientation_tolerance_radians: float = 0.12
    family_scale_relative_tolerance: float = 0.15
    min_center_separation_metres: float = 12.0
    separation_penalty: float = 100.0
    optimizer_max_iterations: int = 180
    optimizer_population_size: int = 20
    optimizer_tolerance: float = 1.0e-8
    min_template_score: float = 0.2

    def __post_init__(self) -> None:
        if self.instance_count < 1:
            raise ValueError("instance_count must be positive.")
        if self.blur_sigma_cells <= 0.0 or self.samples_per_metre <= 0.0:
            raise ValueError("Blur and sampling settings must be positive.")
        if not (0.0 < self.min_scale_scene_per_metre < self.max_scale_scene_per_metre):
            raise ValueError("Court scale bounds are invalid.")
        if self.orientation_min_radians >= self.orientation_max_radians:
            raise ValueError("Court orientation bounds are invalid.")
        if not 0.0 < self.family_scale_relative_tolerance < 1.0:
            raise ValueError("family_scale_relative_tolerance must lie in (0, 1).")
        if (
            self.family_orientation_tolerance_radians <= 0.0
            or self.min_center_separation_metres <= 0.0
            or self.separation_penalty <= 0.0
        ):
            raise ValueError("Court-family separation settings must be positive.")
        if (
            self.optimizer_max_iterations < 1
            or self.optimizer_population_size < 1
            or self.optimizer_tolerance <= 0.0
            or self.min_template_score <= 0.0
        ):
            raise ValueError("Optimizer and acceptance settings must be positive.")


@dataclass(frozen=True)
class CourtLocalRefitSettings:
    """Search bounds for stability refits locked to one physical court cluster."""

    seed: int = 20260725
    centre_radius_m: float = 2.0
    orientation_tolerance_radians: float = 0.08726646259971647
    scale_relative_tolerance: float = 0.03
    blur_sigma_cells: float = 2.0
    samples_per_metre: float = 6.0
    optimizer_max_iterations: int = 120
    optimizer_population_size: int = 16
    optimizer_tolerance: float = 1.0e-8

    def __post_init__(self) -> None:
        if (
            self.centre_radius_m <= 0.0
            or self.orientation_tolerance_radians <= 0.0
            or not 0.0 < self.scale_relative_tolerance < 1.0
            or self.blur_sigma_cells <= 0.0
            or self.samples_per_metre <= 0.0
            or self.optimizer_max_iterations < 1
            or self.optimizer_population_size < 1
            or self.optimizer_tolerance <= 0.0
        ):
            raise ValueError("Local court-refit settings are invalid.")


@dataclass(frozen=True)
class CourtFitCandidate:
    """One separately fitted physical court in ground-plane coordinates."""

    candidate_id: str
    center_uv: tuple[float, float]
    orientation_radians: float
    scale_scene_per_metre: float
    template_score: float
    optimizer_evaluations: int
    scene_from_court: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


def fit_court_instances(
    evidence_sum: NDArray[np.floating[Any]],
    *,
    bounds: tuple[float, float, float, float],
    grid_spacing: float,
    plane: GroundPlaneEstimate,
    settings: CourtTemplateFitSettings,
) -> tuple[CourtFitCandidate, ...]:
    """Fit court instances without averaging candidates from adjacent courts."""
    evidence = np.asarray(evidence_sum, dtype=np.float32)
    if evidence.ndim != 2 or not np.isfinite(evidence).all():
        raise ValueError("evidence_sum must be a finite 2D array.")
    if not bool(np.any(evidence > 0.0)):
        raise ValueError("Cannot fit a court to empty ground-line evidence.")
    if grid_spacing <= 0.0:
        raise ValueError("grid_spacing must be positive.")
    u_min, u_max, v_min, v_max = (float(value) for value in bounds)
    if u_min >= u_max or v_min >= v_max:
        raise ValueError("Court search bounds must have positive area.")

    template = sample_court_line_template(settings.samples_per_metre)
    score_image = gaussian_filter(
        np.log1p(evidence),
        sigma=settings.blur_sigma_cells,
    )

    def template_score(parameters: NDArray[np.float64]) -> float:
        uv = _transform_template(template, parameters)
        columns = (uv[:, 0] - u_min) / grid_spacing
        rows = (uv[:, 1] - v_min) / grid_spacing
        sampled = map_coordinates(
            score_image,
            [rows, columns],
            order=1,
            mode="constant",
            cval=0.0,
        )
        return float(np.mean(sampled))

    search_bounds = [
        (u_min, u_max),
        (v_min, v_max),
        (
            settings.orientation_min_radians,
            settings.orientation_max_radians,
        ),
        (
            settings.min_scale_scene_per_metre,
            settings.max_scale_scene_per_metre,
        ),
    ]
    parameters_and_evaluations: list[tuple[NDArray[np.float64], int]] = []
    first = differential_evolution(
        lambda parameters: -template_score(np.asarray(parameters, dtype=np.float64)),
        search_bounds,
        seed=settings.seed,
        maxiter=settings.optimizer_max_iterations,
        popsize=settings.optimizer_population_size,
        tol=settings.optimizer_tolerance,
        polish=True,
        workers=1,
    )
    first_parameters = np.asarray(first.x, dtype=np.float64)
    parameters_and_evaluations.append((first_parameters, int(first.nfev)))

    for instance_index in range(1, settings.instance_count):
        reference = first_parameters
        angle = float(reference[2])
        scale = float(reference[3])
        relative = settings.family_scale_relative_tolerance
        family_bounds = [
            (u_min, u_max),
            (v_min, v_max),
            (
                max(
                    settings.orientation_min_radians,
                    angle - settings.family_orientation_tolerance_radians,
                ),
                min(
                    settings.orientation_max_radians,
                    angle + settings.family_orientation_tolerance_radians,
                ),
            ),
            (
                max(settings.min_scale_scene_per_metre, scale * (1.0 - relative)),
                min(settings.max_scale_scene_per_metre, scale * (1.0 + relative)),
            ),
        ]

        def separated_objective(
            parameters: NDArray[np.float64],
            reference_scale: float = scale,
        ) -> float:
            candidate = np.asarray(parameters, dtype=np.float64)
            minimum_separation = (
                settings.min_center_separation_metres
                * 0.5
                * (float(candidate[3]) + reference_scale)
            )
            distances = [
                float(np.linalg.norm(candidate[:2] - existing[:2]))
                for existing, _ in parameters_and_evaluations
            ]
            shortfall = max(0.0, minimum_separation - min(distances))
            return -template_score(candidate) + settings.separation_penalty * shortfall

        result = differential_evolution(
            separated_objective,
            family_bounds,
            seed=settings.seed + instance_index,
            maxiter=settings.optimizer_max_iterations,
            popsize=settings.optimizer_population_size,
            tol=settings.optimizer_tolerance,
            polish=True,
            workers=1,
        )
        parameters_and_evaluations.append(
            (np.asarray(result.x, dtype=np.float64), int(result.nfev))
        )

    candidates = [
        CourtFitCandidate(
            candidate_id=f"court-{index}",
            center_uv=(float(parameters[0]), float(parameters[1])),
            orientation_radians=float(parameters[2]),
            scale_scene_per_metre=float(parameters[3]),
            template_score=template_score(parameters),
            optimizer_evaluations=evaluations,
            scene_from_court=tuple(
                float(value)
                for value in scene_from_court_matrix(
                    plane,
                    center_uv=(float(parameters[0]), float(parameters[1])),
                    orientation_radians=float(parameters[2]),
                    scale_scene_per_metre=float(parameters[3]),
                ).ravel()
            ),
        )
        for index, (parameters, evaluations) in enumerate(parameters_and_evaluations)
    ]
    candidates.sort(key=lambda candidate: candidate.template_score, reverse=True)
    for candidate in candidates:
        if candidate.template_score < settings.min_template_score:
            raise ValueError(
                f"{candidate.candidate_id} template score "
                f"{candidate.template_score:.6f} is below "
                f"{settings.min_template_score:.6f}."
            )
    return tuple(candidates)


def fit_court_instance_near_reference(
    evidence_sum: NDArray[np.floating[Any]],
    *,
    bounds: tuple[float, float, float, float],
    grid_spacing: float,
    plane: GroundPlaneEstimate,
    reference_center_uv: tuple[float, float],
    reference_orientation_radians: float,
    reference_scale_scene_per_metre: float,
    settings: CourtLocalRefitSettings,
) -> CourtFitCandidate:
    """Refit one frozen physical cluster without reopening court selection."""
    evidence = np.asarray(evidence_sum, dtype=np.float32)
    if evidence.ndim != 2 or not np.isfinite(evidence).all():
        raise ValueError("evidence_sum must be a finite 2D array.")
    if not bool(np.any(evidence > 0.0)):
        raise ValueError("Cannot refit a court to empty evidence.")
    if grid_spacing <= 0.0 or reference_scale_scene_per_metre <= 0.0:
        raise ValueError("Grid spacing and reference scale must be positive.")
    u_min, u_max, v_min, v_max = bounds
    radius_scene = settings.centre_radius_m * reference_scale_scene_per_metre
    search_bounds = [
        (
            max(u_min, reference_center_uv[0] - radius_scene),
            min(u_max, reference_center_uv[0] + radius_scene),
        ),
        (
            max(v_min, reference_center_uv[1] - radius_scene),
            min(v_max, reference_center_uv[1] + radius_scene),
        ),
        (
            reference_orientation_radians - settings.orientation_tolerance_radians,
            reference_orientation_radians + settings.orientation_tolerance_radians,
        ),
        (
            reference_scale_scene_per_metre * (1.0 - settings.scale_relative_tolerance),
            reference_scale_scene_per_metre * (1.0 + settings.scale_relative_tolerance),
        ),
    ]
    template = sample_court_line_template(settings.samples_per_metre)
    score_image = gaussian_filter(
        np.log1p(evidence),
        sigma=settings.blur_sigma_cells,
    )

    def score(parameters: NDArray[np.float64]) -> float:
        uv = _transform_template(template, parameters)
        columns = (uv[:, 0] - u_min) / grid_spacing
        rows = (uv[:, 1] - v_min) / grid_spacing
        sampled = map_coordinates(
            score_image,
            [rows, columns],
            order=1,
            mode="constant",
            cval=0.0,
        )
        return float(np.mean(sampled))

    result = differential_evolution(
        lambda parameters: -score(np.asarray(parameters, dtype=np.float64)),
        search_bounds,
        seed=settings.seed,
        maxiter=settings.optimizer_max_iterations,
        popsize=settings.optimizer_population_size,
        tol=settings.optimizer_tolerance,
        polish=True,
        workers=1,
    )
    parameters = np.asarray(result.x, dtype=np.float64)
    centre_shift_scene = float(
        np.linalg.norm(parameters[:2] - np.asarray(reference_center_uv))
    )
    if centre_shift_scene > radius_scene * (1.0 + 1.0e-8):
        raise ValueError("Local refit escaped its physical court cluster.")
    return CourtFitCandidate(
        candidate_id="local-court",
        center_uv=(float(parameters[0]), float(parameters[1])),
        orientation_radians=float(parameters[2]),
        scale_scene_per_metre=float(parameters[3]),
        template_score=score(parameters),
        optimizer_evaluations=int(result.nfev),
        scene_from_court=tuple(
            float(value)
            for value in scene_from_court_matrix(
                plane,
                center_uv=(float(parameters[0]), float(parameters[1])),
                orientation_radians=float(parameters[2]),
                scale_scene_per_metre=float(parameters[3]),
            ).ravel()
        ),
    )


def scene_from_court_matrix(
    plane: GroundPlaneEstimate,
    *,
    center_uv: tuple[float, float],
    orientation_radians: float,
    scale_scene_per_metre: float,
) -> NDArray[np.float64]:
    """Build a proper-handed Sim(3) mapping metric court coordinates to scene."""
    if scale_scene_per_metre <= 0.0:
        raise ValueError("scale_scene_per_metre must be positive.")
    basis_u = np.asarray(plane.basis_u, dtype=np.float64)
    basis_v = np.asarray(plane.basis_v, dtype=np.float64)
    normal = np.asarray(plane.normal, dtype=np.float64)
    origin = np.asarray(plane.origin, dtype=np.float64)
    cosine = math.cos(orientation_radians)
    sine = math.sin(orientation_radians)
    rotation = np.column_stack(
        (
            cosine * basis_u + sine * basis_v,
            -sine * basis_u + cosine * basis_v,
            normal,
        )
    )
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError("Court-to-scene rotation must be proper-handed.")
    translation = origin + center_uv[0] * basis_u + center_uv[1] * basis_v
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale_scene_per_metre * rotation
    transform[:3, 3] = translation
    return transform


def publish_court_geometry_artifact(
    payload: dict[str, Any],
    *,
    output_dir: Path,
) -> Path:
    """Publish one immutable fingerprinted court-geometry JSON artifact."""
    _validate_payload(payload)
    manifest = dict(payload)
    fingerprint = _canonical_fingerprint(manifest)
    manifest["artifact_fingerprint"] = fingerprint
    destination = output_dir / f"{payload['artifact_id']}-{fingerprint[:16]}.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite court geometry: {destination}")
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{payload['artifact_id']}-",
        suffix=".json",
        dir=output_dir,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                manifest,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.rename(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def load_court_geometry_artifact(path: Path) -> dict[str, Any]:
    """Load and fingerprint-verify a court-geometry artifact."""
    with path.open(encoding="utf-8") as handle:
        raw: Any = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Court-geometry artifact must be a JSON object.")
    payload = dict(raw)
    _validate_payload(payload)
    fingerprint = payload.get("artifact_fingerprint")
    expected = _canonical_fingerprint(payload)
    if fingerprint != expected:
        raise ValueError(
            "Court-geometry fingerprint mismatch: "
            f"declared {fingerprint}, computed {expected}."
        )
    return payload


def court_line_segments() -> tuple[
    tuple[tuple[float, float], tuple[float, float]],
    ...,
]:
    """Return the painted ITF court segments in metric court coordinates."""
    xd = HALF_DOUBLES_WIDTH
    xs = HALF_SINGLES_WIDTH
    yb = HALF_LENGTH
    ys = SERVICE_LINE_DISTANCE
    return (
        ((-xd, -yb), (-xd, yb)),
        ((xd, -yb), (xd, yb)),
        ((-xs, -yb), (-xs, yb)),
        ((xs, -yb), (xs, yb)),
        ((-xd, -yb), (xd, -yb)),
        ((-xd, yb), (xd, yb)),
        ((-xs, -ys), (xs, -ys)),
        ((-xs, ys), (xs, ys)),
        ((0.0, -ys), (0.0, ys)),
    )


def sample_court_line_template(
    samples_per_metre: float,
) -> NDArray[np.float64]:
    """Sample all painted court segments at deterministic metric intervals."""
    if samples_per_metre <= 0.0:
        raise ValueError("samples_per_metre must be positive.")
    points: list[NDArray[np.float64]] = []
    for start, end in court_line_segments():
        start_array = np.asarray(start, dtype=np.float64)
        end_array = np.asarray(end, dtype=np.float64)
        count = max(
            16,
            int(np.linalg.norm(end_array - start_array) * samples_per_metre),
        )
        fraction = np.linspace(0.0, 1.0, count)[:, None]
        points.append(start_array * (1.0 - fraction) + end_array * fraction)
    return np.concatenate(points)


def _transform_template(
    template: NDArray[np.float64],
    parameters: NDArray[np.float64],
) -> NDArray[np.float64]:
    center_u, center_v, orientation, scale = parameters
    cosine = math.cos(float(orientation))
    sine = math.sin(float(orientation))
    rotation_transpose = np.asarray(
        ((cosine, sine), (-sine, cosine)),
        dtype=np.float64,
    )
    return template @ rotation_transpose * float(scale) + np.asarray(
        (center_u, center_v)
    )


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema") != COURT_GEOMETRY_SCHEMA:
        raise ValueError(
            f"Unsupported court-geometry schema: {payload.get('schema')!r}."
        )
    required = {
        "schema",
        "artifact_id",
        "created_at_utc",
        "ground_line_artifact",
        "fit_settings",
        "candidates",
        "selection",
        "acceptance_status",
        "provenance",
    }
    optional = {"artifact_fingerprint"}
    if not required.issubset(payload) or not set(payload).issubset(required | optional):
        raise ValueError("Court-geometry manifest keys do not match v1 schema.")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Court-geometry candidates must be a non-empty list.")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Court-geometry selection must be an object.")
    candidate_ids = {candidate.get("candidate_id") for candidate in candidates}
    if selection.get("selected_candidate_id") not in candidate_ids:
        raise ValueError("Selected court candidate does not exist.")
    if payload.get("acceptance_status") != "fit_candidate_holdout_not_run":
        raise ValueError("Court geometry must remain a fit candidate before holdout.")


def _canonical_fingerprint(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("artifact_fingerprint", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
