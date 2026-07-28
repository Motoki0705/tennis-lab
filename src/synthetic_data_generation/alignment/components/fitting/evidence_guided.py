"""Evidence-guided multi-start fitting for an unknown number of tennis courts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
from scipy.optimize import differential_evolution

from src.synthetic_data_generation.alignment.components.fitting.court_template import (
    CourtFitCandidate,
    CourtLocalRefitSettings,
    court_line_segments,
    fit_court_instance_near_reference,
    sample_court_line_template,
    scene_from_court_matrix,
    transform_court_template,
)
from src.synthetic_data_generation.alignment.components.ground.plane import (
    GroundPlaneEstimate,
)

Parameters = tuple[float, float, float, float]


@dataclass(frozen=True)
class CourtMultiStartFitSettings:
    """Settings for proposal, clustering, validation, and residual fitting."""

    seed: int = 20260725
    num_global_runs: int = 32
    bootstrap_runs: int = 16
    proposal_fraction: float = 0.60
    uniform_fraction: float = 0.25
    orthogonal_fraction: float = 0.15
    cluster_center_tolerance_m: float = 1.5
    cluster_orientation_tolerance_deg: float = 4.0
    cluster_scale_relative_tolerance: float = 0.04
    min_cluster_support_rate: float = 0.30
    min_bootstrap_survival_rate: float = 0.70
    orientation_ambiguity_margin: float = 0.08
    residual_suppression_strength: float = 0.65
    min_residual_gain: float = 0.05
    max_instances: int = 8
    blur_sigma_cells: float = 2.0
    samples_per_metre: float = 6.0
    min_scale_scene_per_metre: float = 0.055
    max_scale_scene_per_metre: float = 0.085
    orientation_min_radians: float = -math.pi / 2.0
    orientation_max_radians: float = math.pi / 2.0
    optimizer_max_iterations: int = 180
    optimizer_population_size: int = 20
    optimizer_tolerance: float = 1.0e-8
    local_optimizer_max_iterations: int = 100
    local_optimizer_population_size: int = 12
    min_template_score: float = 0.2
    min_confidence: float = 0.62
    min_line_coverage: float = 0.45
    min_internal_line_coverage: float = 0.40
    duplicate_center_tolerance_m: float = 5.0
    bootstrap_block_rows: int = 4
    bootstrap_block_columns: int = 4
    bootstrap_keep_fraction: float = 0.75
    template_support_width_m: float = 0.35
    background_ring_width_m: float = 0.70

    def __post_init__(self) -> None:
        fractions = (
            self.proposal_fraction,
            self.uniform_fraction,
            self.orthogonal_fraction,
        )
        if any(value <= 0.0 for value in fractions) or not math.isclose(
            sum(fractions),
            1.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("Proposal fractions must be positive and sum to one.")
        positive_integers = {
            "num_global_runs": self.num_global_runs,
            "bootstrap_runs": self.bootstrap_runs,
            "max_instances": self.max_instances,
            "optimizer_max_iterations": self.optimizer_max_iterations,
            "optimizer_population_size": self.optimizer_population_size,
            "local_optimizer_max_iterations": self.local_optimizer_max_iterations,
            "local_optimizer_population_size": self.local_optimizer_population_size,
            "bootstrap_block_rows": self.bootstrap_block_rows,
            "bootstrap_block_columns": self.bootstrap_block_columns,
        }
        for name, value in positive_integers.items():
            if isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.blur_sigma_cells <= 0.0 or self.samples_per_metre <= 0.0:
            raise ValueError("Blur and sampling settings must be positive.")
        if not (0.0 < self.min_scale_scene_per_metre < self.max_scale_scene_per_metre):
            raise ValueError("Court scale bounds are invalid.")
        if self.orientation_min_radians >= self.orientation_max_radians:
            raise ValueError("Court orientation bounds are invalid.")
        unit_interval_settings = {
            "min_cluster_support_rate": self.min_cluster_support_rate,
            "min_bootstrap_survival_rate": self.min_bootstrap_survival_rate,
            "orientation_ambiguity_margin": self.orientation_ambiguity_margin,
            "residual_suppression_strength": self.residual_suppression_strength,
            "min_residual_gain": self.min_residual_gain,
            "min_confidence": self.min_confidence,
            "min_line_coverage": self.min_line_coverage,
            "min_internal_line_coverage": self.min_internal_line_coverage,
            "bootstrap_keep_fraction": self.bootstrap_keep_fraction,
        }
        for name, unit_value in unit_interval_settings.items():
            if not 0.0 < unit_value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1].")
        positive_settings = {
            "cluster_center_tolerance_m": self.cluster_center_tolerance_m,
            "cluster_orientation_tolerance_deg": (
                self.cluster_orientation_tolerance_deg
            ),
            "cluster_scale_relative_tolerance": (self.cluster_scale_relative_tolerance),
            "duplicate_center_tolerance_m": self.duplicate_center_tolerance_m,
            "optimizer_tolerance": self.optimizer_tolerance,
            "min_template_score": self.min_template_score,
            "template_support_width_m": self.template_support_width_m,
            "background_ring_width_m": self.background_ring_width_m,
        }
        for name, positive_value in positive_settings.items():
            if positive_value <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if (
            self.cluster_scale_relative_tolerance >= 1.0
            or self.background_ring_width_m <= self.template_support_width_m
        ):
            raise ValueError("Scale tolerance or background ring width is invalid.")


@dataclass(frozen=True)
class CourtFitRun:
    """One global optimizer run and the conditions that produced it."""

    run_id: str
    seed: int
    evidence_subset_id: str
    initialisation_kind: str
    parameters: Parameters
    component_scores: dict[str, float]
    optimizer_evaluations: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic payload."""
        return asdict(self)


@dataclass(frozen=True)
class CourtFitCluster:
    """A symmetry-aware cluster of global fits and its validation diagnostics."""

    cluster_id: str
    representative_parameters: Parameters
    member_run_ids: tuple[str, ...]
    support_rate: float
    bootstrap_survival_rate: float = 0.0
    parameter_dispersion: dict[str, float] = field(default_factory=dict)
    component_scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    residual_gain: float = 0.0
    rejection_reasons: tuple[str, ...] = ()
    candidate: CourtFitCandidate | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible diagnostic payload."""
        payload = asdict(self)
        if self.candidate is not None:
            payload["candidate"] = self.candidate.to_dict()
        return payload


@dataclass(frozen=True)
class CourtMultiStartFitResult:
    """Accepted courts plus complete per-iteration fitting diagnostics."""

    accepted_candidates: tuple[CourtFitCandidate, ...]
    runs_by_iteration: tuple[tuple[CourtFitRun, ...], ...]
    clusters_by_iteration: tuple[tuple[CourtFitCluster, ...], ...]
    residual_evidence_sums: tuple[float, ...]
    stop_status: str

    def diagnostics_dict(self) -> dict[str, Any]:
        """Serialize all trial, cluster, and residual diagnostics."""
        return {
            "runs_by_iteration": [
                [run.to_dict() for run in runs] for runs in self.runs_by_iteration
            ],
            "clusters_by_iteration": [
                [cluster.to_dict() for cluster in clusters]
                for clusters in self.clusters_by_iteration
            ],
            "residual_evidence_sums": list(self.residual_evidence_sums),
            "stop_status": self.stop_status,
        }


def build_proposal_distribution(
    evidence: NDArray[np.floating[Any]],
    *,
    bounds: tuple[float, float, float, float],
    grid_spacing: float,
    settings: CourtMultiStartFitSettings,
    seed: int,
) -> NDArray[np.float64]:
    """Build a mixed DE population from evidence, uniform, and 90-degree proposals."""
    image = _validated_evidence(evidence)
    _validate_geometry(bounds, grid_spacing)
    rng = np.random.default_rng(seed)
    population_size = max(20, settings.optimizer_population_size * 4)
    guided_count = max(1, round(population_size * settings.proposal_fraction))
    orthogonal_count = max(
        1,
        round(population_size * settings.orthogonal_fraction),
    )
    uniform_count = population_size - guided_count - orthogonal_count
    if uniform_count < 1:
        uniform_count = 1
        guided_count = population_size - uniform_count - orthogonal_count

    proposal_image = gaussian_filter(
        np.log1p(image),
        sigma=max(0.5, settings.blur_sigma_cells * 0.5),
    )
    dominant_angles = _dominant_line_orientations(proposal_image)
    pool_size = max(guided_count * 12, 256)
    guided_pool = _random_parameters(
        rng,
        pool_size,
        bounds=bounds,
        settings=settings,
    )
    angle_indices = rng.integers(0, len(dominant_angles), size=pool_size)
    angle_noise = rng.normal(
        0.0,
        math.radians(settings.cluster_orientation_tolerance_deg * 2.0),
        size=pool_size,
    )
    guided_pool[:, 2] = np.asarray(dominant_angles)[angle_indices] + angle_noise
    guided_pool[:, 2] = np.asarray(
        [_normalise_orientation(float(value), settings) for value in guided_pool[:, 2]]
    )
    template = sample_court_line_template(max(1.0, settings.samples_per_metre * 0.35))
    scores = np.asarray(
        [
            _template_score(
                proposal_image,
                template,
                parameters,
                bounds=bounds,
                grid_spacing=grid_spacing,
            )
            for parameters in guided_pool
        ]
    )
    guided = guided_pool[np.argsort(scores)[-guided_count:][::-1]].copy()
    uniform = _random_parameters(
        rng,
        uniform_count,
        bounds=bounds,
        settings=settings,
    )
    orthogonal_source = guided[
        np.arange(orthogonal_count, dtype=np.int64) % len(guided)
    ].copy()
    orthogonal_source[:, 2] = np.asarray(
        [
            _normalise_orientation(float(value) + math.pi / 2.0, settings)
            for value in orthogonal_source[:, 2]
        ]
    )
    population = np.concatenate((guided, uniform, orthogonal_source), axis=0)
    population_scores = np.asarray(
        [
            _template_score(
                proposal_image,
                template,
                parameters,
                bounds=bounds,
                grid_spacing=grid_spacing,
            )
            for parameters in population
        ]
    )
    best_index = int(np.argmax(population_scores))
    population[[0, best_index]] = population[[best_index, 0]]
    return population


def run_multistart_fits(
    evidence: NDArray[np.floating[Any]],
    *,
    bounds: tuple[float, float, float, float],
    grid_spacing: float,
    settings: CourtMultiStartFitSettings,
    iteration_index: int = 0,
) -> tuple[CourtFitRun, ...]:
    """Run deterministic global fits across mixed populations and block subsets."""
    image = _validated_evidence(evidence)
    _validate_geometry(bounds, grid_spacing)
    template = sample_court_line_template(settings.samples_per_metre)
    search_bounds = [
        (bounds[0], bounds[1]),
        (bounds[2], bounds[3]),
        (settings.orientation_min_radians, settings.orientation_max_radians),
        (
            settings.min_scale_scene_per_metre,
            settings.max_scale_scene_per_metre,
        ),
    ]
    runs: list[CourtFitRun] = []
    for run_index in range(settings.num_global_runs):
        seed = settings.seed + iteration_index * 100_000 + run_index
        subset, subset_id = _evidence_subset(
            image,
            settings=settings,
            seed=seed,
            use_full=(run_index % 2 == 0),
        )
        score_image = gaussian_filter(
            np.log1p(subset),
            sigma=settings.blur_sigma_cells,
        )
        population = build_proposal_distribution(
            subset,
            bounds=bounds,
            grid_spacing=grid_spacing,
            settings=settings,
            seed=seed,
        )

        def objective(
            parameters: NDArray[np.float64],
            image_for_run: NDArray[np.floating[Any]] = score_image,
        ) -> float:
            return -_template_score(
                image_for_run,
                template,
                np.asarray(parameters, dtype=np.float64),
                bounds=bounds,
                grid_spacing=grid_spacing,
            )

        result = differential_evolution(
            objective,
            search_bounds,
            seed=seed,
            maxiter=settings.optimizer_max_iterations,
            popsize=settings.optimizer_population_size,
            tol=settings.optimizer_tolerance,
            polish=True,
            workers=1,
            init=population,
        )
        parameters_array = np.asarray(result.x, dtype=np.float64)
        parameters = _parameters_tuple(parameters_array)
        runs.append(
            CourtFitRun(
                run_id=f"instance-{iteration_index}-run-{run_index}",
                seed=seed,
                evidence_subset_id=subset_id,
                initialisation_kind="mixed_evidence_uniform_orthogonal",
                parameters=parameters,
                component_scores={
                    "template_score": -float(result.fun),
                },
                optimizer_evaluations=int(result.nfev),
            )
        )
    return tuple(runs)


def cluster_fit_runs(
    runs: tuple[CourtFitRun, ...],
    settings: CourtMultiStartFitSettings,
) -> tuple[CourtFitCluster, ...]:
    """Cluster global fits with 180-degree court symmetry."""
    if not runs:
        raise ValueError("At least one fit run is required for clustering.")
    parents = list(range(len(runs)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(runs):
        for second_index in range(first_index + 1, len(runs)):
            if (
                _cluster_distance(
                    first.parameters,
                    runs[second_index].parameters,
                    settings,
                )
                <= 1.0
            ):
                union(first_index, second_index)
    groups: dict[int, list[CourtFitRun]] = {}
    for index, run in enumerate(runs):
        groups.setdefault(find(index), []).append(run)

    clusters: list[CourtFitCluster] = []
    for cluster_index, members in enumerate(groups.values()):
        representative = _weighted_medoid(members, settings)
        clusters.append(
            CourtFitCluster(
                cluster_id=f"cluster-{cluster_index}",
                representative_parameters=representative.parameters,
                member_run_ids=tuple(member.run_id for member in members),
                support_rate=len(members) / len(runs),
            )
        )
    clusters.sort(key=lambda cluster: cluster.support_rate, reverse=True)
    return tuple(
        replace(cluster, cluster_id=f"cluster-{index}")
        for index, cluster in enumerate(clusters)
    )


def refit_and_validate_cluster(
    evidence: NDArray[np.floating[Any]],
    cluster: CourtFitCluster,
    *,
    bounds: tuple[float, float, float, float],
    grid_spacing: float,
    plane: GroundPlaneEstimate,
    settings: CourtMultiStartFitSettings,
    iteration_index: int,
    cluster_index: int,
) -> CourtFitCluster:
    """Locally refit a cluster and measure block-bootstrap stability."""
    image = _validated_evidence(evidence)
    reference = cluster.representative_parameters
    local_settings = CourtLocalRefitSettings(
        seed=settings.seed + iteration_index * 100_000 + 50_000 + cluster_index,
        centre_radius_m=settings.cluster_center_tolerance_m,
        orientation_tolerance_radians=math.radians(
            settings.cluster_orientation_tolerance_deg
        ),
        scale_relative_tolerance=settings.cluster_scale_relative_tolerance,
        blur_sigma_cells=settings.blur_sigma_cells,
        samples_per_metre=settings.samples_per_metre,
        optimizer_max_iterations=settings.local_optimizer_max_iterations,
        optimizer_population_size=settings.local_optimizer_population_size,
        optimizer_tolerance=settings.optimizer_tolerance,
    )
    candidate = fit_court_instance_near_reference(
        image,
        bounds=bounds,
        grid_spacing=grid_spacing,
        plane=plane,
        reference_center_uv=(reference[0], reference[1]),
        reference_orientation_radians=reference[2],
        reference_scale_scene_per_metre=reference[3],
        settings=local_settings,
    )
    stable_parameters: list[Parameters] = [
        (
            candidate.center_uv[0],
            candidate.center_uv[1],
            candidate.orientation_radians,
            candidate.scale_scene_per_metre,
        )
    ]
    survived = 0
    for bootstrap_index in range(settings.bootstrap_runs):
        seed = (
            settings.seed
            + iteration_index * 100_000
            + 60_000
            + cluster_index * settings.bootstrap_runs
            + bootstrap_index
        )
        subset, _ = _evidence_subset(
            image,
            settings=settings,
            seed=seed,
            use_full=False,
        )
        bootstrap_candidate = fit_court_instance_near_reference(
            subset,
            bounds=bounds,
            grid_spacing=grid_spacing,
            plane=plane,
            reference_center_uv=candidate.center_uv,
            reference_orientation_radians=candidate.orientation_radians,
            reference_scale_scene_per_metre=candidate.scale_scene_per_metre,
            settings=replace(local_settings, seed=seed),
        )
        parameters = (
            bootstrap_candidate.center_uv[0],
            bootstrap_candidate.center_uv[1],
            bootstrap_candidate.orientation_radians,
            bootstrap_candidate.scale_scene_per_metre,
        )
        if (
            _cluster_distance(
                stable_parameters[0],
                parameters,
                settings,
            )
            <= 1.0
            and bootstrap_candidate.template_score >= settings.min_template_score
        ):
            survived += 1
            stable_parameters.append(parameters)
    survival_rate = survived / settings.bootstrap_runs
    component_scores = score_court_hypothesis(
        image,
        parameters=stable_parameters[0],
        bounds=bounds,
        grid_spacing=grid_spacing,
        settings=settings,
    )
    dispersion = _parameter_dispersion(stable_parameters)
    confidence = _confidence(
        component_scores,
        support_rate=cluster.support_rate,
        survival_rate=survival_rate,
        dispersion=dispersion,
        settings=settings,
    )
    reasons: list[str] = []
    if cluster.support_rate < settings.min_cluster_support_rate:
        reasons.append("rejected_low_support")
    if survival_rate < settings.min_bootstrap_survival_rate:
        reasons.append("rejected_unstable")
    if (
        component_scores["line_coverage"] < settings.min_line_coverage
        or component_scores["internal_line_coverage"]
        < settings.min_internal_line_coverage
    ):
        reasons.append("rejected_low_coverage")
    if component_scores["template_score"] < settings.min_template_score:
        reasons.append("rejected_low_template_score")
    if component_scores["explained_evidence"] < settings.min_residual_gain:
        reasons.append("stopped_low_residual_gain")
    if confidence < settings.min_confidence:
        reasons.append("rejected_low_confidence")
    return replace(
        cluster,
        representative_parameters=stable_parameters[0],
        bootstrap_survival_rate=survival_rate,
        parameter_dispersion=dispersion,
        component_scores=component_scores,
        confidence=confidence,
        residual_gain=component_scores["explained_evidence"],
        rejection_reasons=tuple(reasons),
        candidate=candidate,
    )


def score_court_hypothesis(
    evidence: NDArray[np.floating[Any]],
    *,
    parameters: Parameters,
    bounds: tuple[float, float, float, float],
    grid_spacing: float,
    settings: CourtMultiStartFitSettings,
) -> dict[str, float]:
    """Score template support, line coverage, explanation, and background contrast."""
    image = _validated_evidence(evidence)
    _validate_geometry(bounds, grid_spacing)
    score_image = gaussian_filter(
        np.log1p(image),
        sigma=settings.blur_sigma_cells,
    )
    segment_samples = _sample_court_segments(settings.samples_per_metre)
    parameter_array = np.asarray(parameters, dtype=np.float64)
    segment_scores = [
        _template_score(
            score_image,
            segment,
            parameter_array,
            bounds=bounds,
            grid_spacing=grid_spacing,
        )
        for segment in segment_samples
    ]
    maximum_segment_score = max(segment_scores)
    support_threshold = maximum_segment_score * 0.30
    supported = [score >= support_threshold for score in segment_scores]
    internal_indices = (2, 3, 6, 7, 8)
    line_coverage = sum(supported) / len(supported)
    internal_line_coverage = sum(supported[index] for index in internal_indices) / len(
        internal_indices
    )
    template = np.concatenate(segment_samples)
    template_score = _template_score(
        score_image,
        template,
        parameter_array,
        bounds=bounds,
        grid_spacing=grid_spacing,
    )
    distance = _template_distance_image(
        image.shape,
        template,
        parameter_array,
        bounds=bounds,
        grid_spacing=grid_spacing,
    )
    support_radius = settings.template_support_width_m * parameters[3]
    ring_radius = settings.background_ring_width_m * parameters[3]
    near = distance <= support_radius
    ring = (distance > support_radius) & (distance <= ring_radius)
    total_evidence = float(np.sum(image))
    explained = float(np.sum(image[near])) / max(total_evidence, 1.0e-12)
    background = float(np.mean(score_image[ring])) if bool(np.any(ring)) else 0.0
    contrast = template_score - background
    contrast_ratio = max(0.0, contrast) / max(abs(template_score), 1.0e-12)
    return {
        "template_score": template_score,
        "line_coverage": line_coverage,
        "internal_line_coverage": internal_line_coverage,
        "explained_evidence": explained,
        "background_contrast": contrast,
        "background_contrast_ratio": min(1.0, contrast_ratio),
    }


def select_reliable_cluster(
    clusters: tuple[CourtFitCluster, ...],
    *,
    existing: tuple[CourtFitCandidate, ...],
    settings: CourtMultiStartFitSettings,
) -> tuple[CourtFitCluster | None, tuple[CourtFitCluster, ...]]:
    """Select one non-duplicate cluster or reject an unresolved 90-degree pair."""
    evaluated: list[CourtFitCluster] = []
    for cluster in clusters:
        reasons = list(cluster.rejection_reasons)
        if cluster.candidate is not None and any(
            _is_duplicate(cluster.candidate, candidate, settings)
            for candidate in existing
        ):
            reasons.append("rejected_duplicate")
        evaluated.append(
            replace(cluster, rejection_reasons=tuple(dict.fromkeys(reasons)))
        )
    viable = [cluster for cluster in evaluated if not cluster.rejection_reasons]
    viable.sort(key=lambda cluster: cluster.confidence, reverse=True)
    if not viable:
        return None, tuple(evaluated)
    best = viable[0]
    for alternative in viable[1:]:
        difference = math.degrees(
            _orientation_distance(
                best.representative_parameters[2],
                alternative.representative_parameters[2],
            )
        )
        weak_internal_support = (
            best.component_scores["internal_line_coverage"]
            < settings.min_internal_line_coverage + 0.15
        )
        if (
            80.0 <= difference <= 100.0
            and abs(best.confidence - alternative.confidence)
            < settings.orientation_ambiguity_margin
            and weak_internal_support
        ):
            ambiguous_ids = {best.cluster_id, alternative.cluster_id}
            evaluated = [
                replace(
                    cluster,
                    rejection_reasons=(
                        *cluster.rejection_reasons,
                        "ambiguous_orientation",
                    ),
                )
                if cluster.cluster_id in ambiguous_ids
                else cluster
                for cluster in evaluated
            ]
            return None, tuple(evaluated)
    return best, tuple(evaluated)


def suppress_explained_evidence(
    evidence: NDArray[np.floating[Any]],
    *,
    parameters: Parameters,
    bounds: tuple[float, float, float, float],
    grid_spacing: float,
    settings: CourtMultiStartFitSettings,
) -> NDArray[np.float32]:
    """Softly suppress evidence near an accepted court without zeroing it."""
    image = _validated_evidence(evidence)
    template = sample_court_line_template(settings.samples_per_metre)
    distance = _template_distance_image(
        image.shape,
        template,
        np.asarray(parameters, dtype=np.float64),
        bounds=bounds,
        grid_spacing=grid_spacing,
    )
    radius = settings.template_support_width_m * parameters[3]
    suppression = np.exp(-0.5 * np.square(distance / max(radius, 1.0e-12)))
    residual = image * (1.0 - settings.residual_suppression_strength * suppression)
    return np.asarray(residual, dtype=np.float32)


def fit_unknown_number_of_courts(
    evidence: NDArray[np.floating[Any]],
    *,
    bounds: tuple[float, float, float, float],
    grid_spacing: float,
    plane: GroundPlaneEstimate,
    settings: CourtMultiStartFitSettings,
) -> CourtMultiStartFitResult:
    """Sequentially fit stable court clusters until residual evidence stops improving."""
    residual = _validated_evidence(evidence).copy()
    accepted: list[CourtFitCandidate] = []
    all_runs: list[tuple[CourtFitRun, ...]] = []
    all_clusters: list[tuple[CourtFitCluster, ...]] = []
    residual_sums = [float(np.sum(residual))]
    stop_status = "stopped_max_instances"
    for iteration_index in range(settings.max_instances):
        runs = run_multistart_fits(
            residual,
            bounds=bounds,
            grid_spacing=grid_spacing,
            settings=settings,
            iteration_index=iteration_index,
        )
        clusters = cluster_fit_runs(runs, settings)
        validated = tuple(
            refit_and_validate_cluster(
                residual,
                cluster,
                bounds=bounds,
                grid_spacing=grid_spacing,
                plane=plane,
                settings=settings,
                iteration_index=iteration_index,
                cluster_index=cluster_index,
            )
            for cluster_index, cluster in enumerate(clusters)
        )
        selected, evaluated = select_reliable_cluster(
            validated,
            existing=tuple(accepted),
            settings=settings,
        )
        all_runs.append(runs)
        if selected is None or selected.candidate is None:
            all_clusters.append(evaluated)
            statuses = {
                reason for cluster in evaluated for reason in cluster.rejection_reasons
            }
            if "ambiguous_orientation" in statuses:
                stop_status = "ambiguous_orientation"
            elif "stopped_low_residual_gain" in statuses:
                stop_status = "stopped_low_residual_gain"
            else:
                stop_status = "stopped_no_reliable_cluster"
            break
        candidate = replace(
            selected.candidate,
            candidate_id=f"court-{len(accepted)}",
        )
        accepted.append(candidate)
        selected = replace(selected, candidate=candidate)
        evaluated = tuple(
            selected if cluster.cluster_id == selected.cluster_id else cluster
            for cluster in evaluated
        )
        all_clusters.append(evaluated)
        parameters = (
            candidate.center_uv[0],
            candidate.center_uv[1],
            candidate.orientation_radians,
            candidate.scale_scene_per_metre,
        )
        residual = suppress_explained_evidence(
            residual,
            parameters=parameters,
            bounds=bounds,
            grid_spacing=grid_spacing,
            settings=settings,
        )
        residual_sums.append(float(np.sum(residual)))
    return CourtMultiStartFitResult(
        accepted_candidates=tuple(accepted),
        runs_by_iteration=tuple(all_runs),
        clusters_by_iteration=tuple(all_clusters),
        residual_evidence_sums=tuple(residual_sums),
        stop_status=stop_status,
    )


def _validated_evidence(
    evidence: NDArray[np.floating[Any]],
) -> NDArray[np.float32]:
    image = np.asarray(evidence, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Evidence must be a finite 2D array.")
    if bool(np.any(image < 0.0)):
        raise ValueError("Evidence must be non-negative.")
    if not bool(np.any(image > 0.0)):
        raise ValueError("Cannot fit a court to empty evidence.")
    return image


def _validate_geometry(
    bounds: tuple[float, float, float, float],
    grid_spacing: float,
) -> None:
    u_min, u_max, v_min, v_max = bounds
    if u_min >= u_max or v_min >= v_max:
        raise ValueError("Court search bounds must have positive area.")
    if grid_spacing <= 0.0:
        raise ValueError("grid_spacing must be positive.")


def _dominant_line_orientations(
    proposal_image: NDArray[np.float32],
) -> tuple[float, ...]:
    gradient_v, gradient_u = np.gradient(proposal_image)
    magnitude = np.hypot(gradient_u, gradient_v)
    line_angles = np.mod(np.arctan2(gradient_v, gradient_u) + math.pi / 2.0, math.pi)
    histogram, edges = np.histogram(
        line_angles,
        bins=36,
        range=(0.0, math.pi),
        weights=magnitude,
    )
    top_indices = np.argsort(histogram)[-4:][::-1]
    return tuple(
        float((edges[index] + edges[index + 1]) * 0.5 - math.pi / 2.0)
        for index in top_indices
    )


def _random_parameters(
    rng: np.random.Generator,
    count: int,
    *,
    bounds: tuple[float, float, float, float],
    settings: CourtMultiStartFitSettings,
) -> NDArray[np.float64]:
    values: NDArray[np.float64] = np.empty((count, 4), dtype=np.float64)
    values[:, 0] = rng.uniform(bounds[0], bounds[1], size=count)
    values[:, 1] = rng.uniform(bounds[2], bounds[3], size=count)
    values[:, 2] = rng.uniform(
        settings.orientation_min_radians,
        settings.orientation_max_radians,
        size=count,
    )
    values[:, 3] = rng.uniform(
        settings.min_scale_scene_per_metre,
        settings.max_scale_scene_per_metre,
        size=count,
    )
    return values


def _normalise_orientation(
    angle: float,
    settings: CourtMultiStartFitSettings,
) -> float:
    canonical = (angle + math.pi / 2.0) % math.pi - math.pi / 2.0
    return min(
        settings.orientation_max_radians,
        max(settings.orientation_min_radians, canonical),
    )


def _template_score(
    image: NDArray[np.floating[Any]],
    template: NDArray[np.float64],
    parameters: NDArray[np.float64],
    *,
    bounds: tuple[float, float, float, float],
    grid_spacing: float,
) -> float:
    uv = transform_court_template(template, parameters)
    columns = (uv[:, 0] - bounds[0]) / grid_spacing
    rows = (uv[:, 1] - bounds[2]) / grid_spacing
    values = map_coordinates(
        image,
        [rows, columns],
        order=1,
        mode="constant",
        cval=0.0,
    )
    return float(np.mean(values))


def _evidence_subset(
    evidence: NDArray[np.float32],
    *,
    settings: CourtMultiStartFitSettings,
    seed: int,
    use_full: bool,
) -> tuple[NDArray[np.float32], str]:
    if use_full:
        return evidence, "full"
    rng = np.random.default_rng(seed)
    block_count = settings.bootstrap_block_rows * settings.bootstrap_block_columns
    keep_count = max(1, round(block_count * settings.bootstrap_keep_fraction))
    kept: NDArray[np.bool_] = np.zeros(block_count, dtype=bool)
    kept[rng.choice(block_count, size=keep_count, replace=False)] = True
    row_blocks = np.array_split(
        np.arange(evidence.shape[0]), settings.bootstrap_block_rows
    )
    column_blocks = np.array_split(
        np.arange(evidence.shape[1]),
        settings.bootstrap_block_columns,
    )
    mask = np.zeros(evidence.shape, dtype=bool)
    for row_index, rows in enumerate(row_blocks):
        for column_index, columns in enumerate(column_blocks):
            block_index = row_index * settings.bootstrap_block_columns + column_index
            if kept[block_index]:
                mask[np.ix_(rows, columns)] = True
    subset = np.where(mask, evidence, 0.0).astype(np.float32, copy=False)
    kept_ids = np.flatnonzero(kept)
    identifier = "blocks-" + "-".join(str(int(index)) for index in kept_ids)
    return subset, identifier


def _orientation_distance(first: float, second: float) -> float:
    difference = abs((first - second) % math.pi)
    return min(difference, math.pi - difference)


def _cluster_distance(
    first: Parameters,
    second: Parameters,
    settings: CourtMultiStartFitSettings,
) -> float:
    mean_scale = 0.5 * (first[3] + second[3])
    center_metres = math.hypot(first[0] - second[0], first[1] - second[1]) / (
        mean_scale
    )
    orientation_degrees = math.degrees(_orientation_distance(first[2], second[2]))
    scale_log_difference = abs(math.log(first[3] / second[3]))
    scale_log_tolerance = math.log1p(settings.cluster_scale_relative_tolerance)
    return math.sqrt(
        (center_metres / settings.cluster_center_tolerance_m) ** 2
        + (orientation_degrees / settings.cluster_orientation_tolerance_deg) ** 2
        + (scale_log_difference / scale_log_tolerance) ** 2
    )


def _weighted_medoid(
    members: list[CourtFitRun],
    settings: CourtMultiStartFitSettings,
) -> CourtFitRun:
    weights = np.asarray(
        [max(member.component_scores["template_score"], 1.0e-12) for member in members]
    )
    costs = [
        sum(
            weights[other_index]
            * _cluster_distance(member.parameters, other.parameters, settings)
            for other_index, other in enumerate(members)
        )
        for member in members
    ]
    return members[int(np.argmin(costs))]


def _sample_court_segments(
    samples_per_metre: float,
) -> tuple[NDArray[np.float64], ...]:
    samples: list[NDArray[np.float64]] = []
    for start, end in court_line_segments():
        start_array = np.asarray(start, dtype=np.float64)
        end_array = np.asarray(end, dtype=np.float64)
        count = max(
            16,
            int(np.linalg.norm(end_array - start_array) * samples_per_metre),
        )
        fraction = np.linspace(0.0, 1.0, count)[:, None]
        samples.append(start_array * (1.0 - fraction) + end_array * fraction)
    return tuple(samples)


def _template_distance_image(
    shape: tuple[int, int],
    template: NDArray[np.float64],
    parameters: NDArray[np.float64],
    *,
    bounds: tuple[float, float, float, float],
    grid_spacing: float,
) -> NDArray[np.float64]:
    uv = transform_court_template(template, parameters)
    columns = np.rint((uv[:, 0] - bounds[0]) / grid_spacing).astype(np.int64)
    rows = np.rint((uv[:, 1] - bounds[2]) / grid_spacing).astype(np.int64)
    valid = (rows >= 0) & (rows < shape[0]) & (columns >= 0) & (columns < shape[1])
    seeds: NDArray[np.bool_] = np.zeros(shape, dtype=bool)
    seeds[rows[valid], columns[valid]] = True
    if not bool(np.any(seeds)):
        return np.full(shape, np.inf, dtype=np.float64)
    return np.asarray(distance_transform_edt(~seeds) * grid_spacing, dtype=np.float64)


def _parameter_dispersion(parameters: list[Parameters]) -> dict[str, float]:
    values = np.asarray(parameters, dtype=np.float64)
    scale = float(np.mean(values[:, 3]))
    centers_metres = values[:, :2] / scale
    doubled_angles = 2.0 * values[:, 2]
    resultant = float(abs(np.mean(np.exp(1j * doubled_angles))))
    circular_std = 0.5 * math.sqrt(max(0.0, -2.0 * math.log(max(resultant, 1.0e-12))))
    return {
        "center_std_m": float(
            np.sqrt(
                np.mean(
                    np.sum(np.square(centers_metres - centers_metres.mean(0)), axis=1)
                )
            )
        ),
        "orientation_std_deg": math.degrees(circular_std),
        "scale_cv": float(np.std(values[:, 3]) / scale),
    }


def _confidence(
    component_scores: dict[str, float],
    *,
    support_rate: float,
    survival_rate: float,
    dispersion: dict[str, float],
    settings: CourtMultiStartFitSettings,
) -> float:
    template_quality = component_scores["template_score"] / (
        component_scores["template_score"] + settings.min_template_score
    )
    residual_quality = min(
        1.0,
        component_scores["explained_evidence"] / settings.min_residual_gain,
    )
    stability = 0.5 * support_rate + 0.5 * survival_rate
    dispersion_penalty = min(
        1.0,
        0.4 * dispersion["center_std_m"] / settings.cluster_center_tolerance_m
        + 0.4
        * dispersion["orientation_std_deg"]
        / settings.cluster_orientation_tolerance_deg
        + 0.2 * dispersion["scale_cv"] / settings.cluster_scale_relative_tolerance,
    )
    confidence = (
        0.20 * template_quality
        + 0.15 * component_scores["line_coverage"]
        + 0.15 * component_scores["internal_line_coverage"]
        + 0.15 * component_scores["background_contrast_ratio"]
        + 0.15 * residual_quality
        + 0.20 * stability
        - 0.10 * dispersion_penalty
    )
    return float(np.clip(confidence, 0.0, 1.0))


def _is_duplicate(
    first: CourtFitCandidate,
    second: CourtFitCandidate,
    settings: CourtMultiStartFitSettings,
) -> bool:
    mean_scale = 0.5 * (first.scale_scene_per_metre + second.scale_scene_per_metre)
    center_distance_metres = float(
        np.linalg.norm(np.asarray(first.center_uv) - np.asarray(second.center_uv))
        / mean_scale
    )
    orientation_difference = math.degrees(
        _orientation_distance(
            first.orientation_radians,
            second.orientation_radians,
        )
    )
    scale_difference = (
        abs(first.scale_scene_per_metre - second.scale_scene_per_metre) / mean_scale
    )
    return (
        center_distance_metres < settings.duplicate_center_tolerance_m
        and orientation_difference < settings.cluster_orientation_tolerance_deg * 2.0
        and scale_difference < settings.cluster_scale_relative_tolerance * 2.0
    )


def _parameters_tuple(parameters: NDArray[np.float64]) -> Parameters:
    return (
        float(parameters[0]),
        float(parameters[1]),
        float(parameters[2]),
        float(parameters[3]),
    )


def candidate_from_parameters(
    parameters: Parameters,
    *,
    candidate_id: str,
    template_score: float,
    optimizer_evaluations: int,
    plane: GroundPlaneEstimate,
) -> CourtFitCandidate:
    """Build a serializable court candidate from fitted ground-plane parameters."""
    transform = scene_from_court_matrix(
        plane,
        center_uv=(parameters[0], parameters[1]),
        orientation_radians=parameters[2],
        scale_scene_per_metre=parameters[3],
    )
    return CourtFitCandidate(
        candidate_id=candidate_id,
        center_uv=(parameters[0], parameters[1]),
        orientation_radians=parameters[2],
        scale_scene_per_metre=parameters[3],
        template_score=template_score,
        optimizer_evaluations=optimizer_evaluations,
        scene_from_court=tuple(float(value) for value in transform.ravel()),
    )
