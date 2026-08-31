"""Tests for fail-closed measured alignment evidence preflight."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from numpy.typing import NDArray
from PIL import Image

from src.synthetic_data_generation.alignment.contracts import (
    AlignmentAcceptancePolicy,
    AlignmentEvidence,
    CorrespondenceSet,
    FixedCameraSelectionDiagnostics,
    LineInferenceDeterminismDiagnostics,
    ProposalSearchStopReason,
)
from src.synthetic_data_generation.alignment.evidence_source import (
    _MAXIMUM_RESIDUAL_STATE_COUNT,
    _MAXIMUM_RETAINED_PROPOSAL_STATE_COUNT,
    _MAXIMUM_TILE_STATE_COUNT,
    MeasuredAlignmentEvidenceSource,
    ProductionAlignmentEvidenceSource,
    ProductionCourtLineDetector,
    _assign_candidate_evidence,
    _center_space_tiles,
    _CenterTile,
    _court_line_model_config,
    _court_line_model_state,
    _CourtHypothesis,
    _deduplicate_tiled_proposals,
    _fit_court_hypotheses,
    _fit_reliable_hypothesis_indices,
    _fixed_camera_selection,
    _GroundPlane,
    _maximum_center_tile_width_scene_units,
    _NativeProposal,
    _optimize_court,
    _orientation_search_bands,
    _partition_cameras,
    _partition_cameras_with_holdout_tail,
    _prepare_residual_evidence,
    _project_probability_to_ground,
    _ProjectedLineEvidence,
    _proposal_branch_seed,
    _proposal_frontier_snapshot,
    _proposal_search_after_fit_reliability,
    _proposal_search_resource_bounds,
    _proposal_topology_compatible,
    _ProposalSearchState,
    _refine_complete_proposal_state,
    _RefinedCompleteState,
    _repair_one_lattice_outlier,
    _ResidualEvidenceContext,
    _resolve_common_scale,
    _retain_fit_reliable_hypotheses,
    _retain_observable_cameras,
    _scale_bound_saturation_reason,
    _tile_is_geometrically_impossible,
    _TiledProposal,
)
from src.synthetic_data_generation.alignment.fitting import fit_alignment
from src.synthetic_data_generation.alignment.heatmaps import (
    AlignmentLineHeatmaps,
    AlignmentLineHeatmapView,
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
from src.synthetic_data_generation.alignment.whole_court import (
    evaluate_boundary_lattice_identifiability,
    evaluate_court_identifiability,
    evaluate_court_topology,
    evaluate_whole_template,
    sample_court_line_template,
    transform_template_2d,
)
from src.synthetic_data_generation.reconstruction.scene_export import (
    StandardSceneExport,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.utils.schema.court import HALF_DOUBLES_WIDTH


@dataclass
class _Detector:
    error: Exception | None = None
    preflight_calls: int = 0
    predict_calls: int = 0

    def preflight(self) -> None:
        self.preflight_calls += 1
        if self.error is not None:
            raise self.error

    def predict_probability(
        self,
        image_rgb: NDArray[np.uint8],
    ) -> NDArray[np.float32]:
        self.predict_calls += 1
        return np.ones(image_rgb.shape[:2], dtype=np.float32)

    def inference_cache_identity(self) -> dict[str, object]:
        return {"schema": "test_line_inference_identity_v1"}

    def determinism_diagnostics(self) -> LineInferenceDeterminismDiagnostics:
        return LineInferenceDeterminismDiagnostics(
            seed=42,
            device="cpu",
            model_eval=True,
            inference_mode=True,
            deterministic_algorithms=True,
            deterministic_warn_only=False,
            cudnn_benchmark=False,
            cudnn_deterministic=True,
            cuda_matmul_allow_tf32=False,
            cudnn_allow_tf32=False,
            cublas_workspace_config=None,
            torch_version="test",
            cuda_version=None,
            device_name="cpu",
            cross_hardware_bit_identity_claimed=False,
        )


def test_production_line_config_rebuilds_legacy_checkpoint_with_strict_fields(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path).line_model

    config = _court_line_model_config(settings)

    assert config.decoder.name == "dpt"
    assert config.decoder.size == "base"
    assert config.decoder.channels == 256
    assert config.transformer_encoder.name == "none"
    assert not config.transformer_encoder.enabled


def test_production_line_detector_accepts_explicit_cpu_runtime_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path).line_model
    monkeypatch.setenv("TENNIS_LAB_ALIGNMENT_LINE_DEVICE", "cpu")

    detector = ProductionCourtLineDetector(
        settings,
        cast(Any, object()),
        seed=42,
    )

    assert detector._settings.device == "cpu"


def test_production_line_detector_rejects_unknown_runtime_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TENNIS_LAB_ALIGNMENT_LINE_DEVICE", "cuda:1")

    with pytest.raises(ValueError, match="must be unset or 'cpu'"):
        ProductionCourtLineDetector(
            _settings(tmp_path).line_model,
            cast(Any, object()),
            seed=42,
        )


def test_production_source_accepts_audited_holdout_prefix_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TENNIS_LAB_ALIGNMENT_HOLDOUT_CAMERA_PREFIX_COUNT", "72")

    source = ProductionAlignmentEvidenceSource(
        _settings(tmp_path),
        cast(Any, object()),
        cast(Any, object()),
    )

    assert source._settings.camera_prefix_count == 3
    assert source._holdout_camera_prefix_count == 72


def test_production_source_rejects_unbounded_holdout_prefix_expansion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TENNIS_LAB_ALIGNMENT_HOLDOUT_CAMERA_PREFIX_COUNT", "120")

    with pytest.raises(ValueError, match="must lie between 3 and 96"):
        ProductionAlignmentEvidenceSource(
            _settings(tmp_path),
            cast(Any, object()),
            cast(Any, object()),
        )


def test_line_checkpoint_maps_only_the_exact_historical_single_head() -> None:
    weight = torch.ones((1, 4, 1, 1))
    bias = torch.ones(1)

    with pytest.warns(UserWarning, match="historical court-line final_conv"):
        state = _court_line_model_state(
            {
                "model.encoder.value": torch.ones(1),
                "model.final_conv.weight": weight,
                "model.final_conv.bias": bias,
                "optimizer.value": "ignored",
            }
        )

    assert state["heads.line.weight"] is weight
    assert state["heads.line.bias"] is bias
    assert "final_conv.weight" not in state
    assert "final_conv.bias" not in state


def test_line_checkpoint_rejects_mixed_head_schemas() -> None:
    with pytest.raises(ValueError, match="mixes or incompletely defines"):
        _court_line_model_state(
            {
                "model.final_conv.weight": torch.ones((1, 4, 1, 1)),
                "model.final_conv.bias": torch.ones(1),
                "model.heads.line.weight": torch.ones((1, 4, 1, 1)),
                "model.heads.line.bias": torch.ones(1),
            }
        )


@pytest.mark.parametrize(
    "head_state",
    (
        {"model.final_conv.weight": torch.ones((1, 4, 1, 1))},
        {"model.final_conv.bias": torch.ones(1)},
    ),
)
def test_line_checkpoint_rejects_incomplete_historical_head(
    head_state: dict[str, torch.Tensor],
) -> None:
    with pytest.raises(ValueError, match="incompletely defines"):
        _court_line_model_state(head_state)


def test_line_checkpoint_accepts_complete_canonical_head_without_remapping() -> None:
    weight = torch.ones((1, 4, 1, 1))
    bias = torch.ones(1)

    state = _court_line_model_state(
        {
            "model.heads.line.weight": weight,
            "model.heads.line.bias": bias,
        }
    )

    assert set(state) == {"heads.line.weight", "heads.line.bias"}
    assert state["heads.line.weight"] is weight
    assert state["heads.line.bias"] is bias


@pytest.mark.parametrize(
    "head_state",
    (
        {"model.encoder.value": torch.ones(1)},
        {"model.heads.line.weight": torch.ones((1, 4, 1, 1))},
        {
            "model.heads.line.weight": torch.ones((1, 4, 1, 1)),
            "model.heads.line.bias": torch.ones(1),
            "model.heads.line.extra": torch.ones(1),
        },
    ),
)
def test_line_checkpoint_rejects_noncanonical_current_head(
    head_state: dict[str, torch.Tensor],
) -> None:
    with pytest.raises(ValueError, match="exactly one complete heads.line"):
        _court_line_model_state(head_state)


def test_production_line_config_rejects_channels_without_a_dpt_size(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path).line_model
    settings = replace(
        settings,
        architecture=replace(settings.architecture, decoder_channels=96),
    )

    with pytest.raises(ValueError, match="strict DPT size preset"):
        _court_line_model_config(settings)


def test_measured_source_preflight_checks_real_images_and_detector(
    tmp_path: Path,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    scene = _scene(tmp_path, camera_count=4)
    detector = _Detector()
    source = MeasuredAlignmentEvidenceSource(
        _settings(tmp_path), detector, alignment_policy
    )

    source.preflight(scene)

    assert detector.preflight_calls == 1


def test_measured_source_preflight_fails_before_detector_when_partitions_unavailable(
    tmp_path: Path,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    scene = _scene(tmp_path, camera_count=2)
    detector = _Detector()
    source = MeasuredAlignmentEvidenceSource(
        _settings(tmp_path), detector, alignment_policy
    )

    with pytest.raises(ValueError, match="fixed alignment selection"):
        source.preflight(scene)
    assert detector.preflight_calls == 0


def test_measured_source_has_no_detector_fallback(
    tmp_path: Path,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    scene = _scene(tmp_path, camera_count=4)
    detector = _Detector(error=RuntimeError("trained detector unavailable"))
    source = MeasuredAlignmentEvidenceSource(
        _settings(tmp_path), detector, alignment_policy
    )

    with pytest.raises(RuntimeError, match="trained detector unavailable"):
        source.preflight(scene)
    assert detector.preflight_calls == 1


def test_fixed_selection_is_candidate_count_independent_with_stable_ownership(
    tmp_path: Path,
) -> None:
    base = _settings(tmp_path)
    settings = replace(
        base,
        minimum_fit_cameras=8,
        minimum_holdout_cameras=4,
        camera_prefix_count=48,
        candidate_fit=replace(
            base.candidate_fit,
            maximum_candidate_count=8,
            orientation_minimum_radians=-np.pi / 2.0,
            orientation_maximum_radians=np.pi / 2.0,
        ),
    )
    scene = _scene(tmp_path, camera_count=100)
    fixed = _fixed_camera_selection(
        scene.cameras,
        settings=settings,
    )
    fit, holdout = _partition_cameras(fixed.ordered_cameras, settings=settings)

    assert len(fixed.ordered_cameras) == 48
    assert len(fit) == 32
    assert len(holdout) == 16
    assert (
        fixed.ordered_cameras
        == _fixed_camera_selection(
            scene.cameras,
            settings=settings,
        ).ordered_cameras
    )
    assert settings.camera_partition_unit_count() == 4
    assert settings.candidate_fit.maximum_candidate_count == 8


def test_holdout_tail_preserves_fixed_fit_prefix_and_adds_only_evaluation_cameras(
    tmp_path: Path,
) -> None:
    base = _settings(tmp_path)
    settings = replace(
        base,
        minimum_fit_cameras=8,
        minimum_holdout_cameras=4,
        camera_prefix_count=48,
    )
    selected = _fixed_camera_selection(
        _scene(tmp_path, camera_count=100).cameras,
        settings=settings,
        camera_prefix_count=72,
    ).ordered_cameras

    fit, holdout = _partition_cameras_with_holdout_tail(selected, settings=settings)
    original_fit, original_holdout = _partition_cameras(
        selected[:48], settings=settings
    )

    assert fit == original_fit
    assert holdout == (*original_holdout, *selected[48:])
    assert len(fit) == 32
    assert len(holdout) == 40


def test_fixed_collection_measures_once_and_evaluates_holdout_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    base = _settings(tmp_path)
    settings = replace(
        base,
        minimum_fit_cameras=8,
        minimum_holdout_cameras=4,
        camera_prefix_count=48,
        candidate_fit=replace(
            base.candidate_fit,
            maximum_candidate_count=2,
            orientation_minimum_radians=-np.pi / 2.0,
            orientation_maximum_radians=np.pi / 2.0,
        ),
    )
    detector = _Detector()
    source = MeasuredAlignmentEvidenceSource(settings, detector, alignment_policy)
    selections: list[object] = []
    validation_calls = 0
    expected_result = fit_alignment(alignment_evidence, policy=alignment_policy)

    def evidence_for_selection(**kwargs: object) -> AlignmentEvidence:
        selections.append(kwargs["selection"])
        return alignment_evidence

    def validate(*_args: object, **_kwargs: object) -> object:
        nonlocal validation_calls
        validation_calls += 1
        return expected_result

    _stub_fixed_geometry(monkeypatch)
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_alignment_evidence_for_fixed_selection",
        evidence_for_selection,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source.fit_alignment",
        validate,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_alignment_line_heatmaps",
        lambda **_kwargs: _line_heatmaps(alignment_evidence),
    )

    assert source.collect(_scene(tmp_path, camera_count=60)) is alignment_evidence
    assert detector.predict_calls == 48
    assert validation_calls == 1
    assert len(selections) == 1
    selection = cast(FixedCameraSelectionDiagnostics, selections[0])
    assert selection.requested_camera_count == 48
    assert len(selection.fit_camera_ids) == 32
    assert len(selection.holdout_camera_ids) == 16


def test_fixed_collection_failure_does_not_reselect_or_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    base = _settings(tmp_path)
    settings = replace(
        base,
        minimum_fit_cameras=8,
        minimum_holdout_cameras=4,
        camera_prefix_count=48,
        candidate_fit=replace(
            base.candidate_fit,
            maximum_candidate_count=2,
            orientation_minimum_radians=-np.pi / 2.0,
            orientation_maximum_radians=np.pi / 2.0,
        ),
    )
    detector = _Detector()
    source = MeasuredAlignmentEvidenceSource(settings, detector, alignment_policy)
    validation_calls = 0

    def reject(*_args: object, **_kwargs: object) -> object:
        nonlocal validation_calls
        validation_calls += 1
        raise ValueError("complete validation rejected")

    _stub_fixed_geometry(monkeypatch)
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_alignment_evidence_for_fixed_selection",
        lambda **_kwargs: alignment_evidence,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source.fit_alignment",
        reject,
    )

    with pytest.raises(ValueError, match="without holdout-driven reselection or refit"):
        source.collect(_scene(tmp_path, camera_count=60))

    assert detector.predict_calls == 48
    assert validation_calls == 1


def test_observable_camera_acquisition_excludes_holdout_without_backfill(
    tmp_path: Path,
) -> None:
    settings = replace(
        _settings(tmp_path),
        minimum_fit_cameras=8,
        minimum_holdout_cameras=4,
        camera_prefix_count=24,
        candidate_fit=replace(
            _settings(tmp_path).candidate_fit,
            maximum_candidate_count=2,
            orientation_minimum_radians=-0.5,
            orientation_maximum_radians=0.5,
        ),
        projection=replace(
            _settings(tmp_path).projection,
            minimum_projected_points_per_camera=20,
        ),
    )
    selected = _scene(tmp_path, camera_count=24).cameras
    fit, holdout = _partition_cameras(selected, settings=settings)
    excluded = holdout[0]
    projected = {
        camera.camera_id: _projected_evidence(
            19 if camera.camera_id == excluded.camera_id else 20
        )
        for camera in selected
    }

    retained_fit, retained_holdout, exclusions = _retain_observable_cameras(
        fit,
        holdout,
        projected_by_camera=projected,
        settings=settings,
    )

    assert retained_fit == fit
    assert retained_holdout == holdout[1:]
    assert len(retained_fit) == 16
    assert len(retained_holdout) == 7
    assert [item.to_dict() for item in exclusions] == [
        {
            "camera_id": excluded.camera_id,
            "original_partition": "holdout",
            "selected_line_pixel_count": 20,
            "projected_line_point_count": 19,
            "reason": "insufficient_projected_points",
        }
    ]


def test_observable_camera_acquisition_fails_with_every_exclusion_listed(
    tmp_path: Path,
) -> None:
    settings = replace(
        _settings(tmp_path),
        minimum_fit_cameras=8,
        minimum_holdout_cameras=4,
        camera_prefix_count=24,
        candidate_fit=replace(
            _settings(tmp_path).candidate_fit,
            maximum_candidate_count=2,
            orientation_minimum_radians=-0.5,
            orientation_maximum_radians=0.5,
        ),
        projection=replace(
            _settings(tmp_path).projection,
            minimum_projected_points_per_camera=20,
        ),
    )
    selected = _scene(tmp_path, camera_count=24).cameras
    fit, holdout = _partition_cameras(selected, settings=settings)
    excluded_ids = {camera.camera_id for camera in holdout[:5]}
    projected = {
        camera.camera_id: _projected_evidence(
            0 if camera.camera_id in excluded_ids else 20,
            selected_count=0 if camera.camera_id in excluded_ids else 20,
        )
        for camera in selected
    }

    with pytest.raises(ValueError, match="fit=16/8,holdout=3/4") as error:
        _retain_observable_cameras(
            fit,
            holdout,
            projected_by_camera=projected,
            settings=settings,
        )

    rendered = str(error.value)
    for camera_id in sorted(excluded_ids):
        assert camera_id in rendered
    assert rendered.count("reason=no_detected_line_pixels") == 5


def test_empty_probability_is_explicit_projected_evidence_not_an_exception(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    camera = _scene(tmp_path, camera_count=3).cameras[0]
    plane = _GroundPlane(
        normal=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        offset=0.0,
        origin=np.zeros(3, dtype=np.float64),
        basis_u=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        basis_v=np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
        support_uv_bounds=(-2.0, 2.0, -2.0, 2.0),
    )

    projected = _project_probability_to_ground(
        np.zeros((8, 8), dtype=np.float32),
        camera=camera,
        plane=plane,
        model_settings=settings.line_model,
        projection_settings=settings.projection,
    )

    assert projected.selected_line_pixel_count == 0
    assert projected.points_nht_scene.shape == (0, 3)
    assert projected.points_uv.shape == (0, 2)
    assert projected.probabilities.shape == (0,)
    assert projected.proximity_weights.shape == (0,)


def test_projection_retains_probability_and_applies_camera_range_weight(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    camera = _scene(tmp_path, camera_count=3).cameras[0]
    plane = _GroundPlane(
        normal=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        offset=-1.0,
        origin=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        basis_u=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        basis_v=np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
        support_uv_bounds=(-10.0, 10.0, -10.0, 10.0),
    )
    probability: NDArray[np.float32] = np.zeros((8, 8), dtype=np.float32)
    probability[4, 4] = 0.8
    probability[4, 5] = 0.9

    projected = _project_probability_to_ground(
        probability,
        camera=camera,
        plane=plane,
        model_settings=settings.line_model,
        projection_settings=settings.projection,
    )

    ranges = np.linalg.norm(projected.points_nht_scene, axis=1)
    expected_weights = 1.0 / (
        1.0
        + np.power(
            ranges / settings.projection.proximity_scale,
            settings.projection.proximity_power,
        )
    )
    np.testing.assert_array_equal(
        projected.probabilities,
        np.asarray((0.8, 0.9), dtype=np.float32),
    )
    np.testing.assert_allclose(projected.proximity_weights, expected_weights)


@pytest.mark.parametrize("assignment_distance", (0.24, 0.685))
def test_assignment_corridor_is_cross_validated(
    tmp_path: Path,
    assignment_distance: float,
) -> None:
    settings = _settings(tmp_path)

    with pytest.raises(ValueError, match="evidence_assignment_distance_metres"):
        replace(
            settings,
            candidate_fit=replace(
                settings.candidate_fit,
                evidence_assignment_distance_metres=assignment_distance,
            ),
        )


def test_common_scale_does_not_privilege_the_first_hypothesis() -> None:
    scale, maximum_deviation = _resolve_common_scale(
        np.asarray((0.076, 0.058), dtype=np.float64),
        maximum_relative_deviation=0.3,
    )

    assert scale == pytest.approx(0.067)
    assert maximum_deviation == pytest.approx(abs(0.058 / 0.067 - 1.0))


def test_bound_saturated_and_scale_inconsistent_hypotheses_are_explicit() -> None:
    settings = _settings(Path("/tmp")).candidate_fit

    reason = _scale_bound_saturation_reason(
        settings.minimum_nht_scene_units_per_metre,
        settings=settings,
    )

    assert reason is not None
    assert "scale_bound_saturated(lower" in reason
    with pytest.raises(ValueError, match="scale-inconsistent native hypotheses"):
        _resolve_common_scale(
            np.asarray((0.06, 0.09), dtype=np.float64),
            maximum_relative_deviation=0.1,
        )


def test_spatial_tiles_recover_valid_pair_when_both_band_global_maxima_are_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(
        _settings(tmp_path).candidate_fit,
        maximum_candidate_count=2,
        minimum_explained_evidence_fraction=0.11,
        orientation_minimum_radians=-np.pi / 2.0,
        orientation_maximum_radians=np.pi / 2.0,
        minimum_nht_scene_units_per_metre=0.8,
        maximum_nht_scene_units_per_metre=1.2,
        minimum_template_score=0.1,
        maximum_fit_points=1_000,
    )
    bands = _orientation_search_bands(settings)
    np.testing.assert_allclose(
        bands,
        ((-np.pi / 2.0, 0.0), (0.0, np.pi / 2.0)),
    )
    calls: list[tuple[int, tuple[float, float], int, int]] = []
    prepared_contexts: list[_ResidualEvidenceContext] = []
    prepare_residual_evidence = _prepare_residual_evidence

    def counted_prepare_residual_evidence(
        points: NDArray[np.float64],
        evidence_weights: NDArray[np.float64],
    ) -> _ResidualEvidenceContext:
        context = prepare_residual_evidence(points, evidence_weights)
        prepared_contexts.append(context)
        return context

    def fake_optimize(
        points: NDArray[np.float64],
        *_args: object,
        **kwargs: object,
    ) -> tuple[NDArray[np.float64], float]:
        search_bounds = cast(list[tuple[float, float]], kwargs["bounds"])
        fixed_scale = cast(float | None, kwargs.get("fixed_scale"))
        if fixed_scale is not None:
            center = (search_bounds[0][0] + search_bounds[0][1]) / 2.0
            angle = (search_bounds[2][0] + search_bounds[2][1]) / 2.0
            return np.asarray((center, 0.0, angle)), 0.95
        selected = cast(tuple[_CourtHypothesis, ...], kwargs["selected"])
        band = search_bounds[2]
        if kwargs.get("polish") is False:
            context = cast(_ResidualEvidenceContext, kwargs["evidence_context"])
            assert points is context.points
            assert kwargs["evidence_weights"] is context.evidence_weights
            calls.append(
                (
                    len(points),
                    band,
                    cast(int, kwargs["seed"]),
                    id(context.nearest_tree),
                )
            )
        upper_family = band[0] >= 0.0
        lower, upper = search_bounds[0]
        if lower <= 0.0 <= upper:
            center, score = (0.0, 0.99 if not selected else 0.98)
        elif lower <= -8.0 <= upper:
            center, score = (-8.0, 0.80)
        elif lower <= 8.0 <= upper:
            center, score = (8.0, 0.79)
        else:
            center, score = ((lower + upper) / 2.0, 0.05)
        angle = 0.25 if upper_family else -0.25
        return np.asarray((center, 0.0, angle, 1.0)), score

    def fake_suppress_points(
        points: NDArray[np.float64],
        *,
        parameters: NDArray[np.float64],
        **_kwargs: object,
    ) -> NDArray[np.float64]:
        removed = 10 if abs(float(parameters[0])) < 1.0 else 45
        return points[min(removed, len(points)) :]

    def fake_suppress_evidence(
        points: NDArray[np.float64],
        evidence_weights: NDArray[np.float64],
        *,
        parameters: NDArray[np.float64],
        **_kwargs: object,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        removed = 10 if abs(float(parameters[0])) < 1.0 else 45
        offset = min(removed, len(points))
        return points[offset:], evidence_weights[offset:]

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source._optimize_court",
        fake_optimize,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_prepare_residual_evidence",
        counted_prepare_residual_evidence,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source._suppress_assigned_points",
        fake_suppress_points,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_suppress_assigned_evidence",
        fake_suppress_evidence,
    )

    hypotheses, common_scale, _deviation, search, _trace = _fit_court_hypotheses(
        np.column_stack((np.linspace(-4.0, 4.0, 100), np.zeros(100))),
        evidence_weights=np.ones(100, dtype=np.float64),
        bounds=(-10.0, 10.0, -0.1, 0.1),
        seed=42,
        settings=settings,
    )

    assert sorted(item.native_center_uv[0] for item in hypotheses) == [-8.0, 8.0]
    assert 0.0 not in [item.native_center_uv[0] for item in hypotheses]
    assert common_scale == pytest.approx(1.0)
    assert search.center_tile_count == 5
    assert search.maximum_tile_state_count == 90
    assert search.feasible_complete_state_count >= 1
    assert set(search.selected_center_tile_indices) == {0, 4}
    assert search.selected_explained_point_count == 90
    assert search.explored_tile_state_count == len(calls)
    assert search.retained_proposal_count >= 4
    assert search.residual_state_count == len(prepared_contexts)
    assert search.residual_tree_build_count == len(prepared_contexts)
    assert {tree_id for *_prefix, tree_id in calls} == {
        id(context.nearest_tree) for context in prepared_contexts
    }


def test_production_tiled_search_resource_cap_is_exact() -> None:
    resources = _proposal_search_resource_bounds(
        maximum_candidate_count=8,
        maximum_retained_state_count=128,
        orientation_band_count=2,
        center_tile_count=64,
    )

    assert resources.branch_factor == 128
    assert resources.maximum_tile_state_count == 114_816
    assert resources.maximum_residual_state_count == 897
    assert _MAXIMUM_RETAINED_PROPOSAL_STATE_COUNT == 128
    assert _MAXIMUM_TILE_STATE_COUNT == 131_072
    assert _MAXIMUM_RESIDUAL_STATE_COUNT == 1_024


def test_frontier_history_snapshot_does_not_retain_residual_arrays() -> None:
    points: NDArray[np.float64] = np.zeros((100, 2), dtype=np.float64)
    weights: NDArray[np.float64] = np.ones(len(points), dtype=np.float64)
    state = _ProposalSearchState(
        selected=(_hypothesis_for_topology(center=(0.0, 0.0)),),
        residual=points,
        residual_evidence_weights=weights,
        explained_evidence_fractions=(0.5,),
        orientation_band_indices=(0,),
        center_tile_indices=(0,),
    )

    snapshot = _proposal_frontier_snapshot(state)

    assert not hasattr(snapshot, "residual")
    assert not hasattr(snapshot, "residual_evidence_weights")
    assert snapshot.residual_point_count == 100
    assert snapshot.residual_evidence_sum == pytest.approx(100.0)


@pytest.mark.parametrize(
    (
        "maximum_candidate_count",
        "maximum_retained_state_count",
        "orientation_band_count",
        "center_tile_count",
    ),
    ((9, 128, 2, 64), (8, 129, 2, 64), (2, 128, 3, 64), (2, 128, 2, 65)),
)
def test_tiled_search_resource_cap_rejects_overflow(
    maximum_candidate_count: int,
    maximum_retained_state_count: int,
    orientation_band_count: int,
    center_tile_count: int,
) -> None:
    with pytest.raises(ValueError, match="bounded resource cap"):
        _proposal_search_resource_bounds(
            maximum_candidate_count=maximum_candidate_count,
            maximum_retained_state_count=maximum_retained_state_count,
            orientation_band_count=orientation_band_count,
            center_tile_count=center_tile_count,
        )


def test_prepared_residual_tree_preserves_optimizer_result(tmp_path: Path) -> None:
    settings = replace(
        _settings(tmp_path).candidate_fit,
        samples_per_metre=1.0,
        optimizer_maximum_iterations=2,
        optimizer_population_size=5,
    )
    template = sample_court_line_template(settings.samples_per_metre)
    points = transform_template_2d(
        template,
        np.asarray((0.2, -0.1, 0.05, 0.07), dtype=np.float64),
    )
    weights: NDArray[np.float64] = np.ones(len(points), dtype=np.float64)
    bounds = ((-1.0, 1.0), (-1.0, 1.0), (-0.2, 0.2), (0.06, 0.08))
    baseline_parameters, baseline_score = _optimize_court(
        points,
        evidence_weights=weights,
        template=template,
        bounds=bounds,
        seed=42,
        settings=settings,
    )
    evidence = _prepare_residual_evidence(points, weights)

    reused_parameters, reused_score = _optimize_court(
        evidence.points,
        evidence_weights=evidence.evidence_weights,
        template=template,
        bounds=bounds,
        seed=42,
        settings=settings,
        evidence_context=evidence,
    )

    np.testing.assert_array_equal(reused_parameters, baseline_parameters)
    assert reused_score == baseline_score


@pytest.mark.parametrize(
    "weights",
    (
        np.asarray((1.0, 9.0), dtype=np.float64),
        np.asarray((9.0, 1.0), dtype=np.float64),
    ),
)
def test_optimizer_score_uses_weighted_coverage_with_unweighted_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    weights: NDArray[np.float64],
) -> None:
    settings = replace(
        _settings(tmp_path).candidate_fit,
        score_distance_metres=1.0,
        optimizer_maximum_iterations=1,
        optimizer_population_size=2,
    )
    template = np.asarray(((0.0, 0.0), (1.5, 0.0)), dtype=np.float64)
    points = np.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=np.float64)
    candidate = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    distant_kernel = math.exp(-0.5 * 0.5**2)
    unweighted_score = (1.0 + distant_kernel) / 2.0
    weighted_score = float((weights[0] + weights[1] * distant_kernel) / np.sum(weights))
    expected = min(unweighted_score, weighted_score)

    def fake_differential_evolution(
        objective: Callable[[NDArray[np.float64]], float],
        _bounds: object,
        **_kwargs: object,
    ) -> SimpleNamespace:
        assert -objective(candidate) == pytest.approx(expected)
        return SimpleNamespace(x=candidate)

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "differential_evolution",
        fake_differential_evolution,
    )

    optimized, score = _optimize_court(
        points,
        evidence_weights=weights,
        template=template,
        bounds=((-1.0, 1.0), (-1.0, 1.0), (-0.1, 0.1), (0.9, 1.1)),
        seed=42,
        settings=settings,
    )

    np.testing.assert_array_equal(optimized, candidate)
    assert score == pytest.approx(expected)


@pytest.mark.parametrize(
    "weights",
    (
        np.asarray((1.0, 0.0), dtype=np.float64),
        np.asarray((1.0, np.nan), dtype=np.float64),
        np.asarray((1.0,), dtype=np.float64),
    ),
)
def test_residual_evidence_rejects_invalid_weights(
    weights: NDArray[np.float64],
) -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=np.float64)

    with pytest.raises(ValueError, match="positive finite value per point"):
        _prepare_residual_evidence(points, weights)


def test_center_tiles_cover_exact_two_dimensional_bounds_with_half_open_edges() -> None:
    tiles = _center_space_tiles((0.0, 1.0, -0.2, 0.5), maximum_width=0.3)

    assert len(tiles) == 12
    assert [(tile.u_index, tile.v_index) for tile in tiles] == [
        (u_index, v_index) for u_index in range(4) for v_index in range(3)
    ]
    assert tiles[0].u_bounds[0] == 0.0
    assert tiles[0].v_bounds[0] == -0.2
    assert tiles[-1].logical_u_upper == 1.0
    assert tiles[-1].logical_v_upper == 0.5
    assert tiles[0].u_bounds[1] < tiles[3].u_bounds[0]
    assert np.nextafter(tiles[0].u_bounds[1], math.inf) == tiles[3].u_bounds[0]
    assert all(
        tile.logical_u_upper - tile.u_bounds[0] <= 0.3
        and tile.logical_v_upper - tile.v_bounds[0] <= 0.3
        for tile in tiles
    )


def test_weighted_coverage_floor_preserves_unweighted_tile_bound(
    tmp_path: Path,
) -> None:
    settings = replace(
        _candidate_count_settings(tmp_path, maximum_candidate_count=2),
        minimum_template_score=0.8,
    )
    points = np.asarray(((0.0, 0.0), (0.0, 0.1)), dtype=np.float64)
    relative_bounds = np.asarray(
        ((0.0, 0.0, 0.0, 0.0), (10.0, 0.0, 10.0, 0.0)),
        dtype=np.float64,
    )
    tile = _CenterTile(
        flat_index=0,
        u_index=0,
        v_index=0,
        u_bounds=(0.0, 0.0),
        v_bounds=(0.0, 0.0),
        logical_u_upper=0.0,
        logical_v_upper=0.0,
    )
    uniform = _prepare_residual_evidence(
        points,
        np.ones(len(points), dtype=np.float64),
    )
    weighted = _prepare_residual_evidence(
        points,
        np.asarray((9.0, 1.0), dtype=np.float64),
    )

    assert _tile_is_geometrically_impossible(
        evidence=uniform,
        tile=tile,
        template_relative_bounds=relative_bounds,
        settings=settings,
        selected=(),
    )
    assert _tile_is_geometrically_impossible(
        evidence=weighted,
        tile=tile,
        template_relative_bounds=relative_bounds,
        settings=settings,
        selected=(),
    )


def test_adjacent_tile_boundary_duplicates_are_removed_without_losing_basins() -> None:
    shared = _NativeProposal(
        parameters=np.asarray((0.5, 0.0, 0.1, 1.0)),
        measured_score=0.9,
        orientation_band_radians=(-0.5, 0.5),
        residual_point_count=100,
    )
    distinct = replace(
        shared,
        parameters=np.asarray((8.0, 0.0, 0.1, 1.0)),
        measured_score=0.8,
    )
    retained, duplicate_count = _deduplicate_tiled_proposals(
        (
            _TiledProposal(shared, 0, 0),
            _TiledProposal(shared, 0, 1),
            _TiledProposal(distinct, 0, 2),
        ),
        center_tolerance=1.0e-8,
    )

    assert duplicate_count == 1
    assert [item.proposal.parameters[0] for item in retained] == [0.5, 8.0]


def test_geometry_derives_center_tile_width_from_minimum_compatible_separation(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path).candidate_fit

    assert _maximum_center_tile_width_scene_units(settings) == pytest.approx(
        settings.minimum_center_separation_metres
        * settings.minimum_nht_scene_units_per_metre
        / 2.0
    )


def test_proposal_seeds_are_bound_to_branch_geometry_not_iteration_position() -> None:
    lower = (-np.pi / 2.0, 0.0)
    upper = (0.0, np.pi / 2.0)
    first = replace(
        _hypothesis_for_topology(center=(-8.0, 0.0)),
        proposal_orientation_band_radians=lower,
        native_orientation_radians=-0.25,
        orientation_radians=-0.25,
    )

    tile = _CenterTile(0, 0, 0, (-1.0, 0.0), (-1.0, 0.0), 0.0, 0.0)
    direct = {
        band: _proposal_branch_seed(
            42, selected=(), orientation_band=band, center_tile=tile
        )
        for band in (lower, upper)
    }
    reversed_iteration = {
        band: _proposal_branch_seed(
            42, selected=(), orientation_band=band, center_tile=tile
        )
        for band in (upper, lower)
    }

    assert direct == reversed_iteration
    assert direct[upper] != _proposal_branch_seed(
        42, selected=(first,), orientation_band=upper, center_tile=tile
    )
    adjacent = replace(
        tile,
        flat_index=1,
        u_index=1,
        u_bounds=(0.0, 1.0),
        logical_u_upper=1.0,
    )
    assert direct[lower] != _proposal_branch_seed(
        42, selected=(), orientation_band=lower, center_tile=adjacent
    )


def test_adjacent_shared_line_proposals_pass_but_overlapping_duplicate_fails(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path).candidate_fit
    first = _hypothesis_for_topology(center=(0.0, 0.0))
    adjacent = _hypothesis_for_topology(center=(0.07 * 10.97, 0.0))
    duplicate = _hypothesis_for_topology(center=(0.01, 0.0))

    assert _proposal_topology_compatible(
        adjacent,
        selected=(first,),
        settings=settings,
    )

    assert not _proposal_topology_compatible(
        duplicate,
        selected=(first,),
        settings=settings,
    )


def test_native_optimizer_penalizes_duplicate_basin_inside_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path).candidate_fit
    template = sample_court_line_template(settings.samples_per_metre)
    invalid = np.asarray((0.0, 0.0, 0.0, 0.07), dtype=np.float64)
    valid = np.asarray((0.07 * 10.97, 0.0, 0.0, 0.07), dtype=np.float64)
    points = transform_template_2d(template, invalid)

    def fake_differential_evolution(
        objective: Callable[[NDArray[np.float64]], float],
        _bounds: object,
        **_kwargs: object,
    ) -> SimpleNamespace:
        assert objective(invalid) == pytest.approx(0.0)
        assert objective(valid) < 0.0
        return SimpleNamespace(x=valid)

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source.differential_evolution",
        fake_differential_evolution,
    )

    optimized, score = _optimize_court(
        points,
        evidence_weights=np.ones(len(points), dtype=np.float64),
        template=template,
        bounds=((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5), (0.05, 0.1)),
        seed=42,
        settings=settings,
        selected=(_hypothesis_for_topology(center=(0.0, 0.0)),),
    )

    np.testing.assert_array_equal(optimized, valid)
    assert score > 0.0


def test_optimizer_retains_a_better_valid_initial_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(tmp_path).candidate_fit
    template = sample_court_line_template(settings.samples_per_metre)
    initial = np.asarray((0.0, 0.0, 0.0, 0.07), dtype=np.float64)
    missed = np.asarray((0.8, 0.8, 0.4, 0.06), dtype=np.float64)
    points = transform_template_2d(template, initial)

    def fake_differential_evolution(
        _objective: Callable[[NDArray[np.float64]], float],
        _bounds: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        np.testing.assert_array_equal(kwargs["x0"], initial)
        return SimpleNamespace(x=missed)

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source.differential_evolution",
        fake_differential_evolution,
    )

    optimized, score = _optimize_court(
        points,
        evidence_weights=np.ones(len(points), dtype=np.float64),
        template=template,
        bounds=((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5), (0.05, 0.1)),
        seed=42,
        settings=settings,
        initial_parameters=initial,
    )

    np.testing.assert_array_equal(optimized, initial)
    assert score > settings.minimum_template_score


def test_one_court_proposal_search_infers_one_and_stops_from_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(
        _settings(tmp_path).candidate_fit,
        maximum_candidate_count=4,
        minimum_explained_evidence_fraction=0.17,
        samples_per_metre=2.0,
        minimum_nht_scene_units_per_metre=0.8,
        maximum_nht_scene_units_per_metre=1.2,
        orientation_minimum_radians=-0.2,
        orientation_maximum_radians=0.2,
        score_distance_metres=0.2,
        minimum_template_score=0.4,
        optimizer_maximum_iterations=70,
        optimizer_population_size=8,
        maximum_fit_points=10_000,
        scale_bound_margin_relative=0.005,
        evidence_assignment_distance_metres=0.35,
    )
    template = sample_court_line_template(settings.samples_per_metre)
    observed = transform_template_2d(
        template,
        np.asarray((0.0, 0.0, 0.04, 1.0)),
    )

    def one_court_optimum(
        *_args: object, **kwargs: object
    ) -> tuple[NDArray[np.float64], float]:
        search_bounds = cast(list[tuple[float, float]], kwargs["bounds"])
        lower, upper = search_bounds[0]
        v_center = (search_bounds[1][0] + search_bounds[1][1]) / 2.0
        angle = (search_bounds[2][0] + search_bounds[2][1]) / 2.0
        if lower <= 0.0 <= upper and search_bounds[1][0] <= 0.0 <= search_bounds[1][1]:
            return np.asarray((0.0, 0.0, angle, 1.0)), 1.0
        return np.asarray(((lower + upper) / 2.0, v_center, angle, 1.0)), 0.0

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source._optimize_court",
        one_court_optimum,
    )

    hypotheses, _scale, _deviation, search, _trace = _fit_court_hypotheses(
        observed,
        evidence_weights=np.ones(len(observed), dtype=np.float64),
        bounds=(-15.0, 15.0, -15.0, 15.0),
        seed=42,
        settings=settings,
    )

    assert len(hypotheses) == 1
    assert search.inferred_candidate_count == 1
    assert (
        search.stopping_reason
        is ProposalSearchStopReason.RESIDUAL_EVIDENCE_BELOW_MINIMUM
    )


def test_three_court_proposal_search_infers_three_before_no_reliable_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _candidate_count_settings(
        tmp_path,
        maximum_candidate_count=4,
    )
    _stub_linear_candidate_search(
        monkeypatch,
        reliable_candidate_count=3,
    )
    points = np.column_stack(
        (np.linspace(-40.0, 40.0, 270), np.linspace(-10.0, 10.0, 270))
    )

    hypotheses, common_scale, maximum_deviation, search, _trace = _fit_court_hypotheses(
        points,
        evidence_weights=np.ones(len(points), dtype=np.float64),
        bounds=(-50.0, 50.0, -20.0, 20.0),
        seed=42,
        settings=settings,
    )

    assert len(hypotheses) == 3
    assert [item.center_uv[0] for item in hypotheses] == [-24.0, 0.0, 24.0]
    assert common_scale == pytest.approx(1.0)
    assert maximum_deviation == pytest.approx(0.0)
    assert search.inferred_candidate_count == 3
    assert search.stopping_reason is ProposalSearchStopReason.NO_RELIABLE_PROPOSAL
    assert search.selected_candidate_explained_evidence_fractions == pytest.approx(
        (60.0 / 270.0, 40.0 / 270.0, 90.0 / 270.0)
    )
    assert all(
        fraction >= settings.minimum_explained_evidence_fraction
        for fraction in search.selected_candidate_explained_evidence_fractions
    )
    assert search.frontier_state_counts == (1, 1, 1)
    assert search.feasible_complete_state_counts == (1, 1, 1)
    assert search.refinement_attempt_count == 3
    assert search.refinement_rejected_state_count == 2
    assert search.selected_complete_state_rank == 2
    assert search.selected_complete_state_candidate_count == 3


def test_refined_shallow_frontier_wins_before_deeper_proposals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(
        _candidate_count_settings(tmp_path, maximum_candidate_count=3),
        minimum_explained_evidence_fraction=0.2,
    )

    def optimize(
        *_args: object,
        **kwargs: object,
    ) -> tuple[NDArray[np.float64], float]:
        fixed_scale = kwargs.get("fixed_scale")
        if fixed_scale is not None:
            fixed_center, _maximum_distance = cast(
                tuple[NDArray[np.float64], float], kwargs["center_limit"]
            )
            return np.asarray((fixed_center[0], fixed_center[1], 0.0)), 0.9
        selected = cast(tuple[_CourtHypothesis, ...], kwargs.get("selected", ()))
        if len(selected) >= 2:
            return np.asarray((48.0, 0.0, 0.0, 1.0)), 0.0
        proposal_center = 0.0 if not selected else 24.0
        return np.asarray((proposal_center, 0.0, 0.0, 1.0)), 0.9

    def suppress_evidence(
        points: NDArray[np.float64],
        evidence_weights: NDArray[np.float64],
        *,
        parameters: NDArray[np.float64],
        **_kwargs: object,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        explained_fraction = 0.9 if float(parameters[0]) == 1.0 else 0.5
        offset = round(len(points) * explained_fraction)
        return points[offset:], evidence_weights[offset:]

    def refine(
        selected_state: _ProposalSearchState,
        **_kwargs: object,
    ) -> tuple[_CourtHypothesis, ...]:
        if len(selected_state.selected) == 1:
            item = selected_state.selected[0]
            return (
                replace(
                    item,
                    center_uv=(1.0, 0.0),
                    native_center_uv=(1.0, 0.0),
                ),
            )
        return selected_state.selected

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source._optimize_court",
        optimize,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_maximum_center_tile_width_scene_units",
        lambda _settings: 200.0,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_tile_is_geometrically_impossible",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_suppress_assigned_evidence",
        suppress_evidence,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_refine_selected_native_hypotheses",
        refine,
    )
    points = np.column_stack(
        (np.linspace(-50.0, 50.0, 100), np.zeros(100, dtype=np.float64))
    )

    hypotheses, _scale, _deviation, search, _trace = _fit_court_hypotheses(
        points,
        evidence_weights=np.ones(len(points), dtype=np.float64),
        bounds=(-60.0, 60.0, -5.0, 5.0),
        seed=42,
        settings=settings,
    )

    assert len(hypotheses) == 1
    assert hypotheses[0].native_center_uv == (1.0, 0.0)
    assert search.frontier_state_counts == (1, 1)
    assert search.feasible_complete_state_counts == (1, 1)
    assert search.selected_complete_state_candidate_count == 1
    assert search.refinement_attempt_count == 1
    assert search.selected_residual_evidence_sum == pytest.approx(10.0)


def test_complete_state_residual_is_recomputed_after_common_scale_refit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(
        _candidate_count_settings(tmp_path, maximum_candidate_count=2),
        minimum_explained_evidence_fraction=0.2,
    )
    native = replace(
        _hypothesis_for_topology(center=(1.0, 0.0)),
        nht_scene_units_per_metre=1.0,
        native_nht_scene_units_per_metre=1.0,
    )
    points = np.column_stack(
        (np.linspace(-10.0, 10.0, 100), np.zeros(100, dtype=np.float64))
    )
    weights: NDArray[np.float64] = np.ones(len(points), dtype=np.float64)
    selected_state = _ProposalSearchState(
        selected=(native,),
        residual=points[90:],
        residual_evidence_weights=weights[90:],
        explained_evidence_fractions=(0.9,),
        orientation_band_indices=(0,),
        center_tile_indices=(0,),
    )

    def suppress_evidence(
        values: NDArray[np.float64],
        evidence_weights: NDArray[np.float64],
        *,
        parameters: NDArray[np.float64],
        **_kwargs: object,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        explained_fraction = 0.9 if float(parameters[0]) == 1.0 else 0.5
        offset = round(len(values) * explained_fraction)
        return values[offset:], evidence_weights[offset:]

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_refine_selected_native_hypotheses",
        lambda *_args, **_kwargs: (native,),
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source._optimize_court",
        lambda *_args, **_kwargs: (np.asarray((0.0, 0.0, 0.0)), 0.9),
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_suppress_assigned_evidence",
        suppress_evidence,
    )

    refined = _refine_complete_proposal_state(
        selected_state,
        points=points,
        evidence_weights=weights,
        bounds=(-20.0, 20.0, -5.0, 5.0),
        template=sample_court_line_template(settings.samples_per_metre),
        orientation_bands=_orientation_search_bands(settings),
        center_tiles=_center_space_tiles(
            (-20.0, 20.0, -5.0, 5.0),
            maximum_width=100.0,
        ),
        seed=42,
        settings=settings,
    )

    assert refined.hypotheses[0].center_uv == (0.0, 0.0)
    assert len(refined.residual) == 50
    assert np.sum(refined.residual_evidence_weights) == pytest.approx(50.0)


def test_ranked_complete_state_refinement_runs_in_parallel_and_retains_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _candidate_count_settings(
        tmp_path,
        maximum_candidate_count=3,
    )
    _stub_multiple_complete_state_search(monkeypatch)

    refinement_calls: list[tuple[float, ...]] = []
    refinement_threads: set[int] = set()
    refinement_barrier = threading.Barrier(3)

    def refine(
        selected_state: _ProposalSearchState,
        **_kwargs: object,
    ) -> tuple[_CourtHypothesis, ...]:
        selected = selected_state.selected
        refinement_calls.append(tuple(item.center_uv[0] for item in selected))
        refinement_threads.add(threading.get_ident())
        refinement_barrier.wait(timeout=2.0)
        if selected[0].center_uv[0] < 0.0:
            raise ValueError("ranked basin saturated during refinement")
        return selected

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_refine_selected_native_hypotheses",
        refine,
    )

    hypotheses, _scale, _deviation, search, _trace = _fit_court_hypotheses(
        np.column_stack((np.linspace(-10.0, 10.0, 100), np.zeros(100))),
        evidence_weights=np.ones(100, dtype=np.float64),
        bounds=(-10.0, 10.0, -0.1, 0.1),
        seed=42,
        settings=settings,
    )

    assert sorted(refinement_calls) == [(-8.0,), (0.0,), (8.0,)]
    assert len(refinement_threads) == 3
    assert hypotheses[0].native_center_uv[0] == pytest.approx(-8.0)
    assert search.feasible_complete_state_count >= 2
    assert search.refinement_attempt_count == 1
    assert search.refinement_rejected_state_count == 0
    assert search.selected_complete_state_rank == 0


def test_fit_validator_backtracks_to_next_complete_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _candidate_count_settings(
        tmp_path,
        maximum_candidate_count=3,
    )
    _stub_multiple_complete_state_search(monkeypatch)
    validated_centers: list[float] = []

    def require_reliable(
        hypotheses: tuple[_CourtHypothesis, ...],
        _common_scale: float,
    ) -> None:
        center = hypotheses[0].native_center_uv[0]
        validated_centers.append(center)
        if center < 0.0:
            raise ValueError("fit-unreliable complete state")

    hypotheses, _scale, _deviation, search, _trace = _fit_court_hypotheses(
        np.column_stack((np.linspace(-10.0, 10.0, 100), np.zeros(100))),
        evidence_weights=np.ones(100, dtype=np.float64),
        bounds=(-10.0, 10.0, -0.1, 0.1),
        seed=42,
        settings=settings,
        complete_state_validator=require_reliable,
    )

    assert sorted(validated_centers) == pytest.approx([-8.0, 0.0, 8.0])
    assert hypotheses[0].native_center_uv[0] == pytest.approx(8.0)
    assert search.refinement_attempt_count == 2
    assert search.refinement_rejected_state_count == 1
    assert search.selected_complete_state_rank == 1


def test_fit_validator_selects_largest_complete_set_despite_noise_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _candidate_count_settings(
        tmp_path,
        maximum_candidate_count=3,
    )
    _stub_linear_candidate_search(
        monkeypatch,
        reliable_candidate_count=4,
    )
    validated_counts: list[int] = []

    def require_three_complete_courts(
        hypotheses: tuple[_CourtHypothesis, ...],
        _common_scale: float,
    ) -> None:
        validated_counts.append(len(hypotheses))
        if len(hypotheses) != 3:
            raise ValueError("not the largest independently complete court set")

    hypotheses, _scale, _deviation, search, _trace = _fit_court_hypotheses(
        np.column_stack((np.linspace(-40.0, 40.0, 270), np.linspace(-10.0, 10.0, 270))),
        evidence_weights=np.ones(270, dtype=np.float64),
        bounds=(-60.0, 60.0, -20.0, 20.0),
        seed=42,
        settings=settings,
        complete_state_validator=require_three_complete_courts,
    )

    assert len(hypotheses) == 3
    assert validated_counts
    assert max(validated_counts) == 3
    assert len(search.frontier_state_counts) == 3
    assert search.selected_complete_state_candidate_count == 3
    assert (
        search.stopping_reason is ProposalSearchStopReason.NO_ADDITIONAL_COMPLETE_COURT
    )
    assert (
        search.selected_residual_evidence_sum / search.original_evidence_sum
        > settings.minimum_explained_evidence_fraction
    )


def test_one_partial_court_is_repaired_from_two_valid_lattice_centers(
    tmp_path: Path,
) -> None:
    settings = _candidate_count_settings(tmp_path, maximum_candidate_count=4)
    template = sample_court_line_template(settings.samples_per_metre)

    def hypothesis(candidate_id: str, center: tuple[float, float]) -> _CourtHypothesis:
        return replace(
            _hypothesis_for_topology(center=center),
            candidate_id=candidate_id,
            nht_scene_units_per_metre=1.0,
            native_nht_scene_units_per_metre=1.0,
            template_score=1.0,
            native_template_score=1.0,
            native_center_uv=center,
            proposal_orientation_band_radians=(-0.2, 0.2),
        )

    hypotheses = (
        hypothesis("candidate-000", (0.0, 0.0)),
        hypothesis("candidate-001", (12.0, 0.0)),
        hypothesis("candidate-002", (60.0, 0.0)),
    )
    points = np.concatenate(
        [
            transform_template_2d(template, (center, 0.0, 0.0, 1.0))
            for center in (0.0, 12.0, 24.0)
        ]
    )
    weights: NDArray[np.float64] = np.ones(len(points), dtype=np.float64)
    state = _RefinedCompleteState(
        hypotheses=hypotheses,
        selected_proposals=hypotheses,
        native_refined=hypotheses,
        common_scale=1.0,
        maximum_scale_deviation=0.0,
        explained_evidence_fractions=(0.2, 0.2, 0.2),
        residual=points,
        residual_evidence_weights=weights,
        selected_orientation_band_indices=(0, 0, 0),
        selected_center_tile_indices=(0, 1, 2),
        native_score_sum=3.0,
    )

    def require_lattice_court(candidate: _CourtHypothesis) -> None:
        if (
            min(abs(candidate.center_uv[0] - value) for value in (0.0, 12.0, 24.0))
            > 1e-9
        ):
            raise ValueError("not a complete lattice court")

    def require_complete_lattice(
        candidates: tuple[_CourtHypothesis, ...], _common_scale: float
    ) -> None:
        for candidate in candidates:
            require_lattice_court(candidate)

    repaired, _rejections = _repair_one_lattice_outlier(
        state,
        points=points,
        evidence_weights=weights,
        bounds=(-10.0, 70.0, -20.0, 20.0),
        settings=settings,
        candidate_validator=require_lattice_court,
        complete_state_validator=require_complete_lattice,
    )

    assert repaired is not None
    assert repaired.hypotheses[2].center_uv == pytest.approx((24.0, 0.0))
    assert len(repaired.residual) < len(points)


def test_boundary_lattice_assistance_requires_positive_half_court_semantics(
    tmp_path: Path,
) -> None:
    longitudinal = np.concatenate(
        [
            np.column_stack(
                (
                    np.full(7, offset),
                    np.linspace(-6.0, 6.0, 7),
                    np.zeros(7),
                )
            )
            for offset in (0.0, 4.115, 5.485)
        ]
    )
    transverse = np.concatenate(
        [
            np.column_stack(
                (
                    np.linspace(-4.0, 4.0, 7),
                    np.full(7, offset),
                    np.zeros(7),
                )
            )
            for offset in (-11.885, -6.4, 6.4)
        ]
    )
    points = np.concatenate((longitudinal, transverse))
    camera_ids = tuple(f"camera-{index % 4}" for index in range(len(points)))
    correspondences = CorrespondenceSet(
        points_court=points,
        points_scene=points,
        camera_ids=camera_ids,
    )
    settings = replace(
        _settings(tmp_path).candidate_fit,
        common_scale_relative_tolerance=0.07,
    ).whole_court_evidence(
        required_court_count=3,
        minimum_matches_per_offset_level=3,
    )
    metrics = evaluate_court_identifiability(
        correspondences,
        minimum_camera_count=4,
        settings=settings,
    )

    assert (
        metrics.to_dict(
            minimum_camera_count=4,
            settings=settings,
        )["accepted"]
        is False
    )
    assistance = evaluate_boundary_lattice_identifiability(
        metrics,
        minimum_camera_count=4,
    )
    assert assistance["accepted"] is True
    assert all(cast(dict[str, bool], assistance["threshold_checks"]).values())


def test_candidate_validator_filters_partial_court_before_beam_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _candidate_count_settings(
        tmp_path,
        maximum_candidate_count=3,
    )
    _stub_multiple_complete_state_search(monkeypatch)
    validated_centers: list[float] = []

    def require_complete_court(hypothesis: _CourtHypothesis) -> None:
        center = hypothesis.center_uv[0]
        validated_centers.append(center)
        if center < 0.0:
            raise ValueError("partial-court proposal")

    hypotheses, _scale, _deviation, search, _trace = _fit_court_hypotheses(
        np.column_stack((np.linspace(-10.0, 10.0, 100), np.zeros(100))),
        evidence_weights=np.ones(100, dtype=np.float64),
        bounds=(-10.0, 10.0, -0.1, 0.1),
        seed=42,
        settings=settings,
        candidate_validator=require_complete_court,
    )

    assert sorted(validated_centers) == pytest.approx([-8.0, 0.0, 8.0])
    assert hypotheses[0].native_center_uv[0] == pytest.approx(8.0)
    assert search.frontier_state_counts == (2,)


def test_candidate_validator_does_not_reject_coarse_residual_proposals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _candidate_count_settings(
        tmp_path,
        maximum_candidate_count=4,
    )
    _stub_linear_candidate_search(
        monkeypatch,
        reliable_candidate_count=3,
    )
    validated_centers: list[float] = []

    hypotheses, _scale, _deviation, _search, _trace = _fit_court_hypotheses(
        np.column_stack((np.linspace(-40.0, 40.0, 270), np.linspace(-10.0, 10.0, 270))),
        evidence_weights=np.ones(270, dtype=np.float64),
        bounds=(-50.0, 50.0, -20.0, 20.0),
        seed=42,
        settings=settings,
        candidate_validator=lambda hypothesis: validated_centers.append(
            hypothesis.center_uv[0]
        ),
    )

    assert len(hypotheses) == 3
    assert validated_centers == pytest.approx([24.0])


def test_complete_state_refinement_falls_back_to_valid_ranked_proposal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _candidate_count_settings(
        tmp_path,
        maximum_candidate_count=3,
    )
    _stub_multiple_complete_state_search(monkeypatch)

    def reject_refinement(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[_CourtHypothesis, ...]:
        raise ValueError("refined basin rejected")

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_refine_selected_native_hypotheses",
        reject_refinement,
    )
    points = np.column_stack((np.linspace(-10.0, 10.0, 100), np.zeros(100)))

    hypotheses, _scale, _deviation, search, _trace = _fit_court_hypotheses(
        points,
        evidence_weights=np.ones(len(points), dtype=np.float64),
        bounds=(-10.0, 10.0, -0.1, 0.1),
        seed=42,
        settings=settings,
    )

    assert len(hypotheses) == 1
    assert hypotheses[0].native_center_uv[0] == pytest.approx(-8.0)
    assert search.selected_complete_state_rank == 0
    assert search.refinement_rejected_state_count == 0


def test_court_count_inference_rejects_zero_reliable_proposals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _candidate_count_settings(
        tmp_path,
        maximum_candidate_count=4,
    )
    _stub_linear_candidate_search(
        monkeypatch,
        reliable_candidate_count=0,
    )
    points = np.column_stack((np.linspace(-10.0, 10.0, 30), np.linspace(-3.0, 3.0, 30)))

    with pytest.raises(ValueError, match="found no reliable court proposal"):
        _fit_court_hypotheses(
            points,
            evidence_weights=np.ones(len(points), dtype=np.float64),
            bounds=(-50.0, 50.0, -20.0, 20.0),
            seed=42,
            settings=settings,
        )


def test_court_count_inference_fails_at_maximum_with_reliable_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _candidate_count_settings(
        tmp_path,
        maximum_candidate_count=2,
    )
    _stub_linear_candidate_search(
        monkeypatch,
        reliable_candidate_count=3,
    )
    points = np.column_stack(
        (np.linspace(-40.0, 40.0, 270), np.linspace(-10.0, 10.0, 270))
    )

    with pytest.raises(ValueError, match="reached maximum_candidate_count"):
        _fit_court_hypotheses(
            points,
            evidence_weights=np.ones(len(points), dtype=np.float64),
            bounds=(-50.0, 50.0, -20.0, 20.0),
            seed=42,
            settings=settings,
        )


def test_fit_only_reliability_rejects_semantically_incomplete_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alignment_evidence: AlignmentEvidence,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    valid = alignment_evidence.candidates[0].fit
    longitudinal: NDArray[np.bool_] = np.asarray(
        np.isclose(
            np.abs(valid.points_court[:, 0]),
            HALF_DOUBLES_WIDTH,
            atol=1.0e-8,
            rtol=0.0,
        ),
        dtype=np.bool_,
    )
    incomplete = CorrespondenceSet(
        points_court=valid.points_court[longitudinal],
        points_scene=valid.points_scene[longitudinal],
        camera_ids=tuple(
            camera_id
            for camera_id, retained in zip(
                valid.camera_ids,
                longitudinal,
                strict=True,
            )
            if retained
        ),
    )
    hypotheses = (
        replace(
            _hypothesis_for_topology(center=(0.0, 0.0)),
            candidate_id="candidate-000",
        ),
        replace(
            _hypothesis_for_topology(center=(2.0, 0.0)),
            candidate_id="candidate-001",
        ),
    )
    correspondences = {
        "candidate-000": valid,
        "candidate-001": incomplete,
    }
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_assign_candidate_evidence",
        lambda candidates, **_kwargs: {
            candidate.candidate_id: {} for candidate in candidates
        },
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_candidate_correspondences",
        lambda hypothesis, **_kwargs: correspondences[hypothesis.candidate_id],
    )
    plane = _GroundPlane(
        normal=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        offset=0.0,
        origin=np.zeros(3, dtype=np.float64),
        basis_u=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        basis_v=np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
        support_uv_bounds=(-5.0, 5.0, -5.0, 5.0),
    )
    fit_cameras = _scene(tmp_path, camera_count=2).cameras

    reliability_settings = replace(
        _settings(tmp_path),
        candidate_fit=replace(
            _settings(tmp_path).candidate_fit,
            common_scale_relative_tolerance=0.05,
        ),
    )
    retained, transforms, rejections = _fit_reliable_hypothesis_indices(
        hypotheses,
        common_scale=1.0,
        plane=plane,
        fit_cameras=fit_cameras,
        projected_by_camera={
            camera.camera_id: _projected_evidence(20) for camera in fit_cameras
        },
        settings=reliability_settings,
        policy=alignment_policy,
    )

    assert retained == (0,)
    assert len(transforms) == 1
    assert len(rejections) == 1
    assert rejections[0].startswith("candidate-001:semantic_identifiability_rejected(")
    assert "'longitudinal'" in rejections[0]
    assert "'transverse'" in rejections[0]


def test_fit_only_reliability_refits_and_rebuilds_selected_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alignment_policy: AlignmentAcceptancePolicy,
) -> None:
    candidate_settings = _candidate_count_settings(
        tmp_path,
        maximum_candidate_count=4,
    )
    candidate_settings = replace(
        candidate_settings,
        common_scale_relative_tolerance=0.05,
    )
    settings = replace(
        _settings(tmp_path),
        candidate_fit=candidate_settings,
    )
    _stub_linear_candidate_search(
        monkeypatch,
        reliable_candidate_count=3,
    )
    points = np.column_stack(
        (np.linspace(-40.0, 40.0, 270), np.linspace(-10.0, 10.0, 270))
    )
    hypotheses, common_scale, maximum_deviation, search, fit_trace = (
        _fit_court_hypotheses(
            points,
            evidence_weights=np.ones(len(points), dtype=np.float64),
            bounds=(-50.0, 50.0, -20.0, 20.0),
            seed=42,
            settings=candidate_settings,
        )
    )
    reliability_counts: list[int] = []

    def fit_reliability(
        candidates: tuple[_CourtHypothesis, ...],
        **_kwargs: object,
    ) -> tuple[tuple[int, ...], tuple[RigidTransform, ...], tuple[str, ...]]:
        reliability_counts.append(len(candidates))
        if len(candidates) == 3:
            return (0, 1), (), ("candidate-002:semantic_identifiability_rejected",)
        first = np.eye(4, dtype=np.float64)
        second = np.eye(4, dtype=np.float64)
        second[0, 3] = 30.0
        return (
            (0, 1),
            (
                RigidTransform.from_matrix(first),
                RigidTransform.from_matrix(second),
            ),
            (),
        )

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_fit_reliable_hypothesis_indices",
        fit_reliability,
    )
    plane = _GroundPlane(
        normal=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        offset=0.0,
        origin=np.zeros(3, dtype=np.float64),
        basis_u=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        basis_v=np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
        support_uv_bounds=(-50.0, 50.0, -20.0, 20.0),
    )

    retained, retained_scale, retained_deviation, retained_search, _retained_trace = (
        _retain_fit_reliable_hypotheses(
            hypotheses,
            fit_trace=fit_trace,
            common_scale=common_scale,
            maximum_deviation=maximum_deviation,
            proposal_search=search,
            fit_points_uv=points,
            evidence_weights=np.ones(len(points), dtype=np.float64),
            bounds=plane.support_uv_bounds,
            seed=42,
            plane=plane,
            fit_cameras=(),
            projected_by_camera={},
            settings=settings,
            policy=alignment_policy,
        )
    )

    assert reliability_counts == [3, 2]
    assert [item.candidate_id for item in retained] == [
        "candidate-000",
        "candidate-001",
    ]
    assert retained_scale == pytest.approx(1.0)
    assert retained_deviation == pytest.approx(0.0)
    assert retained_search.inferred_candidate_count == 2
    assert (
        retained_search.stopping_reason is ProposalSearchStopReason.NO_RELIABLE_PROPOSAL
    )
    assert len(retained_search.selected_orientation_band_indices) == 2
    assert len(retained_search.selected_center_tile_indices) == 2
    assert retained_search.selected_explained_evidence_fraction == pytest.approx(
        sum(retained_search.selected_candidate_explained_evidence_fractions)
    )
    assert (
        retained_search.selected_residual_evidence_sum
        + retained_search.selected_explained_evidence_sum
        == pytest.approx(retained_search.original_evidence_sum)
    )


def test_fit_reliability_cannot_relabel_residual_stop_to_accept_missing_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alignment_evidence: AlignmentEvidence,
) -> None:
    proposal_search = alignment_evidence.diagnostics.proposal_search
    assert (
        proposal_search.stopping_reason
        is ProposalSearchStopReason.RESIDUAL_EVIDENCE_BELOW_MINIMUM
    )
    hypothesis = replace(
        _hypothesis_for_topology(center=(0.0, 0.0)),
        candidate_id="candidate-000",
    )
    points = np.column_stack(
        (np.linspace(-10.0, 10.0, 100), np.zeros(100, dtype=np.float64))
    )
    weights: NDArray[np.float64] = np.ones(len(points), dtype=np.float64)

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_proposal_explanation_for_hypotheses",
        lambda *_args, **_kwargs: ((0.4,), points[40:], weights[40:]),
    )

    with pytest.raises(ValueError, match="Residual-evidence stop requires"):
        _proposal_search_after_fit_reliability(
            proposal_search,
            (hypothesis,),
            source_indices=(0,),
            fit_points_uv=points,
            evidence_weights=weights,
            seed=42,
            settings=_settings(tmp_path).candidate_fit,
        )


def test_common_scale_replacement_refits_pose_and_recomputes_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(
        _settings(tmp_path).candidate_fit,
        maximum_candidate_count=2,
        minimum_explained_evidence_fraction=0.2,
        evidence_assignment_distance_metres=1.0e-4,
        common_scale_relative_tolerance=0.1,
        orientation_minimum_radians=-0.3,
        orientation_maximum_radians=0.3,
    )
    calls: list[float | None] = []
    answers = iter(
        (
            (np.asarray((0.0, 0.0, 0.0, 0.070)), 0.81),
            (np.asarray((2.0, 0.0, 0.0, 0.072)), 0.79),
            (np.asarray((0.0, 0.0, 0.0, 0.070)), 0.81),
            (np.asarray((0.01, 0.0, 0.0)), 0.90),
            (np.asarray((0.0, 0.0, 0.0, 0.070)), 0.81),
            (np.asarray((2.0, 0.0, 0.0, 0.072)), 0.79),
            (np.asarray((0.02, 0.01, 0.1)), 0.91),
            (np.asarray((2.02, 0.01, 0.1)), 0.89),
        )
    )

    def fake_optimize(
        *_args: object, **kwargs: object
    ) -> tuple[NDArray[np.float64], float]:
        fixed_scale = kwargs.get("fixed_scale")
        calls.append(None if fixed_scale is None else cast(float, fixed_scale))
        return next(answers)

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source._optimize_court",
        fake_optimize,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_maximum_center_tile_width_scene_units",
        lambda _settings: 100.0,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source._suppress_assigned_points",
        lambda points, **_kwargs: points[round(len(points) * 0.6) :],
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_suppress_assigned_evidence",
        lambda points, evidence_weights, **_kwargs: (
            points[round(len(points) * 0.6) :],
            evidence_weights[round(len(points) * 0.6) :],
        ),
    )
    points = np.column_stack(
        (np.linspace(-10.0, 10.0, 200), np.linspace(-8.0, 8.0, 200))
    )

    hypotheses, common_scale, _deviation, _search, _trace = _fit_court_hypotheses(
        points,
        evidence_weights=np.ones(len(points), dtype=np.float64),
        bounds=(-12.0, 12.0, -10.0, 10.0),
        seed=42,
        settings=settings,
    )

    assert calls == [
        None,
        None,
        None,
        pytest.approx(0.070),
        None,
        None,
        pytest.approx(0.071),
        pytest.approx(0.071),
    ]
    assert common_scale == pytest.approx(0.071)
    assert [item.center_uv for item in hypotheses] == [(0.02, 0.01), (2.02, 0.01)]
    assert [item.template_score for item in hypotheses] == [0.91, 0.89]
    assert all(
        item.nht_scene_units_per_metre == pytest.approx(common_scale)
        for item in hypotheses
    )


def test_common_scale_refit_bounds_preserve_parallel_court_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(
        _settings(tmp_path).candidate_fit,
        maximum_candidate_count=2,
        minimum_explained_evidence_fraction=0.2,
        evidence_assignment_distance_metres=1.0e-4,
        common_scale_relative_tolerance=0.07,
        orientation_minimum_radians=-0.3,
        orientation_maximum_radians=0.3,
    )
    native_answers = iter(
        (
            (np.asarray((0.0, 0.0, 0.0, 0.070)), 0.9),
            (np.asarray((1.0, 0.0, 0.0, 0.071)), 0.89),
            (np.asarray((0.0, 0.0, 0.0, 0.070)), 0.9),
            (np.asarray((0.0, 0.0, 0.0, 0.070)), 0.9),
            (np.asarray((1.0, 0.0, 0.0, 0.071)), 0.89),
        )
    )
    refit_bounds: list[list[tuple[float, float]]] = []

    def fake_optimize(
        *_args: object,
        **kwargs: object,
    ) -> tuple[NDArray[np.float64], float]:
        if kwargs.get("fixed_scale") is None:
            return next(native_answers)
        bounds = cast(list[tuple[float, float]], kwargs["bounds"])
        refit_bounds.append(bounds)
        candidate_index = len(refit_bounds) - 1
        x = bounds[0][1] if candidate_index == 0 else bounds[0][0]
        return np.asarray((x, 0.0, 0.0)), 0.95

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source._optimize_court",
        fake_optimize,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_maximum_center_tile_width_scene_units",
        lambda _settings: 100.0,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source._suppress_assigned_points",
        lambda points, **_kwargs: points[round(len(points) * 0.6) :],
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_suppress_assigned_evidence",
        lambda points, evidence_weights, **_kwargs: (
            points[round(len(points) * 0.6) :],
            evidence_weights[round(len(points) * 0.6) :],
        ),
    )
    points = np.column_stack(
        (np.linspace(-10.0, 10.0, 200), np.linspace(-8.0, 8.0, 200))
    )

    hypotheses, common_scale, _deviation, _search, _trace = _fit_court_hypotheses(
        points,
        evidence_weights=np.ones(len(points), dtype=np.float64),
        bounds=(-12.0, 12.0, -10.0, 10.0),
        seed=42,
        settings=settings,
    )

    maximum_scene_displacement = (
        common_scale * settings.maximum_center_refit_displacement_metres()
    )
    assert len(refit_bounds) == 3
    assert refit_bounds[1][0] == pytest.approx(
        (-maximum_scene_displacement, maximum_scene_displacement)
    )
    assert refit_bounds[2][0] == pytest.approx(
        (1.0 - maximum_scene_displacement, 1.0 + maximum_scene_displacement)
    )
    assert hypotheses[0].center_uv[0] < hypotheses[1].center_uv[0]
    assert [item.candidate_id for item in hypotheses] == [
        "candidate-000",
        "candidate-001",
    ]
    assert all(
        item.common_scale_refit_center_displacement_metres
        <= item.maximum_common_scale_refit_center_displacement_metres
        for item in hypotheses
    )


def test_two_side_by_side_courts_are_deterministic_under_clutter(
    tmp_path: Path,
) -> None:
    settings = replace(
        _settings(tmp_path).candidate_fit,
        maximum_candidate_count=4,
        samples_per_metre=2.0,
        minimum_nht_scene_units_per_metre=0.8,
        maximum_nht_scene_units_per_metre=1.2,
        orientation_minimum_radians=-0.3,
        orientation_maximum_radians=0.3,
        score_distance_metres=0.2,
        minimum_template_score=0.35,
        family_orientation_tolerance_radians=0.15,
        family_scale_relative_tolerance=0.15,
        optimizer_maximum_iterations=70,
        optimizer_population_size=8,
        maximum_fit_points=10_000,
        common_scale_relative_tolerance=0.07,
        scale_bound_margin_relative=0.005,
        evidence_assignment_distance_metres=0.3,
    )
    template = sample_court_line_template(settings.samples_per_metre)
    rng = np.random.default_rng(19)
    first = transform_template_2d(template, np.asarray((-8.0, 0.0, 0.04, 1.0)))
    second = transform_template_2d(template, np.asarray((8.0, 0.0, 0.04, 1.0)))
    observed = np.concatenate(
        (
            first + rng.normal(scale=0.01, size=first.shape),
            second + rng.normal(scale=0.01, size=second.shape),
            rng.uniform((-20.0, -15.0), (20.0, 15.0), size=(100, 2)),
        )
    )

    first_run = _fit_court_hypotheses(
        observed,
        evidence_weights=np.ones(len(observed), dtype=np.float64),
        bounds=(-20.0, 20.0, -15.0, 15.0),
        seed=42,
        settings=settings,
    )
    second_run = _fit_court_hypotheses(
        observed,
        evidence_weights=np.ones(len(observed), dtype=np.float64),
        bounds=(-20.0, 20.0, -15.0, 15.0),
        seed=42,
        settings=settings,
    )

    hypotheses, common_scale, maximum_deviation, proposal_search, trace = first_run
    assert first_run == second_run
    assert trace.common_scale_refitted == hypotheses
    assert common_scale == pytest.approx(1.0, abs=0.03)
    assert maximum_deviation < settings.common_scale_relative_tolerance
    assert proposal_search.feasible_complete_state_count >= 1
    centers = np.asarray([item.center_uv for item in hypotheses])
    assert np.linalg.norm(centers[0] - centers[1]) > 15.0
    assert sorted(centers[:, 0]) == pytest.approx([-8.0, 8.0], abs=0.15)
    plane = _GroundPlane(
        normal=np.asarray((0.0, 0.0, 1.0)),
        offset=0.0,
        origin=np.zeros(3, dtype=np.float64),
        basis_u=np.asarray((1.0, 0.0, 0.0)),
        basis_v=np.asarray((0.0, 1.0, 0.0)),
        support_uv_bounds=(-20.0, 20.0, -15.0, 15.0),
    )
    observed_3d = np.column_stack((observed, np.zeros(len(observed))))
    projected = {
        "camera-0": _ProjectedLineEvidence(
            points_nht_scene=observed_3d,
            points_uv=observed,
            probabilities=np.ones(len(observed), dtype=np.float32),
            proximity_weights=np.ones(len(observed), dtype=np.float64),
            selected_line_pixel_count=len(observed),
        )
    }
    first_assignment = _assign_candidate_evidence(
        hypotheses,
        plane=plane,
        projected_by_camera=projected,
        settings=settings,
    )
    second_assignment = _assign_candidate_evidence(
        hypotheses,
        plane=plane,
        projected_by_camera=projected,
        settings=settings,
    )
    for candidate_id in first_assignment:
        np.testing.assert_array_equal(
            first_assignment[candidate_id]["camera-0"],
            second_assignment[candidate_id]["camera-0"],
        )
    transforms: list[tuple[str, RigidTransform]] = []
    for hypothesis in hypotheses:
        cosine = np.cos(hypothesis.orientation_radians)
        sine = np.sin(hypothesis.orientation_radians)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:2, :2] = ((cosine, -sine), (sine, cosine))
        matrix[:2, 3] = np.asarray(hypothesis.center_uv) / common_scale
        transform = RigidTransform.from_matrix(matrix)
        transforms.append((hypothesis.candidate_id, transform))
        metrics = evaluate_whole_template(
            scene_from_court=transform,
            measured_points_scene=observed_3d / common_scale,
            settings=settings.whole_court_evidence(
                required_court_count=2,
                minimum_matches_per_offset_level=3,
            ),
        )
        assert all(
            metrics.threshold_checks(
                settings.whole_court_evidence(
                    required_court_count=2,
                    minimum_matches_per_offset_level=3,
                )
            ).values()
        )
    topology = evaluate_court_topology(transforms)
    assert len(topology) == 1
    assert all(
        topology[0]
        .threshold_checks(
            settings.whole_court_evidence(
                required_court_count=2,
                minimum_matches_per_offset_level=3,
            )
        )
        .values()
    )


def test_adjacent_courts_both_receive_their_shared_sideline(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path).candidate_fit
    scale = 0.07
    separation = 2.0 * HALF_DOUBLES_WIDTH * scale
    first = replace(
        _hypothesis_for_topology(center=(0.0, 0.0)),
        candidate_id="candidate-000",
    )
    second = replace(
        _hypothesis_for_topology(center=(separation, 0.0)),
        candidate_id="candidate-001",
        native_center_uv=(separation, 0.0),
    )
    template = sample_court_line_template(settings.samples_per_metre)
    shared_template = template[
        np.isclose(template[:, 0], HALF_DOUBLES_WIDTH, atol=1.0e-12, rtol=0.0)
    ]
    assert len(shared_template) > 0
    shared_uv = transform_template_2d(
        shared_template,
        np.asarray((0.0, 0.0, 0.0, scale), dtype=np.float64),
    )
    shared_nht = np.column_stack((shared_uv, np.zeros(len(shared_uv))))
    plane = _GroundPlane(
        normal=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        offset=0.0,
        origin=np.zeros(3, dtype=np.float64),
        basis_u=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        basis_v=np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
        support_uv_bounds=(-2.0, 2.0, -2.0, 2.0),
    )
    projected = _ProjectedLineEvidence(
        points_nht_scene=shared_nht,
        points_uv=shared_uv,
        probabilities=np.ones(len(shared_uv), dtype=np.float32),
        proximity_weights=np.ones(len(shared_uv), dtype=np.float64),
        selected_line_pixel_count=len(shared_uv),
    )

    assigned = _assign_candidate_evidence(
        (first, second),
        plane=plane,
        projected_by_camera={"camera-0": projected},
        settings=settings,
    )

    np.testing.assert_array_equal(assigned[first.candidate_id]["camera-0"], shared_nht)
    np.testing.assert_array_equal(assigned[second.candidate_id]["camera-0"], shared_nht)


def _stub_fixed_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    plane = _GroundPlane(
        normal=np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
        offset=0.0,
        origin=np.zeros(3, dtype=np.float64),
        basis_u=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        basis_v=np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
        support_uv_bounds=(-2.0, 2.0, -2.0, 2.0),
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source._estimate_ground_plane",
        lambda *_args, **_kwargs: plane,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source._project_probability_to_ground",
        lambda *_args, **_kwargs: _projected_evidence(20),
    )


def _projected_evidence(
    projected_count: int, *, selected_count: int = 20
) -> _ProjectedLineEvidence:
    points = np.column_stack(
        (
            np.linspace(-1.0, 1.0, projected_count),
            np.linspace(1.0, -1.0, projected_count),
            np.zeros(projected_count),
        )
    )
    return _ProjectedLineEvidence(
        points_nht_scene=np.asarray(points, dtype=np.float64),
        points_uv=np.asarray(points[:, :2], dtype=np.float64),
        probabilities=np.full(projected_count, 0.75, dtype=np.float32),
        proximity_weights=np.full(projected_count, 0.8, dtype=np.float64),
        selected_line_pixel_count=selected_count,
    )


def _line_heatmaps(evidence: AlignmentEvidence) -> AlignmentLineHeatmaps:
    selection = evidence.diagnostics.selection
    projected_counts = {
        item.camera_id: item.projected_line_point_count
        for item in evidence.diagnostics.cameras
    }
    projected_counts.update(
        {
            item.camera_id: item.projected_line_point_count
            for item in selection.excluded_cameras
        }
    )
    measured = {
        item.camera_id: evidence.ground_plane_frame.to_uv(
            evidence.metric_adapter.metric_from_nht_points(item.points_nht_scene)
        )
        for item in evidence.measured_camera_lines
    }
    fit_ids = set(evidence.diagnostics.evaluation.fit_camera_ids)
    return AlignmentLineHeatmaps(
        bounds_uv=evidence.ground_plane_frame.bounds_uv_metres,
        grid_spacing=0.25,
        proximity_scale=0.35,
        proximity_power=2.0,
        views=tuple(
            AlignmentLineHeatmapView(
                camera_id=camera_id,
                probability=np.asarray(
                    ((0.0, 0.25), (0.5, 1.0)),
                    dtype=np.float32,
                ),
                points_uv=measured.get(
                    camera_id,
                    np.column_stack(
                        (
                            np.linspace(-0.9, 0.9, projected_counts[camera_id]),
                            np.linspace(0.9, -0.9, projected_counts[camera_id]),
                        )
                    ).astype(np.float64),
                ),
                projected_probabilities=np.full(
                    projected_counts[camera_id],
                    0.75,
                    dtype=np.float32,
                ),
                proximity_weights=np.full(
                    projected_counts[camera_id],
                    0.8,
                    dtype=np.float64,
                ),
                included_in_aggregate=camera_id in fit_ids,
            )
            for camera_id in selection.camera_prefix_ids
        ),
    )


def _stub_multiple_complete_state_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def optimize(
        *_args: object,
        **kwargs: object,
    ) -> tuple[NDArray[np.float64], float]:
        search_bounds = cast(list[tuple[float, float]], kwargs["bounds"])
        if kwargs.get("fixed_scale") is not None:
            return (
                np.asarray(
                    tuple((lower + upper) / 2.0 for lower, upper in search_bounds),
                    dtype=np.float64,
                ),
                0.9,
            )
        selected = cast(tuple[_CourtHypothesis, ...], kwargs.get("selected", ()))
        center_lower, center_upper = search_bounds[0]
        if selected:
            center = (center_lower + center_upper) / 2.0
            return np.asarray((center, 0.0, 0.0, 1.0)), 0.0
        if center_lower <= -8.0 <= center_upper:
            center, score = -8.0, 0.95
        elif center_lower <= 8.0 <= center_upper:
            center, score = 8.0, 0.9
        elif center_lower <= 0.0 <= center_upper:
            center, score = 0.0, 0.85
        else:
            center, score = (center_lower + center_upper) / 2.0, 0.0
        return np.asarray((center, 0.0, 0.0, 1.0)), score

    def suppress_evidence(
        points: NDArray[np.float64],
        evidence_weights: NDArray[np.float64],
        *,
        parameters: NDArray[np.float64],
        **_kwargs: object,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        center = float(parameters[0])
        removed = 30 if center < -4.0 else 25 if center > 4.0 else 20
        offset = min(removed, len(points))
        return points[offset:], evidence_weights[offset:]

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source._optimize_court",
        optimize,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_tile_is_geometrically_impossible",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_suppress_assigned_evidence",
        suppress_evidence,
    )


def _candidate_count_settings(
    tmp_path: Path,
    *,
    maximum_candidate_count: int,
) -> CourtCandidateFitSettings:
    return replace(
        _settings(tmp_path).candidate_fit,
        maximum_candidate_count=maximum_candidate_count,
        maximum_retained_state_count=4,
        minimum_explained_evidence_fraction=0.1,
        samples_per_metre=2.0,
        minimum_nht_scene_units_per_metre=0.8,
        maximum_nht_scene_units_per_metre=1.2,
        orientation_minimum_radians=-0.2,
        orientation_maximum_radians=0.2,
        score_distance_metres=0.2,
        minimum_template_score=0.4,
        optimizer_maximum_iterations=4,
        optimizer_population_size=4,
        maximum_fit_points=10_000,
        common_scale_relative_tolerance=0.1,
        scale_bound_margin_relative=0.005,
        evidence_assignment_distance_metres=0.35,
    )


def _stub_linear_candidate_search(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reliable_candidate_count: int,
) -> None:
    centers = (24.0, -24.0, 0.0, 48.0)

    def optimize(
        *_args: object,
        **kwargs: object,
    ) -> tuple[NDArray[np.float64], float]:
        fixed_scale = kwargs.get("fixed_scale")
        bounds = cast(list[tuple[float, float]], kwargs["bounds"])
        if fixed_scale is not None:
            return (
                np.asarray(
                    (
                        (bounds[0][0] + bounds[0][1]) / 2.0,
                        (bounds[1][0] + bounds[1][1]) / 2.0,
                        (bounds[2][0] + bounds[2][1]) / 2.0,
                    ),
                    dtype=np.float64,
                ),
                0.9,
            )
        selected = cast(tuple[_CourtHypothesis, ...], kwargs.get("selected", ()))
        index = len(selected)
        score = 0.9 if index < reliable_candidate_count else 0.0
        return np.asarray((centers[index], 0.0, 0.0, 1.0)), score

    def suppress_points(
        points: NDArray[np.float64],
        **_kwargs: object,
    ) -> NDArray[np.float64]:
        return points[max(1, len(points) // 3) :]

    def suppress_evidence(
        points: NDArray[np.float64],
        evidence_weights: NDArray[np.float64],
        **_kwargs: object,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        cut = max(1, len(points) // 3)
        return points[cut:], evidence_weights[cut:]

    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source._optimize_court",
        optimize,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_maximum_center_tile_width_scene_units",
        lambda _settings: 200.0,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_tile_is_geometrically_impossible",
        lambda **_kwargs: False,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_suppress_assigned_points",
        suppress_points,
    )
    monkeypatch.setattr(
        "src.synthetic_data_generation.alignment.evidence_source."
        "_suppress_assigned_evidence",
        suppress_evidence,
    )


def _hypothesis_for_topology(
    *,
    center: tuple[float, float],
) -> _CourtHypothesis:
    return _CourtHypothesis(
        candidate_id="proposal",
        center_uv=center,
        orientation_radians=0.0,
        nht_scene_units_per_metre=0.07,
        template_score=0.9,
        native_nht_scene_units_per_metre=0.07,
        native_template_score=0.9,
        native_center_uv=center,
        native_orientation_radians=0.0,
        common_scale_refit_center_displacement_metres=0.0,
        maximum_common_scale_refit_center_displacement_metres=1.0,
        proposal_orientation_band_radians=(-0.5, 0.5),
        proposal_residual_point_count_before_suppression=100,
        proposal_residual_point_count_after_suppression=50,
    )


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
        camera_prefix_count=3,
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
            proximity_scale=0.35,
            proximity_power=2.0,
            grid_spacing=0.25,
            minimum_projected_points_per_camera=3,
        ),
        candidate_fit=CourtCandidateFitSettings(
            maximum_candidate_count=4,
            maximum_retained_state_count=8,
            minimum_explained_evidence_fraction=0.05,
            samples_per_metre=2.0,
            minimum_nht_scene_units_per_metre=0.05,
            maximum_nht_scene_units_per_metre=0.1,
            orientation_minimum_radians=-0.5,
            orientation_maximum_radians=0.5,
            score_distance_metres=0.25,
            minimum_template_score=0.1,
            family_orientation_tolerance_radians=0.1,
            family_scale_relative_tolerance=0.1,
            minimum_center_separation_metres=10.97,
            optimizer_maximum_iterations=2,
            optimizer_population_size=2,
            optimizer_tolerance=1.0e-4,
            maximum_fit_points=100,
            common_scale_relative_tolerance=0.1,
            scale_bound_margin_relative=0.01,
            evidence_assignment_distance_metres=0.35,
            whole_template_inlier_distance_metres=0.3,
            minimum_whole_template_inlier_fraction=0.6,
            maximum_whole_template_q95_error_metres=1.5,
            minimum_semantic_segment_inlier_fraction=0.5,
            maximum_court_footprint_overlap_fraction=1.0e-9,
        ),
        correspondences=CorrespondenceSettings(
            maximum_match_distance_metres=0.25,
            maximum_correspondences_per_camera=20,
            minimum_correspondences_per_camera=3,
        ),
    )
