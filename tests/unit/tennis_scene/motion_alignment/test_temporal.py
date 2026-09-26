"""Independent geometry, sparse derivatives, temporal noise and missing-data checks."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation

from src.tennis_scene.motion_alignment.temporal import (
    PlacementRejection,
    PlacementUnavailable,
    TemporalPlacementConfig,
    _Objective,
    estimate_body_scale,
    fit_supported_track,
    fit_temporal_placement,
)


def source_points(count: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    pose = rng.normal(size=(17, 3)) * [0.3, 0.2, 0.5]
    pose -= pose[[11, 12]].mean(0)
    result: np.ndarray = np.asarray(np.repeat(pose[None], count, 0), dtype=np.float64)
    return result


def placed(
    source: np.ndarray, yaw: np.ndarray, root: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    result: np.ndarray = np.asarray(
        scale
        * np.einsum(
            "tij,tkj->tki", Rotation.from_euler("z", yaw[:, None]).as_matrix(), source
        )
        + root[:, None],
        dtype=np.float64,
    )
    return result


def test_known_temporal_motion_and_scale_do_not_fit_travel_distance() -> None:
    source = source_points(31)
    yaw = np.linspace(2.9, 3.4, len(source))
    root = np.column_stack(
        (np.linspace(0, 25, len(source)), np.zeros(len(source)), np.ones(len(source)))
    )
    target = placed(source, yaw, root, 1.12)
    weight = np.ones(source.shape[:2])
    weight[::3, :5] = 0
    target[weight == 0] = np.nan
    cfg = TemporalPlacementConfig()
    scale, count = estimate_body_scale(source, target, weight, cfg)
    assert scale == pytest.approx(1.12) and count > 30
    fit = fit_temporal_placement(
        source, target, weight, scale=scale, fps=30.0, config=cfg
    )
    np.testing.assert_allclose(fit.joints, placed(source, yaw, root, 1.12), atol=1e-7)


def test_sparse_jacobian_and_loss_match_independent_scalar_definition() -> None:
    rng = np.random.default_rng(71)
    source = source_points(4)
    target = source + rng.normal(size=source.shape) * 0.15
    weights = rng.uniform(0.2, 1.0, size=source.shape[:2])
    cfg = TemporalPlacementConfig()
    objective = _Objective(source, target, weights, 24.0, cfg)
    params = rng.normal(size=(4, 4)) * 0.1
    numeric = approx_derivative(objective.residual, params.ravel(), method="3-point")
    np.testing.assert_allclose(
        objective.jacobian(params.ravel()).toarray(), numeric, atol=1e-7, rtol=1e-6
    )
    points = placed(source, params[:, 0], params[:, 1:])
    distance = ((points - target) / cfg.data_sigma_m) ** 2
    expected = ((np.sqrt(1 + distance.sum(-1)) - 1) * weights).sum() / weights.sum()
    expected += (
        cfg.temporal_weight
        * (
            (np.diff(params[:, 1:], n=2, axis=0) * 24**2 / cfg.root_acceleration_sigma)
            ** 2
        ).mean()
    )
    expected += (
        cfg.temporal_weight
        * (
            (np.diff(params[:, 0], n=2) * 24**2 / cfg.yaw_acceleration_sigma) ** 2
        ).mean()
    )
    assert 0.5 * np.sum(objective.residual(params.ravel()) ** 2) == pytest.approx(
        expected
    )


def test_temporal_fit_reduces_translation_noise_without_changing_articulation() -> None:
    source = source_points(45)
    rng = np.random.default_rng(3)
    root = np.column_stack(
        (np.linspace(0, 2, len(source)), np.zeros(len(source)), np.ones(len(source)))
    )
    target = source + (root + rng.normal(0, 0.035, root.shape))[:, None]
    fit = fit_temporal_placement(
        source,
        target,
        np.ones(source.shape[:2]),
        scale=1.0,
        fps=30.0,
        config=TemporalPlacementConfig(),
    )
    assert (
        np.linalg.norm(fit.hip_position - root, axis=-1).mean()
        < np.linalg.norm(target[:, 11:13].mean(1) - root, axis=-1).mean() * 0.7
    )
    # Rigid per-frame placement preserves all pairwise distances.
    np.testing.assert_allclose(
        np.linalg.norm(fit.joints[:, :, None] - fit.joints[:, None], axis=-1),
        np.linalg.norm(source[:, :, None] - source[:, None], axis=-1),
        atol=1e-8,
    )


def test_no_temporal_bridge_over_missing_data_or_body_segment_boundary() -> None:
    source = source_points(12)
    root = np.zeros((12, 3))
    root[:, 2] = 1
    root[4:8, 0] = 10
    root[8:, 0] = -10
    target = placed(source, np.zeros(12), root, 1.1)
    weights = np.ones((12, 17))
    weights[3] = 0
    ids = np.r_[np.zeros(8, np.int64), np.ones(4, np.int64)]
    result = fit_supported_track(
        source, target, weights, ids, fps=60.0, config=TemporalPlacementConfig()
    )
    assert (
        not result.valid[3]
        and result.reasons[3] == PlacementRejection.INSUFFICIENT_JOINTS
    )
    np.testing.assert_array_equal(result.hip_position[3], 0)
    np.testing.assert_allclose(
        result.hip_position[result.valid], root[result.valid], atol=1e-8
    )
    assert result.scale == pytest.approx(1.1) and len(result.intervals) == 3


def test_solver_failure_is_recorded_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = source_points(5)
    monkeypatch.setattr(
        "src.tennis_scene.motion_alignment.temporal.least_squares",
        lambda *a, **k: SimpleNamespace(
            success=False, message="injected solver failure"
        ),
    )
    result = fit_supported_track(
        source,
        source,
        np.ones((5, 17)),
        np.zeros(5, np.int64),
        fps=30.0,
        config=TemporalPlacementConfig(),
    )
    assert not result.valid.any()
    assert (result.reasons == PlacementRejection.SOLVER_FAILED).all()
    assert "injected" in result.intervals[0]["message"]


def test_scale_failure_and_zero_joint_support_are_explicit() -> None:
    source = source_points(5)
    cfg = TemporalPlacementConfig()
    with pytest.raises(PlacementUnavailable, match="outside configured bounds"):
        estimate_body_scale(source, 3 * source, np.ones((5, 17)), cfg)
    result = fit_supported_track(
        source, source, np.zeros((5, 17)), np.zeros(5, np.int64), fps=30.0, config=cfg
    )
    assert result.scale is None and not result.valid.any()
    with pytest.raises(ValueError, match="positive"):
        replace(cfg, temporal_weight=float("nan"))


def test_single_supported_frame_has_no_acceleration_term() -> None:
    source = source_points(1)
    expected = source + [1, 2, 3]
    result = fit_temporal_placement(
        source,
        expected,
        np.ones((1, 17)),
        scale=1.0,
        fps=30.0,
        config=TemporalPlacementConfig(),
    )
    np.testing.assert_allclose(result.joints, expected, atol=1e-7)
