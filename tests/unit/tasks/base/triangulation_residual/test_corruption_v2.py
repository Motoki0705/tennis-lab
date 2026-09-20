"""Counterfactual noise streams and observation-only camera-fit boundaries."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from omegaconf import OmegaConf

from src.tasks.base.triangulation_residual.cameras import (
    fit_court_rig,
    fixed_six_camera_rig,
)
from src.tasks.base.triangulation_residual.configuration import (
    CorruptionConfig,
    FeatureConfig,
    V2Config,
)
from src.tasks.base.triangulation_residual.corruption_v2 import (
    ErrorFamily,
    corrupt_candidates,
)
from src.tasks.base.triangulation_residual.geometry import prepare_geometry
from src.utils.geometry.triangulation import project_multiview
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d


def configs(mode="clean"):
    directory = (
        Path(__file__).resolve().parents[5]
        / "src/tasks/base/triangulation_residual/configs"
    )
    noise = CorruptionConfig(
        **OmegaConf.to_container(
            OmegaConf.load(directory / "geometric_residual.yaml").corruption
        )
    )
    noise = replace(
        noise,
        camera_rotation_std_deg=0,
        camera_center_std_m=0,
        camera_focal_log_std=0,
        camera_principal_std_px=0,
        focal_scale_min=1.0,
        focal_scale_max=1.0,
    )
    v2 = V2Config(
        **OmegaConf.to_container(
            OmegaConf.load(directory / "geometric_residual_v2.yaml").v2
        )
    )
    return noise, replace(
        v2,
        error_mode=mode,
        true_camera_position_jitter_m=0,
        true_camera_height_jitter_m=0,
    )


def world(task="plcs", frames=60):
    joints = 17 if task == "plcs" else 1
    values = np.random.default_rng(1).uniform(
        [-0.3, -5.2, 0.5], [0.3, -4.8, 1.9], (frames, joints, 3)
    )
    values[:] = values[0]
    values[..., 0] += np.arange(frames)[:, None] * 0.01
    if joints == 17:
        values[:, (5, 6), 2] = 1.5
        values[:, (11, 12), 2] = 0.9
    return values


def draw(mode="clean", *, task="plcs", seed=9, noise=None, v2=None, target=None):
    default_noise, default_v2 = configs(mode)
    return corrupt_candidates(
        world(task) if target is None else target,
        fixed_six_camera_rig((1920, 1080)),
        np.random.default_rng(seed),
        default_noise if noise is None else noise,
        default_v2 if v2 is None else v2,
        fps=30,
        task=task,
    )


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_persistent_only_confidence_is_not_an_exact_error_label(task):
    _, cfg = configs("persistent")
    generated = draw(
        task=task, v2=replace(cfg, persistent_high_confidence_probability=1.0)
    )
    ordinary = generated.scores[(generated.scores > 0) & ~generated.persistent_mask]
    wrong = generated.scores[(generated.scores > 0) & generated.persistent_mask]
    assert len(wrong) > 0
    assert ordinary.std() > 0.05
    assert ordinary.min() < wrong.max()
    assert wrong.min() < ordinary.max()


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_clean_draw_fits_court_and_recovers_exact_projection_and_world(task):
    target = world(task)
    generated = draw(task=task, target=target)
    assert generated.family == ErrorFamily.CLEAN
    assert generated.severity == 0
    assert_array_equal(generated.valid_indices, np.arange(6))
    projected, _ = project_multiview(target, generated.true_rig.matrices)
    expected = projected.transpose(2, 0, 1, 3)
    assert_allclose(
        generated.observations_px[generated.scores > 0],
        expected[generated.scores > 0],
        atol=1e-10,
    )
    assert not generated.persistent_mask.any()
    assert not generated.dropout_mask.any()
    assert not generated.radial_coefficients.any()
    selected = generated.subset(np.array([5, 0, 2]))
    assert_array_equal(selected.observations_px, generated.observations_px[[5, 0, 2]])
    geometry = prepare_geometry(
        selected.observations_px,
        selected.scores,
        selected.court_px,
        selected.court_scores,
        selected.estimated_rig,
        root_indices=(11, 12) if task == "plcs" else (0,),
        fps=30,
        feature_config=FeatureConfig("raw", 1.0),
    )
    assert_allclose(geometry.init_world_m, target, atol=1e-5)
    assert not np.shares_memory(selected.true_rig.K, selected.estimated_rig.K)


def test_all_six_cameras_are_fitted_once_from_exactly_the_returned_court():
    with patch(
        "src.tasks.base.triangulation_residual.corruption_v2.fit_court_rig",
        wraps=fit_court_rig,
    ) as fitter:
        generated = draw("calibration")
        fitter.assert_called_once()
        args, kwargs = fitter.call_args
    assert len(args) == 3
    assert args[0] is generated.court_px
    assert args[1] is generated.court_scores
    assert_array_equal(args[2], generated.true_rig.image_size)
    assert kwargs == {"min_points": 6}
    fitted = fit_court_rig(
        generated.court_px, generated.court_scores, generated.true_rig.image_size
    )
    indices = generated.valid_indices
    assert len(indices) >= 2
    assert_allclose(
        generated.subset(indices).estimated_rig.matrices,
        fitted.subset(indices).matrices,
    )
    assert not np.allclose(
        generated.subset(indices).estimated_rig.matrices,
        generated.true_rig.matrices[indices],
    )
    # Object GT cannot influence a fit: change every body point with the same
    # RNG and court configuration and the fitted cameras remain identical.
    another = draw("calibration", target=world() + [1.0, 1.0, 0.2])
    assert_array_equal(generated.court_px, another.court_px)
    assert_allclose(
        generated.subset(indices).estimated_rig.matrices,
        another.subset(indices).estimated_rig.matrices,
    )


def test_disabling_object_noise_keeps_cameras_court_events_and_temporal_draws():
    noise, v2 = configs("combined")
    v2 = replace(v2, persistent_rate_per_view_second=4)
    baseline = draw(noise=noise, v2=v2)
    quiet = draw(
        noise=replace(
            noise,
            observation_sigma_px=0,
            temporal_sigma_px=0,
            view_bias_px=0,
            outlier_probability=0,
            confidence_noise=0,
        ),
        v2=v2,
    )
    assert_array_equal(baseline.true_rig.matrices, quiet.true_rig.matrices)
    assert_array_equal(baseline.clean_uv, quiet.clean_uv)
    assert_array_equal(baseline.court_px, quiet.court_px)
    assert_array_equal(baseline.court_scores, quiet.court_scores)
    assert baseline.persistent_events == quiet.persistent_events
    assert_array_equal(baseline.persistent_kind, quiet.persistent_kind)
    assert_array_equal(baseline.dropout_mask, quiet.dropout_mask)
    assert_array_equal(baseline.time_shifts, quiet.time_shifts)
    assert baseline.persistent_mask.any()
    assert_array_equal(
        baseline.observations_px[baseline.persistent_mask],
        quiet.observations_px[quiet.persistent_mask],
    )
    common = (baseline.scores > 0) & (quiet.scores > 0) & ~baseline.persistent_mask
    assert not np.allclose(
        baseline.observations_px[common], quiet.observations_px[common]
    )


def test_disabling_court_noise_does_not_change_object_or_radial_draws():
    noise, v2 = configs("combined")
    baseline = draw(noise=noise, v2=v2)
    quiet = draw(
        noise=noise,
        v2=replace(
            v2,
            court_noise_px=0,
            court_bias_px=0,
            court_outlier_probability=0,
            court_dropout_probability=0,
        ),
    )
    assert_array_equal(baseline.observations_px, quiet.observations_px)
    assert_array_equal(baseline.scores, quiet.scores)
    assert_array_equal(baseline.radial_coefficients, quiet.radial_coefficients)
    assert_array_equal(baseline.true_rig.K, quiet.true_rig.K)
    common = (baseline.court_scores > 0) & (quiet.court_scores > 0)
    assert not np.allclose(baseline.court_px[common], quiet.court_px[common])


def test_radial_error_is_shared_by_object_and_court_before_calibration():
    noise, v2 = configs("calibration")
    v2 = replace(
        v2,
        court_noise_px=0,
        court_bias_px=0,
        court_outlier_probability=0,
        court_dropout_probability=0,
    )
    generated = draw(noise=replace(noise, radial_std=0.08), v2=v2)
    court = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()[:14]
    clean_object, _ = project_multiview(world(), generated.true_rig.matrices)
    clean_court, _ = project_multiview(court, generated.true_rig.matrices)
    for points, actual, score in (
        (
            clean_object.transpose(2, 0, 1, 3),
            generated.observations_px,
            generated.scores,
        ),
        (clean_court.transpose(1, 0, 2), generated.court_px, generated.court_scores),
    ):
        for view in range(6):
            center = generated.true_rig.K[view, :2, 2]
            focal = generated.true_rig.K[view, [0, 1], [0, 1]]
            normalized = (points[view] - center) / focal
            expected = center + focal * normalized * (
                1
                + generated.radial_coefficients[view]
                * np.sum(normalized**2, axis=-1, keepdims=True)
            )
            valid = score[view] > 0
            assert_allclose(actual[view][valid], expected[valid], atol=1e-10)


def test_missing_court_is_nan_and_calibration_failures_are_retained():
    noise, v2 = configs("calibration")
    generated = draw(noise=noise, v2=replace(v2, court_dropout_probability=1.0))
    assert np.isnan(generated.court_px).all()
    assert not generated.court_scores.any()
    assert len(generated.valid_indices) == 0
    assert all(
        error is not None and error.reason == "insufficient_points"
        for error in generated.court_fit.failures
    )
    with pytest.raises(ValueError, match="no successful court fit"):
        generated.subset(np.array([0, 1]))


@pytest.mark.parametrize(
    "mode,family",
    [
        ("calibration", 1),
        ("observation", 2),
        ("temporal", 3),
        ("persistent", 4),
        ("combined", 5),
    ],
)
def test_named_diagnostic_modes_have_fixed_severity_and_isolated_court_error(
    mode, family
):
    generated = draw(mode)
    assert generated.family == family
    assert generated.severity == 1
    if mode not in ("calibration", "combined"):
        clean = draw("clean")
        assert_array_equal(generated.court_px, clean.court_px)
        assert not generated.radial_coefficients.any()
    if mode == "persistent":
        assert len(generated.persistent_events) >= 1
        assert generated.persistent_mask.any()
        assert not generated.dropout_mask.any()
        assert not generated.time_shifts.any()


def test_mixture_clean_and_hard_weights_leave_true_camera_draw_unchanged():
    noise, v2 = configs("mixed")
    v2 = replace(
        v2, true_camera_position_jitter_m=0.75, true_camera_height_jitter_m=0.5
    )
    clean = draw(noise=replace(noise, clean_probability=1, hard_probability=0), v2=v2)
    hard = draw(noise=replace(noise, clean_probability=0, hard_probability=1), v2=v2)
    assert clean.family == 0 and clean.severity == 0
    assert hard.family == 5 and 2 <= hard.severity <= 4
    assert_array_equal(clean.true_rig.matrices, hard.true_rig.matrices)
    delta = clean.true_rig.centers - fixed_six_camera_rig((1920, 1080)).centers
    assert (np.abs(delta[:, :2]) <= 0.75).all()
    assert (np.abs(delta[:, 2]) <= 0.5).all()
    direction = -clean.true_rig.centers / np.linalg.norm(
        clean.true_rig.centers, axis=-1, keepdims=True
    )
    assert_allclose(clean.true_rig.R[:, 2], direction, atol=1e-12)


def test_time_shift_carries_source_visibility_instead_of_using_target_gt_visibility():
    noise, v2 = configs("temporal")
    noise = replace(
        noise,
        dropout_probability=0,
        burst_probability=0,
        time_shift_probability=1,
        time_shift_max_frames=5,
    )
    target = world("blcs")
    target[:, 0, 0] = np.linspace(-20, 20, len(target))
    generated = draw(task="blcs", noise=noise, v2=v2, target=target)
    projected, _ = project_multiview(target, generated.true_rig.matrices)
    projected = projected.transpose(2, 0, 1, 3)
    for view, shift in enumerate(generated.time_shifts):
        index = np.arange(len(target)) + shift
        inside = (index >= 0) & (index < len(target))
        visible = generated.scores[view, :, 0] > 0
        assert not visible[~inside].any()
        assert_allclose(
            generated.observations_px[view, visible], projected[view, index[visible]]
        )
    assert ((generated.scores > 0) & ~generated.clean_visible).any()


def test_combined_draw_is_seed_reproducible():
    first = draw("combined")
    second = draw("combined")
    for name in (
        "observations_px",
        "scores",
        "court_px",
        "court_scores",
        "persistent_mask",
        "persistent_kind",
        "persistent_source_missing",
        "dropout_mask",
        "time_shifts",
    ):
        assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.persistent_events == second.persistent_events
    assert_allclose(
        first.subset(first.valid_indices).estimated_rig.matrices,
        second.subset(second.valid_indices).estimated_rig.matrices,
    )


def test_independent_camera_perturbation_is_explicitly_rejected():
    noise, v2 = configs()
    with pytest.raises(ValueError, match="Court14"):
        draw(noise=replace(noise, camera_rotation_std_deg=0.5), v2=v2)


def test_retry_holds_error_family_and_severity_without_changing_other_rng_streams():
    noise, v2 = configs("mixed")
    force_clean = replace(noise, clean_probability=1, hard_probability=0)
    arguments = (world(), fixed_six_camera_rig((1920, 1080)))
    first = corrupt_candidates(
        *arguments,
        np.random.default_rng(6),
        replace(noise, clean_probability=0, hard_probability=1),
        v2,
        fps=30,
        task="plcs",
    )
    fixed = corrupt_candidates(
        *arguments,
        np.random.default_rng(10),
        force_clean,
        v2,
        fps=30,
        task="plcs",
        fixed_error=(first.family, first.severity),
    )
    otherwise_clean = corrupt_candidates(
        *arguments, np.random.default_rng(10), force_clean, v2, fps=30, task="plcs"
    )
    assert fixed.family == first.family == ErrorFamily.COMBINED
    assert fixed.severity == first.severity > 1
    assert otherwise_clean.family == ErrorFamily.CLEAN
    assert_array_equal(fixed.true_rig.matrices, otherwise_clean.true_rig.matrices)
    assert_array_equal(fixed.clean_uv, otherwise_clean.clean_uv)
    unchanged = corrupt_candidates(
        *arguments,
        np.random.default_rng(6),
        replace(noise, clean_probability=0, hard_probability=1),
        v2,
        fps=30,
        task="plcs",
        fixed_error=(first.family, first.severity),
    )
    assert_array_equal(first.observations_px, unchanged.observations_px)
    assert_array_equal(first.court_px, unchanged.court_px)


@pytest.mark.parametrize(
    "invalid",
    [
        (6, 1.0),
        (0, 1.0),
        (1, 0.0),
        (1, 0.5),
        (1, -1.0),
        (1, np.nan),
        (True, 1.0),
        (1.0, 1.0),
    ],
)
def test_invalid_fixed_error_is_rejected(invalid):
    noise, v2 = configs()
    with pytest.raises(ValueError, match="fixed_error"):
        corrupt_candidates(
            world(),
            fixed_six_camera_rig((1920, 1080)),
            np.random.default_rng(2),
            noise,
            v2,
            fps=30,
            task="plcs",
            fixed_error=invalid,
        )
