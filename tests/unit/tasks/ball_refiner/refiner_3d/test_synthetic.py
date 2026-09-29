from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
import yaml

from src.tasks.ball_refiner.refiner_3d.synthetic.observations import (
    make_distribution,
    perturb_cameras,
)
from src.tasks.ball_refiner.refiner_3d.synthetic.timebase import (
    Event,
    event_masks,
    resample,
    retained_events,
)
from src.tasks.ball_refiner.refiner_3d.triangulation import frame_observations
from src.tasks.blcs.generate_dataset.simulation.rally_simulator import RallyResult
from src.utils.geometry.probabilistic_triangulation import (
    GaussianPrior3D,
    LaplaceConfig,
    triangulate_gmm,
)
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.paths import PROJECT_ROOT


def test_rational_resampling_has_no_stride_drift_or_extrapolation():
    native_times = np.arange(2401) / 240
    positions = np.column_stack([native_times, native_times * 2, 3 - native_times])
    times, sampled = resample(positions, native_hz=240, numerator=60000, denominator=1001, max_frames=10000)
    assert len(times) == 600
    np.testing.assert_array_equal(times, np.arange(600) * 1001 / 60000)
    np.testing.assert_allclose(sampled, np.column_stack([times, 2 * times, 3 - times]), atol=1e-6)
    assert times[-1] <= native_times[-1]
    assert times[-1] != pytest.approx(599 / 60, abs=1e-4)


def test_events_filter_discarded_future_bounces_and_preserve_native_seconds():
    shots = [
        SimpleNamespace(t_start=0, t_net=-1, t_bounce1=120, t_bounce2=300, t_bounce3=-1),
        SimpleNamespace(t_start=240, t_net=-1, t_bounce1=350, t_bounce2=450, t_bounce3=-1),
    ]
    result = cast(RallyResult, SimpleNamespace(fps_out=240, sim_fps=240, trajectory_sim=torch.zeros(401, 3), shot_events=shots))
    times = np.arange(100) * 1001 / 60000
    events = retained_events(result, times)
    assert [event.native_frame for event in events] == [0, 120, 240, 350]
    for event in events:
        assert event.seconds == event.native_frame / 240
        assert abs(times[event.frame] - event.seconds) <= 1001 / 120000
    result.fps_out = 30
    with pytest.raises(ValueError, match="native-rate"):
        retained_events(result, times)


def test_event_masks_cover_interpolation_and_acceleration_stencil():
    event = Event("bounce", 41, 41 / 240, 10)
    labels, region, free = event_masks(30, [event], radius=5)
    np.testing.assert_array_equal(np.flatnonzero(region), np.arange(5, 16))
    np.testing.assert_array_equal(np.flatnonzero(labels[:, 1]), [10])
    assert not free[0] and not free[-1]
    # Linear interpolation straddling the impact lies at frames 10/11.
    assert not free[9:13].any()
    assert free[4] and free[16]
    _, hit_region, net_free = event_masks(30, [Event("net", 41, 41 / 240, 10)], radius=5)
    assert not hit_region.any()
    assert not net_free[5:16].any()


def _fixture():
    root = PROJECT_ROOT / "src/tasks/ball_refiner/refiner_3d"
    fixture = json.loads((root / "fixtures/meiji_video_002_clip_010.json").read_text())
    cameras = tuple(PinholeCamera(c["camera_id"], np.array(c["K"]), np.array(c["R"]), np.array(c["t"])) for c in fixture["cameras"])
    sizes = np.asarray([[c["w"], c["h"]] for c in fixture["cameras"]], dtype=float)
    plan = yaml.safe_load((root / "dataset_plan.yaml").read_text())
    return cameras, sizes, plan


def test_long_occlusion_retains_presence_full_covariance_and_all_64_modes():
    cameras, sizes, plan = _fixture()
    settings = plan["degradation"]
    positions = np.tile([0., 0., 2.], (96, 1))
    distribution, masks, metadata = make_distribution(positions, cameras, sizes, settings, np.random.default_rng(936), rally_index=3)
    assert metadata["shared_gap_length"] == 64
    assert masks["occlusion_mask"].all(0).sum() >= 64
    assert not masks["out_of_frame_mask"].any()
    torch.testing.assert_close(distribution.presence_probability, distribution.presence_probability[:, :1].expand(3, 96))
    torch.testing.assert_close(distribution.scale_tril[:, 40], distribution.scale_tril[:, 0] * settings["gap_sigma_multiplier"])
    assert bool((distribution.covariance[..., 0, 1] != 0).all())
    observations = frame_observations(distribution, torch.from_numpy(sizes), frame=40)
    result = triangulate_gmm(observations, cameras, prior=GaussianPrior3D(np.array(settings["prior_mean_m"]), np.diag(settings["prior_covariance_diagonal_m2"])), config=LaplaceConfig(64, 100))
    assert result.distribution.means.shape == (64, 3)
    assert len(np.unique(result.camera_subsets, axis=0)) == 8
    assert result.prior_only_probability > 0
    np.linalg.cholesky(result.distribution.covariance.astype(np.float32))
    np.testing.assert_allclose(result.distribution.weights.sum(), 1)


def test_camera_perturbation_changes_geometry_without_breaking_rotation():
    cameras, _, plan = _fixture()
    perturbed = perturb_cameras(cameras, plan["geometry"]["perturbation_per_scene"], np.random.default_rng(936))
    repeat = perturb_cameras(cameras, plan["geometry"]["perturbation_per_scene"], np.random.default_rng(936))
    for base, actual, second in zip(cameras, perturbed, repeat, strict=True):
        assert not np.array_equal(base.matrix, actual.matrix)
        assert not np.array_equal(base.center, actual.center)
        np.testing.assert_allclose(actual.rotation @ actual.rotation.T, np.eye(3), atol=1e-6)
        np.testing.assert_array_equal(actual.matrix, second.matrix)


def test_short_rally_gap_fails_instead_of_shortening_requested_gap():
    cameras, sizes, plan = _fixture()
    with pytest.raises(ValueError, match="too short"):
        make_distribution(np.tile([0., 0., 2.], (50, 1)), cameras, sizes, plan["degradation"], np.random.default_rng(0), rally_index=3)
