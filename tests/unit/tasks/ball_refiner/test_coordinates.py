"""Scientific/data contracts for the shared coordinate refiner strategy."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.tasks.ball_refiner.coordinates.config import (
    CorruptionConfig,
    DiscriminatorConfig,
    ModelConfig,
    parse_section,
)
from src.tasks.ball_refiner.coordinates.corruption import (
    coordinate_noise,
    corrupt_trajectory,
)
from src.tasks.ball_refiner.coordinates.evaluation import error_metrics
from src.tasks.ball_refiner.coordinates.generation import (
    CameraSampling,
    event_frames,
    sample_cameras,
    sample_visible_cameras,
)
from src.tasks.ball_refiner.coordinates.inference import refine_coordinates
from src.tasks.ball_refiner.coordinates.model import (
    CoordinateRefiner,
)
from src.tasks.ball_refiner.coordinates.models.discriminators import (
    TrajectoryDiscriminator,
)
from src.tasks.base.training.gan_loss import LSGANLoss
from src.utils.geometry.triangulation import triangulate_multiview
from src.utils.schema.court import BASELINE_CLEAR, HALF_LENGTH


@pytest.fixture
def corruption():
    return CorruptionConfig(0.5, 0.04, 3, 10, 200.0, 3.0, 0.1, 3)


def projected(frames=600):
    cameras = sample_cameras(CameraSampling(4, 1280, 720, 3, 8, 60, 90, 2, 3, 0.5, 2, 1024), np.random.default_rng(41))
    time = np.arange(frames) / 60
    xyz = np.column_stack((np.sin(time), np.sin(time * 0.4) * 7, 2 + np.sin(time)))
    uv = np.stack([c.project(xyz)[0] for c in cameras]).astype(np.float32)
    visible = np.ones((4, frames), dtype=bool)
    events = np.zeros(frames, dtype=np.uint8)
    events[np.arange(25, frames - 25, 40)] = 1
    return xyz, uv, visible, np.stack([c.matrix for c in cameras]), events


def test_camera_planes_and_roundtrip():
    cameras = sample_cameras(CameraSampling(40, 1280, 720, 3, 8, 60, 90, 2, 3, 0.5, 2, 1024), np.random.default_rng(42))
    centers = np.stack([c.center for c in cameras])
    np.testing.assert_allclose(np.abs(centers[:, 1]), HALF_LENGTH + BASELINE_CLEAR, atol=1e-5)
    assert np.ptp(centers[:, 0]) > 12
    assert np.ptp(centers[:, 2]) > 3
    assert len(np.unique([c.intrinsic[0, 0] for c in cameras])) == 40
    xyz, uv, visible, projections, _ = projected()
    result = triangulate_multiview(uv.transpose(1, 0, 2), visible.T.astype(float), projections)
    assert result.valid.all()
    np.testing.assert_allclose(result.points, xyz, atol=1e-4)


def test_noise_radial_p95_is_measured_over_all_observations(corruption):
    noise = coordinate_noise((250000,), corruption, np.random.default_rng(3))
    distance = np.linalg.norm(noise, axis=-1)
    assert np.quantile(distance, 0.95) == pytest.approx(200, abs=5)
    assert np.median(distance) < 5
    assert np.quantile(distance, 0.99) > 300


def test_camera_selection_keeps_clean_trajectory_in_frame_before_corruption():
    xyz, _, _, _, _ = projected()
    config = CameraSampling(4, 1280, 720, 3, 12, 60, 120, 2, 12, 0.5, 2, 1024)
    cameras = sample_visible_cameras(config, xyz, np.random.default_rng(5))
    assert cameras is not None
    for camera in cameras:
        uv, front = camera.project(xyz)
        assert front.all() and (uv >= 0).all()
        assert (uv[:, 0] < 1280).all() and (uv[:, 1] < 720).all()
    impossible = xyz + np.array([0, 1000, 0])
    assert sample_visible_cameras(replace(config, maximum_attempts=1), impossible, np.random.default_rng(5)) is None


def test_asymmetric_event_gaps_include_events_and_match_3d(corruption):
    _, uv, visible, cameras, events = projected()
    config = replace(corruption, event_probability=1, isolated_probability=0)
    result = corrupt_trajectory(uv, visible, cameras, events, config=config, seed=7)
    assert len(result.intervals) == np.count_nonzero(events)
    for event, start, end in result.intervals:
        assert 3 <= event - start <= 10
        assert 3 <= end - event - 1 <= 10
        assert 7 <= end - start <= 21
        assert event - start != end - event - 1
        assert result.missing_2d[:, start:end].all()
        assert result.missing_3d[start:end].all()
    assert np.all(result.uv_px[result.missing_2d] == 0)
    assert np.all(result.xyz_m[result.missing_3d] == 0)


def test_ablation_changes_selection_only_not_noise_or_gap_geometry(corruption):
    _, uv, visible, cameras, events = projected()
    low = corrupt_trajectory(uv, visible, cameras, events, config=replace(corruption, event_probability=0.25), seed=15)
    high = corrupt_trajectory(uv, visible, cameras, events, config=replace(corruption, event_probability=0.75), seed=15)
    np.testing.assert_array_equal(low.noise_px, high.noise_px)
    assert set(map(tuple, low.intervals)) <= set(map(tuple, high.intervals))
    both = ~low.missing_2d & ~high.missing_2d
    np.testing.assert_array_equal(low.uv_px[both], high.uv_px[both])


def test_phantom_bounces_after_return_are_excluded_before_rounding():
    def shot(start, returned, bounces):
        return SimpleNamespace(t_start=start, t_return=returned, t_bounce1=bounces[0], t_bounce2=bounces[1], t_bounce3=bounces[2])
    result = SimpleNamespace(trajectory=np.zeros((80, 3)), shot_events=[shot(0, 35, [20, 36, 60]), shot(36, -1, [64, -1, -1])])
    events = event_frames(result, stride=4)
    # 36 is a real next shot but a hypothetical previous bounce, in same output bin.
    assert events[9] == 1
    assert events[5] == 2
    assert events[15] == 0
    assert events[16] == 2


@pytest.mark.parametrize("dimensions,architecture", [(2, "regression"), (3, "regression"), (3, "flow")])
def test_masked_values_cannot_leak_and_observed_frames_are_predicted(dimensions, architecture):
    torch.set_num_threads(1)
    model = CoordinateRefiner(ModelConfig(dimensions, architecture, 16, 1, 2, 0, 32, 3, 64, 8, 10000.0)).eval()
    coordinates = torch.randn(1, 43, dimensions)
    missing = torch.zeros(1, 43, dtype=torch.bool)
    missing[:, 10:25] = True
    first = refine_coordinates(model, coordinates, missing, seed=91)
    coordinates[missing] = float("nan")
    second = refine_coordinates(model, coordinates, missing, seed=91)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == coordinates.shape and torch.isfinite(first).all()
    assert not torch.allclose(first[~missing], coordinates[~missing])
    # All missing windows are valid inputs and still return one complete trajectory.
    prediction = refine_coordinates(model, coordinates, torch.ones_like(missing), seed=8)
    assert torch.isfinite(prediction).all()


def test_future_observation_influences_past_offline_prediction():
    model = CoordinateRefiner(ModelConfig(2, "regression", 16, 1, 2, 0, 32, 3, 64, 8, 10000.0)).eval()
    values = torch.zeros(1, 16, 2, requires_grad=True)
    model(values, torch.zeros(1, 16, dtype=torch.bool))[0, 0].sum().backward()
    assert values.grad[0, -1].abs().sum() > 0


@pytest.mark.parametrize("architecture,with_state,with_time", [
    ("regression", True, False), ("regression", False, True),
    ("flow", False, False), ("flow", True, False), ("flow", False, True),
])
def test_flow_arguments_are_rejected_before_tensor_computation(architecture, with_state, with_time):
    model = CoordinateRefiner(ModelConfig(3, architecture, 16, 1, 2, 0, 32, 3, 64, 8, 10000.0)).eval()
    coordinates = torch.zeros(1, 16, 3)
    missing = torch.zeros(1, 16, dtype=torch.bool)
    computed = []
    model.input.register_forward_pre_hook(lambda *_: computed.append(True))
    with pytest.raises(ValueError, match="Flow forward requires|Regression does not accept"):
        model(coordinates, missing, state=torch.zeros_like(coordinates) if with_state else None,
              time=torch.zeros(1) if with_time else None)
    assert not computed


def test_flow_and_gan_both_supply_trainable_gradients():
    coords, target = torch.randn(2, 16, 3), torch.randn(2, 16, 3)
    missing = torch.rand(2, 16) < 0.5
    model = CoordinateRefiner(ModelConfig(3, "flow", 16, 1, 2, 0, 32, 3, 64, 8, 10000.0))
    loss = model.flow_loss(coords, missing, target, torch.Generator().manual_seed(4))
    loss.backward()
    assert torch.isfinite(loss) and model.input.weight.grad.abs().sum() > 0
    predicted = torch.randn(2, 16, 3, requires_grad=True)
    discriminator = TrajectoryDiscriminator(3, DiscriminatorConfig("trajectory_transformer", 16, 1, 2, 64, 0.0, 8, 10000.0, "swiglu", 32, 0.02, 0.02))
    LSGANLoss().generator_loss(discriminator(predicted)).backward()
    assert predicted.grad.abs().sum() > 0


def test_metric_rmse_is_euclidean_and_counts_every_frame():
    prediction = np.array([[3, 4], [0, 0], [0, 12]], dtype=float)
    report = error_metrics(prediction, np.zeros_like(prediction), np.array([True, False, False]), np.array([True, False, True]))
    assert report["all"]["rmse"] == pytest.approx(np.sqrt(169 / 3))
    assert report["missing"]["rmse"] == 5
    assert report["observed"]["count"] == 2
    assert report["frame_missing_rate"] == pytest.approx(1 / 3)


def test_config_rejects_implicit_or_unknown_fields():
    with pytest.raises(ValueError, match="missing"):
        parse_section(ModelConfig, {"dimensions": 2})
    with pytest.raises(TypeError):
        parse_section(ModelConfig, {"dimensions": True, "architecture": "regression", "width": 16, "layers": 1, "heads": 2, "dropout": 0, "window_length": 32, "flow_steps": 3, "ffn_dim": 64, "rope_dim": 8, "rope_theta": 10000.0})


def test_event_only_inputs_preserve_observations_and_shared_3d(corruption):
    xyz, uv, visible, cameras, events = projected()
    config = replace(corruption, noise_p95_px=0.0, jitter_sigma_px=0.0, outlier_probability=0.0, isolated_probability=0.0)
    result = corrupt_trajectory(uv, visible, cameras, events, config=config, seed=7)
    expected: np.ndarray = np.zeros(len(events), dtype=bool)
    for _, start, end in result.intervals:
        expected[start:end] = True
    np.testing.assert_array_equal(result.missing_2d, np.broadcast_to(expected, visible.shape))
    np.testing.assert_array_equal(result.missing_3d, expected)
    assert not result.isolated.any() and not result.noise_px.any()
    np.testing.assert_array_equal(result.uv_px[~result.missing_2d], uv[~result.missing_2d])
    np.testing.assert_allclose(result.xyz_m[~expected], xyz[~expected], atol=1e-4)
