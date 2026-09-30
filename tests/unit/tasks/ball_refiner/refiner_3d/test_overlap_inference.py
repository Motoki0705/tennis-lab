"""Noise/absolute-time preservation, short tails, and actual blend seam behavior."""
from dataclasses import replace

import pytest
import torch
from torch import Tensor

from src.tasks.ball_refiner.refiner_3d.diffusion.context_inference import (
    predict_context,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.memory_fixture import (
    analytic_memory_batch,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.model import (
    DenoiserOutput,
    MixtureCondition,
    ModelConfig,
    TrajectoryDenoiser,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.overlap_inference import (
    predict_overlap,
)
from src.utils.paths import PROJECT_ROOT


class Pointwise(TrajectoryDenoiser):
    def forward(self, state: Tensor, time: Tensor, condition: MixtureCondition) -> DenoiserOutput:
        return DenoiserOutput(state + condition.timestamps_seconds[..., None], state[..., :2])


@pytest.mark.parametrize('length', [16, 32, 33, 65])
@pytest.mark.parametrize('objective', ['flow', 'regression'])
def test_overlap_preserves_pointwise_function_all_samples_and_short_tail(length, objective) -> None:
    torch.set_num_threads(1)
    batch = analytic_memory_batch(PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d/fixtures/meiji_video_002_clip_010.json', batch_size=1, frames=length, seed=936)
    batch = replace(batch, condition=replace(batch.condition, timestamps_seconds=batch.condition.timestamps_seconds + 7.))
    model = Pointwise(ModelConfig(16, 1, 4, 2, 4, 0.)).eval()
    noise = torch.randn((3, 1, length, 3), generator=torch.Generator().manual_seed(3))
    before = noise.clone()
    actual, spans = predict_overlap(model, batch, objective=objective, initial_noise=noise, steps=1, frames=32, stride=16, check_budget=lambda: None)
    expected, _, _ = predict_context(model, batch, objective=objective, initial_noise=noise,
        probe_state=torch.zeros((1, length, 3)), probe_time=torch.zeros(1), steps=1, frames=None, check_budget=lambda: None)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(noise, before, rtol=0, atol=0)
    assert spans[-1]['real_stop'] == length
    assert actual.shape == (3 if objective == 'flow' else 1, length, 3)
    changed = replace(batch, target_positions_m=batch.target_positions_m + 100, free_flight_mask=~batch.free_flight_mask)
    again, _ = predict_overlap(model, changed, objective=objective, initial_noise=noise, steps=1, frames=32, stride=16, check_budget=lambda: None)
    torch.testing.assert_close(actual, again, rtol=0, atol=0)


def test_blend_reduces_a_known_window_boundary_jump_without_hiding_frames() -> None:
    torch.set_num_threads(1)
    batch = analytic_memory_batch(PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d/fixtures/meiji_video_002_clip_010.json', batch_size=1, frames=24, seed=936)

    class WindowBias(TrajectoryDenoiser):
        def forward(self, state: Tensor, time: Tensor, condition: MixtureCondition) -> DenoiserOutput:
            bias = condition.timestamps_seconds[:, :1, None].expand_as(state)
            return DenoiserOutput(bias, torch.zeros_like(state[..., :2]))

    model = WindowBias(ModelConfig(16, 1, 4, 2, 4, 0.)).eval()
    noise = torch.zeros(2, 1, 24, 3)
    original, _, _ = predict_context(model, batch, objective='regression', initial_noise=noise,
        probe_state=noise[0], probe_time=torch.zeros(1), steps=1, frames=8, check_budget=lambda: None)
    blended, _ = predict_overlap(model, batch, objective='regression', initial_noise=noise,
        steps=1, frames=8, stride=4, check_budget=lambda: None)
    assert blended.shape == original.shape
    assert torch.diff(blended, n=2, dim=1).square().sum() < torch.diff(original, n=2, dim=1).square().sum() / 2
    with pytest.raises(ValueError, match='stride'):
        predict_overlap(model, batch, objective='regression', initial_noise=noise, steps=1, frames=8, stride=8, check_budget=lambda: None)
    with pytest.raises(ValueError, match='initial noise'):
        predict_overlap(model, batch, objective='flow', initial_noise=noise[:, :, :-1], steps=1, frames=8, stride=4, check_budget=lambda: None)
