"""Fixed-noise inference, window ownership, and preservation of absolute time."""
from dataclasses import replace

import pytest
import torch
from torch import Tensor

from src.tasks.ball_refiner.refiner_3d.diffusion.context_probe import predict_context
from src.tasks.ball_refiner.refiner_3d.diffusion.flow import sample_trajectories
from src.tasks.ball_refiner.refiner_3d.diffusion.memory_fixture import (
    analytic_memory_batch,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.model import (
    DenoiserOutput,
    MixtureCondition,
    ModelConfig,
    TrajectoryDenoiser,
)
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import denormalize_court_position


def test_explicit_noise_matches_original_sampler_and_does_not_mutate_it() -> None:
    torch.set_num_threads(1)
    batch = analytic_memory_batch(PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d/fixtures/meiji_video_002_clip_010.json', batch_size=1, frames=17, seed=936)
    model = TrajectoryDenoiser(ModelConfig(16, 1, 4, 2, 4, 0.)).eval()
    generator = torch.Generator().manual_seed(123)
    noise = torch.stack([torch.randn((1, 17, 3), generator=generator) for _ in range(3)])
    before = noise.clone()
    implicit = sample_trajectories(model, batch.condition, samples=3, steps=2, generator=torch.Generator().manual_seed(123))
    explicit = sample_trajectories(model, batch.condition, samples=3, steps=2, generator=torch.Generator().manual_seed(999), initial_noise=noise)
    torch.testing.assert_close(implicit.positions_m, explicit.positions_m, rtol=0, atol=0)
    torch.testing.assert_close(noise, before, rtol=0, atol=0)
    actual, _, owners = predict_context(model, batch, objective='flow', initial_noise=noise,
        probe_state=noise[0], probe_time=torch.tensor([.3]), steps=2, frames=None, check_budget=lambda: None)
    torch.testing.assert_close(actual, implicit.positions_m[:, 0], rtol=0, atol=0)
    assert owners == [{'window_start': 0, 'real_stop': 17, 'owned_start': 0, 'owned_stop': 17, 'padded_frames': 0}]
    with pytest.raises(ValueError, match='initial noise'):
        sample_trajectories(model, batch.condition, samples=3, steps=2, generator=generator, initial_noise=noise[:, :, :-1])


def test_short_tail_keeps_absolute_time_noise_and_earliest_ownership() -> None:
    torch.set_num_threads(1)
    batch = analytic_memory_batch(PROJECT_ROOT / 'src/tasks/ball_refiner/refiner_3d/fixtures/meiji_video_002_clip_010.json', batch_size=1, frames=17, seed=936)
    batch = replace(batch, condition=replace(batch.condition, timestamps_seconds=batch.condition.timestamps_seconds + 7.))

    class Pointwise(TrajectoryDenoiser):
        def forward(self, state: Tensor, time: Tensor, condition: MixtureCondition) -> DenoiserOutput:
            positions = state + condition.timestamps_seconds[..., None]
            return DenoiserOutput(positions, state[..., :2])

    model = Pointwise(ModelConfig(16, 1, 4, 2, 4, 0.)).eval()
    noise = torch.randn((2, 1, 17, 3), generator=torch.Generator().manual_seed(3))
    state, time = noise[0], torch.tensor([.25])
    outputs = []
    for width in (None, 4):
        prediction, probe, owners = predict_context(model, batch, objective='flow', initial_noise=noise,
            probe_state=state, probe_time=time, steps=1, frames=width, check_budget=lambda: None)
        outputs.append(prediction)
        torch.testing.assert_close(probe.positions_norm, state + batch.condition.timestamps_seconds[..., None])
        assert sum(o['owned_stop'] - o['owned_start'] for o in owners) == 17
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], denormalize_court_position(noise[:, 0] + batch.condition.timestamps_seconds[..., None]), rtol=0, atol=0)
    assert owners[-1] == {'window_start': 14, 'real_stop': 17, 'owned_start': 16, 'owned_stop': 17, 'padded_frames': 1}
