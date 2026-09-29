from dataclasses import replace

import pytest
import torch
import yaml

from src.tasks.ball_refiner.refiner_3d.diffusion.flow import (
    sample_trajectories,
    training_objective,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.losses import LossConfig
from src.tasks.ball_refiner.refiner_3d.diffusion.memory_fixture import (
    analytic_memory_batch,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.model import (
    ModelConfig,
    TrajectoryDenoiser,
    validate_flow_state,
)
from src.utils.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT / "src/tasks/ball_refiner/refiner_3d"
FIXTURE = ROOT / "fixtures/meiji_video_002_clip_010.json"


def small_model():
    return TrajectoryDenoiser(ModelConfig(32, 1, 4, 2, 4, 0.))


def test_memory_profile_cpu_forward_backward_all_four_losses():
    torch.set_num_threads(1)
    config = yaml.safe_load((ROOT / "memory_smoke.yaml").read_text())
    batch = analytic_memory_batch(FIXTURE, batch_size=config["batch_size"], frames=config["frames"], seed=config["seed"])
    model = TrajectoryDenoiser(ModelConfig(**config["model"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    before = model.position_head.weight.detach().clone()
    loss, terms = training_objective(model, batch, LossConfig(**config["loss"]), torch.Generator().manual_seed(936), objective="flow")
    assert set(terms) == {"x0", "reprojection", "physics", "event"}
    assert all(bool(torch.isfinite(value)) for value in terms.values())
    loss.backward()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.position_head.weight)
    assert batch.condition.means_m.shape == (2, 128, 64, 3)


def test_output_head_is_absolute_x0_and_not_a_residual():
    batch = analytic_memory_batch(FIXTURE, batch_size=2, frames=16, seed=4)
    model = small_model().eval()
    with torch.no_grad():
        model.position_head.weight.zero_()
        model.position_head.bias.copy_(torch.tensor([.1, .2, .3]))
    result = model(torch.randn(2, 16, 3) * 10, torch.tensor([.1, .8]), batch.condition)
    torch.testing.assert_close(result.positions_norm, torch.tensor([.1, .2, .3]).expand(2, 16, 3))


def test_full_mixture_permutation_invariance_and_last_component_influence():
    batch = analytic_memory_batch(FIXTURE, batch_size=1, frames=16, seed=5)
    condition = batch.condition
    model = small_model().eval()
    state, time = torch.zeros(1, 16, 3), torch.tensor([.5])
    baseline = model(state, time, condition).positions_norm
    permutation = torch.arange(63, -1, -1)
    permuted = replace(condition, means_m=condition.means_m[:, :, permutation], covariance_m2=condition.covariance_m2[:, :, permutation], weights=condition.weights[:, :, permutation], camera_subsets=condition.camera_subsets[:, :, permutation])
    torch.testing.assert_close(model(state, time, permuted).positions_norm, baseline, atol=1e-6, rtol=1e-5)
    changed = condition.means_m.clone()
    changed[:, :, -1, 0] += 20
    assert not torch.allclose(model(state, time, replace(condition, means_m=changed)).positions_norm, baseline)


def test_padding_never_changes_real_frames_or_becomes_a_gap():
    batch = analytic_memory_batch(FIXTURE, batch_size=1, frames=24, seed=8)
    condition = batch.condition
    padding = condition.padding_mask.clone()
    padding[:, -4:] = True
    first = replace(condition, padding_mask=padding)
    changed = condition.means_m.clone()
    changed[:, -4:] += 1000
    second = replace(first, means_m=changed)
    model = small_model().eval()
    state, time = torch.zeros(1, 24, 3), torch.zeros(1)
    a, b = model(state, time, first), model(state, time, second)
    torch.testing.assert_close(a.positions_norm[:, :-4], b.positions_norm[:, :-4])
    assert not bool(a.positions_norm[:, -4:].any())
    assert bool(torch.isfinite(a.positions_norm[:, 8:16]).all())  # genuine gap remains real
    invalid = padding.clone()
    invalid[:, -1] = False
    with pytest.raises(ValueError, match="right padding"):
        replace(condition, padding_mask=invalid)


def test_sampling_reports_empirical_uncertainty_and_replays_explicit_seed():
    batch = analytic_memory_batch(FIXTURE, batch_size=1, frames=16, seed=6)
    model = small_model()
    a = sample_trajectories(model, batch.condition, samples=3, steps=4, generator=torch.Generator().manual_seed(9))
    b = sample_trajectories(model, batch.condition, samples=3, steps=4, generator=torch.Generator().manual_seed(9))
    assert model.training
    assert a.positions_m.shape == (3, 1, 16, 3)
    torch.testing.assert_close(a.positions_m, b.positions_m)
    torch.testing.assert_close(a.mean_m, a.positions_m.mean(0))
    torch.testing.assert_close(a.covariance_m2.diagonal(dim1=-2, dim2=-1), a.positions_m.var(0))
    assert bool((a.covariance_m2.diagonal(dim1=-2, dim2=-1) > 0).any())
    assert not a.positions_m.requires_grad


def test_same_backbone_one_step_regression_ignores_flow_rng():
    batch = analytic_memory_batch(FIXTURE, batch_size=1, frames=16, seed=7)
    model, weights = small_model(), LossConfig(1., .01, .0001, .1)
    a, _ = training_objective(model, batch, weights, torch.Generator().manual_seed(1), objective="regression")
    b, _ = training_objective(model, batch, weights, torch.Generator().manual_seed(2), objective="regression")
    torch.testing.assert_close(a, b)
    a.backward()
    assert model.position_head.weight.grad is not None


@pytest.mark.parametrize("kind", ["shape", "nan", "time"])
def test_flow_boundary_rejects_invalid_state(kind):
    batch = analytic_memory_batch(FIXTURE, batch_size=1, frames=16, seed=9)
    state, time = torch.zeros(1, 16, 3), torch.zeros(1)
    if kind == "shape":
        state = state[:, :-1]
    elif kind == "nan":
        state[0, 0, 0] = float("nan")
    else:
        time[:] = 1.1
    with pytest.raises(ValueError):
        validate_flow_state(state, time, batch.condition)
