from dataclasses import replace

import torch

from src.tasks.ball_refiner.refiner_3d.diffusion.losses import (
    gravity_residual,
    robust_reprojection,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.memory_fixture import (
    analytic_memory_batch,
)
from src.utils.paths import PROJECT_ROOT

FIXTURE = PROJECT_ROOT / "src/tasks/ball_refiner/refiner_3d/fixtures/meiji_video_002_clip_010.json"


def test_physics_uses_seconds_and_excludes_impact_stencils():
    batch = analytic_memory_batch(FIXTURE, batch_size=1, frames=40, seed=0)
    assert gravity_residual(batch.target_positions_m, batch).item() < 1e-6
    impulse = batch.target_positions_m.clone()
    impulse[:, 16, 0] += 3
    assert gravity_residual(impulse, batch).item() > 1
    masked = batch.free_flight_mask.clone()
    masked[:, 11:22] = False  # event +/-5
    assert gravity_residual(impulse, replace(batch, free_flight_mask=masked)).item() < 1e-6


def test_robust_reprojection_preserves_modes_and_has_finite_outlier_gradients():
    batch = analytic_memory_batch(FIXTURE, batch_size=1, frames=16, seed=0)
    clean = robust_reprojection(batch.target_positions_m, batch)
    outlier = (batch.target_positions_m + torch.tensor([100., 100., 100.])).requires_grad_()
    loss = robust_reprojection(outlier, batch)
    assert bool(torch.isfinite(loss)) and loss > clean
    loss.backward()
    assert outlier.grad is not None and bool(torch.isfinite(outlier.grad).all())
    # Removing all amodal presence explicitly removes observation supervision.
    absent = replace(batch, presence_2d=torch.zeros_like(batch.presence_2d))
    assert robust_reprojection(outlier.detach(), absent).item() == 0


def test_padding_and_absent_views_need_no_fake_2d_covariance_or_mixture():
    batch = analytic_memory_batch(FIXTURE, batch_size=1, frames=16, seed=0)
    padding = batch.condition.padding_mask.clone()
    padding[:, -4:] = True
    condition = replace(batch.condition, padding_mask=padding)
    baseline = robust_reprojection(batch.target_positions_m, replace(batch, condition=condition))
    covariance, weights = batch.covariance_2d_px2.clone(), batch.weights_2d.clone()
    covariance[:, :, -4:] = 0
    weights[:, :, -4:] = 0
    actual = robust_reprojection(batch.target_positions_m, replace(batch, condition=condition, covariance_2d_px2=covariance, weights_2d=weights))
    torch.testing.assert_close(actual, baseline)
    absent = replace(batch, presence_2d=torch.zeros_like(batch.presence_2d), covariance_2d_px2=torch.zeros_like(covariance), weights_2d=torch.zeros_like(weights))
    assert robust_reprojection(batch.target_positions_m, absent).item() == 0
