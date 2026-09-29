import numpy as np
import pytest
import torch

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.refiner_3d.triangulation import frame_observations


def test_consumes_all_components_and_source_pixel_correlations():
    means = torch.full((2, 3, 2, 2), 0.5)
    chol = torch.tensor([[0.02, 0.0], [0.01, 0.03]]).expand(2, 3, 2, 2, 2).clone()
    logits = torch.tensor([0.2, -0.3]).expand(2, 3, 2)
    distribution = BallGMM2D(means, chol, logits, torch.zeros(2, 3))
    result = frame_observations(
        distribution, torch.tensor([[1920.0, 1080.0], [640.0, 480.0]]), frame=1
    )
    assert result.means_px.shape == (2, 2, 2)
    np.testing.assert_allclose(result.means_px[:, 0], [[959.5, 539.5], [319.5, 239.5]])
    scale = np.diag([1919, 1079])
    np.testing.assert_allclose(
        result.covariance_px2[0, 0],
        scale @ (chol[0, 0, 0] @ chol[0, 0, 0].T).numpy() @ scale,
        rtol=1e-6,
    )
    np.testing.assert_allclose(result.weights, distribution.weights[:, 1].numpy())
    np.testing.assert_allclose(result.presence, [0.5, 0.5])
    with pytest.raises(ValueError, match="Frame index"):
        frame_observations(
            distribution, torch.tensor([[1920.0, 1080.0], [640.0, 480.0]]), frame=3
        )


def test_pixel_covariance_is_promoted_before_asymmetric_float32_scaling():
    chol = torch.tensor([[0.013017, 0.0], [0.01703, 0.0730103]]).expand(3, 1, 3, 2, 2).clone()
    dist = BallGMM2D(torch.full((3, 1, 3, 2), 0.5), chol, torch.zeros(3, 1, 3), torch.full((3, 1), 4.))
    sizes = torch.tensor([[1920., 1080.]]).expand(3, 2)
    observed = frame_observations(dist, sizes, frame=0)
    pixel_chol = chol[:, 0].double() * (sizes.double() - 1)[:, None, :, None]
    expected = pixel_chol @ pixel_chol.transpose(-1, -2)
    np.testing.assert_allclose(observed.covariance_px2, expected.numpy(), rtol=1e-14)
    np.testing.assert_allclose(observed.covariance_px2, observed.covariance_px2.swapaxes(-1, -2), rtol=1e-14)
    np.linalg.cholesky(observed.covariance_px2)
