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
