"""Positive-domain fitting and explicit nonregular-component integration."""

import numpy as np
import pytest

from src.utils.geometry.probabilistic_triangulation import (
    CameraGMM,
    GaussianPrior3D,
    LaplaceConfig,
    triangulate_gmm,
)
from src.utils.geometry.probabilistic_triangulation.optimization import (
    NonregularComponentError,
)
from src.utils.geometry.probabilistic_triangulation.solver import (
    HybridConfig,
    fit_component,
    triangulate_hybrid,
)
from src.utils.geometry.probabilistic_triangulation.volume import (
    VoxelConfig,
    integrate_component,
)
from src.utils.geometry.triangulation import PinholeCamera
from src.utils.paths import PROJECT_ROOT


def test_frozen_behind_camera_case_uses_volume_and_keeps_its_mass(monkeypatch):
    case = np.load(PROJECT_ROOT / "knowledge/runs/run-i936-synthetic-smoke-v1-s936/behind-mode-case.npz", allow_pickle=False)
    matrices, means, covariance = case["P"], case["means"], case["cov"]
    prior = GaussianPrior3D(np.array([0., 0., 2.]), np.diag([36., 144., 9.]))
    from src.utils.geometry.probabilistic_triangulation import solver
    original = solver.project_with_jacobian
    def checked(point, matrices):
        assert (matrices[:, 2, :3] @ point + matrices[:, 2, 3] > 0).all()
        return original(point, matrices)
    monkeypatch.setattr(solver, "project_with_jacobian", checked)
    with pytest.raises(NonregularComponentError):
        fit_component(matrices, means, covariance, prior, max_nfev=100)
    # Factor the fixture's P into the camera API without modifying projection.
    from scipy.linalg import rq
    camera_list = []
    for i, matrix in enumerate(matrices):
        intrinsic, rotation = rq(matrix[:, :3])
        signs = np.diag(np.sign(intrinsic.diagonal()))
        intrinsic, rotation = intrinsic @ signs, signs @ rotation
        camera_list.append(PinholeCamera(str(i), intrinsic, rotation, np.linalg.solve(intrinsic, matrix[:, 3])))
    cameras = tuple(camera_list)
    obs = CameraGMM(means[:, None], covariance[:, None], np.ones((2, 1)), np.array([.8, .7]))
    config = HybridConfig(LaplaceConfig(4, 100), VoxelConfig(16, 7, 512, 4.))
    result = triangulate_hybrid(obs, cameras, prior=prior, config=config)
    assert len(result.distribution.weights) == 4
    assert any(m.startswith("volume:") for m in result.component_methods)
    assert result.distribution.weights == pytest.approx([.06, .14, .24, .56])
    for mean, active in zip(result.distribution.means, result.camera_subsets, strict=True):
        assert all(cameras[i].project(mean)[1] for i in np.flatnonzero(active))
    np.linalg.cholesky(result.distribution.covariance.astype(np.float32))


def test_boundary_volume_evidence_and_moments_match_refined_integral():
    camera = PinholeCamera("a", np.diag([10., 10., 1.]), np.eye(3), np.zeros(3))
    prior = GaussianPrior3D(np.array([0., 0., -.5]), np.eye(3))
    means, covariance = np.zeros((1, 2)), np.tile(np.eye(2)*100, (1, 1, 1))
    a = integrate_component((camera,), means, covariance, prior, VoxelConfig(16, 6, 512, 4.))
    b = integrate_component((camera,), means, covariance, prior, VoxelConfig(24, 7, 1024, 4.))
    assert a[0][2] > 0
    np.testing.assert_allclose(a[0], b[0], atol=.015)
    np.testing.assert_allclose(a[1], b[1], atol=.02)
    assert abs(a[2]-b[2]) < .02


def test_impossible_positive_region_fails_without_component_removal():
    # Coincident cameras face opposite directions, so no common positive point.
    k=np.diag([10.,10.,1.])
    cameras=(PinholeCamera("a", k, np.eye(3), np.zeros(3)), PinholeCamera("b", k, np.diag([1.,-1.,-1.]), np.zeros(3)))
    obs=CameraGMM(np.zeros((2,1,2)),np.tile(np.eye(2),(2,1,1,1)),np.ones((2,1)),np.ones(2))
    prior=GaussianPrior3D(np.zeros(3),np.eye(3))
    with pytest.raises(NonregularComponentError):
        triangulate_gmm(obs,cameras,prior=prior,config=LaplaceConfig(1,100))
    with pytest.raises(RuntimeError,match="no finite likelihood"):
        triangulate_hybrid(obs,cameras,prior=prior,config=HybridConfig(LaplaceConfig(1,100),VoxelConfig(8,3,32,4.)))
