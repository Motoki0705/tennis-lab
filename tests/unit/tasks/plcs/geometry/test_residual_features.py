"""Raw residual geometry feature contract."""

import numpy as np

from src.tasks.plcs.data.augmentation.residual import fixed_six_camera_rig
from src.tasks.plcs.geometry.residual_features import prepare_geometry
from src.tasks.plcs.model_io.residual_contracts import feature_dimension
from src.utils.geometry.triangulation import project_multiview
from src.utils.schema.court import STANDARD_COURT_CONFIG, court_keypoints_3d


def test_prepare_geometry_uses_fixed_raw_residual_features() -> None:
    rig = fixed_six_camera_rig((1920, 1080)).subset(np.array([0, 1, 4], dtype=np.int64))
    world: np.ndarray = np.zeros((4, 17, 3), dtype=np.float64)
    world[..., 1] = -5.0
    world[..., 2] = 1.2
    world[:, (11, 12), 2] = 0.9
    pixels, depth = project_multiview(world, rig.matrices)
    observations = pixels.transpose(2, 0, 1, 3)
    scores = (depth.transpose(2, 0, 1) > 0).astype(np.float64)
    court = court_keypoints_3d(STANDARD_COURT_CONFIG).numpy()[:14].astype(float)
    court_px, court_depth = project_multiview(court, rig.matrices)
    geometry = prepare_geometry(
        observations,
        scores,
        court_px.transpose(1, 0, 2),
        (court_depth.T > 0).astype(np.float64),
        rig,
        root_indices=(11, 12),
        fps=30.0,
    )
    assert geometry.features.shape == (3, 4, feature_dimension(17))
    np.testing.assert_allclose(geometry.residual_uv, 0.0, atol=1e-5)
