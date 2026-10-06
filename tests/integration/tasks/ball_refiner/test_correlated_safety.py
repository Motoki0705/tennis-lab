"""Joint residual transport must preserve the confidence/error row pairing."""
import numpy as np

from tests.benchmarks.court_side_correlated import ResidualBank
from tests.benchmarks.legacy_ball_confidence import (
    PointConfidenceRule,
    point_confidence,
)


def test_joint_confidence_follows_its_error_not_an_independent_mask():
    bank = ResidualBank.__new__(ResidualBank)
    # An accurate/confident and an inaccurate/uncertain real-frame surrogate.
    bank.arrays = {
        "error_uv": np.array([[[.001, .001]] * 4, [[.25, .2]] * 4], np.float32),
        "scale_tril_uv": np.tile(np.eye(2) * .001, (2, 4, 1, 1)).astype(np.float32),
        "mixture_logits": np.array([[0, 1, 2, 3], [3, 2, 1, 0]], np.float32),
        "presence_logits": np.array([6., -6.], np.float32),
    }
    rows = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]])
    uv: np.ndarray = np.full((3, 3, 2), 500., np.float32)
    distribution = bank.transport(uv, (1920, 1080), rows)
    point, presence, area = point_confidence(distribution, (1920, 1080))
    np.testing.assert_array_equal(PointConfidenceRule(.9, 30000.).rejection_codes(presence, area) == 0, rows == 0)
    error = np.linalg.norm(point - uv, axis=-1)
    assert error[rows == 0].max() < 3 and error[rows == 1].min() > 500
    np.testing.assert_array_equal(distribution.presence_logits.numpy(), bank.arrays["presence_logits"][rows])
    np.testing.assert_array_equal(distribution.scale_tril.numpy(), bank.arrays["scale_tril_uv"][rows])


def test_blocks_keep_one_empirical_camera_per_view_with_explicit_fourth():
    bank = ResidualBank.__new__(ResidualBank)
    bank.runs = {c: [np.arange(c * 1000, c * 1000 + 200), np.arange(c * 1000 + 300, c * 1000 + 320)] for c in range(3)}
    result = bank.sample(300, 4, np.random.default_rng(30001))
    assert result.shape == (4, 300)
    assert len(set(result[:3, 0] // 1000)) == 3
    for row in result:
        assert np.unique(row // 1000).size == 1
        offsets = row % 1000
        assert ((offsets < 200) | ((offsets >= 300) & (offsets < 320))).all()
    np.testing.assert_array_equal(result, bank.sample(300, 4, np.random.default_rng(30001)))
