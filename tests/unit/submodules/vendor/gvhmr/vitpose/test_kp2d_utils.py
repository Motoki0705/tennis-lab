"""Tests for vendored ViTPose heatmap decoding (UDP post-processing)."""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.submodules.vendor.gvhmr.vitpose.kp2d_utils import keypoints_from_heatmaps

Float32Array: TypeAlias = NDArray[np.float32]

def gaussian_heatmap(
    height: int, width: int, cx: float, cy: float, sigma: float = 2.0
) -> Float32Array:
    ys, xs = np.mgrid[0:height, 0:width]
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2)).astype(  # type: ignore[no-any-return]
        np.float32
    )


class TestKeypointsFromHeatmaps:
    def test_udp_decodes_peak_to_full_image_coords(self) -> None:
        H, W = 64, 48
        num_joints = 3
        peaks = [(10.0, 20.0), (24.0, 32.0), (40.0, 55.0)]  # (x, y) in heatmap space

        heatmaps = np.stack(
            [gaussian_heatmap(H, W, cx, cy) for cx, cy in peaks], axis=0
        )[None]  # (1, J, H, W)

        center = np.array([[100.0, 200.0]])
        # bbox of 192x256 pixels (scale unit = 200 px)
        scale = np.array([[192.0 / 200.0, 256.0 / 200.0]])

        preds, maxvals = keypoints_from_heatmaps(
            heatmaps=heatmaps, center=center, scale=scale, use_udp=True
        )

        assert preds.shape == (1, num_joints, 2)
        assert maxvals.shape == (1, num_joints, 1)

        # UDP mapping: full = peak * (scale*200) / (heatmap_size - 1) + center - scale*200/2
        for j, (cx, cy) in enumerate(peaks):
            expected_x = cx * 192.0 / (W - 1) + 100.0 - 96.0
            expected_y = cy * 256.0 / (H - 1) + 200.0 - 128.0
            np.testing.assert_allclose(preds[0, j, 0], expected_x, atol=1.0)
            np.testing.assert_allclose(preds[0, j, 1], expected_y, atol=1.0)

        assert (maxvals > 0.9).all()
