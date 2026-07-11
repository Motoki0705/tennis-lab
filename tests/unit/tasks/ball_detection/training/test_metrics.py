"""Unit tests for :mod:`src.tasks.ball_detection.training.metrics`."""

from __future__ import annotations

import torch

from src.tasks.ball_detection.training.metrics import BallDetectionMetrics
from src.utils.data.heatmaps import generate_gaussian_heatmaps

# 72x128 heatmap lattice mapped to a 1280x720 frame: one cell ~= 10px, so a
# 4px matching threshold separates lattice-argmax from sub-pixel decoding.
_HEATMAP_HW = (72, 128)
_ORIGINAL_WH = (1280.0, 720.0)
_SIGMA_RATIO = 0.012


def _build_batch(center_norm: tuple[float, float]) -> dict[str, torch.Tensor]:
    heatmap = generate_gaussian_heatmaps(
        _HEATMAP_HW, torch.tensor(center_norm), sigma_ratio=_SIGMA_RATIO
    )
    target_px = torch.tensor(
        [
            center_norm[0] * (_ORIGINAL_WH[0] - 1.0),
            center_norm[1] * (_ORIGINAL_WH[1] - 1.0),
        ]
    )
    return {
        "pred_heatmaps": heatmap[None, None],  # (B=1, T=1, H, W)
        "target_coords": target_px[None, None, None],  # (B, T, K=1, 2)
        "target_visibility": torch.ones(1, 1, 1),
        "original_size": torch.tensor([list(_ORIGINAL_WH)]),
    }


def _run_metric(*, subpixel_refine: bool, center_norm: tuple[float, float]) -> dict:
    metric = BallDetectionMetrics(
        peak_threshold=0.5,
        ball_distance_threshold=4.0,
        nms_kernel=3,
        max_predictions_per_frame=8,
        subpixel_refine=subpixel_refine,
    )
    batch = _build_batch(center_norm)
    metric.update(
        batch["pred_heatmaps"],
        batch["target_coords"],
        batch["target_visibility"],
        batch["original_size"],
    )
    return {name: float(value) for name, value in metric.compute().items()}


class TestBallDetectionMetricsSubpixel:
    def test_off_lattice_target_matches_only_with_refinement(self) -> None:
        # Mid-cell center: lattice argmax is ~5px off in both axes (~7px radial),
        # beyond the 4px threshold; sub-pixel decoding recovers it exactly.
        center = (0.4130, 0.5020)
        quantized = _run_metric(subpixel_refine=False, center_norm=center)
        refined = _run_metric(subpixel_refine=True, center_norm=center)
        assert quantized["f1"] == 0.0
        assert refined["f1"] == 1.0
        assert refined["mean_distance_px"] < 0.1

    def test_on_lattice_target_matches_either_way(self) -> None:
        # Lattice-aligned center: both modes must match with ~zero distance.
        center = (64.0 / 127.0, 36.0 / 71.0)
        for subpixel_refine in (False, True):
            values = _run_metric(subpixel_refine=subpixel_refine, center_norm=center)
            assert values["f1"] == 1.0
            assert values["mean_distance_px"] < 1.0e-3
