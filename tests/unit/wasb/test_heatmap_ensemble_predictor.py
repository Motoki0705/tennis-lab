import torch

from src.wasb.inference.heatmap_ensemble_predictor import ensemble_heatmaps_argmax


def test_ensemble_heatmaps_argmax_threshold_sum() -> None:
    # Build two 1x3x3 heatmaps (logits) with different peaks.
    hm1 = torch.full((1, 3, 3), -10.0)
    hm2 = torch.full((1, 3, 3), -10.0)

    # hm1: strong peak at (row=1, col=2)
    hm1[0, 1, 2] = 10.0
    # hm2: strong peak at (row=0, col=1)
    hm2[0, 0, 1] = 10.0

    # With threshold 0.5 after sigmoid, both peaks survive; summed peaks tie at 1.0.
    idx, peak = ensemble_heatmaps_argmax([hm1, hm2], heatmap_threshold=0.5, apply_sigmoid=True)
    assert idx.shape == (1,)
    assert peak.shape == (1,)
    assert float(peak[0]) > 0.9

    # Make hm2 peak below threshold by lowering its logit.
    hm2[0, 0, 1] = 0.0  # sigmoid(0)=0.5 passes; lower slightly to fail
    idx2, _ = ensemble_heatmaps_argmax([hm1, hm2], heatmap_threshold=0.51, apply_sigmoid=True)
    # Should pick hm1 peak at flat index 1*3+2 = 5.
    assert int(idx2[0].item()) == 5

