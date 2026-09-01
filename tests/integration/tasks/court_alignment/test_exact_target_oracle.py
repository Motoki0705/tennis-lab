"""Oracle test for exact synthetic KP14 decoding and vote grouping."""

from typing import cast

import pytest
import torch

from src.tasks.court_alignment.data.dataset import (
    GroundCourtDataset,
    GroundCourtDatasetConfig,
)
from src.tasks.court_alignment.data.splits import GroundCourtSplitConfig
from src.tasks.court_alignment.inference.decoder import decode_court_instances

pytestmark = pytest.mark.integration


def test_exact_targets_decode_and_group_for_one_hundred_fixed_samples() -> None:
    # A generous canvas/margin and 1 px/m scale make every KP visible.  This
    # isolates decoder/association correctness from the intentional clipped
    # court distribution used by the production procedural configuration.
    config = GroundCourtDatasetConfig(
        image_size=128,
        max_courts=2,
        min_courts=1,
        min_center_distance_px=32.0,
        max_sampling_attempts=64,
        court_margin_px=32.0,
        scale_px_per_metre_range=(1.0, 1.0),
        split=GroundCourtSplitConfig(
            train_size=100,
            val_size=0,
            test_size=0,
            seed=20260901,
        ),
        sigma_px=1.0,
        vote_radius_px=3.0,
    )
    dataset = GroundCourtDataset(config, split="train")
    for index in range(100):
        sample = dataset[index]
        target_heatmaps = cast(torch.Tensor, sample["target_heatmaps"])
        target_center_votes = cast(torch.Tensor, sample["target_center_votes"])
        num_courts = cast(int, sample["num_courts"])
        # Convert exact [0,1] targets into finite logits for the model decoder.
        logits = torch.logit(
            target_heatmaps.clamp(1.0e-6, 1.0 - 1.0e-6)
        ).unsqueeze(0)
        result = decode_court_instances(
            logits,
            target_center_votes.unsqueeze(0),
            threshold=0.25,
            nms_kernel=3,
            max_peaks=4,
            subpixel_refine=True,
            cluster_distance_px=12.0,
            max_instances=2,
        )
        assert result[0].num_instances == num_courts, index
        assert torch.equal(
            result[0].valid, torch.ones_like(result[0].valid, dtype=torch.bool)
        ), index
