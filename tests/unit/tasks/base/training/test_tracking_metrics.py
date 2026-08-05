from __future__ import annotations

import torch

from src.tasks.base.training.tracking_metrics import (
    TrackingMetricConfig,
    common_lifecycle_tracking_metrics,
)


def test_metrics_measure_two_segments_as_one_legal_query_reuse() -> None:
    target_presence = torch.zeros(1, 12, 1, dtype=torch.bool)
    target_presence[:, 1:4, 0] = True
    target_presence[:, 7:10, 0] = True
    target_instance_id = torch.full((1, 12, 1), -1, dtype=torch.long)
    target_instance_id[:, 1:4, 0] = 20
    target_instance_id[:, 7:10, 0] = 21
    target_position = torch.zeros(1, 12, 1, 3)
    target_position[:, 1:4, 0, 0] = 1.0
    target_position[:, 7:10, 0, 0] = 2.0
    prediction_position = torch.zeros(1, 12, 2, 3)
    prediction_position[:, :, 0] = target_position[:, :, 0]
    presence_logits = torch.full((1, 12, 2), -20.0)
    presence_logits[:, 1:4, 0] = 20.0
    presence_logits[:, 7:10, 0] = 20.0

    metrics = common_lifecycle_tracking_metrics(
        {"position": prediction_position, "presence_logits": presence_logits},
        {
            "target_position": target_position,
            "target_presence": target_presence,
            "target_instance_id": target_instance_id,
            "frame_mask": torch.ones(1, 12, dtype=torch.bool),
        },
        [(torch.tensor([0]), torch.tensor([0]))],
        config=TrackingMetricConfig(
            presence_threshold=0.5,
            duplicate_distance=0.1,
        ),
    )

    assert metrics["birth_frame_error"].item() == 0.0
    assert metrics["death_frame_error"].item() == 0.0
    assert metrics["lifecycle_presence_f1"].item() == 1.0
    assert metrics["query_reuse_count"].item() == 1.0
    assert metrics["segment_id_switches"].item() == 0.0
    assert metrics["illegal_overlap_count"].item() == 0.0
