from typing import Any

import pytest
import torch

from src.tasks.ball_detection.training.heatmap_pretraining.diagnostics import (
    batch_identity,
)
from tests.benchmarks.ball_dpt_checkpoint_replay import verify_target


def prefix() -> list[dict[str, Any]]:
    return [dict(clip_id=[f"clip-{i}"], start=[i], frame_step=[1], frame_indices=torch.arange(i, i + 32)[None]) for i in range(8)]


def test_replay_requires_the_exact_failed_window_at_the_correct_update() -> None:
    batches = prefix()
    failure = dict(attempted_update=42004, batch=batch_identity(batches[3]))
    verify_target(batches, first_step=42000, failure=failure)
    with pytest.raises(ValueError, match="failed batch"):
        verify_target(batches, first_step=42001, failure=failure)
    failure["batch"]["frame_step"] = [2]
    with pytest.raises(ValueError, match="failed batch"):
        verify_target(batches, first_step=42000, failure=failure)


def test_replay_that_stops_before_the_failure_is_rejected() -> None:
    batches = prefix()
    with pytest.raises(ValueError, match="include the failed"):
        verify_target(batches[:3], first_step=42000, failure=dict(attempted_update=42004, batch=batch_identity(batches[3])))
