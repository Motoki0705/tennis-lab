"""Historical two-channel checkpoints use the same real resume contract."""

import json
from pathlib import Path

import pytest

from tests.integration.tasks.ball_detection.lifecycle_contracts import (
    exercise_mdd_resume,
)
from tests.integration.tasks.ball_detection.mdd_training_contracts import (
    omit_explicit_channels,
    read_json,
)
from tests.integration.tasks.ball_detection.model_contracts import FAMILIES

pytestmark = pytest.mark.integration


def test_historical_pretraining_restores_then_matches_the_next_update(tmp_path: Path) -> None:
    exercise_mdd_resume(
        FAMILIES["mdd_dpt_pretrain"], tmp_path, epochs=2, mutate=omit_explicit_channels,
    )
    output = tmp_path / "interrupted"
    stopped = read_json(output / "DIAGNOSTIC_STOP.json")
    assert stopped["global_step"] == 1 and stopped["total_updates"] == 2
    assert stopped["complete"] is False
    assert read_json(output / "COMPLETED.json")["global_step"] == 2
    logs = [
        [json.loads(line) for line in (tmp_path / name / "train.jsonl").read_text().splitlines()]
        for name in ("continuous", "interrupted")
    ]
    assert [row["global_step"] for row in logs[1]] == [1, 2]
    assert [(row["learning_rate"], row["train_loss"]) for row in logs[0]] == [
        (row["learning_rate"], row["train_loss"]) for row in logs[1]
    ]
