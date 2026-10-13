"""The seed-repeat decision must use validation, including when test disagrees."""

import json
from pathlib import Path

import pytest

from src.tasks.blcs.experiments.issue1044 import ARMS
from src.tasks.blcs.experiments.repeat_selected import select_additional_arm


def test_repeat_selection_excludes_controls_and_never_reads_test(
    tmp_path: Path,
) -> None:
    for arm, score in zip(ARMS, (0.01, 0.02, 0.3, 0.2, 0.1, 0.4), strict=True):
        folder = tmp_path / f"{arm}-s42-u4000"
        folder.mkdir()
        receipt = {
            "arm": arm,
            "seed": 42,
            "optimizer_updates": 4000,
            "smoke": False,
            "selection_used_test": False,
            "protocol": "issue1044-v1",
            "validation_trajectory_rmse_m": score,
            "test_trajectory_rmse_m": 0 if arm == "parameters" else 100,
        }
        (folder / "selection.json").write_text(json.dumps(receipt))
    result = select_additional_arm(tmp_path)
    assert result["selected_arm"] == "rollout_velocity"
    assert result["used_test"] is False
    assert select_additional_arm(tmp_path) == result
    path = tmp_path / "velocity-s42-u4000/selection.json"
    changed = json.loads(path.read_text())
    changed["validation_trajectory_rmse_m"] = 0.05
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Previously frozen"):
        select_additional_arm(tmp_path)
