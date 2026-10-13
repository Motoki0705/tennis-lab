"""The seed-repeat decision must use validation, including when test disagrees."""

import json
from hashlib import sha256
from pathlib import Path

import pytest

from src.tasks.blcs.experiments.final_evaluation import trained_runs
from src.tasks.blcs.experiments.issue1044 import ARMS
from src.tasks.blcs.experiments.repeat_selected import select_additional_arm


def _write_receipt(root: Path, arm: str, seed: int, score: float) -> None:
    folder = root / f"{arm}-s{seed}-u4000"
    folder.mkdir()
    checkpoint = folder / "selected.ckpt"
    checkpoint.write_bytes(f"{arm}-{seed}".encode())
    receipt = {
        "arm": arm,
        "seed": seed,
        "optimizer_updates": 4000,
        "smoke": False,
        "chunked": False,
        "gan": False,
        "selection_used_test": False,
        "protocol": "issue1044-v1",
        "validation_trajectory_rmse_m": score,
        "test_trajectory_rmse_m": 0 if arm == "parameters" else 100,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint.read_bytes()).hexdigest(),
    }
    (folder / "selection.json").write_text(json.dumps(receipt))


def _write_seed42_receipts(root: Path) -> None:
    for arm, score in zip(ARMS, (0.01, 0.02, 0.3, 0.2, 0.1, 0.4), strict=True):
        _write_receipt(root, arm, 42, score)


def test_repeat_selection_excludes_controls_and_never_reads_test(
    tmp_path: Path,
) -> None:
    _write_seed42_receipts(tmp_path)
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


@pytest.mark.parametrize("mode", ["chunked", "gan"])
@pytest.mark.parametrize("missing", [False, True])
def test_repeat_selection_rejects_nonformal_mode(
    tmp_path: Path, mode: str, missing: bool
) -> None:
    _write_seed42_receipts(tmp_path)
    path = tmp_path / "velocity-s42-u4000/selection.json"
    receipt = json.loads(path.read_text())
    if missing:
        receipt.pop(mode)
    else:
        receipt[mode] = True
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Incomplete or inconsistent"):
        select_additional_arm(tmp_path)
    assert not (tmp_path / "additional-seed-selection.json").exists()


def test_final_evaluation_accepts_ten_formal_receipts(tmp_path: Path) -> None:
    _write_seed42_receipts(tmp_path)
    for seed in (43, 44):
        for arm in ("reprojection", "rollout_velocity"):
            _write_receipt(tmp_path, arm, seed, 0.2)
    assert len(trained_runs(tmp_path)) == 10


@pytest.mark.parametrize("seed", [43, 44])
@pytest.mark.parametrize("arm", ["reprojection", "rollout_velocity"])
@pytest.mark.parametrize("mode", ["chunked", "gan"])
@pytest.mark.parametrize("missing", [False, True])
def test_final_evaluation_rejects_nonformal_confirmation_run(
    tmp_path: Path, seed: int, arm: str, mode: str, missing: bool
) -> None:
    _write_seed42_receipts(tmp_path)
    for run_seed in (43, 44):
        for run_arm in ("reprojection", "rollout_velocity"):
            _write_receipt(tmp_path, run_arm, run_seed, 0.2)
    path = tmp_path / f"{arm}-s{seed}-u4000/selection.json"
    receipt = json.loads(path.read_text())
    if missing:
        receipt.pop(mode)
    else:
        receipt[mode] = True
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Invalid completed run"):
        trained_runs(tmp_path)
