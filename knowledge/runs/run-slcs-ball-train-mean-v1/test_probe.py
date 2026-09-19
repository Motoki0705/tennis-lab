"""CPU checks for train-only fit and exactly matched evaluation metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.utils.checksum import FileIntegrityError

SPEC = importlib.util.spec_from_file_location(
    "train_mean_probe", Path(__file__).with_name("probe.py")
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def example() -> dict[str, Any]:
    return {
        "video_ids": np.array(["train", "train"]),
        "clip_ids": np.array(["clip", "clip"]),
        "camera_ids": np.array(["cam", "cam"]),
        "frame_idx": np.array([[0, 1, 2], [1, 2, -1]]),
        "padding_mask": np.array([[False, False, False], [False, False, True]]),
        "ball_mask": np.array([[True, True, True], [True, True, False]]),
        "ball_weight": np.array([[1.0, 0.5, 0.0], [0.5, 0.0, 0.0]]),
        "target_ball_position": np.array(
            [
                [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [90.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [90.0, 0.0, 0.0], [999.0, 0.0, 0.0]],
            ]
        ),
    }


def test_fit_deduplicates_padding_and_zero_weight() -> None:
    mean, evidence, report = probe.fit_mean(example())
    np.testing.assert_array_equal(mean, [1.0, 0.0, 0.0])
    assert report["positive_weight_count"] == 2
    assert report["unique_valid_count"] == 3
    assert report["weight_sum"] == 1.5
    assert evidence["frame_idx"].tolist() == [0, 1]
    np.testing.assert_allclose(report["constant_m"], mean * probe.SCALE)


@pytest.mark.parametrize(
    "key,value",
    [
        ("target_ball_position", [4.0, 0.0, 0.0]),
        ("ball_mask", False),
        ("ball_weight", 0.7),
    ],
)
def test_duplicate_disagreement_rejected(key: str, value: Any) -> None:
    arrays = example()
    arrays[key][1, 0] = value
    with pytest.raises(ValueError):
        probe.fit_mean(arrays)


def test_distinct_cameras_remain_separate() -> None:
    arrays = example()
    arrays["camera_ids"][1] = "alt"
    mean, _, report = probe.fit_mean(arrays)
    np.testing.assert_allclose(mean, [1.5, 0, 0])
    assert report["positive_weight_count"] == 3


def test_no_positive_weights_rejected() -> None:
    arrays = example()
    arrays["ball_weight"][:] = 0
    with pytest.raises(ValueError, match="No positive"):
        probe.fit_mean(arrays)


def test_eval_leakage_and_changed_target_rejected() -> None:
    arrays = example()
    with pytest.raises(ValueError, match="leakage"):
        probe.validate_evaluation(arrays, arrays, {"train"})
    expected = {key: value.copy() for key, value in arrays.items()}
    arrays["target_ball_position"][0, 0, 0] = 2
    with pytest.raises(ValueError, match="target_ball_position"):
        probe.validate_evaluation(arrays, expected, set())


def test_model_metric_recalculation_and_unweighted_window_scoring() -> None:
    arrays = example()
    arrays["pred_ball_position"] = np.zeros_like(arrays["target_ball_position"])
    arrays["ball_pos_error_m"] = np.linalg.norm(
        arrays["target_ball_position"] * probe.SCALE, axis=-1
    )
    headline = float(arrays["ball_pos_error_m"][arrays["ball_mask"]].mean())
    result = probe.score(arrays, np.zeros(3), headline)
    assert result["valid_window_occurrences"] == 5
    assert result["model_error_m"] == result["fixed_train_mean_error_m"]
    assert result["model_error_m"]["mean"] == pytest.approx(
        (3 + 90 + 3 + 90) / 5 * probe.SCALE[0]
    )
    assert result["prediction_std_xyz_m"] == [0, 0, 0]
    with pytest.raises(ValueError, match="headline"):
        probe.score(arrays, np.zeros(3), headline + 1)
    arrays["ball_pos_error_m"][0, 0] = 1
    with pytest.raises(ValueError, match="Saved model ball error"):
        probe.score(arrays, np.zeros(3), headline)


def test_failure_status_and_existing_directory_not_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "diagnostic"
    with pytest.raises(FileIntegrityError, match="File could not be verified"):
        probe.run(tmp_path / "missing.yaml", [], output)
    before = (output / "status.json").read_text()
    assert "failed" in before
    with pytest.raises(FileExistsError):
        probe.run(tmp_path / "missing.yaml", [], output)
    assert (output / "status.json").read_text() == before
