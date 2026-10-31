"""Synthetic threshold-survey inputs, without datasets or checkpoints."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.tasks.plcs.scripts.analysis.analyze_skeleton_validity import (
    FrequencySurvey,
    main,
)
from tests.support.geometry.skeleton import coco_pose_m


def _archive_arrays() -> tuple[np.ndarray, np.ndarray]:
    pose = coco_pose_m().numpy()
    return np.stack([pose, np.zeros_like(pose), np.full_like(pose, np.nan)]), np.array(
        [False, False, True]
    )


def test_frequency_counts_by_bone_plane_and_axis_exclude_padding_and_use_candidates() -> (
    None
):
    pose, padding = _archive_arrays()
    survey = FrequencySurvey([0.0, 4.0], [0.0, 0.9])
    survey.update(pose, padding, chunk_size=1)
    result = survey.result()
    assert (
        result["total_frames"],
        result["nonpadding_frames"],
        result["padding_frames"],
    ) == (3, 2, 1)
    metrics = result["metrics"]
    assert metrics["bone_length_m"][0]["invalid_count"] == [1] * 12
    assert metrics["bone_length_m"][0]["invalid_fraction"] == [0.5] * 12
    assert metrics["bone_length_m"][1]["invalid_count"] == [
        2,
        2,
        2,
        2,
        2,
        1,
        2,
        1,
        2,
        2,
        1,
        1,
    ]
    assert metrics["torsion_plane_sine"][0]["invalid_count"] == [[1, 1]] * 4
    # The left forearm plane has normalized cross magnitude 4/5, independently
    # of the raw cross magnitude 12. The larger candidate must count it.
    assert metrics["torsion_plane_sine"][1]["invalid_count"] == [
        [1, 2],
        [1, 1],
        [1, 1],
        [1, 1],
    ]
    assert metrics["torsion_plane_sine"][1]["invalid_fraction"] == [
        [0.5, 1.0],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ]
    assert metrics["torso_axis_length_m"][0]["invalid_count"] == [1, 1, 1]
    assert metrics["torso_axis_length_m"][1]["invalid_count"] == [2, 2, 2]
    assert metrics["torso_plane_sine"][0]["invalid_count"] == [1, 1]
    assert metrics["bone_length_m"][0]["any_invalid_fraction"] == 0.5


def test_cli_surveys_target_prediction_exports_and_saved_scene_separately(
    tmp_path: Path,
) -> None:
    pose, padding = _archive_arrays()
    archive = tmp_path / "coco17.npz"
    np.savez(archive, pose_m=pose[None], padding_mask=padding[None, None])
    scene_dir = tmp_path / "scenes" / "sample"
    scene_dir.mkdir(parents=True)
    np.save(scene_dir / "human_kp_3d.npy", pose[:2])
    output = tmp_path / "report.json"
    assert (
        main(
            [
                "--target",
                str(archive),
                "--prediction",
                str(archive),
                "--target-scenes",
                str(scene_dir.parent),
                "--output",
                str(output),
                "--length-thresholds-m",
                "0",
                "4",
                "--plane-thresholds",
                "0",
                ".5",
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    # The report records the adopted API domain (#1539) next to the candidates.
    assert result["api_thresholds"] == {
        "min_bone_length_m": 1e-6,
        "min_plane_sine": 1e-6,
    }
    assert set(result["sources"]) == {"target", "prediction", "target_scenes"}
    assert (
        result["sources"]["target"]["metrics"]
        == result["sources"]["prediction"]["metrics"]
    )
    assert (
        result["sources"]["target_scenes"]["metrics"]
        == result["sources"]["target"]["metrics"]
    )


def test_nonfinite_observation_and_all_padding_are_errors() -> None:
    pose, padding = _archive_arrays()
    survey = FrequencySurvey([0.0], [0.0])
    with pytest.raises(ValueError, match="finite"):
        survey.update(pose, np.zeros_like(padding))
    survey = FrequencySurvey([0.0], [0.0])
    survey.update(pose, np.ones_like(padding))
    with pytest.raises(ValueError, match="No valid"):
        survey.result()


@pytest.mark.parametrize("thresholds", [[], [-1.0], [float("nan")], [0.0, 0.0]])
def test_threshold_candidates_are_explicit_and_validated(
    thresholds: list[float],
) -> None:
    with pytest.raises(ValueError, match="Threshold"):
        FrequencySurvey(thresholds, [0.0])
