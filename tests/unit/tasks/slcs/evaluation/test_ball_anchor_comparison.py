"""Hand-computed anchor classes, frame distances and physical velocity units."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.tasks.slcs.evaluation.ball_anchor_comparison import (
    _anchor_layout,
    compare_ball_anchors,
    save_ball_anchor_comparison,
)
from src.utils.schema.court import COURT_COORD_SCALE_XYZ
from tests.unit.tasks.slcs.evaluation.test_ball_transition_comparison import _write


def _arrays(multiplier: float = 2.0) -> dict[str, np.ndarray]:
    n, t = 4, 30
    lengths = np.array([30, 30, 30, 5])
    starts = np.array([100, 200, 300, 400])
    padding = np.arange(t)[None] >= lengths[:, None]
    observed: np.ndarray = np.zeros((n, t), dtype=bool)
    observed[0, [2, 4]] = True
    observed[1, 3] = True
    observed[3, 0] = True
    observed[padding] = True  # Must never create an anchor on the right.
    target = np.zeros((n, t, 3))
    target[..., 0] = (
        np.arange(t)[None]
        / np.array([30, 30, 60, 60])[:, None]
        / COURT_COORD_SCALE_XYZ[0]
    )
    return {
        "scene_ids": np.array(
            ["v0/c@cam@100", "v0/c@cam@200", "v1/c@cam@300", "v1/c@cam@400"]
        ),
        "video_ids": np.array(["v0", "v0", "v1", "v1"]),
        "clip_ids": np.array(["v0/c", "v0/c", "v1/c", "v1/c"]),
        "camera_ids": np.array(["cam"] * n),
        "window_start": starts,
        "window_length": lengths,
        "frame_idx": np.where(padding, -1, starts[:, None] + np.arange(t)[None]),
        "padding_mask": padding,
        "ball_mask": ~padding,
        "player_mask": ~padding[:, None],
        "ball_weight": np.where(padding, 0.0, 0.25),
        "player_weight": np.where(padding[:, None], 0.0, 0.5),
        "ball_observed": observed,
        "player_observed": ~padding[:, None],
        "target_ball_position": target,
        "pred_ball_position": target * multiplier,
        "target_player_position": np.zeros((n, 1, t, 3)),
        "target_player_rotation": np.ones((n, 1, t, 2)),
    }


def _row(
    report: dict[str, Any],
    anchor_class: str,
    bucket: str = "all",
    *,
    group: str = "all",
    transition: str | None = None,
) -> dict[str, Any]:
    return next(
        row
        for row in report["position_rows" if transition is None else "boundary_rows"]
        if row["anchor_class"] == anchor_class
        and row["distance_bucket"] == bucket
        and row["group"] == group
        and (transition is None or row["transition"] == transition)
    )


def test_layout_single_both_none_and_padding() -> None:
    labels, distances = _anchor_layout(_arrays())
    assert labels[0, :6].tolist() == [
        "missing_right_only",
        "missing_right_only",
        "observed",
        "missing_both_sides",
        "observed",
        "missing_left_only",
    ]
    assert distances[0, :6].tolist() == [2, 1, None, 1, None, 1]
    assert labels[1, :5].tolist() == ["missing_right_only"] * 3 + [
        "observed",
        "missing_left_only",
    ]
    assert distances[1, :5].tolist() == [3, 2, 1, None, 1]
    assert (labels[2] == "missing_no_anchor").all()
    assert distances[2].tolist() == [None] * 30
    assert labels[3, :5].tolist() == ["observed"] + ["missing_left_only"] * 4
    assert (labels[3, 5:] == "padding").all()
    assert distances[3, 5:].tolist() == [None] * 25
    # Unequal two-anchor distances choose the nearer side, not the mean.
    arrays = _arrays()
    arrays["ball_observed"][0] = False
    arrays["ball_observed"][0, [2, 8]] = True
    _, distances = _anchor_layout(arrays)
    assert distances[0, 3:8].tolist() == [1, 2, 3, 2, 1]


def test_hand_computed_position_distance_bins_and_velocity(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "a", _arrays())
    candidate = _write(tmp_path / "b", _arrays(1))
    # Model/checkpoint identity is intentionally allowed to differ.
    for path, checkpoint in ((baseline, "base.ckpt"), (candidate, "other.ckpt")):
        config = OmegaConf.load(path / "evaluation_config.yaml")
        config.evaluate.checkpoint = checkpoint
        OmegaConf.save(config, path / "evaluation_config.yaml")
    report = compare_ball_anchors(baseline, candidate)
    assert report["split"] == "val"
    for label, path in (("baseline", baseline), ("candidate", candidate)):
        for name, digest in report["sources"][label]["sha256"].items():
            assert digest == hashlib.sha256((path / name).read_bytes()).hexdigest()
    right = _row(report, "missing_right_only")
    assert right["baseline"]["count"] == 5
    assert right["baseline"]["mean"] == pytest.approx(4 / 150)
    assert right["baseline"]["p95"] == pytest.approx(
        np.percentile(np.array([0, 1, 0, 1, 2]) / 30, 95)
    )
    assert right["baseline"]["max"] == pytest.approx(2 / 30)
    assert right["candidate_minus_baseline"]["mean"] == pytest.approx(-4 / 150)
    assert _row(report, "missing_both_sides")["baseline"]["mean"] == pytest.approx(0.1)
    assert _row(report, "missing_left_only", ">=25")["baseline"]["count"] == 3
    assert _row(report, "missing_left_only", ">=25")["baseline"][
        "mean"
    ] == pytest.approx(86 / 90)
    assert _row(report, "missing_left_only", "9-24")["baseline"]["count"] == 32
    assert _row(report, "missing_left_only", "2-8")["baseline"]["count"] == 17
    assert _row(report, "observed")["baseline"]["count"] == 4
    assert _row(report, "missing_no_anchor")["baseline"]["count"] == 30
    assert {
        row["distance_bucket"]
        for row in report["position_rows"]
        if row["anchor_class"] in {"observed", "missing_no_anchor"}
    } == {"all"}
    edge = _row(report, "missing_left_only", "1", transition="observed_to_missing")
    for name, expected in (
        ("velocity_vector_error_mps", 1),
        ("pred_speed_mps", 2),
        ("teacher_speed_mps", 1),
        ("signed_speed_bias_mps", 1),
    ):
        assert edge["metrics"][name]["baseline"]["count"] == 3
        for stat in ("mean", "p95", "max"):
            assert edge["metrics"][name]["baseline"][stat] == pytest.approx(expected)
    assert edge["metrics"]["velocity_vector_error_mps"]["candidate_minus_baseline"][
        "mean"
    ] == pytest.approx(-1)
    assert (
        _row(report, "missing_right_only", transition="missing_to_observed")["metrics"][
            "pred_speed_mps"
        ]["baseline"]["count"]
        == 2
    )
    for transition in ("observed_to_missing", "missing_to_observed"):
        assert (
            _row(report, "missing_both_sides", transition=transition)["metrics"][
                "pred_speed_mps"
            ]["baseline"]["count"]
            == 1
        )
    assert _row(
        report, "missing_left_only", group="v1", transition="observed_to_missing"
    )["metrics"]["velocity_vector_error_mps"]["baseline"]["mean"] == pytest.approx(1)
    empty = _row(report, "missing_left_only", "2-8", transition="observed_to_missing")
    assert empty["metrics"]["pred_speed_mps"]["baseline"] == {
        "count": 0,
        "mean": None,
        "p95": None,
        "max": None,
    }
    assert (
        empty["metrics"]["pred_speed_mps"]["candidate_minus_baseline"]["mean"] is None
    )
    assert _row(report, "missing_right_only", ">=25")["baseline"]["mean"] is None
    json.dumps(report, allow_nan=False)


def test_teacher_masks_exclude_statistics_not_observation_anchors(
    tmp_path: Path,
) -> None:
    arrays = _arrays()
    arrays["ball_mask"][0, 2] = False
    arrays["ball_weight"][0, 2] = 0
    report = compare_ball_anchors(
        _write(tmp_path / "a", arrays), _write(tmp_path / "b", arrays)
    )
    assert _row(report, "observed")["baseline"]["count"] == 3
    assert _row(report, "missing_both_sides")["baseline"]["count"] == 1
    assert (
        _row(report, "missing_both_sides", transition="observed_to_missing")["metrics"][
            "pred_speed_mps"
        ]["baseline"]["count"]
        == 0
    )
    # Wrong direction has zero speed bias but nonzero vector error.
    candidate = {key: value.copy() for key, value in arrays.items()}
    candidate["pred_ball_position"] = -candidate["target_ball_position"]
    report = compare_ball_anchors(
        tmp_path / "a", _write(tmp_path / "reverse", candidate)
    )
    metrics = _row(report, "missing_left_only", transition="observed_to_missing")[
        "metrics"
    ]
    assert metrics["signed_speed_bias_mps"]["candidate"]["mean"] == pytest.approx(0)
    assert metrics["velocity_vector_error_mps"]["candidate"]["mean"] == pytest.approx(2)


@pytest.mark.parametrize(
    "issue",
    [
        "teacher",
        "observed",
        "weight",
        "shape",
        "dtype",
        "nan",
        "inf",
        "padding",
        "noncontiguous",
        "fps",
        "condition",
        "split",
    ],
)
def test_reject_invalid_or_unpaired_sources(tmp_path: Path, issue: str) -> None:
    base, other = _arrays(), _arrays(1)
    if issue == "teacher":
        other["target_ball_position"][0, 0, 0] += 1
    elif issue == "observed":
        other["ball_observed"][0, 0] = True
    elif issue == "weight":
        other["ball_weight"][0, 0] = 0.75
    elif issue == "shape":
        other["pred_ball_position"] = other["pred_ball_position"][..., :2]
    elif issue == "dtype":
        other["ball_observed"] = other["ball_observed"].astype(np.int64)
    elif issue in {"nan", "inf"}:
        other["pred_ball_position"][0, 0, 0] = float(issue)
    elif issue == "padding":
        other["padding_mask"][0, 3] = True
    elif issue == "noncontiguous":
        base["frame_idx"][0, 2] += 1
        other["frame_idx"][0, 2] += 1
    a, b = _write(tmp_path / "a", base), _write(tmp_path / "b", other)
    if issue == "fps":
        motion = json.loads((b / "motion.json").read_text())
        motion["fps_by_clip"][0]["fps"] = 0
        (b / "motion.json").write_text(json.dumps(motion))
    if issue in {"condition", "split"}:
        config = OmegaConf.load(b / "evaluation_config.yaml")
        config.evaluate["input_mode" if issue == "condition" else "split"] = (
            "no_rgb" if issue == "condition" else "train"
        )
        OmegaConf.save(config, b / "evaluation_config.yaml")
    with pytest.raises(ValueError):
        compare_ball_anchors(a, b)


def test_only_validation_and_new_absolute_json_cli(tmp_path: Path) -> None:
    a, b = _write(tmp_path / "a", _arrays()), _write(tmp_path / "b", _arrays(1))
    output = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.tasks.slcs.scripts.compare_ball_anchors",
            "--baseline",
            str(a),
            "--candidate",
            str(b),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.read_text())["split"] == "val"
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        save_ball_anchor_comparison(a, b, output=output)
    assert output.read_bytes() == original
    dangling = tmp_path / "symlink.json"
    dangling.symlink_to(tmp_path / "missing.json")
    with pytest.raises(FileExistsError):
        save_ball_anchor_comparison(a, b, output=dangling)
    with pytest.raises(ValueError, match="absolute JSON"):
        save_ball_anchor_comparison(a, b, output=Path("relative.json"))
    for path in (a, b):
        config = OmegaConf.load(path / "evaluation_config.yaml")
        config.evaluate.split = "test"
        OmegaConf.save(config, path / "evaluation_config.yaml")
    with pytest.raises(ValueError, match="validation split"):
        save_ball_anchor_comparison(a, b, output=tmp_path / "test.json")
    assert not (tmp_path / "test.json").exists()
