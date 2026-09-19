"""Physical units, paired provenance, and visibility-stratified motion checks."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.tasks.slcs.evaluation.ball_transition_comparison import (
    _PAIRED_KEYS,
    compare_ball_transitions,
    save_ball_transition_comparison,
)
from src.tasks.slcs.evaluation.motion import summarize_motion
from src.utils.schema.court import COURT_COORD_SCALE_XYZ


def _arrays(multiplier: float) -> dict[str, np.ndarray]:
    teacher = np.zeros((2, 5, 3))
    for i, fps in enumerate((30.0, 60.0)):
        teacher[i, :, 0] = (
            np.array([0.0, 1.0, 3.0, 6.0, 10.0]) / fps / COURT_COORD_SCALE_XYZ[0]
        )
    return {
        "scene_ids": np.array(["v0/c@cam@0", "v1/c@cam@0"]),
        "video_ids": np.array(["v0", "v1"]),
        "clip_ids": np.array(["v0/c", "v1/c"]),
        "camera_ids": np.array(["cam", "cam"]),
        "window_start": np.zeros(2, dtype=np.int64),
        "window_length": np.full(2, 5, dtype=np.int64),
        "frame_idx": np.tile(np.arange(5), (2, 1)),
        "padding_mask": np.zeros((2, 5), dtype=bool),
        "ball_mask": np.ones((2, 5), dtype=bool),
        "player_mask": np.ones((2, 1, 5), dtype=bool),
        "ball_weight": np.full((2, 5), 0.25),
        "player_weight": np.ones((2, 1, 5)),
        "ball_observed": np.array([[True, True, False, False, True]] * 2),
        "player_observed": np.ones((2, 1, 5), dtype=bool),
        "target_ball_position": teacher,
        "pred_ball_position": teacher * multiplier,
        "target_player_position": np.zeros((2, 1, 5, 3)),
        "pred_player_position": np.zeros((2, 1, 5, 3)),
        "target_player_rotation": np.ones((2, 1, 5, 2)),
    }


def _motion() -> dict[str, Any]:
    return {
        "position_representation": "normalized_court",
        "position_scale_xyz_m": list(COURT_COORD_SCALE_XYZ),
        "fps_by_clip": [
            {"video_id": "v0", "clip_id": "v0/c", "fps": 30.0},
            {"video_id": "v1", "clip_id": "v1/c", "fps": 60.0},
        ],
    }


def _write(
    root: Path,
    arrays: dict[str, np.ndarray],
    motion: dict[str, Any] | None = None,
    *,
    mode: str = "full",
) -> Path:
    root.mkdir(parents=True)
    np.savez(root / "eval_arrays.npz", **cast(dict[str, Any], arrays))
    (root / "motion.json").write_text(
        json.dumps(_motion() if motion is None else motion)
    )
    OmegaConf.save(
        OmegaConf.create({"evaluate": {"input_mode": mode, "split": "val"}}),
        root / "evaluation_config.yaml",
    )
    return root


def _row(
    report: dict[str, Any],
    stratum: str = "all",
    subset: str = "all",
    group: str = "all",
) -> dict[str, Any]:
    return next(
        row
        for row in report["rows"]
        if row["stratum"] == stratum
        and row["subset"] == subset
        and row["group"] == group
    )


def test_real_fps_matches_motion_and_four_visibility_patterns(tmp_path: Path) -> None:
    base, candidate = _arrays(2.0), _arrays(1.0)
    report = compare_ball_transitions(
        _write(tmp_path / "base", base),
        _write(tmp_path / "new", candidate),
        fast_speed_mps=2.5,
    )
    row = _row(report)
    assert row["baseline"]["count"] == 8
    assert row["baseline"]["pred_speed_mean_mps"] == pytest.approx(5.0)
    assert row["baseline"]["teacher_speed_mean_mps"] == pytest.approx(2.5)
    assert row["baseline"]["signed_speed_bias_mean_mps"] == pytest.approx(2.5)
    assert row["candidate"]["velocity_vector_error_max_mps"] == 0
    assert row["candidate_minus_baseline"][
        "velocity_vector_error_mean_mps"
    ] == pytest.approx(-2.5)
    reference = summarize_motion(
        base,
        {("v0", "v0/c"): 30.0, ("v1", "v1/c"): 60.0},
        position_representation="normalized_court",
    )["rows"][0]["entities"]["ball"]["velocity"]
    for stat in ("mean", "p95", "max"):
        assert row["baseline"][f"pred_speed_{stat}_mps"] == pytest.approx(
            reference["pred_norm"][stat]
        )
    for stratum, speed in (
        ("both_observed", 1.0),
        ("observed_to_missing", 2.0),
        ("both_missing", 3.0),
        ("missing_to_observed", 4.0),
    ):
        assert _row(report, stratum)["candidate"]["count"] == 2
        assert _row(report, stratum)["candidate"][
            "pred_speed_mean_mps"
        ] == pytest.approx(speed)
    for video in ("v0", "v1"):
        assert _row(report, group=video)["candidate"][
            "teacher_speed_mean_mps"
        ] == pytest.approx(2.5)
    assert _row(report, subset="fast_teacher")["baseline"]["count"] == 4
    empty = _row(report, "both_observed", "fast_teacher")
    assert empty["candidate"]["count"] == 0
    assert all(
        value is None for key, value in empty["candidate"].items() if key != "count"
    )
    assert empty["candidate_minus_baseline"]["pred_speed_mean_mps"] is None


@pytest.mark.parametrize(
    "exclusion,count", [("gap", 4), ("mask", 4), ("padding", 6), ("empty", 0)]
)
def test_only_consecutive_valid_nonpadding_pairs_count(
    tmp_path: Path, exclusion: str, count: int
) -> None:
    base = _arrays(2.0)
    if exclusion == "gap":
        base["frame_idx"][:, 2] += 1
    elif exclusion == "mask":
        base["ball_mask"][:, 2] = False
        base["ball_weight"][:, 2] = 0
    elif exclusion == "padding":
        base["padding_mask"][:, -1] = True
        base["window_length"][:] = 4
        base["frame_idx"][:, -1] = -1
        for entity in ("ball", "player"):
            base[f"{entity}_mask"][..., -1] = False
            base[f"{entity}_weight"][..., -1] = 0
    else:
        base["ball_mask"][:] = False
        base["ball_weight"][:] = 0
    report = compare_ball_transitions(
        _write(tmp_path / "a", base), _write(tmp_path / "b", base), fast_speed_mps=3.0
    )
    assert _row(report)["baseline"]["count"] == count


@pytest.mark.parametrize("key", _PAIRED_KEYS)
def test_mismatched_teachers_inputs_and_metadata_rejected(
    tmp_path: Path, key: str
) -> None:
    base, candidate = _arrays(2.0), _arrays(1.0)
    value = candidate[key]
    if value.dtype.kind == "U":
        value.flat[0] = "changed"
    elif value.dtype == bool:
        value.flat[0] = not value.flat[0]
    else:
        value.flat[0] += 0.1 if value.dtype.kind == "f" else 1
    with pytest.raises(ValueError):
        compare_ball_transitions(
            _write(tmp_path / "a", base),
            _write(tmp_path / "b", candidate),
            fast_speed_mps=3.0,
        )


@pytest.mark.parametrize(
    "issue",
    [
        "fps_mismatch",
        "fps_missing",
        "fps_extra",
        "fps_duplicate",
        "fps_zero",
        "fps_nan",
        "representation",
        "scale",
        "condition",
    ],
)
def test_metadata_rejects_unit_rate_and_condition_ambiguity(
    tmp_path: Path, issue: str
) -> None:
    motion = _motion()
    mode = "full"
    if issue == "fps_mismatch":
        motion["fps_by_clip"][0]["fps"] = 25.0
    elif issue == "fps_missing":
        motion["fps_by_clip"].pop()
    elif issue == "fps_extra":
        motion["fps_by_clip"].append(
            {"video_id": "extra", "clip_id": "extra/c", "fps": 30.0}
        )
    elif issue == "fps_duplicate":
        motion["fps_by_clip"].append(motion["fps_by_clip"][0])
    elif issue in {"fps_zero", "fps_nan"}:
        motion["fps_by_clip"][0]["fps"] = 0 if issue == "fps_zero" else float("nan")
    elif issue == "representation":
        motion["position_representation"] = "meters"
    elif issue == "scale":
        motion["position_scale_xyz_m"] = [1.0, 1.0, 1.0]
    else:
        mode = "no_rgb"
    with pytest.raises(ValueError):
        compare_ball_transitions(
            _write(tmp_path / "a", _arrays(2.0)),
            _write(tmp_path / "b", _arrays(1.0), motion, mode=mode),
            fast_speed_mps=3.0,
        )


def test_new_absolute_json_output_only(tmp_path: Path) -> None:
    base = _write(tmp_path / "base", _arrays(2.0))
    other = _write(tmp_path / "other", _arrays(1.0))
    output = tmp_path / "report.json"
    result = save_ball_transition_comparison(
        base, other, output=output, fast_speed_mps=3.0
    )
    assert json.loads(result.read_text())["fast_speed_mps"] == 3.0
    with pytest.raises(FileExistsError):
        save_ball_transition_comparison(base, other, output=output, fast_speed_mps=3.0)
    with pytest.raises(ValueError, match="absolute JSON"):
        save_ball_transition_comparison(
            base, other, output=Path("relative.json"), fast_speed_mps=3.0
        )


@pytest.mark.parametrize("threshold", [0.0, -1.0, float("nan"), float("inf")])
def test_threshold_is_required_finite_positive(
    tmp_path: Path, threshold: float
) -> None:
    with pytest.raises(ValueError, match="threshold"):
        compare_ball_transitions(tmp_path, tmp_path, fast_speed_mps=threshold)
