from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.tasks.court_side.review.store import StoreCase
from src.utils.checksum import dual_sha256
from tests.unit.tasks.court_side.review.conftest import write_artifact


def test_missing_decision_does_not_become_zero_cost(stored_case: Path) -> None:
    case = StoreCase(stored_case)
    assert case.decision is None
    assert not case.summary()["frame_scores_saved"]
    assert case.summary()["point_sources"]["cam0"]["schema"] == "ball_detections v1"


def test_historical_side_is_explicitly_missing_scores(stored_case: Path) -> None:
    index = stored_case / "scene.json"
    document = json.loads(index.read_text())
    document["artifacts"]["court_side"] = write_artifact(
        stored_case,
        "court_side",
        "court_side",
        2,
        {
            "camera_ids": ["cam0", "cam1"],
            "reference_camera": "cam0",
            "view_half_turns": [False, True],
        },
    )
    index.write_text(json.dumps(document))
    decision = StoreCase(stored_case).decision
    assert decision is not None
    assert (
        decision["hypotheses"] is None
        and decision["frames"] is None
        and decision["margin"] is None
    )
    assert decision["schema_version"] == 2


def test_current_points_do_not_filter_on_low_presence(stored_case: Path) -> None:
    index = stored_case / "scene.json"
    document = json.loads(index.read_text())
    document["artifacts"]["ball_points/cam0"] = write_artifact(
        stored_case,
        "ball_points/cam0",
        "ball_points",
        2,
        {
            "camera_id": "cam0",
            "source_size_wh": [128, 72],
            "frame_indices": np.arange(4, dtype=np.int64),
            "uv_px": np.full((4, 2), 20, np.float32),
            "presence_probability": np.full(4, 0.001, np.float64),
        },
    )
    index.write_text(json.dumps(document))
    points = StoreCase(stored_case).points["cam0"]
    assert points is not None and points["observed"].all()
    assert points["schema"] == "ball_points v2"
    assert (points["uv"] == 20).all()


def test_unsupported_current_points_do_not_fall_back_to_detector(
    stored_case: Path,
) -> None:
    index = stored_case / "scene.json"
    document = json.loads(index.read_text())
    document["artifacts"]["ball_points/cam0"] = write_artifact(
        stored_case, "ball_points/cam0", "ball_points", 99, {}
    )
    index.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Unsupported ball_points"):
        StoreCase(stored_case)


def test_side_input_lineage_cannot_show_unrelated_points(stored_case: Path) -> None:
    index = stored_case / "scene.json"
    document = json.loads(index.read_text())
    reference = write_artifact(
        stored_case,
        "court_side",
        "court_side",
        2,
        {
            "camera_ids": ["cam0", "cam1"],
            "reference_camera": "cam0",
            "view_half_turns": [False, True],
        },
    )
    descriptor_path = stored_case / reference["path"]
    descriptor = json.loads(descriptor_path.read_text())
    descriptor["dependencies"] = {
        "ball_cam0": document["artifacts"]["ball_detection/cam0"]
    }
    descriptor_path.write_text(json.dumps(descriptor))
    reference["sha256"] = dual_sha256(descriptor_path)
    document["artifacts"]["court_side"] = reference
    document["artifacts"]["ball_points/cam0"] = write_artifact(
        stored_case,
        "ball_points/cam0",
        "ball_points",
        2,
        {
            "camera_id": "cam0",
            "source_size_wh": [128, 72],
            "frame_indices": np.arange(4, dtype=np.int64),
            "uv_px": np.full((4, 2), 20, np.float32),
            "presence_probability": np.full(4, 0.001, np.float64),
        },
    )
    index.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="side artifact lineage"):
        StoreCase(stored_case)


def test_corrupt_array_is_rejected(stored_case: Path) -> None:
    path = stored_case / "components/ball_detection_cam0/array_0001.npy"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        StoreCase(stored_case)


def test_stale_dependency_is_rejected(stored_case: Path) -> None:
    index = stored_case / "scene.json"
    document = json.loads(index.read_text())
    reference = document["artifacts"]["ball_detection/cam0"]
    path = stored_case / reference["path"]
    descriptor = json.loads(path.read_text())
    descriptor["dependencies"] = {"old": {"artifact_id": "superseded"}}
    path.write_text(json.dumps(descriptor))
    reference["sha256"] = dual_sha256(path)
    index.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="superseded"):
        StoreCase(stored_case)


def test_snapshot_cannot_silently_follow_index_changes(stored_case: Path) -> None:
    case = StoreCase(stored_case)
    with (stored_case / "scene.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        case.assert_unchanged()
