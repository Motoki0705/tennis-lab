from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.submodules.visualization.review.store import PoseReviewStore, exact_sample

from .conftest import replace_payload


def test_unsampled_frame_keeps_dense_placement_and_does_not_use_nearest_input(snapshot: Path) -> None:
    store = PoseReviewStore(snapshot)
    frame = store.frame(7, "cam0", 1)
    assert frame["request"] is None and frame["parameters"] is None
    assert frame["request_camera"] == "cam0"
    assert frame["placement"]["smpl_valid"] is True
    assert len(frame["vertices"]) == 6890
    assert frame["seconds"] == 1 / 30
    assert store.frame(7, "cam0", 0)["parameters"] is not None


def test_invalid_mask_hides_numerical_coordinates_and_mesh_but_keeps_valid_joints(snapshot: Path) -> None:
    frame = PoseReviewStore(snapshot).frame(7, "cam0", 3)
    assert frame["vertices"] is None
    assert frame["placement"]["position"] is None
    assert frame["placement"]["yaw"] is None
    assert frame["placement"]["reprojection_px"] is None
    assert frame["placement"]["reason"] == "INSUFFICIENT_JOINTS"
    assert frame["joints_3d"][10] is None
    assert frame["joint_reasons_3d"][10] == 1
    assert sum(j is not None for j in frame["joints_3d"]) == 16


def test_observed_mask_and_per_frame_identity_are_authoritative(snapshot: Path) -> None:
    store = PoseReviewStore(snapshot)
    assert store.frame(7, "cam0", 2)["observation"]["state"] == "unobserved"
    assert store.frame(7, "cam0", 4)["observation"]["state"] == "unobserved"
    frame = store.frame(7, "cam0", 0)
    assert frame["observation"]["track_id"] == 42
    assert frame["observation"]["joints"][10][2] == pytest.approx(.1)
    assert store.metadata()["identity_policy"] == "per_frame_v3"
    assert frame["observations_by_camera"]["cam0"]["state"] == "observed"
    assert store.metadata()["triangulation_rejection_codes"]["1"] == "INSUFFICIENT_VIEWS"


def test_historical_v2_identity_is_explicit_and_not_upgraded_silently(snapshot: Path) -> None:
    def historical(value: dict[str, object]) -> dict[str, object]:
        value["player_ids"] = np.array([[7]], np.int64)
        return value
    replace_payload(snapshot, "player_association", historical, version=2)
    store = PoseReviewStore(snapshot)
    assert store.metadata()["identity_policy"] == "historical_static_v2"
    assert store.frame(7, "cam0", 4)["observation"]["state"] == "observed"


def test_missing_rgb_and_optional_artifact_are_explicit(snapshot: Path) -> None:
    index = snapshot / "scene.json"
    document = json.loads(index.read_text())
    del document["artifacts"]["player_triangulation"]
    index.write_text(json.dumps(document))
    store = PoseReviewStore(snapshot)
    assert store.metadata()["cameras"][0]["media"] == "missing"
    assert store.frame(7, "cam0", 0)["joints_3d"] is None
    with pytest.raises(FileNotFoundError, match="RGB"):
        store.image("cam0", 0)


def test_timeline_does_not_bridge_rejected_frames(snapshot: Path) -> None:
    timeline = PoseReviewStore(snapshot).timeline(7, "cam0")
    assert timeline["observed"] == [True, True, False, True, False]
    assert timeline["sampled"] == [True, False, True, False, True]
    assert timeline["mesh"] == [True, True, True, False, True]


def test_corrupt_array_stops_instead_of_displaying_it(snapshot: Path) -> None:
    array = next((snapshot / "gvhmr").glob("*.npy"))
    with array.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        PoseReviewStore(snapshot)


def test_changed_snapshot_stops_before_serving_another_frame(snapshot: Path) -> None:
    store = PoseReviewStore(snapshot)
    index = snapshot / "scene.json"
    index.write_text(index.read_text() + " ")
    with pytest.raises(ValueError, match="snapshot changed"):
        store.frame(7, "cam0", 1)


def test_superseded_input_is_not_paired_with_current_body(snapshot: Path) -> None:
    index = snapshot / "scene.json"
    document = json.loads(index.read_text())
    del document["artifacts"]["body_view_selection"]
    index.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="superseded"):
        PoseReviewStore(snapshot)


def test_superseded_ancestor_outside_displayed_families_is_rejected(snapshot: Path) -> None:
    from src.utils.checksum import dual_sha256

    index = snapshot / "scene.json"
    document = json.loads(index.read_text())
    path = snapshot / "calibration.json"
    descriptor = {"node": "court_calibration", "artifact_id": "calibration", "output_schema": "local_court_calibration", "output_version": 1, "source_sha256": "a" * 64, "dependencies": {"old_court": {"artifact_id": "no-longer-adopted"}}}
    path.write_text(json.dumps(descriptor))
    reference = {"artifact_id": "calibration", "schema": "local_court_calibration", "version": 1, "path": path.name, "sha256": dual_sha256(path)}
    document["artifacts"]["court_calibration"] = reference
    selected = document["artifacts"]["body_view_selection"]
    selected_path = snapshot / selected["path"]
    selected_descriptor = json.loads(selected_path.read_text())
    selected_descriptor["dependencies"]["calibration"] = reference
    selected_path.write_text(json.dumps(selected_descriptor))
    selected["sha256"] = dual_sha256(selected_path)
    gvhmr = document["artifacts"]["gvhmr"]
    gvhmr_path = snapshot / gvhmr["path"]
    gvhmr_descriptor = json.loads(gvhmr_path.read_text())
    gvhmr_descriptor["dependencies"]["selection"] = selected
    gvhmr_path.write_text(json.dumps(gvhmr_descriptor))
    gvhmr["sha256"] = dual_sha256(gvhmr_path)
    placement = document["artifacts"]["body_placement"]
    placement_path = snapshot / placement["path"]
    placement_descriptor = json.loads(placement_path.read_text())
    placement_descriptor["dependencies"]["recovered"] = gvhmr
    placement_path.write_text(json.dumps(placement_descriptor))
    placement["sha256"] = dual_sha256(placement_path)
    index.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="superseded"):
        PoseReviewStore(snapshot)


def test_frame_axis_mismatch_is_rejected(snapshot: Path) -> None:
    def wrong_axis(value: dict[str, object]) -> dict[str, object]:
        value["player_ids"] = np.array([[[7, 7]]], np.int64)
        return value
    replace_payload(snapshot, "player_association", wrong_axis)
    with pytest.raises(ValueError, match="frame axis"):
        PoseReviewStore(snapshot)


def test_source_frames_are_exact_and_ordered() -> None:
    frames = np.array([0, 2, 4], np.int64)
    assert exact_sample(frames, 2) == 1
    assert exact_sample(frames, 3) is None
    with pytest.raises(ValueError, match="increasing"):
        exact_sample(np.array([2, 2], np.int64), 2)


@pytest.mark.parametrize("person,camera,frame", [(8, "cam0", 0), (7, "unknown", 0), (7, "cam0", 5), (7, "cam0", -1)])
def test_invalid_selection_is_rejected(snapshot: Path, person: int, camera: str, frame: int) -> None:
    with pytest.raises(ValueError):
        PoseReviewStore(snapshot).frame(person, camera, frame)
