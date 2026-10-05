"""Inspection contracts use tiny array fixtures, never screenshot evidence."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.tasks.base.generate_dataset import (
    CourtKeypointArtifactMetadata,
    build_court_view_record,
    inject_court_keypoint_artifact_metadata,
    inject_scene_court_keypoint_metadata,
    resolve_court_keypoint_contract,
)
from src.tasks.base.visualization.review.court import court_keypoints
from src.tasks.blcs.generate_dataset.io.dataset_io import BLCS_DATASET_SCHEMA_ID
from src.tasks.blcs.visualization.review import (
    BLCSDatasetReviewService,
    create_dataset_app,
)
from src.tasks.blcs.visualization.review.inspection import (
    normalization_diagnostic,
    shot_events,
)
from src.utils.projection.camera_projector import make_look_at_camera, project_points
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
    normalize_court_position,
    normalize_court_velocity,
)


@pytest.fixture
def review_root(tmp_path: Path) -> Path:
    form = tmp_path / "blcs" / "single_object"
    scene = form / "scenes" / "scene_000000"
    scene.mkdir(parents=True)
    contract = resolve_court_keypoint_contract("physical_v1")
    header = CourtKeypointArtifactMetadata.from_contract(
        contract, dataset_schema_id=BLCS_DATASET_SCHEMA_ID
    )
    root_meta = inject_court_keypoint_artifact_metadata(
        {
            "config": {
                "generation": {"mode": "single_object"},
                "court_keypoints": {"selector": "physical_v1"},
            }
        },
        header,
        location="fixture root",
    )
    (form / "meta.json").write_text(json.dumps(root_meta), encoding="utf-8")
    camera = make_look_at_camera(
        (0.0, -20.0, 6.0),
        look_at=(0.0, 0.0, 0.0),
        image_size=(1280, 720),
        hfov_deg=60.0,
    )
    view = build_court_view_record(
        camera_id="cam_0", camera_center_court_m=camera.C.tolist(), contract=contract
    )
    meta: dict[str, object] = {
        "scene_id": "scene_000000",
        "num_frames": 4,
        "num_cameras": 1,
        "fps_out": 30,
        "court_coordinate_normalization": court_coordinate_normalization_metadata(),
        "court_config": {"net_post_offset_x": 0.492},
        "shots": [
            {
                "shot_index": 0,
                "t_start": 0,
                "t_net": -1,
                "t_bounce1": 1,
                "t_bounce2": 3,
                "t_bounce3": 8,
                "t_return": 2,
            },
            {
                "shot_index": 1,
                "t_start": 2,
                "t_net": -1,
                "t_bounce1": -1,
                "t_bounce2": -1,
                "t_bounce3": -1,
                "t_return": -1,
            },
        ],
    }
    meta = inject_scene_court_keypoint_metadata(
        meta, header, [view], location="fixture scene"
    )
    (scene / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    params = {
        "C": camera.C.tolist(),
        "R": camera.R.tolist(),
        "f": camera.f,
        "cx": camera.cx,
        "cy": camera.cy,
        "w": camera.w,
        "h": camera.h,
    }
    (scene / "scalars.json").write_text(
        json.dumps({"num_balls": 1, "num_cameras": 1, "cam_0_params": params}),
        encoding="utf-8",
    )
    position = np.asarray(
        [[0, 0, 1], [1, 2, 2], [30, 0, 1], [0, -40, 1]], dtype=np.float32
    )
    velocity = np.asarray([[8, 20, 3]] * 4, dtype=np.float32)
    for name, value in {
        "ball_pos_world": position,
        "ball_pos_norm": normalize_court_position(position),
        "ball_vel_world": velocity,
        "ball_vel_norm": normalize_court_velocity(velocity),
    }.items():
        np.save(scene / f"{name}.npy", value)
    for name, points in (("ball", position), ("court_kp", court_keypoints(0.492))):
        pixels, front = project_points(camera, torch.as_tensor(points))
        uv = pixels.numpy() / [camera.w, camera.h]
        vis = front.numpy() & (uv >= 0).all(axis=-1) & (uv <= 1).all(axis=-1)
        np.save(scene / f"cam_0_{name}_uv.npy", uv.astype(np.float32))
        np.save(scene / f"cam_0_{name}_vis.npy", vis)
    for split in ("train", "val", "test"):
        (form / f"{split}.txt").write_text(
            "scene_000000\n" if split == "train" else "", encoding="utf-8"
        )
    return tmp_path


def _scene(root: Path) -> Path:
    return root / "blcs" / "single_object" / "scenes" / "scene_000000"


def _client(root: Path) -> TestClient:
    return TestClient(create_dataset_app(BLCSDatasetReviewService(root)))


def _evidence(client: TestClient) -> dict[str, Any]:
    params = {"form": "single_object", "scene": "scene_000000"}
    document = client.get("/api/scene", params=params).json()
    response = client.get(
        "/api/inspection", params={**params, "revision": document["revision"]}
    )
    assert response.status_code == 200, response.text
    evidence: dict[str, Any] = response.json()
    return evidence


def test_saved_uv_teachers_and_units_are_preserved(review_root: Path) -> None:
    evidence = _evidence(_client(review_root))
    assert evidence["source"]["rgb"] == "not_saved"
    assert evidence["contract"]["court_saved_points"] == 20
    assert evidence["contract"]["normalization"]["scale_xyz_m"] == [11.885] * 3
    assert evidence["dataset"] == {
        "scene_count": 1,
        "split_counts": {"train": 1, "val": 0, "test": 0},
        "scene_splits": ["train"],
    }
    np.testing.assert_array_equal(
        evidence["ball"]["position_m"],
        np.load(_scene(review_root) / "ball_pos_world.npy"),
    )
    camera = evidence["cameras"][0]
    assert len(camera["court"]["saved_uv"]) == 20
    assert camera["ball"]["summary"]["max_error_px"] < 0.001
    assert camera["ball"]["summary"]["visibility_mismatches"] == 0
    assert camera["ball"]["saved_uv"][2][0] > 1  # keep the outside observation
    assert camera["ball"]["saved_visibility"][2] is False
    assert camera["ball"]["projected_uv"][3] == [
        None,
        None,
    ]  # behind-camera UV is undefined
    assert camera["ball"]["error_px"][3] is None
    assert evidence["ball"]["normalization"]["position"]["status"] == "ok"


def test_reprojection_detects_saved_outlier_and_visibility_disagreement(
    review_root: Path,
) -> None:
    scene = _scene(review_root)
    uv = np.load(scene / "cam_0_ball_uv.npy")
    uv[0, 0] += 0.1
    np.save(scene / "cam_0_ball_uv.npy", uv)
    vis = np.load(scene / "cam_0_ball_vis.npy")
    vis[1] = False
    np.save(scene / "cam_0_ball_vis.npy", vis)
    camera = _evidence(_client(review_root))["cameras"][0]
    assert camera["ball"]["error_px"][0] == pytest.approx(128, abs=0.001)
    assert camera["ball"]["summary"]["visibility_mismatches"] == 1
    assert camera["ball"]["saved_visibility"][1] is False
    assert camera["ball"]["expected_visibility"][1] is True


def test_missing_observations_remain_missing_while_3d_is_available(
    review_root: Path,
) -> None:
    scene = _scene(review_root)
    for name in (
        "cam_0_ball_uv",
        "cam_0_court_kp_vis",
        "ball_vel_world",
        "ball_vel_norm",
    ):
        (scene / f"{name}.npy").unlink()
    evidence = _evidence(_client(review_root))
    camera = evidence["cameras"][0]
    assert camera["ball"]["saved_uv"] is None
    assert camera["ball"]["error_px"] is None
    assert camera["court"]["saved_visibility"] is None
    assert camera["court"]["summary"]["visibility_mismatches"] is None
    assert camera["ball"]["projected_uv"][0] is not None
    assert evidence["ball"]["velocity_mps"] is None
    assert evidence["ball"]["normalization"]["velocity"]["status"] == "missing"


def test_nonfinite_observations_are_json_null_with_a_count(review_root: Path) -> None:
    scene = _scene(review_root)
    uv = np.load(scene / "cam_0_ball_uv.npy")
    uv[1, 0] = np.nan
    np.save(scene / "cam_0_ball_uv.npy", uv)
    camera = _evidence(_client(review_root))["cameras"][0]
    assert camera["ball"]["saved_uv"][1][0] is None
    assert camera["ball"]["error_px"][1] is None
    assert camera["ball"]["summary"]["invalid_saved_uv"] == 1


@pytest.mark.parametrize(
    "name,value",
    [
        ("cam_0_ball_vis", np.ones(4, dtype=np.float32)),
        ("cam_0_court_kp_uv", np.zeros((14, 2), dtype=np.float32)),
    ],
)
def test_invalid_array_contract_is_an_explicit_error(
    review_root: Path, name: str, value: np.ndarray
) -> None:
    np.save(_scene(review_root) / f"{name}.npy", value)
    client = _client(review_root)
    params = {"form": "single_object", "scene": "scene_000000"}
    document = client.get("/api/scene", params=params).json()
    response = client.get(
        "/api/inspection", params={**params, "revision": document["revision"]}
    )
    assert response.status_code == 422
    assert f"{name}.npy" in response.text


def test_inspection_requires_the_same_revision_as_the_3d_view(
    review_root: Path,
) -> None:
    client = _client(review_root)
    params = {"form": "single_object", "scene": "scene_000000"}
    revision = client.get("/api/scene", params=params).json()["revision"]
    np.save(
        _scene(review_root) / "cam_0_ball_uv.npy", np.zeros((4, 2), dtype=np.float32)
    )
    assert (
        client.get(
            "/api/inspection", params={**params, "revision": revision}
        ).status_code
        == 409
    )
    assert client.get("/api/inspection", params=params).status_code == 422
    assert (
        client.get(
            "/api/inspection",
            params={**params, "scene": "../escape", "revision": revision},
        ).status_code
        == 422
    )


def test_only_current_single_object_physical_v1_can_be_reviewed(
    review_root: Path,
) -> None:
    with pytest.raises(ValueError, match="only.*single_object"):
        BLCSDatasetReviewService(review_root, forms=["multi_object"])
    root_meta = review_root / "blcs" / "single_object" / "meta.json"
    meta = json.loads(root_meta.read_text())
    meta["config"]["court_keypoints"]["selector"] = "camera_view_v2"
    root_meta.write_text(json.dumps(meta))
    assert (
        _client(review_root)
        .get("/api/scene", params={"form": "single_object", "scene": "scene_000000"})
        .status_code
        == 422
    )


def test_nonfinite_teacher_fails_before_the_3d_buffer_is_displayed(
    review_root: Path,
) -> None:
    position = np.load(_scene(review_root) / "ball_pos_world.npy")
    position[0, 0] = np.nan
    np.save(_scene(review_root) / "ball_pos_world.npy", position)
    response = _client(review_root).get(
        "/api/scene", params={"form": "single_object", "scene": "scene_000000"}
    )
    assert response.status_code == 422
    assert "finite" in response.text


def test_discarded_continuations_are_not_active_events(review_root: Path) -> None:
    events = _evidence(_client(review_root))["events"]
    first = {e["field"]: e for e in events if e["shot_index"] == 0}
    assert first["t_start"]["kind"] == "hit"
    assert first["t_bounce1"]["status"] == "on_trajectory"
    assert first["t_bounce1"]["time_seconds"] == pytest.approx(1 / 30)
    assert first["t_bounce2"]["status"] == "after_shot"
    assert first["t_bounce2"]["frame"] == 3
    assert first["t_bounce3"]["status"] == "outside_scene"
    assert first["t_return"]["status"] == "on_trajectory"  # retained boundary
    assert first["t_net"]["status"] == "not_recorded"


def test_missing_and_malformed_shots_do_not_become_invented_events() -> None:
    assert shot_events({}, frames=4, fps=30) is None
    events = shot_events({"shots": [{"t_start": 0}]}, frames=4, fps=30)
    assert events is not None and events[1]["status"] == "missing"
    assert events[0]["shot_index"] is None  # do not invent a missing saved ID
    with pytest.raises(ValueError, match="integer"):
        shot_events({"shots": [{"t_start": 0.0}]}, frames=4, fps=30)
    with pytest.raises(ValueError, match="output frame"):
        shot_events({"shots": [{"t_start": 0, "t_net": -2}]}, frames=4, fps=30)


def test_normalized_pairs_are_compared_without_replacing_saved_world_values() -> None:
    value = np.array([[5.485, 11.885, 1.07]], dtype=np.float32)
    normalized = normalize_court_position(value)
    assert normalization_diagnostic(value, normalized, velocity=False)["status"] == "ok"
    assert (
        normalization_diagnostic(value, value / [5.485, 11.885, 1.07], velocity=False)[
            "status"
        ]
        == "mismatch"
    )
    assert normalization_diagnostic(value, None, velocity=False)["status"] == "missing"


def test_assets_are_declared_for_wheels_and_shared_assets_are_preserved(
    review_root: Path,
) -> None:
    client = _client(review_root)
    index = client.get("/").text
    assert index.index("blcs-inspection.mjs") < index.index("/static/app.js")
    assert "/static/style.css" in index
    for name in ("blcs-inspection.mjs", "blcs-observation.mjs", "blcs-inspection.css"):
        assert client.get(f"/static/{name}").status_code == 200
    assert client.get("/static/unknown.mjs").status_code == 404
    config = tomllib.loads(
        (Path(__file__).resolve().parents[6] / "pyproject.toml").read_text()
    )
    assert (
        "static/*.mjs"
        in config["tool"]["setuptools"]["package-data"][
            "src.tasks.blcs.visualization.review"
        ]
    )
