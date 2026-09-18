"""BLCS review data must reproduce the stored 2D projections for all forms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.tasks.base.visualization.review.camera import parse_cameras
from src.tasks.base.visualization.review.court import court_keypoints
from src.tasks.blcs.visualization.review.dataset_service import (
    BLCSDatasetReviewService,
)
from src.tasks.blcs.visualization.review.web import create_dataset_app
from src.utils.projection.camera_projector import project_points

DATA_ROOT = Path("/home/kamimura/projects/tennis-lab/data")
SCENE = "scene_000000"
FORMS = (
    "single_object",
    "multi_object",
    "single_object_broadcast",
    "multi_object_broadcast",
    "multi_object_camera_view_v2",
    "single_object_camera_view_v2",
)

pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "blcs").is_dir(), reason="BLCS dataset is unavailable"
)


def scene_dir(form: str) -> Path:
    return DATA_ROOT / "blcs" / form / "scenes" / SCENE


def load_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def project(record: Any, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = record.to_camera()
    uv, in_front = project_points(
        camera, torch.as_tensor(np.asarray(points), dtype=torch.float32)
    )
    uv = (uv / torch.tensor([camera.w, camera.h], dtype=torch.float32)).numpy()
    in_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] <= 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= 1)
    )
    return uv, (in_front.numpy() & in_bounds)


@pytest.mark.parametrize("form", FORMS)
def test_court_keypoints_reproject_exactly(form: str) -> None:
    directory = scene_dir(form)
    meta = load_json(directory / "meta.json")
    cameras = parse_cameras(load_json(directory / "scalars.json"))
    keypoints = court_keypoints(meta["court_config"]["net_post_offset_x"])
    for index, record in enumerate(cameras):
        uv, visible = project(record, keypoints)
        saved = np.load(directory / f"cam_{index}_court_kp_uv.npy")
        saved_vis = np.load(directory / f"cam_{index}_court_kp_vis.npy")
        order = np.asarray(
            meta["court_keypoint_views"][index]["semantic_to_physical"]
        )
        physical = np.empty_like(uv)
        physical[order] = saved
        physical_vis = np.empty_like(visible)
        physical_vis[order] = saved_vis
        assert np.abs(physical - uv).max() < 1e-5
        assert np.array_equal(physical_vis, visible)


@pytest.mark.parametrize("form", FORMS)
def test_ball_reprojects_exactly(form: str) -> None:
    directory = scene_dir(form)
    cameras = parse_cameras(load_json(directory / "scalars.json"))
    positions = np.load(directory / "ball_pos_world.npy")
    multi = positions.ndim == 3
    presence = np.load(directory / "ball_present.npy") if multi else None
    points = positions.reshape(-1, 3)
    for index, record in enumerate(cameras):
        uv, visible = project(record, points)
        saved = np.load(directory / f"cam_{index}_ball_uv.npy").reshape(-1, 2)
        saved_vis = np.load(directory / f"cam_{index}_ball_vis.npy").reshape(-1)
        selected = visible.copy()
        if presence is not None:
            selected &= presence.reshape(-1)
        assert np.array_equal(selected, saved_vis)
        assert np.abs(uv[selected] - saved[selected]).max() < 1e-5


def test_api_shapes_and_errors() -> None:
    service = BLCSDatasetReviewService(DATA_ROOT, forms=["single_object", "multi_object"])
    client = TestClient(create_dataset_app(service))

    catalog = client.get("/api/catalog").json()
    assert catalog["task"] == "blcs"
    assert catalog["entity"] == "ball"
    assert catalog["skeleton"] is None
    assert {form["name"] for form in catalog["forms"]} == {"single_object", "multi_object"}

    scenes = client.get("/api/scenes", params={"form": "single_object"}).json()
    assert scenes["form"] == "single_object"
    assert scenes["scenes"][0] == "scene_000000"

    document = client.get(
        "/api/scene", params={"form": "single_object", "scene": SCENE}
    ).json()
    assert document["form"] == "single_object"
    assert document["mode"] == "single"
    assert document["entity"]["kind"] == "ball"
    assert document["entity"]["slots"] == 1
    assert document["entity"]["joint_count"] == 1
    assert document["entity"]["presence"] is False
    assert document["coordinate_frame"] == "physical_court_v1"
    assert len(document["cameras"]) == 6
    assert len(document["court"]["keypoints"]) == 20

    revision = document["revision"]
    response = client.get(
        "/api/scene/buffer",
        params={"form": "single_object", "scene": SCENE, "revision": revision},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    frames = document["entity"]["frames"]
    assert len(response.content) == frames * 3 * 4

    multi = client.get(
        "/api/scene", params={"form": "multi_object", "scene": SCENE}
    ).json()
    slots = multi["entity"]["slots"]
    frames = multi["entity"]["frames"]
    assert multi["entity"]["presence"] is True
    multi_buffer = client.get(
        "/api/scene/buffer",
        params={"form": "multi_object", "scene": SCENE, "revision": multi["revision"]},
    ).content
    assert len(multi_buffer) == slots * frames * 3 * 4 + slots * frames

    stale = client.get(
        "/api/scene",
        params={"form": "single_object", "scene": SCENE, "revision": "0" * 20},
    )
    assert stale.status_code == 409

    escaped = client.get(
        "/api/scene", params={"form": "single_object", "scene": "../escape"}
    )
    assert escaped.status_code in {404, 422}
    assert client.get("/api/scenes", params={"form": "../escape"}).status_code == 422
    assert client.get("/static/unknown.js").status_code == 404
