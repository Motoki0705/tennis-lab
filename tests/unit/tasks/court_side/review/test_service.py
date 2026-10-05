from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.tasks.court_side.review.service import ReviewService
from src.tasks.court_side.review.web import create_app


def test_synchronized_frame_points_and_rgb(
    stored_case: Path, saved_diagnostics: Path
) -> None:
    service = ReviewService(stored_case, diagnostics=saved_diagnostics)
    client = TestClient(create_app(service))
    response = client.get("/api/frame?case=0&frame=2")
    assert response.status_code == 200
    frame = response.json()
    assert frame["frame"] == 2 and frame["seconds"] == 2 / 60
    assert frame["observing_cameras"] == ["cam0", "cam1"]
    assert frame["evidence"]["scores"]["supports"] == [1, 0]
    for camera in ("cam0", "cam1"):
        image = client.get(f"/api/image/0/{camera}/2")
        assert image.status_code == 200
        decoded = cv2.imdecode(np.frombuffer(image.content, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == (72, 128, 3)
        assert 90 < decoded.mean() < 110


def test_unknown_camera_frame_and_mutation_rejected(stored_case: Path) -> None:
    client = TestClient(create_app(ReviewService(stored_case)))
    assert client.get("/api/frame?case=0&frame=4").status_code == 404
    assert client.get("/api/frame?case=3&frame=0").status_code == 404
    assert client.get("/api/image/0/other/2").status_code == 404
    assert client.post("/api/frame?case=0&frame=0").status_code == 405
    assert client.get("/static/other.js").status_code == 404


def test_unresolved_and_interpolated_not_claimed_as_observed(stored_case: Path) -> None:
    service = ReviewService(stored_case)
    unresolved = service.frame(0, 0)["cameras"][0]["point"]
    assert unresolved["uv_px"] is None and not unresolved["observed"]
    interpolated = service.frame(0, 3)["cameras"][0]["point"]
    assert interpolated["uv_px"] == [30, 30] and not interpolated["observed"]
    assert interpolated["kind"] == "interpolated"
    assert service.frame(0, 2)["evidence"]["state"] == "not_saved"


def test_changed_snapshot_is_rejected_even_for_cached_images(stored_case: Path) -> None:
    service = ReviewService(stored_case)
    service.image(0, "cam0", 2)
    with (stored_case / "scene.json").open("a") as stream:
        stream.write("\n")
    client = TestClient(create_app(service))
    assert client.get("/api/image/0/cam0/2").status_code == 409
    assert client.get("/api/frame?case=0&frame=2").status_code == 409


def test_missing_rgb_keeps_saved_coordinates_reviewable(stored_case: Path) -> None:
    service = ReviewService(stored_case)
    Path(service.cases[0].videos[0]["path"]).unlink()
    data = service.frame(0, 2)
    assert not data["cameras"][0]["image_available"]
    assert data["cameras"][0]["point"]["uv_px"] == [30, 30]
    client = TestClient(create_app(service))
    assert client.get("/api/image/0/cam0/2").status_code == 404
