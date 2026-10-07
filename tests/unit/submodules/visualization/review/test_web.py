from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.submodules.visualization.review.store import PoseReviewStore
from src.submodules.visualization.review.web import create_app


def test_web_serves_real_reader_and_only_get_routes(snapshot: Path) -> None:
    with TestClient(create_app(PoseReviewStore(snapshot))) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/meta").json()["identity_policy"] == "per_frame_v3"
        frame = client.get("/api/frame", params={"person": 7, "camera": "cam0", "frame": 3})
        assert frame.status_code == 200
        assert frame.json()["vertices"] is None
        assert frame.json()["placement"]["reason_code"] == 101
        assert client.get("/api/frame", params={"person": 7, "camera": "cam0", "frame": 5}).status_code == 422
        assert client.get("/api/image", params={"camera": "cam0", "frame": 0}).status_code == 404
        assert client.post("/api/frame").status_code == 405
