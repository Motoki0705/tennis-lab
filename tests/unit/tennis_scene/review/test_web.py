"""The RGB API binds a camera/frame and rejects mutable or outside sources."""

import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np
import pytest

from src.tennis_scene.review.web import make_server
from src.tennis_scene.scripts.visualize_component_store import Review
from src.utils.checksum import dual_sha256


def test_rgb_api_binds_source_frame_and_detects_changed_index(tmp_path: Path) -> None:
    review = Review.__new__(Review)
    review.output = tmp_path / "gallery"
    review.output.mkdir()
    review.index_path = tmp_path / "scene.json"
    review.index_path.write_text("{}")
    video = tmp_path / "cam0.mp4"
    video.write_bytes(b"unit test source")
    review.videos = {"cam0": video}
    review.frame_count = 5
    review.snapshot = {"index_sha256": dual_sha256(review.index_path),
                       "sources": [{"camera_id": "cam0", "available": True, "sha256": dual_sha256(video)}]}
    review.frame = lambda camera, index: np.full((48, 64, 3), index * 10, np.uint8)  # type: ignore[method-assign]
    server = make_server(review, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url + "/api/frame?camera=cam0&frame=3", timeout=5) as response:
            assert response.headers["X-Source-Frame"] == "3"
            assert response.headers["X-Source-Camera"] == "cam0"
            decoded = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
            assert decoded is not None
            assert decoded.shape == (48, 64, 3) and abs(float(decoded.mean()) - 30) < 1
        for query in ("camera=../../private&frame=0", "camera=cam0&frame=5", "camera=cam0&frame=-1", "camera=cam0&frame=1&frame=2"):
            with pytest.raises(HTTPError) as error:
                urlopen(url + "/api/frame?" + query, timeout=5)
            assert error.value.code == 400
        review.index_path.write_text('{"changed":true}')
        with pytest.raises(HTTPError) as error:
            urlopen(url + "/api/frame?camera=cam0&frame=0", timeout=5)
        assert error.value.code == 409
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
