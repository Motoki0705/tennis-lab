"""Review keeps partial, unlocated and unstored states distinct on CPU."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.tasks.player_detection.data.detection_dataset import select_detection_frames
from src.tasks.player_detection.generate_dataset.builder import build_dataset
from src.tasks.player_detection.review.service import PlayerReviewService, StoreReview
from src.tasks.player_detection.review.web import create_app
from tests.support.tasks.player_detection.chat_root import SyntheticRoot


@pytest.fixture
def review(synthetic_root: SyntheticRoot) -> StoreReview:
    directory = build_dataset(synthetic_root.config)
    service = PlayerReviewService(directory.parent.parent)
    return service.review(directory.name)


def test_review_preserves_unresolved_and_unstored_frames(review: StoreReview) -> None:
    clip = review.store.clips[0]
    frame = review.frame(clip.clip_id, 2)
    assert frame["reviewed"] is True
    assert frame["selection_reason"] == "unresolved_player"
    unresolved = frame["annotations"][1]
    assert unresolved["bbox_source"] == "unresolved"
    assert unresolved["bbox_xyxy"] is None
    assert unresolved["visible_bbox_xyxy"] is None
    assert unresolved["visible_box_eligible"] is False
    assert unresolved["default_training_box"] is False
    assert frame["annotations"][0]["visible_box_eligible"] is True
    assert frame["annotations"][0]["default_training_box"] is False
    assert "NaN" not in json.dumps(frame, allow_nan=False)
    timeline = review.timeline(clip.clip_id)
    assert timeline["missing_ranges"] == [[0, 0], [3, 3], [5, 5]]
    assert timeline["unstored_frames"] == 3
    assert [frame["frame_index"] for frame in timeline["frames"]] == [1, 2, 4]
    with pytest.raises(FileNotFoundError, match="absence is not a negative"):
        review.frame(clip.clip_id, 0)


def test_review_eligibility_matches_training_selection_and_retains_amodal_boxes(review: StoreReview) -> None:
    selected = select_detection_frames(review.store, "train", review.selection, frame_stride=1)
    assert [row for row, reason in enumerate(review.reasons) if reason == "eligible"] == selected.frames.tolist()
    clip = review.store.clips[0]
    frame = review.frame(clip.clip_id, 4)
    bbox = frame["annotations"][0]
    assert bbox["truncated"] is True
    assert bbox["bbox_xyxy"] == [80.0, 20.0, 120.0, 70.0]
    assert bbox["visible_bbox_xyxy"] == [80.0, 20.0, 96.0, 64.0]
    assert frame["selection_reason"] == "eligible"
    assert review.summary()["counts"]["unresolved"] == 1


def test_review_http_is_readonly_and_has_no_arbitrary_file_or_prediction_access(review: StoreReview) -> None:
    service = PlayerReviewService(review.store.directory.parent.parent)
    with TestClient(create_app(service)) as client:
        params = {"dataset": review.store.directory.name, "clip": review.store.clips[0].clip_id, "frame": 1}
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in review.store.directory.rglob("*") if p.is_file()}
        response = client.get("/api/frame", params=params)
        assert response.status_code == 200
        assert response.json()["source_frame_index"] == int(review.store.frames["source_frame_index"][0])
        jpeg = client.get("/api/image", params=params)
        assert jpeg.headers["content-type"] == "image/jpeg"
        decoded = cv2.imdecode(np.frombuffer(jpeg.content, np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded, review.store.read_bgr(0))
        assert client.get("/api/frame", params={**params, "frame": 0}).status_code == 404
        assert client.get("/api/image", params={**params, "dataset": "../../etc"}).status_code == 404
        assert client.get("/api/clips", params={"dataset": params["dataset"], "flag": "unresolved"}).json()["total"] == 1
        assert client.get("/api/clips", params={"dataset": params["dataset"], "flag": "invented"}).status_code == 422
        assert client.post("/api/infer", json={}).status_code == 405
        assert client.post("/api/frame", json={}).status_code == 405
        assert client.get("/api/catalog", headers={"host": "external.invalid"}).status_code == 400
        assert client.get("/static/not-an-asset.txt").status_code == 404
        after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in review.store.directory.rglob("*") if p.is_file()}
        assert before == after


def test_catalogue_reports_missing_or_invalid_stores_without_substitute_data(tmp_path: Path) -> None:
    empty = PlayerReviewService(tmp_path)
    assert empty.catalog()["datasets"] == []
    invalid = tmp_path / "player_detection/broken"
    invalid.mkdir(parents=True)
    (invalid / "metadata.json").write_text('{"schema_version":"unknown"}')
    service = PlayerReviewService(tmp_path)
    assert service.catalog()["datasets"][0]["available"] is False
    assert "Unsupported player store schema" in service.catalog()["datasets"][0]["reason"]
    with pytest.raises(FileNotFoundError, match="unavailable"):
        service.review("broken")


def test_catalogue_rejects_version_symlink_outside_data_root(review: StoreReview, tmp_path: Path) -> None:
    data_root = tmp_path / "elsewhere"
    base = data_root / "player_detection"
    base.mkdir(parents=True)
    (base / "escaped").symlink_to(review.store.directory, target_is_directory=True)
    service = PlayerReviewService(data_root)
    assert service.catalog()["datasets"][0]["available"] is False
    assert "escapes" in service.catalog()["datasets"][0]["reason"]
