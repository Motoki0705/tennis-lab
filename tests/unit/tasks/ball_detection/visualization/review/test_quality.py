"""Review counts and navigation preserve missing/estimated label semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.tasks.ball_detection.generate_dataset.frame_store.clip import SourceFrame
from src.tasks.ball_detection.visualization.inference.service import (
    DetectionRequestError,
    DetectionService,
)
from src.tasks.base.visualization.detection.web import create_detection_app
from tests.support.tasks.ball_detection.store import ball, frame, write_store_clip


def quality_service(tmp_path: Path) -> DetectionService:
    directory = tmp_path / "data/ball_detection/test-v1"
    write_store_clip(directory, "meiji/video/clip/cam0", [
        frame(0, ball()),
        frame(1, ball("interpolated")),
        frame(2, ball("occlusion_estimated")),
        frame(3, ball("unresolved", None)),
        frame(4, ball("out_of_frame", None)),
        frame(5),
        frame(6, annotated=False),
        frame(7, ball(), ball("unresolved", None, "b002")),
        frame(8, ball(), ball(track="b002")),
        SourceFrame(9, True, False, True, "unlabeled", (ball(),)),
    ], source="meiji")
    write_store_clip(directory, "chat_annotation/video-test/clip", [
        frame(0, ball("out_of_frame", None)), frame(1),
    ], source="chat_annotation", split="test")
    # Generated summaries are not the source of truth for the review counts.
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["counts"] = {"frames": 999, "clips": 777}
    metadata_path.write_text(json.dumps(metadata))
    return DetectionService(tmp_path, data_root=tmp_path / "data")


def test_catalog_counts_frames_separately_from_instances_and_stale_metadata(tmp_path: Path) -> None:
    service = quality_service(tmp_path)
    overview = service.catalog()["datasets"][0]["overview"]
    assert overview["version"] == "test-v1"
    assert overview["clips"] == 2
    assert overview["counts"]["frames"] == 12
    assert {key: overview["counts"][key] for key in (
        "scored_positive", "scored_negative", "reference", "unreviewed",
    )} == {"scored_positive": 3, "scored_negative": 4, "reference": 4, "unreviewed": 1}
    assert overview["counts"]["observed"] == 4
    assert overview["point_counts"]["observed"] == 5
    assert overview["counts"]["context"] == overview["counts"]["segment_break"] == 1
    assert overview["splits"]["train"] == {"clips": 1, "frames": 10, "groups": 1}
    assert overview["splits"]["test"] == {"clips": 1, "frames": 2, "groups": 1}
    sources = {source["id"]: source for source in overview["sources"]}
    assert sources["meiji"]["counts"]["frames"] == 10
    assert sources["chat_annotation"]["splits"]["test"]["frames"] == 2


def test_review_jump_positions_and_frame_states_are_row_aligned(tmp_path: Path) -> None:
    service = quality_service(tmp_path)
    scene = "store/test-v1::meiji/video/clip/cam0"
    review = service.review(scene)
    assert review["source"] == "meiji" and review["split"] == "train"
    assert review["positions"]["reference"] == [1, 2, 3, 7]
    assert review["positions"]["unresolved"] == [3, 7]
    assert review["positions"]["scored_negative"] == [4, 5]
    assert review["positions"]["unreviewed"] == [6]
    assert review["positions"]["context"] == [9]
    items = service.preview(scene, count=10)["items"]
    assert items[3]["review"]["point_kinds"] == ["unresolved"]
    assert not items[3]["gt"]["points"][0]["visible"]
    assert items[4]["review"]["supervision"] == "scored_negative"
    assert items[6]["review"]["supervision"] == "unreviewed"
    assert items[6]["gt"]["points"] == []
    assert items[7]["review"]["supervision"] == "reference"
    assert items[7]["review"]["point_kinds"] == ["observed", "unresolved"]
    # Context/segment flags do not silently change the canonical supervision.
    assert items[9]["review"] == {
        "supervision": "scored_positive", "point_kinds": ["observed"],
        "context": True, "segment_break": True, "event": "unlabeled", "time_seconds": 0.3,
    }


def test_clip_filters_compose_before_pagination_and_keep_source_metadata(tmp_path: Path) -> None:
    service = quality_service(tmp_path)
    result = service.scenes("store/test-v1", source="meiji", split="train", review_state="unresolved")
    assert result["total"] == 1
    assert result["items"][0]["review"]["counts"]["unresolved"] == 2
    assert service.scenes("store/test-v1", offset=1, source="meiji", split="train", review_state="unresolved")["items"] == []
    assert service.scenes("store/test-v1", split="test", review_state="reference")["total"] == 0
    assert service.scenes("store/test-v1", source="chat_annotation", review_state="scored_negative")["total"] == 1
    with pytest.raises(DetectionRequestError, match="Unknown source"):
        service.scenes("store/test-v1", source="unknown")
    with pytest.raises(DetectionRequestError, match="Unknown split"):
        service.scenes("store/test-v1", split="unknown")
    with pytest.raises(DetectionRequestError, match="Unknown review_state"):
        service.scenes("store/test-v1", review_state="unknown")


def test_http_exposes_explicit_review_state_without_gpu_or_checkpoint(tmp_path: Path) -> None:
    service = quality_service(tmp_path)
    client = TestClient(create_detection_app(service, task="ball_detection", mode="review", service_config={}))
    scenes = client.get("/api/scenes", params={"dataset": "store/test-v1", "source": "meiji", "review_state": "unreviewed"})
    assert scenes.status_code == 200
    scene = scenes.json()["items"][0]["id"]
    response = client.get("/api/review", params={"scene": scene})
    assert response.status_code == 200
    assert response.json()["positions"]["unreviewed"] == [6]
    assert client.get("/api/scenes", params={"dataset": "store/test-v1", "split": "invalid"}).status_code == 422
    assert client.get("/static/review.mjs").status_code == 200


def test_catalog_refresh_replaces_annotation_counts_and_jump_targets(tmp_path: Path) -> None:
    service = quality_service(tmp_path)
    scene = "store/test-v1::meiji/video/clip/cam0"
    assert service.review(scene)["positions"]["unresolved"] == [3, 7]
    write_store_clip(tmp_path / "data/ball_detection/test-v1", "meiji/video/clip/cam0", [frame(0)], source="meiji")
    overview = service.catalog()["datasets"][0]["overview"]
    assert overview["counts"]["frames"] == 3
    assert overview["counts"]["reference"] == 0
    assert service.review(scene)["positions"]["unresolved"] == []
