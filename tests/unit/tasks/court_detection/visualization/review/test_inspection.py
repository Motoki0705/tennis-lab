"""Inspection must retain missing visibility and raw annotation semantics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from src.tasks.base.visualization.detection.web import create_detection_app
from src.tasks.court_detection.visualization.inference.service import DetectionService
from src.tasks.court_detection.visualization.review.datasets import (
    CourtDatasetCatalog,
    GroundTruthMasks,
)
from src.tasks.court_detection.visualization.review.inspection import (
    record_states,
    sample_inspection,
)

pytestmark = pytest.mark.unit


def _inspection(catalog: CourtDatasetCatalog, dataset: str) -> dict[str, Any]:
    entry = catalog.entry(dataset)
    record = catalog.records(dataset)[0]
    raw = catalog.input_for(entry).load(record)
    return sample_inspection(
        entry,
        record,
        raw,
        catalog.input_for(entry).spec,
        GroundTruthMasks(seg=None, line=None, semantic_line=None),
    )


def test_real_visibility_is_unknown_even_when_kp_is_supervised(
    catalog: CourtDatasetCatalog,
) -> None:
    review = _inspection(catalog, "tennis_court_detector/train")
    assert review["stored_visibility"] == "not_provided"
    assert review["pose_teacher"] is False
    assert review["kp_supervised"] == 14
    assert review["counts"] == {"in_frame_visibility_unknown": 14}
    for point in review["points"]:
        assert point["renderer_visible"] is None
        assert point["stored_in_frame"] is None
        assert point["in_front"] is None
    assert all(not target["stored"] for target in review["targets"])
    assert all(not target["available"] for target in review["targets"])


def test_out_of_frame_is_saved_coordinate_not_missing_annotation(
    catalog: CourtDatasetCatalog,
) -> None:
    entry = catalog.entry("tennis_court_detector/train")
    record = catalog.records(entry.id)[0]
    raw = catalog.input_for(entry).load(record)
    channels = raw.keypoint_channels
    assert channels is not None
    points = channels.points_xy.clone()
    points[0, 0, 0] = -2.0
    visible = channels.point_visible.clone()
    visible[0, 0] = False
    raw = replace(
        raw,
        keypoint_channels=replace(channels, points_xy=points, point_visible=visible),
    )
    review = sample_inspection(
        entry,
        record,
        raw,
        catalog.input_for(entry).spec,
        GroundTruthMasks(None, None, None),
    )
    row = cast(list[dict[str, Any]], review["points"])[0]
    assert row["x"] == -2.0 and row["state"] == "out_of_frame"
    assert row["kp_supervised"] is False
    assert review["annotation_present"] is True
    assert len(cast(list[Any], review["points"])) == 14


def test_synthetic_target_identity_is_separate_from_semantic_channel(
    catalog: CourtDatasetCatalog,
) -> None:
    review = _inspection(catalog, "synthetic_court/B00/train")
    assert review["target_court"] == "court-b"
    assert review["stored_visibility"] == "renderer"
    assert review["pose_teacher"] is True
    assert len(review["points"]) == 14
    assert {p["physical_index"] for p in review["points"]} == set(range(14))
    for point in review["points"]:
        assert point["kp_supervised"] == (
            point["in_front"] and point["stored_in_frame"] and point["renderer_visible"]
        )
    assert review["source_split"] == "train"


def test_renderer_invisible_point_does_not_become_missing_annotation(
    catalog: CourtDatasetCatalog,
) -> None:
    entry = catalog.entry("synthetic_court/B00/train")
    record = catalog.records(entry.id)[0]
    raw = catalog.input_for(entry).load(record)
    channels = raw.keypoint_channels
    assert channels is not None
    import copy

    payload = dict(record.payload)
    manifest = copy.deepcopy(payload["manifest_record"])
    court = next(
        c
        for c in manifest["projection"]["courts"]
        if c["court_instance_id"] == "court-b"
    )
    stored = court["classes"][0]["points"][0]
    stored.update(uv=[10.0, 10.0], in_frame=True, in_front=True, renderer_visible=False)
    payload["manifest_record"] = manifest
    points = channels.points_xy.clone()
    points[0, 0] = points.new_tensor([10.0, 10.0])
    visible = channels.point_visible.clone()
    visible[0, 0] = False
    review = sample_inspection(
        entry,
        replace(record, payload=payload),
        replace(
            raw,
            keypoint_channels=replace(
                channels, points_xy=points, point_visible=visible
            ),
        ),
        catalog.input_for(entry).spec,
        GroundTruthMasks(None, None, None),
    )
    row = cast(list[dict[str, Any]], review["points"])[0]
    assert row["state"] == "renderer_not_visible"
    assert row["in_frame"] is True and row["kp_supervised"] is False
    assert review["annotation_present"] is True
    assert "renderer_not_visible" in record_states(replace(record, payload=payload))


def test_unannotated_is_explicit_and_not_court_absence(
    catalog: CourtDatasetCatalog,
) -> None:
    entry = catalog.entry("tennis_court_detector/train")
    record = catalog.records(entry.id)[0]
    raw = replace(catalog.input_for(entry).load(record), keypoint_channels=None)
    review = sample_inspection(
        entry,
        record,
        raw,
        catalog.input_for(entry).spec,
        GroundTruthMasks(None, None, None),
    )
    assert review["annotation_present"] is False
    assert review["points"] == []
    assert "コート不在と判断しない" in str(review["flags"])


def test_raw_annotation_is_original_sparse_record(review_project: Path) -> None:
    service = DetectionService(project_root=review_project)
    for dataset in ("tennis_court_detector/train", "synthetic_court/B00/val"):
        scene = service.scenes(dataset, limit=1)["items"][0]["id"]
        saved = service.annotation(scene)
        annotation = cast(dict[str, Any], saved["annotation"])
        if dataset.startswith("tennis"):
            assert len(annotation["kps"]) == 14
            assert "visibility" not in annotation
            assert annotation["split"] == "train"
        else:
            assert annotation["schema"] == "canonical_court_sample_v3"
            assert annotation["split"] == "validation"
            assert (
                annotation["target_court"]["binding"]["court_instance_id"] == "court-b"
            )
            # Raw retains both courts; review/derived targets select one.
            assert len(annotation["projection"]["courts"]) == 2


def test_dataset_map_and_read_only_routes(review_project: Path) -> None:
    service = DetectionService(project_root=review_project)
    client = TestClient(
        create_detection_app(
            service, task="court_detection", mode="review", service_config={}
        )
    )
    catalog = client.get("/api/catalog").json()
    assert len(catalog["court_review"]["families"]) == 2
    synthetic = catalog["court_review"]["families"][1]
    assert synthetic["splits"]["val"]["count"] == 1
    assert synthetic["stored_count"] == 3
    assert synthetic["schema"] == "canonical_court_dataset_v3"
    scene = service.scenes("tennis_court_detector/train")["items"][0]["id"]
    assert (
        client.get("/api/court/annotation", params={"scene": scene}).status_code == 200
    )
    assert (
        client.get(
            "/api/court/annotation", params={"scene": "../etc/passwd"}
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/court/annotation",
            params={"scene": "tennis_court_detector/train::../etc"},
        ).status_code
        == 404
    )
    assert (
        client.post("/api/court/annotation", params={"scene": scene}).status_code == 405
    )
    assert client.get("/task-static/review.mjs").status_code == 200
    assert client.get("/task-static/review.css").status_code == 200
    assert client.get("/task-static/inspection.py").status_code == 404
    assert client.get("/task-static/%2e%2e%2finspection.py").status_code == 404
    assert (
        client.get(
            "/api/scenes",
            params={
                "dataset": "tennis_court_detector/train",
                "sample_state": "out_of_frame",
                "source": "tracknet",
            },
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/api/scenes",
            params={
                "dataset": "tennis_court_detector/train",
                "sample_state": "occluded",
            },
        ).status_code
        == 422
    )


def test_candidate_filter_combines_search_and_pagination(
    review_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = DetectionService(project_root=review_project)
    assert (
        service.scenes("tennis_court_detector/train", sample_state="out_of_frame")[
            "total"
        ]
        == 0
    )
    assert (
        service.scenes(
            "tennis_court_detector/train", sample_state="duplicate_coordinates"
        )["total"]
        == 0
    )
    records = service.datasets.records("tennis_court_detector/train")
    payload = dict(records[0].payload)
    coordinates = list(cast(tuple[tuple[float, float], ...], payload["keypoints"]))
    coordinates[0] = (-1.0, 10.0)
    coordinates[1] = coordinates[0]
    payload["keypoints"] = tuple(coordinates)
    assert record_states(replace(records[0], payload=payload)) == {
        "out_of_frame",
        "duplicate_coordinates",
    }
    candidate = replace(records[0], payload=payload)
    monkeypatch.setattr(service.datasets, "records", lambda _: (candidate,))
    assert (
        service.scenes(
            "tennis_court_detector/train",
            search=candidate.sample_id,
            sample_state="out_of_frame",
        )["total"]
        == 1
    )
    assert (
        service.scenes(
            "tennis_court_detector/train",
            search="no-match",
            sample_state="out_of_frame",
        )["total"]
        == 0
    )
    page = service.scenes(
        "tennis_court_detector/train",
        offset=1,
        limit=1,
        sample_state="duplicate_coordinates",
    )
    assert page == {"total": 1, "items": []}
    with pytest.raises(ValueError, match="Unsupported Court sample state"):
        service.scenes("tennis_court_detector/train", sample_state="missing")


def test_unreadable_source_remains_disabled_without_breaking_other_datasets(
    review_project: Path,
) -> None:
    manifest = (
        review_project / "data/court_detection/tennis_court_detector-v1/dataset.json"
    )
    manifest.write_text("{invalid JSON")
    catalog = DetectionService(project_root=review_project).catalog()
    family = catalog["court_review"]["families"][0]
    assert family["reason"] and family["stored_count"] is None
    assert all(not split["available"] for split in family["splits"].values())
    assert catalog["court_review"]["families"][1]["splits"]["train"]["available"]
