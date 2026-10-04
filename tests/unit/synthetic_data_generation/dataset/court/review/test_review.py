"""Review must bind images to labels and never serve files outside an owner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.court.review.service import (
    ReviewService,
    contained_file,
)
from src.synthetic_data_generation.dataset.court.review.web import create_app
from src.synthetic_data_generation.dataset.court.schema import CourtDatasetSchemaVersion
from src.synthetic_data_generation.visualization.overlays import render_court_overlay
from src.synthetic_data_generation.visualization.sources import CourtSourceFrame


@pytest.fixture
def owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ReviewService, dict[str, Any], str]:
    scene = tmp_path / "B00"
    root = scene / "datasets/court"
    root.mkdir(parents=True)
    (scene / "alignment").mkdir()
    (root / "dataset.json").write_text("{}")
    (scene / "alignment/alignment.json").write_text("{}")
    sample = {
        "sample_id": "sample-1",
        "width": 80,
        "height": 40,
        "rgb": "rgb.npy",
        "labels": "label.json",
        "projection": {"courts": []},
        "camera": {
            "camera_id": "sample-1",
            "intrinsics": np.eye(3).ravel().tolist(),
            "camera_to_scene": np.eye(4).ravel().tolist(),
        },
        "trajectory_id": "trajectory-1",
        "view_id": "view-1",
        "trajectory_frame_index": 0,
        "trajectory_group_id": "group-1",
        "split": "train",
    }
    np.save(root / "rgb.npy", np.full((40, 80, 3), 0.5, dtype=np.float32))
    (root / "label.json").write_text(json.dumps(sample))
    service = ReviewService(tmp_path)
    data = {
        "samples": {"sample-1": sample},
        "rejected_samples": {
            "reject-1": dict(
                sample,
                sample_id="reject-1",
                projection=None,
                reasons=["ambiguous_camera_relative_near_far:court-000"],
            )
        },
        "schema": "canonical_court_dataset_v3",
        "summary": {"id": "B00"},
        "image_store": None,
    }
    monkeypatch.setattr(service, "_cached_load", lambda scene, revision: data)
    return service, sample, service.revision("B00")


def test_overlay_reads_float_rgb_and_preserves_aspect(
    owner: tuple[ReviewService, dict[str, Any], str],
) -> None:
    service, _, revision = owner
    result = service.overlay("B00", revision, "sample-1", 40)
    image = cv2.imdecode(np.frombuffer(result, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape == (20, 40, 3)
    assert np.allclose(image, 128, atol=1)


def test_labels_must_match_selected_sample(
    owner: tuple[ReviewService, dict[str, Any], str],
) -> None:
    service, sample, revision = owner
    label = dict(sample, sample_id="different")
    (service.root / "B00/datasets/court/label.json").write_text(json.dumps(label))
    with pytest.raises(ValueError, match="binding mismatch"):
        service.overlay("B00", revision, "sample-1", 40)


def test_nan_rgb_is_an_explicit_error(
    owner: tuple[ReviewService, dict[str, Any], str],
) -> None:
    service, _, revision = owner
    np.save(
        service.root / "B00/datasets/court/rgb.npy",
        np.full((40, 80, 3), np.nan, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="RGB must be finite"):
        service.overlay("B00", revision, "sample-1", 40)


def test_changed_revision_rejects_cached_image(
    owner: tuple[ReviewService, dict[str, Any], str],
) -> None:
    service, _, revision = owner
    service.overlay("B00", revision, "sample-1", 40)
    (service.root / "B00/datasets/court/dataset.json").write_text('{"new": true}')
    with pytest.raises(RuntimeError, match="changed"):
        service.overlay("B00", revision, "sample-1", 40)


def test_file_and_scene_symlinks_cannot_escape(tmp_path: Path) -> None:
    root = tmp_path / "scenes"
    root.mkdir()
    outside = tmp_path / "secret.json"
    outside.write_text("{}")
    (root / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        contained_file(root, "link")
    (root / "external").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        ReviewService(root).scene_root("external")


def test_api_reports_stale_revision_and_unknown_sample(
    owner: tuple[ReviewService, dict[str, Any], str],
) -> None:
    service, _, revision = owner
    with TestClient(create_app(service)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/unknown.js").status_code == 404
        assert (
            client.get("/api/scenes/B00", params={"revision": "old"}).status_code == 409
        )
        assert (
            client.get(
                "/api/scenes/B00/images/missing", params={"revision": revision}
            ).status_code
            == 404
        )
        assert (
            client.get(
                "/api/scenes/B00/images/sample-1",
                params={"revision": revision, "width": -1},
            ).status_code
            == 422
        )
        response = client.get(
            "/api/scenes/B00/images/sample-1",
            params={"revision": revision, "width": 40},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert client.post("/api/scenes").status_code == 405
        assert (
            client.get(
                "/api/scenes/B00/images/sample-1",
                params={"revision": revision, "mode": "unknown"},
            ).status_code
            == 422
        )
        raw = client.get(
            "/api/scenes/B00/images/sample-1",
            params={"revision": revision, "mode": "raw", "width": 40},
        )
        assert raw.status_code == 200
        decoded = cv2.imdecode(
            np.frombuffer(raw.content, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded is not None
        assert decoded.shape == (20, 40, 3)
        detail = client.get(
            "/api/scenes/B00/samples/sample-1", params={"revision": revision}
        )
        assert detail.status_code == 200
        assert detail.json()["target_court"] is None
        assert detail.json()["camera_to_scene"] == np.eye(4).ravel().tolist()
        assert (
            client.get(
                "/api/scenes/B00/samples/missing", params={"revision": revision}
            ).status_code
            == 404
        )
        assert (
            client.get(
                "/api/scenes/B00/samples/sample-1", params={"revision": "stale"}
            ).status_code
            == 409
        )


def test_source_run_change_invalidates_images_and_details(
    owner: tuple[ReviewService, dict[str, Any], str],
) -> None:
    service, _, revision = owner
    (service.root / "B00/run.json").write_text(
        '{"scene_id":"B00","source_video":"changed.mp4"}'
    )
    with pytest.raises(RuntimeError, match="changed"):
        service.sample_detail("B00", revision, "sample-1")


def test_rejected_candidate_never_serves_an_accepted_image(
    owner: tuple[ReviewService, dict[str, Any], str],
) -> None:
    service, _, revision = owner
    with TestClient(create_app(service)) as client:
        rejected = client.get(
            "/api/scenes/B00/rejections/reject-1", params={"revision": revision}
        )
        assert rejected.status_code == 200
        assert rejected.json()["visibility"]["projection_recorded"] is False
        assert rejected.json()["image_state"] == "not_retained"
        assert rejected.json()["visibility"]["counts"] is None
        assert (
            client.get(
                "/api/scenes/B00/images/reject-1", params={"revision": revision}
            ).status_code
            == 404
        )
        assert (
            client.get(
                "/api/scenes/B00/rejections/sample-1", params={"revision": revision}
            ).status_code
            == 404
        )


def test_publication_distinguishes_missing_source_and_reject_details(
    owner: tuple[ReviewService, dict[str, Any], str],
) -> None:
    service, sample, _ = owner
    dataset: dict[str, Any] = {
        "schema": "canonical_court_dataset_v3",
        "status": "completed",
        "samples": [sample],
        "trajectory_groups": [],
        "metrics": {
            "accepted_frame_count": 1,
            "rejected_frame_count": 7,
            "split_frame_counts": {"train": 1, "validation": 0, "test": 0},
        },
    }
    descriptor = service.root / "B00/datasets/court/dataset.json"
    descriptor.write_text(json.dumps(dataset))
    result = service.publication("B00", dataset)
    assert result["source_video"] is None
    assert result["generation_state"] is None
    assert result["rejected_count"] == 7
    assert result["rejected_record_count"] is None
    assert result["renderer_rasters_retained"] is None
    dataset["metrics"]["accepted_frame_count"] = 2
    with pytest.raises(ValueError, match="acceptance metrics"):
        service.publication("B00", dataset)


@pytest.mark.parametrize("version", list(CourtDatasetSchemaVersion))
def test_image_only_overlay_has_no_metadata_panels(
    version: CourtDatasetSchemaVersion,
) -> None:
    rgb: NDArray[np.float32] = np.full((240, 420, 3), 0.5, dtype=np.float32)
    frame = CourtSourceFrame(
        rgb=rgb,
        sample_id="sample",
        view_id="view",
        trajectory_frame_index=0,
        projection={"courts": []},
        schema_version=version,
    )
    plain = render_court_overlay(frame, trajectory_id="path", show_metadata=False)
    assert np.all(plain == 128)
    decorated = render_court_overlay(frame, trajectory_id="path")
    assert not np.array_equal(plain, decorated)
