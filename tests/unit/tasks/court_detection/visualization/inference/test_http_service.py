"""The Court backend must satisfy the shared detection HTTP contract."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.tasks.base.visualization.detection.web import create_detection_app
from src.tasks.court_detection.visualization.inference import service as service_module
from src.tasks.court_detection.visualization.inference.service import DetectionService

pytestmark = pytest.mark.unit


def _client(project: Path) -> TestClient:
    service = DetectionService(project_root=project)
    config = {
        "project_root": str(project),
        "data_root": str(project / "data"),
        "outputs_root": str(project / "outputs" / "court_detection"),
        "checkpoints_root": str(project / "ckpt" / "court_detection"),
    }
    app = create_detection_app(
        service,
        task="court_detection",
        mode="inference",
        service_config=config,
    )
    return TestClient(app)


def test_catalog_scenes_preview_and_image_endpoints(
    materialized_catalog, review_project: Path
) -> None:
    client = _client(review_project)

    catalog = client.get("/api/catalog").json()
    assert catalog["task"] == "court_detection"
    assert catalog["title"]
    assert catalog["mode"] == "inference"
    assert catalog["datasets"]

    scenes = client.get(
        "/api/scenes", params={"dataset": "tennis_court_detector/val", "limit": 5}
    ).json()
    assert scenes["total"] == 1
    scene = scenes["items"][0]
    assert scene["frames"] == 1

    preview = client.get("/api/preview", params={"scene": scene["id"]}).json()
    assert (preview["width"], preview["height"]) == (32, 24)
    assert preview["start"] == 0
    ground_truth = preview["items"][0]["gt"]
    assert ground_truth["points"] and ground_truth["rasters"]
    for raster in ground_truth["rasters"]:
        Image.open(io.BytesIO(base64.b64decode(raster["data"].split(",")[1])))

    image = client.get("/api/image", params={"scene": scene["id"], "frame": 0})
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"
    assert image.content[:2] == b"\xff\xd8"


def test_endpoints_reject_invalid_requests(
    materialized_catalog, review_project: Path
) -> None:
    client = _client(review_project)
    scene = client.get(
        "/api/scenes", params={"dataset": "tennis_court_detector/val"}
    ).json()["items"][0]["id"]

    assert (
        client.get("/api/preview", params={"scene": scene, "count": 2}).status_code
        == 422
    )
    assert client.get("/api/scenes", params={"dataset": "../etc"}).status_code == 422
    # A well-formed scene id that is not in the catalog is a missing resource;
    # an unresolvable id shape is rejected as an invalid request.
    assert (
        client.get(
            "/api/image",
            params={"scene": "tennis_court_detector/val::missing", "frame": 0},
        ).status_code
        == 404
    )
    assert (
        client.get("/api/image", params={"scene": "unknown", "frame": 0}).status_code
        == 422
    )
    blocked = client.post(
        "/api/infer",
        json={
            "checkpoint": "run/checkpoints/epoch.ckpt",
            "scene": scene,
            "start": 0,
            "count": 1,
            "threshold": 0.5,
            "device": "cpu",
        },
        headers={"origin": "http://evil.example"},
    )
    assert blocked.status_code == 403


def test_infer_endpoint_returns_the_service_payload(
    materialized_catalog, review_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = (
        review_project
        / "outputs"
        / "court_detection"
        / "run"
        / "checkpoints"
        / "epoch.ckpt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "hyper_parameters": {"config": {}, "target_bundle_state": {}},
            "state_dict": {},
        },
        checkpoint,
    )

    class _StubRunner:
        def __init__(self, path: Path, *, device: torch.device, **_: object) -> None:
            del path, device

        def predict(self, image) -> dict[str, object]:
            height, width = image.height, image.width
            return {
                "kp": service_module.CourtKeypointPrediction(
                    keypoints=torch.zeros(14, 1, 2),
                    scores=torch.full((14, 1), 0.9),
                    valid=torch.ones(14, 1, dtype=torch.bool),
                    heatmaps=torch.zeros(14, 4, 4),
                ),
                "seg": service_module.CourtSegmentationPrediction(
                    mask=torch.zeros(height, width, dtype=torch.long),
                    logits=torch.zeros(7, height, width),
                ),
            }

        def close(self) -> None:
            return None

    from src.tasks.court_detection.visualization.inference.checkpoints import (
        CourtCheckpointInfo,
        CourtHeadSpec,
    )
    from src.tasks.court_detection.visualization.review.datasets import (
        DENSE_TARGET_SCHEMAS,
    )
    from src.utils.schema.court import GROUND_COURT_KP_NAMES

    stat = checkpoint.stat()
    info = CourtCheckpointInfo(
        id="run/checkpoints/epoch.ckpt",
        path=checkpoint,
        label="run/checkpoints/epoch",
        model="court_hierarchical",
        metadata_source="checkpoint",
        supported=True,
        reason=None,
        heads=(
            CourtHeadSpec(
                kind="kp",
                schema="fixture_kp14",
                output_channels=14,
                channel_names=GROUND_COURT_KP_NAMES,
            ),
            CourtHeadSpec(
                kind="seg",
                schema=DENSE_TARGET_SCHEMAS["seg"],
                output_channels=7,
                channel_names=tuple(f"class_{index}" for index in range(7)),
            ),
        ),
        size_bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
    )
    monkeypatch.setattr(service_module, "CourtHeadRunner", _StubRunner)
    monkeypatch.setattr(service_module, "describe_checkpoint", lambda *_: info)
    client = _client(review_project)
    scene = client.get(
        "/api/scenes", params={"dataset": "tennis_court_detector/val"}
    ).json()["items"][0]["id"]

    response = client.post(
        "/api/infer",
        json={
            "checkpoint": "run/checkpoints/epoch.ckpt",
            "scene": scene,
            "start": 0,
            "count": 1,
            "threshold": 0.5,
            "device": "cpu",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["scene"] == scene
    assert payload["items"][0]["pred"]["rasters"]
    assert "kp_mean_error_px" in payload["metrics"]
