"""Unit tests for the PLCS inference-UI HTTP layer."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from src.tasks.plcs.visualization.inference import web as web_module
from src.tasks.plcs.visualization.inference.service import (
    InferenceService,
    PredictionRequest,
    PredictionResult,
)
from src.tasks.plcs.visualization.inference.web import create_app


def test_cuda_request_is_validated_then_enqueued(
    client: tuple[TestClient, InferenceService, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    http, service, _ = client
    events: list[str] = []

    def validate(request: PredictionRequest) -> dict[str, Any]:
        events.append("validate")
        return {"device": "cuda:0"}

    def queued(task: str, **kwargs: Any) -> bytes:
        events.append("queue")
        assert task == "plcs"
        assert kwargs["request"]["scene"] == "scene_000000"
        assert kwargs["request"]["device"] == "cuda:0"
        assert kwargs["service"]["checkpoint_root"] == str(service.checkpoint_root)
        return b"framed-result"

    monkeypatch.setattr(service, "validate_prediction_request", validate)
    monkeypatch.setattr(web_module, "run_queued_inference", queued)
    response = http.post(
        "/api/predict",
        json={
            "checkpoint": "test.ckpt",
            "family": "single_object",
            "scene": "scene_000000",
            "cameras": [0],
            "device": "cuda:0",
        },
    )
    assert response.status_code == 200
    assert response.content == b"framed-result"
    assert events == ["validate", "queue"]


def _write_dataset(data_root: Path) -> None:
    root = data_root / "single_object"
    (root / "scenes" / "scene_000000").mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "config": {
                    "court_keypoints": {"selector": "physical_v1"},
                    "generation": {"mode": "single_object"},
                    "camera": {"layout": "fixed"},
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "val.txt").write_text("scene_000000\n", encoding="utf-8")
    (root / "train.txt").write_text("", encoding="utf-8")
    (root / "test.txt").write_text("", encoding="utf-8")
    scene = root / "scenes" / "scene_000000"
    (scene / "meta.json").write_text(
        json.dumps(
            {
                "scene_id": "scene_000000",
                "fps": 120.0,
                "num_frames": 400,
                "court_keypoint_views": [
                    {"camera_id": f"camera_{index}"} for index in range(6)
                ],
            }
        ),
        encoding="utf-8",
    )
    (scene / "scalars.json").write_text(
        json.dumps({"num_cameras": 6, "num_persons": 1}), encoding="utf-8"
    )


@pytest.fixture
def client(tmp_path: Path) -> tuple[TestClient, InferenceService, Path]:
    data_root = tmp_path / "data" / "plcs"
    checkpoint_root = tmp_path / "outputs" / "plcs"
    checkpoint_root.mkdir(parents=True)
    _write_dataset(data_root)
    checkpoint = (
        checkpoint_root / "run_a" / "logs" / "version_0" / "checkpoints" / "best.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"")
    (checkpoint.parent.parent / "hparams.yaml").write_text(
        yaml.safe_dump(
            {
                "config": {
                    "model": {
                        "name": "plcs_multiview_axial",
                        "io": {"input_profile": "multiview"},
                        "max_views": 6,
                    },
                    "court_keypoints": {"selector": "physical_v1"},
                    "data": {"scene_dir": "plcs/single_object"},
                }
            }
        ),
        encoding="utf-8",
    )
    service = InferenceService(
        data_root=data_root, checkpoint_root=checkpoint_root, device="cpu"
    )
    return TestClient(create_app(service)), service, checkpoint_root


def test_catalog_returns_families_and_checkpoints(
    client: tuple[TestClient, InferenceService, Path],
) -> None:
    test_client, _, checkpoint_root = client
    response = test_client.get("/api/catalog")
    assert response.status_code == 200
    body = response.json()
    assert body["device"] == "cpu"
    assert [item["id"] for item in body["families"]] == ["single_object"]
    assert len(body["checkpoints"]) == 1
    entry = body["checkpoints"][0]
    assert entry["id"] == "run_a/logs/version_0/checkpoints/best.ckpt"
    assert entry["families"] == ["single_object", "single_object_broadcast"]
    assert entry["supported"] is True
    assert str(checkpoint_root) == body["checkpoint_root"]


def test_scenes_endpoint_lists_and_filters(
    client: tuple[TestClient, InferenceService, Path],
) -> None:
    test_client, _, _ = client
    response = test_client.get("/api/scenes", params={"family": "single_object"})
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["split"] == "val"
    assert body["scenes"][0]["id"] == "scene_000000"
    empty = test_client.get(
        "/api/scenes", params={"family": "single_object", "query": "zzz"}
    )
    assert empty.json()["total"] == 0


def test_scenes_endpoint_rejects_bad_split(
    client: tuple[TestClient, InferenceService, Path],
) -> None:
    test_client, _, _ = client
    response = test_client.get(
        "/api/scenes", params={"family": "single_object", "split": "nope"}
    )
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


def test_scene_detail_endpoint(
    client: tuple[TestClient, InferenceService, Path],
) -> None:
    test_client, _, _ = client
    response = test_client.get(
        "/api/scenes/single_object/scene_000000",
        params={"checkpoint": "run_a/logs/version_0/checkpoints/best.ckpt"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["num_frames"] == 400
    assert body["num_cameras"] == 6
    assert body["supported"] is True
    assert body["window"]["max_length"] == 400
    assert body["reference_required"] is False


def test_scene_detail_reports_unknown_family_and_missing_scene(
    client: tuple[TestClient, InferenceService, Path],
) -> None:
    test_client, _, _ = client
    unknown = test_client.get("/api/scenes/nope/scene_000000")
    assert unknown.status_code == 422
    assert isinstance(unknown.json()["detail"], str)
    missing = test_client.get("/api/scenes/single_object/scene_999999")
    assert missing.status_code == 404
    assert isinstance(missing.json()["detail"], str)


def test_static_allowlist_rejects_unknown_asset(
    client: tuple[TestClient, InferenceService, Path],
) -> None:
    test_client, _, _ = client
    assert test_client.get("/static/nope.js").status_code == 404


def test_predict_returns_framed_binary(
    client: tuple[TestClient, InferenceService, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, service, _ = client

    captured: dict[str, PredictionRequest] = {}

    def fake_predict(request: PredictionRequest) -> PredictionResult:
        captured["request"] = request
        payload: np.ndarray = np.arange(12, dtype=np.float32)
        header = {
            "scene": {"id": request.scene, "family": request.family},
            "tracks": [
                {
                    "kind": "gt",
                    "has_joints": True,
                    "position": {"offset": 0, "count": 6, "shape": [2, 3]},
                },
                {
                    "kind": "pred",
                    "has_joints": True,
                    "position": {"offset": 6, "count": 6, "shape": [2, 3]},
                },
            ],
            "payload_elements": 12,
        }
        return PredictionResult(header=header, payload=payload)

    monkeypatch.setattr(service, "predict", fake_predict)
    body = {
        "checkpoint": "run_a/logs/version_0/checkpoints/best.ckpt",
        "family": "single_object",
        "scene": "scene_000000",
        "cameras": [0, 1, 2, 3, 4, 5],
        "window_start": 10,
        "window_length": 2,
        "canonical_pose_source": "gt",
        "device": "cpu",
    }
    response = test_client.post("/api/predict", json=body)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    raw = response.content
    (header_length,) = struct.unpack("<I", raw[:4])
    header = json.loads(raw[4 : 4 + header_length].decode("utf-8"))
    assert header["scene"]["id"] == "scene_000000"
    payload = np.frombuffer(raw[4 + header_length :], dtype="<f4")
    assert payload.shape == (12,)
    assert payload[0] == pytest.approx(0.0)
    assert payload[11] == pytest.approx(11.0)
    assert captured["request"].cameras == (0, 1, 2, 3, 4, 5)
    assert captured["request"].window_start == 10
    assert captured["request"].reference_camera_id is None


def test_predict_rejects_invalid_body(
    client: tuple[TestClient, InferenceService, Path],
) -> None:
    test_client, _, _ = client
    response = test_client.post(
        "/api/predict",
        json={
            "checkpoint": "run_a/logs/version_0/checkpoints/best.ckpt",
            "family": "single_object",
            "scene": "scene_000000",
            "cameras": [],
        },
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, str)
    assert "cameras" in detail


def test_validate_endpoint_reports_preflight_failures(
    client: tuple[TestClient, InferenceService, Path],
) -> None:
    test_client, _, _ = client
    valid = {
        "checkpoint": "run_a/logs/version_0/checkpoints/best.ckpt",
        "family": "single_object",
        "scene": "scene_000000",
        "cameras": [0, 1, 2],
        "window_start": 0,
        "window_length": 8,
        "canonical_pose_source": "gt",
        "device": "cpu",
    }
    ok = test_client.post("/api/validate", json=valid)
    assert ok.status_code == 200
    assert ok.json() == {"valid": True}
    bad = test_client.post("/api/validate", json={**valid, "cameras": [0, 1, 99]})
    assert bad.status_code == 422
    assert isinstance(bad.json()["detail"], str)


def test_catalog_lists_configured_checkpoint_roots(
    client: tuple[TestClient, InferenceService, Path],
) -> None:
    test_client, _, checkpoint_root = client
    body = test_client.get("/api/catalog").json()
    assert body["checkpoint_roots"] == [str(checkpoint_root)]


def test_scene_preview_endpoint_returns_framed_binary(
    client: tuple[TestClient, InferenceService, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_client, service, _ = client

    def fake_preview(*args: object, **kwargs: object) -> PredictionResult:
        payload: np.ndarray = np.arange(6, dtype=np.float32)
        return PredictionResult(
            header={
                "mode": "preview",
                "scene": {"family": "single_object", "id": "scene_000000"},
                "tracks": [
                    {
                        "kind": "gt",
                        "has_joints": True,
                        "position": {"offset": 0, "count": 3, "shape": [1, 3]},
                        "joints": {"offset": 3, "count": 3, "shape": [1, 1, 3]},
                    }
                ],
                "payload_elements": 6,
                "cameras": [{"index": 0, "id": "cam_0", "center": [0, 0, 0]}],
            },
            payload=payload,
        )

    monkeypatch.setattr(service, "scene_preview", fake_preview)
    response = test_client.get(
        "/api/scenes/single_object/scene_000000/preview",
        params={"cameras": [0, 1, 2], "window_start": 3, "window_length": 1},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    raw = response.content
    (header_length,) = struct.unpack("<I", raw[:4])
    header = json.loads(raw[4 : 4 + header_length].decode("utf-8"))
    assert header["mode"] == "preview"
    payload = np.frombuffer(raw[4 + header_length :], dtype="<f4")
    assert payload.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_preview_endpoint_reports_unknown_scene(
    client: tuple[TestClient, InferenceService, Path],
) -> None:
    test_client, _, _ = client
    # "Not listed in a split" is a request error (422), matching /api/predict;
    # only a missing scene directory is a 404.
    missing = test_client.get("/api/scenes/single_object/scene_999999/preview")
    assert missing.status_code == 422
    assert isinstance(missing.json()["detail"], str)


def test_shared_scene_asset_route_serves_three(
    client: tuple[TestClient, InferenceService, Path],
) -> None:
    test_client, _, _ = client
    response = test_client.get("/shared/three.module.js")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert test_client.get("/shared/not_a_real_asset.js").status_code == 404
