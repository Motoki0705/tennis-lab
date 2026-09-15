"""HTTP contract tests for the BLCS inference FastAPI application."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from src.tasks.blcs.visualization.inference import web as web_module
from src.tasks.blcs.visualization.inference.service import InferenceService
from src.tasks.blcs.visualization.inference.web import create_app


def _write_checkpoint(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hyper_parameters": {"config": config}}, path)


def _trajectory_config(name: str, selector: str, scene_dir: str) -> dict[str, Any]:
    return {
        "model": {
            "name": name,
            "io": {"input_profile": "multiview"},
            "num_court_tokens": 14,
            "max_seq_len": 256,
            "max_num_cameras": 4,
        },
        "court_keypoints": {"selector": selector},
        "data": {
            "scene_dir": scene_dir,
            "seq_len_range": [128, 128],
            "num_views_range": [3, 4],
        },
    }


def _tracking_config(
    name: str,
    selector: str,
    scene_dir: str,
    *,
    num_queries: int = 4,
) -> dict[str, Any]:
    model: dict[str, Any] = {"name": name, "num_queries": num_queries}
    if name.endswith("_reference"):
        model.update(
            {
                "target_frame_contract": "reference_camera_court_rzpi_v1",
                "track_query_rope_contract": "time_camera_reference_selector_v1",
                "reference_selector_mode": "reference",
            }
        )
    return {
        "model": model,
        "court_keypoints": {"selector": selector},
        "data": {
            "scene_dir": scene_dir,
            "seq_len_range": [128, 128],
            "num_views_range": [3, 3],
        },
    }


def _write_light_form(
    data_root: Path,
    form: str,
    *,
    scene_ids: Sequence[str],
    num_cameras: int = 4,
    num_balls: int = 1,
    frames: int = 6,
    split_files: bool = True,
) -> None:
    form_dir = data_root / "blcs" / form
    (form_dir / "scenes").mkdir(parents=True)
    for scene_id in scene_ids:
        scene_dir = form_dir / "scenes" / scene_id
        scene_dir.mkdir()
        (scene_dir / "meta.json").write_text(
            json.dumps(
                {
                    "scene_id": scene_id,
                    "num_frames": frames,
                    "fps_out": 30.0,
                    "num_cameras": num_cameras,
                }
            ),
            encoding="utf-8",
        )
        (scene_dir / "scalars.json").write_text(
            json.dumps({"num_cameras": num_cameras, "num_balls": num_balls}),
            encoding="utf-8",
        )
    if split_files:
        (form_dir / "test.txt").write_text(
            "\n".join(scene_ids) + "\n",
            encoding="utf-8",
        )


@pytest.fixture
def service(tmp_path: Path) -> InferenceService:
    _write_light_form(
        tmp_path / "data",
        "single_object_camera_view_v2",
        scene_ids=["scene_000000", "scene_000001"],
    )
    _write_light_form(
        tmp_path / "data",
        "multi_object",
        scene_ids=["scene_000000"],
        num_cameras=3,
        num_balls=2,
        split_files=False,
    )
    _write_checkpoint(
        tmp_path / "ckpt" / "blcs" / "reference.ckpt",
        _trajectory_config(
            "blcs_multiview_axial_reference",
            "camera_view_v2",
            "blcs/single_object_camera_view_v2",
        ),
    )
    _write_checkpoint(
        tmp_path / "ckpt" / "blcs" / "tracking.ckpt",
        _tracking_config(
            "blcs_track_query",
            "physical_v1",
            "blcs/multi_object",
        ),
    )
    return InferenceService(
        tmp_path / "outputs",
        tmp_path / "ckpt",
        tmp_path / "data",
    )


@pytest.fixture
def static_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "static"
    directory.mkdir()
    (directory / "index.html").write_text(
        "<!doctype html><title>BLCS</title>",
        encoding="utf-8",
    )
    (directory / "app.js").write_text("export {};\n", encoding="utf-8")
    (directory / "style.css").write_text("body{}\n", encoding="utf-8")
    (directory / "scene.mjs").write_text("export {};\n", encoding="utf-8")
    monkeypatch.setattr(web_module, "STATIC", directory)
    return directory


def test_index_and_static_assets_are_served(
    service: InferenceService,
    static_dir: Path,
) -> None:
    client = TestClient(create_app(service))
    index = client.get("/")
    assert index.status_code == 200
    assert "BLCS" in index.text

    script = client.get("/static/app.js")
    assert script.status_code == 200
    assert script.text == "export {};\n"

    assert client.get("/static/evil.txt").status_code == 404
    assert client.get("/static/../index.html").status_code in {404, 400}


def test_catalog_shape(service: InferenceService) -> None:
    client = TestClient(create_app(service))
    response = client.get("/api/catalog")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"roots", "checkpoints", "scene_forms", "court", "world"}
    assert [root["id"] for root in body["roots"]] == ["outputs", "checkpoints"]
    checkpoint_ids = {entry["id"] for entry in body["checkpoints"]}
    assert checkpoint_ids == {
        "checkpoints:blcs/reference.ckpt",
        "checkpoints:blcs/tracking.ckpt",
    }
    forms = {form["id"] for form in body["scene_forms"]}
    assert forms == {"multi_object", "single_object_camera_view_v2"}
    assert len(body["court"]["keypoints"]) == 20
    assert body["world"]["court_contract"] == "physical_v1"


def test_scenes_returns_a_page(service: InferenceService) -> None:
    client = TestClient(create_app(service))
    response = client.get(
        "/api/scenes",
        params={"form": "single_object_camera_view_v2", "limit": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["split"] == "test"
    assert body["total"] == 2
    assert body["limit"] == 1
    assert body["offset"] == 0
    assert [scene["id"] for scene in body["scenes"]] == ["scene_000000"]


def test_scenes_rejects_unknown_form_with_422(service: InferenceService) -> None:
    client = TestClient(create_app(service))
    response = client.get("/api/scenes", params={"form": "nope"})
    assert response.status_code == 422
    assert "Unknown scene form" in response.json()["detail"]


def test_scenes_rejects_missing_split_with_422(service: InferenceService) -> None:
    client = TestClient(create_app(service))
    response = client.get(
        "/api/scenes",
        params={"form": "multi_object", "split": "test"},
    )
    assert response.status_code == 422
    assert "no scene list file" in response.json()["detail"]


def test_scenes_rejects_invalid_limit_with_422(service: InferenceService) -> None:
    client = TestClient(create_app(service))
    response = client.get(
        "/api/scenes",
        params={"form": "multi_object", "limit": 0},
    )
    assert response.status_code == 422


def test_scene_requires_form_and_scene(service: InferenceService) -> None:
    client = TestClient(create_app(service))
    assert client.get("/api/scene").status_code == 422
    response = client.get(
        "/api/scene",
        params={"form": "single_object_camera_view_v2", "scene": "scene_404"},
    )
    assert response.status_code == 404


def test_catalog_marks_tracking_checkpoint_runnable(
    service: InferenceService,
) -> None:
    client = TestClient(create_app(service))
    body = client.get("/api/catalog").json()
    tracking = next(
        entry
        for entry in body["checkpoints"]
        if entry["id"] == "checkpoints:blcs/tracking.ckpt"
    )
    assert tracking["runnable"] is True
    assert tracking["object_mode"] == "multi"
    assert tracking["reference"] is False
    assert tracking["allowed_forms"] == ["multi_object"]
    assert tracking["num_queries"] == 4


def test_infer_rejects_inconsistent_checkpoint_with_422(
    service: InferenceService,
    tmp_path: Path,
) -> None:
    _write_checkpoint(
        tmp_path / "ckpt" / "blcs" / "inconsistent.ckpt",
        _tracking_config(
            "blcs_track_query_reference",
            "physical_v1",
            "blcs/multi_object",
        ),
    )
    client = TestClient(create_app(service))
    response = client.post(
        "/api/infer",
        json={
            "checkpoint": "checkpoints:blcs/inconsistent.ckpt",
            "form": "multi_object",
            "scene": "scene_000000",
        },
    )
    assert response.status_code == 422
    assert "could not be read" in response.json()["detail"]


def test_infer_rejects_mismatched_form_with_422(service: InferenceService) -> None:
    client = TestClient(create_app(service))
    response = client.post(
        "/api/infer",
        json={
            "checkpoint": "checkpoints:blcs/reference.ckpt",
            "form": "multi_object",
            "scene": "scene_000000",
        },
    )
    assert response.status_code == 422
    assert "cannot run form" in response.json()["detail"]


def test_cuda_http_request_uses_shared_queue(
    service: InferenceService, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    def validate(**kwargs: Any) -> SimpleNamespace:
        events.append("validate")
        return SimpleNamespace(device_key="cuda:0")

    def queued(task: str, **kwargs: Any) -> bytes:
        events.append("queue")
        assert task == "blcs"
        assert kwargs["request"]["scene_id"] == "scene_000000"
        assert kwargs["request"]["device"] == "cuda:0"
        assert kwargs["service"]["outputs_root"] == str(service.outputs_root)
        return b'{"queued_result":true}'

    monkeypatch.setattr(service, "validate_inference_request", validate)
    monkeypatch.setattr(web_module, "run_queued_inference", queued)
    response = TestClient(create_app(service)).post(
        "/api/infer",
        json={
            "checkpoint": "checkpoints:blcs/reference.ckpt",
            "form": "single_object_camera_view_v2",
            "scene": "scene_000000",
            "device": "cuda:0",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"queued_result": True}
    assert events == ["validate", "queue"]
