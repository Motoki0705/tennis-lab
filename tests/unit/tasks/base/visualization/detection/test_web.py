from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.tasks.base.visualization.detection.web import create_detection_app


@pytest.fixture
def service():
    backend = Mock()
    backend.catalog.return_value = {"task": "ball_detection"}
    backend.scenes.return_value = {"items": [], "total": 0}
    backend.preview.return_value = {"items": []}
    backend.image.return_value = b"image"
    backend.infer.return_value = {"items": []}
    return backend


def client(service, mode="inference"):
    return TestClient(
        create_detection_app(
            service,
            task="ball_detection",
            mode=mode,
            service_config={"project_root": "/repo"},
        )
    )


def test_review_never_runs_inference(service):
    response = client(service, "review").post(
        "/api/infer", json={"checkpoint": "c", "scene": "s", "device": "cpu"}
    )
    assert response.status_code == 403
    service.infer.assert_not_called()


def test_cpu_preflight_and_execution(service):
    response = client(service).post(
        "/api/infer",
        json={"checkpoint": "c", "scene": "s", "device": "cpu", "count": 8},
    )
    assert response.status_code == 200
    service.validate.assert_called_once_with(
        checkpoint="c", scene="s", device="cpu", count=8, start=0, threshold=0.5
    )
    service.infer.assert_called_once_with(
        checkpoint="c", scene="s", device="cpu", count=8, start=0, threshold=0.5
    )


def test_gpu_uses_queue_only(service, monkeypatch):
    monkeypatch.setattr(
        "src.tasks.base.visualization.detection.web.torch.cuda.is_available",
        lambda: True,
    )
    queue = Mock(return_value=b'{"items":[]}')
    monkeypatch.setattr(
        "src.tasks.base.visualization.detection.web.run_queued_inference", queue
    )
    response = client(service).post(
        "/api/infer", json={"checkpoint": "c", "scene": "s"}
    )
    assert response.status_code == 200
    queue.assert_called_once()
    assert queue.call_args.args == ("ball_detection",)
    service.infer.assert_not_called()


def test_unavailable_cuda_has_no_fallback(service, monkeypatch):
    monkeypatch.setattr(
        "src.tasks.base.visualization.detection.web.torch.cuda.is_available",
        lambda: False,
    )
    response = client(service).post(
        "/api/infer", json={"checkpoint": "c", "scene": "s"}
    )
    assert response.status_code == 422
    service.validate.assert_not_called()
    service.infer.assert_not_called()


def test_validation_failure_releases_job_lock(service):
    service.validate.side_effect = [ValueError("incompatible"), None]
    ui = client(service)
    request = {"checkpoint": "c", "scene": "s", "device": "cpu"}
    assert ui.post("/api/infer", json=request).status_code == 422
    assert ui.post("/api/infer", json=request).status_code == 200


@pytest.mark.parametrize(
    "update",
    [
        {"count": 0},
        {"count": 65},
        {"start": -1},
        {"threshold": 1.1},
        {"checkpoint": ""},
        {"device": "auto"},
        {"unknown": True},
    ],
)
def test_bad_requests(service, update):
    response = client(service).post(
        "/api/infer", json={"checkpoint": "c", "scene": "s", "device": "cpu", **update}
    )
    assert response.status_code == 422
    service.infer.assert_not_called()


def test_cross_origin_rejected(service):
    response = client(service).post(
        "/api/infer",
        json={"checkpoint": "c", "scene": "s", "device": "cpu"},
        headers={"origin": "https://external.example"},
    )
    assert response.status_code == 403
    service.validate.assert_not_called()


def test_same_origin_accepted(service):
    response = client(service).post(
        "/api/infer",
        json={"checkpoint": "c", "scene": "s", "device": "cpu"},
        headers={"origin": "http://testserver"},
    )
    assert response.status_code == 200


def test_catalog_preview_and_image(service):
    ui = client(service, "review")
    assert ui.get("/api/catalog").json()["mode"] == "review"
    assert ui.get("/api/preview?scene=s&start=2&count=1").status_code == 200
    service.preview.assert_called_once_with("s", 2, 1)
    response = ui.get("/api/image?scene=s&frame=2")
    assert response.content == b"image"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def test_scene_search_passes_checkpoint(service):
    assert (
        client(service)
        .get("/api/scenes?dataset=d&checkpoint=c&search=ball&offset=100")
        .status_code
        == 200
    )
    service.scenes.assert_called_once_with("d", "ball", 100, 100, "c")


@pytest.mark.parametrize(
    "path",
    [
        "/static/no.py",
        "/static/web.py",
        "/api/scenes?dataset=d&limit=500",
        "/api/preview?scene=s&count=100",
    ],
)
def test_asset_and_payload_bounds(service, path):
    assert client(service).get(path).status_code in {404, 422}


def test_assets_are_local(service):
    ui = client(service)
    assert ui.get("/").status_code == 200
    for asset in ("app.js", "viewer.mjs", "icons.mjs", "style.css"):
        response = ui.get(f"/static/{asset}")
        assert response.status_code == 200
        assert len(response.content) > 100


@pytest.mark.parametrize("task", ["ball_detection", "court_detection"])
def test_detection_worker_dispatch(task, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from src.tasks.base.visualization.inference_queue import execute_request

    backend = Mock()
    backend.infer.return_value = {"items": [], "metrics": {"distance": None}}
    constructor = Mock(return_value=backend)
    module = SimpleNamespace(DetectionService=constructor)
    monkeypatch.setattr("importlib.import_module", lambda name: module)
    result = execute_request(
        {
            "task": task,
            "service": {"project_root": "/repo"},
            "request": {"checkpoint": "c", "scene": "s", "device": "cuda"},
        }
    )
    constructor.assert_called_once_with(project_root=Path("/repo"))
    backend.infer.assert_called_once_with(checkpoint="c", scene="s", device="cuda")
    assert b'"distance": null' in result
