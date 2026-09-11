"""Real HTTP range delivery, exact-frame seek and export through the browser API."""

from pathlib import Path
from time import monotonic, sleep

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from omegaconf import DictConfig, OmegaConf

from src.tennis_scene.clip_studio.project import ClipSource, ClipStudioProject
from src.tennis_scene.clip_studio.web.app import create_app
from src.tennis_scene.configuration import parse_clip_studio_config


@pytest.fixture
def web_client(tmp_path):
    source = tmp_path / "data/tennis_multivew/raw/test/video_000"
    source.mkdir(parents=True)
    videos = []
    for camera in range(2):
        video = source / f"cam{camera}.mp4"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10, (64, 48)
        )
        for index in range(30):
            writer.write(np.full((48, 64, 3), index * 7, dtype=np.uint8))
        writer.release()
        videos.append(video)
    cfg = OmegaConf.load(
        Path(__file__).parents[3] / "src/tennis_scene/configs/clip_studio.yaml"
    )
    assert isinstance(cfg, DictConfig)
    del cfg["defaults"]
    del cfg["hydra"]
    cfg.paths = {
        "project_root": str(tmp_path),
        "data_root": "data",
        "checkpoint_root": "ckpt",
        "artifact_root": "outputs",
        "output_root": "outputs",
        "cache_root": ".cache",
        "external_asset_root": "third_party",
    }
    cfg.export.fps = 10.0
    cfg.export.width = 64
    cfg.export.height = 48
    cfg.source_directory = "tennis_multivew/raw/test/video_000"
    runtime = parse_clip_studio_config(cfg)
    project = ClipStudioProject(
        dataset_id="test",
        video_id="video_000",
        sources=[
            ClipSource(videos[0], "cam0"),
            ClipSource(videos[1], "cam1", 0.2),
        ],
    )
    with TestClient(create_app(runtime, project)) as client:
        yield client, runtime


def test_ranged_media_and_exact_source_frame(web_client):
    client, runtime = web_client
    assert client.get("/").status_code == 200
    response = client.get("/api/media/0", headers={"Range": "bytes=0-15"})
    assert response.status_code == 206
    assert len(response.content) == 16
    frame = client.get("/api/frame/1?time=0.5&revision=0")
    assert frame.headers["x-frame-index"] == "7"
    image = cv2.imdecode(np.frombuffer(frame.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.mean() == pytest.approx(49, abs=3)
    assert client.get("/api/frame/0?time=-1&revision=0").status_code == 204
    assert client.get("/api/frame/0?time=nan&revision=0").status_code == 422
    assert client.get("/api/media/99").status_code == 404


def test_autosave_revision_and_cross_origin_protection(web_client):
    client, runtime = web_client
    body = {"revision": 0, "action": "create", "start_sec": 0.5, "end_sec": 1.0}
    assert (
        client.post(
            "/api/edit", json=body, headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    assert client.post("/api/edit", json=body).status_code == 200
    assert runtime.export.projects_path.is_file()
    assert client.post("/api/edit", json=body).status_code == 409
    assert client.get("/api/frame/0?time=0.5&revision=0").status_code == 409
    assert (
        client.post("/api/edit", json={"revision": 1, "action": "undo"}).json()["clips"]
        == []
    )


def wait_job(client):
    deadline = monotonic() + 20
    while monotonic() < deadline:
        job = client.get("/api/jobs").json()
        if job["status"] != "running":
            return job
        sleep(0.03)
    pytest.fail("Background job timed out")


def test_real_export_and_whole_batch_preflight(web_client):
    client, runtime = web_client
    client.post(
        "/api/edit",
        json={"revision": 0, "action": "create", "start_sec": 0.5, "end_sec": 1.0},
    )
    client.post("/api/jobs", json={"revision": 1, "kind": "export"})
    assert wait_job(client)["status"] == "done"
    assert (
        runtime.export.output_dir / "videos/video_000/clips/clip_000/media/cam0.mp4"
    ).is_file()
    # A later invalid clip must fail the batch before any earlier new clip is written.
    client.post(
        "/api/edit",
        json={"revision": 1, "action": "create", "start_sec": 1.0, "end_sec": 1.5},
    )
    client.post(
        "/api/edit",
        json={"revision": 2, "action": "create", "start_sec": 2.0, "end_sec": 5.0},
    )
    client.post(
        "/api/jobs",
        json={"revision": 3, "kind": "export", "clip_names": ["clip_001", "clip_002"]},
    )
    assert wait_job(client)["status"] == "failed"
    assert not (runtime.export.output_dir / "videos/video_000/clips/clip_001").exists()


def test_batch_skips_verified_outputs_but_rejects_changed_edits(web_client):
    client, runtime = web_client
    client.post(
        "/api/edit",
        json={"revision": 0, "action": "create", "start_sec": 0.5, "end_sec": 1.0},
    )
    client.post("/api/jobs", json={"revision": 1, "kind": "export"})
    assert wait_job(client)["status"] == "done"
    video = runtime.export.output_dir / "videos/video_000/clips/clip_000/media/cam0.mp4"
    before = video.stat().st_mtime_ns
    client.post(
        "/api/edit",
        json={"revision": 1, "action": "create", "start_sec": 1.0, "end_sec": 1.5},
    )
    client.post("/api/jobs", json={"revision": 2, "kind": "export"})
    result = wait_job(client)
    assert result["status"] == "done"
    assert result["completed"] == 1
    assert result["skipped"] == ["clip_000"]
    assert video.stat().st_mtime_ns == before
    client.post("/api/jobs", json={"revision": 2, "kind": "export"})
    repeated = wait_job(client)
    assert repeated["status"] == "done"
    assert repeated["completed"] == 0
    assert repeated["skipped"] == ["clip_000", "clip_001"]
    # Missing media are not mistaken for a completed export.
    missing = video.with_name("cam1.mp4")
    saved_media = missing.read_bytes()
    missing.unlink()
    client.post("/api/jobs", json={"revision": 2, "kind": "export"})
    assert wait_job(client)["status"] == "failed"
    missing.write_bytes(saved_media)
    # Same name with changed timing is not a valid skip.
    client.post(
        "/api/edit",
        json={
            "revision": 2,
            "action": "update",
            "name": "clip_000",
            "start_sec": 0.6,
            "end_sec": 1.0,
        },
    )
    client.post("/api/jobs", json={"revision": 3, "kind": "export"})
    result = wait_job(client)
    assert result["status"] == "failed"
    assert "differs" in result["message"]
    assert video.stat().st_mtime_ns == before


def test_cancel_stops_active_encoder_and_removes_unpublished_output(web_client):
    from dataclasses import replace

    client, runtime = web_client
    # Long enough to cancel a real active encoder, without depending on sleeps
    # in implementation or mocked progress. The source remains the tiny fixture.
    jobs = client.app.state.jobs
    jobs.export = replace(jobs.export, fps=240.0, width=1920, height=1080)
    client.post(
        "/api/edit",
        json={"revision": 0, "action": "create", "start_sec": 0.2, "end_sec": 2.5},
    )
    client.post("/api/jobs", json={"revision": 1, "kind": "export"})
    deadline = monotonic() + 15
    while monotonic() < deadline:
        job = client.get("/api/jobs").json()
        if job.get("frames_completed", 0) > 0:
            break
        assert job["status"] == "running", job
        sleep(0.02)
    else:
        pytest.fail("No encoding progress")
    assert job["status"] == "running"
    started = monotonic()
    client.post("/api/jobs/cancel")
    job = wait_job(client)
    assert job["status"] == "cancelled", job
    assert monotonic() - started < 5
    assert job["completed"] == 0
    assert not (runtime.export.output_dir / "videos/video_000/clips/clip_000").exists()
    assert not (runtime.export.output_dir / "dataset.json").exists()
    assert not list(runtime.export.output_dir.glob(".clip-studio-export-*"))
    # Cancellation must not poison the worker or prevent retrying the clip.
    jobs.export = replace(jobs.export, fps=10.0, width=64, height=48)
    client.post("/api/jobs", json={"revision": 1, "kind": "export"})
    assert wait_job(client)["status"] == "done"


@pytest.mark.parametrize("missing_timestamp", [False, True])
def test_recording_time_startup_notice_and_saved_sync(web_client, missing_timestamp):
    from dataclasses import replace

    import av

    from src.tennis_scene.clip_studio.initialization import load_or_create_project

    _, runtime = web_client
    paths = []
    for index in range(2):
        path = runtime.export.resolver.roots.data_root / f"camera{index}.mp4"
        with av.open(str(path), mode="w") as output:
            if not (missing_timestamp and index == 1):
                output.metadata["creation_time"] = f"2026-07-09T16:08:0{index}Z"
            stream = output.add_stream("libx264", rate=10)
            stream.width = 64
            stream.height = 48
            stream.pix_fmt = "yuv420p"
            for _ in range(30):
                frame = av.VideoFrame.from_ndarray(
                    np.full((48, 64, 3), 80, dtype=np.uint8), format="rgb24"
                )
                output.mux(stream.encode(frame))
            output.mux(stream.encode())
        paths.append(path)
    runtime = replace(runtime, video_paths=tuple(paths), camera_ids=("cam0", "cam1"))
    project, notice = load_or_create_project(runtime)
    assert notice is not None
    assert notice.warning == missing_timestamp
    assert [s.offset_sec for s in project.sources] == (
        [0, 0] if missing_timestamp else [1, 0]
    )
    with TestClient(create_app(runtime, project, startup_notice=notice)) as client:
        response = client.get("/api/startup-notice").json()
        assert response["warning"] == missing_timestamp
        if missing_timestamp:
            assert (
                "cam1" in response["message"] and "creation_time" in response["message"]
            )
        assert 'id="startup-notice"' in client.get("/").text
        assert client.get("/api/frame/0?time=0&revision=0").headers[
            "x-frame-index"
        ] == ("0" if missing_timestamp else "10")
        assert (
            client.post(
                "/api/edit",
                json={"revision": 0, "action": "offsets", "offsets_sec": [0.75, 0]},
            ).status_code
            == 200
        )
        assert client.get("/api/startup-notice").json() == response
    project, notice = load_or_create_project(runtime)
    assert [s.offset_sec for s in project.sources] == [0.75, 0]
    with TestClient(create_app(runtime, project, startup_notice=notice)) as client:
        assert client.get("/api/startup-notice").json() is None
