from __future__ import annotations

import asyncio
import hashlib
import io
import json
import threading
import time
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.tennis_scene.chat_annotation.runtime.contracts import (
    Ball,
    BallAnnotation,
    ClipManifest,
    FrameMap,
    FrameRange,
    PlayerAnnotation,
    make_template,
)
from src.tennis_scene.chat_annotation.runtime.media import encode_video
from src.tennis_scene.chat_annotation.web.app import create_app
from src.tennis_scene.chat_annotation.web.catalog import Catalog, RevisionConflict
from src.tennis_scene.chat_annotation.web.handoff import make_handoff
from src.tennis_scene.chat_annotation.web.metrics import (
    frame_ranges,
    inspect_annotation,
)
from src.tennis_scene.chat_annotation.web.preview import (
    BoundedBuffer,
    PreviewBusy,
    PreviewCancelled,
    PreviewRenderer,
)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_root(
    tmp_path: Path, original: ClipManifest, vfr: bool = False
) -> tuple[Path, ClipManifest]:
    root = tmp_path / "annotations"
    video = root / "videos/sample/sample.mp4"
    video.parent.mkdir(parents=True)
    pts, duration = 0, 40
    mappings = []
    frames = []
    for index in range(12):
        duration = 80 if vfr and index % 3 == 0 else 40
        mappings.append(
            FrameMap(
                frame_index=index,
                source_frame_index=index,
                source_pts=1000 + pts,
                clip_pts=pts,
                duration_pts=duration,
                is_target=2 <= index < 10,
            )
        )
        frame = av.VideoFrame.from_ndarray(
            np.full((180, 320, 3), 35, dtype=np.uint8), format="bgr24"
        )
        frames.append((frame, pts, duration))
        pts += duration
    encode_video(
        video,
        width=320,
        height=180,
        time_base=Fraction(1, 1000),
        rate=Fraction(25),
        frames=frames,
        crf=18,
        preset="ultrafast",
    )
    manifest = original.model_copy(
        update={
            "source": original.source.model_copy(update={"filename": "sample.mp4"}),
            "width": 320,
            "height": 180,
            "filename": "sample.mp4",
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "bytes": video.stat().st_size,
            "time_base": "1/1000",
            "nominal_fps": "25",
            "frames": mappings,
            "media_range": FrameRange(start=0, stop=12),
            "target_range": FrameRange(start=2, stop=10),
        }
    )
    manifest = ClipManifest.model_validate(manifest.model_dump())
    save_json(
        root / "_preparation/sample/run/clips/clip_000/clip_manifest.json",
        manifest.model_dump(),
    )
    for target in ("ball", "player"):
        path = root / f"project_kits/{target}_detection/REQUEST.txt"
        path.parent.mkdir(parents=True)
        path.write_text(f"Official {target} REQUEST with schema", encoding="utf-8")
    return root, manifest


def target_data(
    manifest: ClipManifest, target: str, reviewed: int = 12
) -> dict[str, Any]:
    data: dict[str, Any] = make_template(manifest, target).model_dump()
    data["status"] = "completed" if reviewed == len(manifest.frames) else "partial"
    data["issues"] = (
        [] if data["status"] == "completed" else ["Unreviewed frames remain"]
    )
    for row in data["frames"][:reviewed]:
        row["reviewed"] = True
    return data


def accepted(
    root: Path, manifest: ClipManifest, target: str, reviewed: int = 12
) -> Path:
    path = root / f"annotated/processed/{target}/sample.json"
    save_json(path, target_data(manifest, target, reviewed))
    return path


def draft(
    root: Path,
    manifest: ClipManifest,
    generation: int,
    reviewed: int,
    *,
    bad_hash: bool = False,
) -> Path:
    directory = root / f"codex_campaign/generation_{generation:03d}"
    path = directory / "ball/annotation_sample.json"
    save_json(path, target_data(manifest, "ball", reviewed))
    manifest_path = root / "_preparation/sample/run/clips/clip_000/clip_manifest.json"
    save_json(
        directory / "assignment.json",
        [
            {
                "clip_id": "sample",
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            }
        ],
    )
    save_json(
        directory / "exported.json",
        [
            {
                "clip_id": "sample",
                "target": "ball",
                "annotation": str(path),
                "annotation_sha256": "0" * 64
                if bad_hash
                else hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    )
    save_json(directory / "parent_result.json", {"validation": {"errors": 0}})
    return path


def test_metrics_distinguish_missing_review_absence_and_uncertainty(
    manifest: ClipManifest,
) -> None:
    value = make_template(manifest, "ball")
    assert isinstance(value, BallAnnotation)
    value.frames[0].reviewed = True  # Confirmed empty, not missing data.
    value.frames[2].reviewed = True
    value.frames[2].balls = [
        Ball(
            track_id="b1",
            center_px=None,
            status="out_of_frame",
            interpolation_frames=None,
        )
    ]
    value.frames[3].reviewed = True
    value.frames[3].balls = [
        Ball(
            track_id="b1",
            center_px=None,
            status="unresolved",
            interpolation_frames=None,
        )
    ]
    stats, timeline = inspect_annotation(value, manifest, "ball")
    assert stats["reviewed"] == 3
    assert stats["absent_frames"] == 1
    assert stats["uncertain_frames"] == stats["null_objects"] == 1
    assert stats["out_of_frame_objects"] == 1
    assert stats["unreviewed_ranges"] == [[1, 2], [4, 12]]
    assert not timeline[2]["uncertain"]
    value.status = "completed"
    assert inspect_annotation(value, manifest, "ball")[0]["state"] == "invalid"
    assert frame_ranges([5, 1, 2, 2, 8]) == [[1, 3], [5, 6], [8, 9]]


def test_catalog_covers_done_missing_and_target_specific_coverage(
    tmp_path: Path, manifest: ClipManifest
) -> None:
    root, m = fixture_root(tmp_path, manifest)
    source = root / "videos/sample/sample.mp4"
    done = root / "done/sample/sample.mp4"
    done.parent.mkdir(parents=True)
    source.rename(done)
    accepted(root, m, "ball", 4)
    c = Catalog(root)
    row = c.snapshot()["clips"][0]
    assert row["location"] == "done" and row["state"] == "pending"
    assert row["targets"]["ball"]["reviewed"] == 4
    assert row["targets"]["player"]["state"] == "missing"
    assert row["targets"]["ball"]["unreviewed_ranges"] == [[4, 12]]
    assert c.snapshot()["summary"]["target_frames"] == 24  # Include context frames.


def test_draft_never_masks_accepted_and_bad_latest_is_not_skipped(
    tmp_path: Path, manifest: ClipManifest
) -> None:
    root, m = fixture_root(tmp_path, manifest)
    path = accepted(root, m, "ball", 2)
    draft(root, m, 1, 7)
    draft(root, m, 2, 12, bad_hash=True)
    c = Catalog(root)
    assert c.snapshot()["clips"][0]["targets"]["ball"]["reviewed"] == 2
    path.unlink()
    c.refresh()
    assert c.snapshot()["clips"][0]["targets"]["ball"]["state"] == "invalid"
    assert c.snapshot("accepted")["clips"][0]["targets"]["ball"]["state"] == "missing"


def test_deferred_draft_is_unreviewed_and_handoff_preserves_prior_rows(
    tmp_path: Path, manifest: ClipManifest
) -> None:
    root, m = fixture_root(tmp_path, manifest)
    draft(root, m, 1, 5)
    c = Catalog(root)
    version = c.default_version(c.clips["sample"], "ball", "working")
    assert version and version.origin == "draft"
    handoff = make_handoff(c, c.revision, [{"clip_id": "sample"}], "working")
    item = handoff["manifest"]["clips"][0]
    assert item["targets"]["ball"]["frame_ranges_half_open"] == [[5, 12]]
    assert item["targets"]["ball"]["reviewed_frames_to_preserve"] == 5
    assert item["targets"]["player"]["frame_ranges_half_open"] == [[0, 12]]
    assert "Official ball REQUEST" in handoff["text"]
    assert "Official player REQUEST" not in handoff["text"]
    assert "Official player REQUEST" in handoff["requests"]["player"]["text"]
    assert set(handoff["requests"]["ball"]["manifest"]["clips"][0]["targets"]) == {
        "ball"
    }
    assert item["targets"]["ball"]["annotation"]["origin"] == "draft"


def test_stale_json_manifest_and_foreign_versions_rejected(
    tmp_path: Path, manifest: ClipManifest
) -> None:
    root, m = fixture_root(tmp_path, manifest)
    path = accepted(root, m, "ball", 4)
    c = Catalog(root)
    version = c.default_version(c.clips["sample"], "ball", "working")
    assert version
    with pytest.raises(KeyError):
        c.select("sample", "player", version.id)
    save_json(path, target_data(m, "ball", 6))
    with pytest.raises(RevisionConflict):
        c.read_version(version)
    with pytest.raises(RevisionConflict):
        make_handoff(c, c.revision, [{"clip_id": "sample"}], "working")
    old_revision = c.revision
    c.refresh()
    with pytest.raises(RevisionConflict):
        c.check_revision(old_revision)
    mp = c.clips["sample"].manifest_path
    data = json.loads(mp.read_text())
    data["source"]["title"] = "updated"
    save_json(mp, data)
    with pytest.raises(RevisionConflict):
        c.check_clip_current(c.clips["sample"])


def test_raw_submission_requires_choice_and_does_not_become_accepted(
    tmp_path: Path, manifest: ClipManifest
) -> None:
    root, m = fixture_root(tmp_path, manifest)
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("ball_sample.json", json.dumps(target_data(m, "ball")))
    content = memory.getvalue()
    path = root / "annotated/raw" / f"{hashlib.sha256(content).hexdigest()}.zip"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    c = Catalog(root)
    row = c.snapshot()["clips"][0]
    assert row["state"] == "selection_required"
    assert row["targets"]["ball"]["accepted_state"] == "missing"
    with pytest.raises(ValueError, match="版を選択"):
        make_handoff(c, c.revision, [{"clip_id": "sample"}], "working")
    with pytest.raises(ValueError, match="版を選択"):
        make_handoff(
            c,
            c.revision,
            [{"clip_id": "sample", "versions": {"ball": None, "player": None}}],
            "working",
        )
    version = c.clips["sample"].versions["ball"][0]
    assert c.read_version(version)[1]["state"] == "completed"


@pytest.mark.parametrize("vfr", [False, True])
def test_web_overlay_is_memory_only_and_preserves_every_pts(
    tmp_path: Path, manifest: ClipManifest, vfr: bool
) -> None:
    root, m = fixture_root(tmp_path, manifest, vfr)
    ball_path = accepted(root, m, "ball")
    value = json.loads(ball_path.read_text())
    for row in value["frames"]:
        row["balls"] = [
            {
                "track_id": "b1",
                "center_px": [100, 60],
                "status": "visible",
                "interpolation_frames": None,
            }
        ]
    save_json(ball_path, value)
    accepted(root, m, "player")
    files_before = {
        p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    app = create_app(root)
    with TestClient(app) as client:
        catalog = client.get("/api/catalog").json()
        row = catalog["clips"][0]
        assert row["state"] == "completed"
        response = client.post(
            "/api/clips/sample/preview",
            json={
                "revision": catalog["revision"],
                "versions": {
                    t: row["targets"][t]["selected_version"] for t in ("ball", "player")
                },
                "target": "both",
            },
        )
        assert response.status_code == 200, (
            response.text[:300] if response.status_code != 200 else ""
        )
        assert response.headers["x-preview-storage"] == "memory-only"
        assert response.headers["cache-control"] == "no-store"
        with av.open(io.BytesIO(response.content)) as video:
            frames = list(video.decode(video=0))
            assert len(frames) == 12 and frames[0].height == m.height + 80
            assert [
                (f.pts * f.time_base, f.duration * f.time_base) for f in frames
            ] == [
                (
                    f.clip_pts * Fraction(m.time_base),
                    f.duration_pts * Fraction(m.time_base),
                )
                for f in m.frames
            ]
            pixel = frames[0].to_ndarray(format="bgr24")[60, 100]
            assert (
                pixel[1] > 140 and pixel[2] > 140
            )  # Yellow center, actually rendered.
        assert not app.state.previews.busy
        original = client.get(
            f"/api/clips/sample/video?revision={catalog['revision']}",
            headers={"Range": "bytes=0-15"},
        )
        assert original.status_code == 206 and len(original.content) == 16
        assert (
            client.post(
                "/api/refresh", headers={"Origin": "https://other.example"}
            ).status_code
            == 403
        )
        assert client.get("/static/../catalog.py").status_code == 404
        assert client.get("/", headers={"Host": "other.example"}).status_code == 400
        assert "Annotation Desk" in client.get("/").text
    assert files_before == {
        p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_memory_buffer_limits_and_render_cancellation_release_worker(
    tmp_path: Path, manifest: ClipManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = threading.Event()
    with BoundedBuffer(4, event) as buffer:
        buffer.write(b"1234")
        with pytest.raises(OSError, match="メモリ上限"):
            buffer.write(b"5")
        event.set()
        with pytest.raises(PreviewCancelled):
            buffer.write(b"")
    renderer = PreviewRenderer()
    exited = threading.Event()

    def slow_render(video, manifest, annotation, cancelled):
        try:
            while not cancelled.is_set():
                time.sleep(0.005)
            raise PreviewCancelled("cancelled")
        finally:
            exited.set()

    monkeypatch.setattr(renderer, "_render", slow_render)

    async def scenario():
        async def disconnected():
            return True

        with pytest.raises(PreviewCancelled):
            await renderer.render(
                tmp_path / "unused.mp4",
                manifest,
                make_template(manifest, "ball"),
                disconnected,
            )
        assert exited.is_set() and not renderer.busy

    try:
        asyncio.run(scenario())
    finally:
        renderer.close()


def test_wrong_target_schema_and_provenance_hash_are_not_complete(
    tmp_path: Path, manifest: ClipManifest
) -> None:
    root, m = fixture_root(tmp_path, manifest)
    path = accepted(root, m, "ball")
    save_json(path, target_data(m, "player"))
    assert Catalog(root).snapshot()["clips"][0]["state"] == "invalid"
    accepted(root, m, "ball")
    save_json(
        root / "annotated/processing/hash.json",
        {
            "state": "completed",
            "members": [
                {
                    "clip_id": "sample",
                    "target": "ball",
                    "decision": "accepted",
                    "output_sha256": "f" * 64,
                }
            ],
        },
    )
    assert Catalog(root).snapshot()["clips"][0]["state"] == "invalid"


def test_unreviewed_coordinates_do_not_count_as_localization(
    manifest: ClipManifest,
) -> None:
    value = make_template(manifest, "player")
    assert isinstance(value, PlayerAnnotation)
    # Template empty/unreviewed must never be called confirmed absence.
    stats, _ = inspect_annotation(value, manifest, "player")
    assert stats["state"] == "unreviewed"
    assert stats["localized_frames"] == stats["absent_frames"] == 0


def test_malformed_receipts_are_diagnostics_not_missing_inventory(
    tmp_path: Path, manifest: ClipManifest
) -> None:
    root, _ = fixture_root(tmp_path, manifest)
    save_json(root / "codex_campaign/state.json", {"active_workers": []})
    save_json(root / "annotated/processing/bad.json", [])
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("bad.json", "[]")
    raw = root / "annotated/raw/invalid.zip"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(memory.getvalue())
    catalog = Catalog(root)
    assert len(catalog.clips) == 1
    assert len(catalog.diagnostics) == 3


def test_api_rejects_malformed_selection_and_empty_refinement(
    tmp_path: Path, manifest: ClipManifest
) -> None:
    root, m = fixture_root(tmp_path, manifest)
    accepted(root, m, "ball")
    accepted(root, m, "player")
    with TestClient(create_app(root)) as client:
        revision = client.get("/api/catalog").json()["revision"]
        assert (
            client.post(
                "/api/handoff",
                json={"revision": revision, "selections": [{"clip_id": []}]},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/handoff",
                json={
                    "revision": revision,
                    "selections": [{"clip_id": "sample", "versions": "bad"}],
                },
            ).status_code
            == 422
        )
        response = client.post(
            "/api/handoff",
            json={
                "revision": revision,
                "selections": [{"clip_id": "sample", "refine": True}],
            },
        )
        assert response.status_code == 422 and "不確実性" in response.json()["detail"]


def test_cancelling_request_releases_encoder_before_next_render(
    tmp_path: Path, manifest: ClipManifest, monkeypatch: pytest.MonkeyPatch
) -> None:
    renderer = PreviewRenderer()
    exited = threading.Event()

    def render_until_cancelled(video, manifest, annotation, cancelled):
        try:
            while not cancelled.is_set():
                time.sleep(0.005)
            raise PreviewCancelled("cancelled")
        finally:
            exited.set()

    monkeypatch.setattr(renderer, "_render", render_until_cancelled)

    async def scenario():
        async def connected():
            return False

        task = asyncio.create_task(
            renderer.render(
                tmp_path / "unused",
                manifest,
                make_template(manifest, "ball"),
                connected,
            )
        )
        await asyncio.sleep(0.02)
        with pytest.raises(PreviewBusy):
            await renderer.render(
                tmp_path / "unused",
                manifest,
                make_template(manifest, "ball"),
                connected,
            )
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited.is_set() and not renderer.busy

    try:
        asyncio.run(scenario())
    finally:
        renderer.close()


def test_preview_render_failure_leaves_no_file_or_busy_slot(
    tmp_path: Path, manifest: ClipManifest
) -> None:
    root, m = fixture_root(tmp_path, manifest)
    accepted(root, m, "ball")
    accepted(root, m, "player")
    renderer = PreviewRenderer(max_bytes=10)
    before = set(root.rglob("*"))
    with TestClient(create_app(root, renderer=renderer)) as client:
        data = client.get("/api/catalog").json()
        row = data["clips"][0]
        response = client.post(
            "/api/clips/sample/preview",
            json={
                "revision": data["revision"],
                "versions": {
                    t: row["targets"][t]["selected_version"] for t in ("ball", "player")
                },
            },
        )
        assert response.status_code == 422
        assert not renderer.busy
    assert set(root.rglob("*")) == before


def test_source_filename_not_source_id_defines_video_directory(
    tmp_path: Path, manifest: ClipManifest
) -> None:
    root, m = fixture_root(tmp_path, manifest)
    source = root / "videos/sample/sample.mp4"
    destination = root / "videos/local-recording/sample.mp4"
    destination.parent.mkdir(parents=True)
    source.rename(destination)
    m.source.filename = "local-recording.mp4"
    save_json(
        root / "_preparation/sample/run/clips/clip_000/clip_manifest.json",
        m.model_dump(),
    )
    catalog = Catalog(root)
    assert catalog.clips["sample"].video == destination
    assert catalog.snapshot()["clips"][0]["video_available"] is True
