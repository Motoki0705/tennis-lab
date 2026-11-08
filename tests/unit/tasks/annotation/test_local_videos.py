from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from src.tasks.annotation import local_videos
from src.tasks.annotation.configuration import configure_local_videos
from src.tasks.annotation.local_videos import LocalVideoConfigurationReceipt
from src.tasks.annotation.reviews import read_local_context, requests
from src.tasks.base.annotation.artifacts import read_document, write_document
from src.tasks.base.annotation.collection.local import LocalInputLedger
from src.tasks.base.annotation.editing.contracts import (
    BlankAnnotationContext,
    ReviewContext,
)
from src.tasks.base.annotation.media import probe_video
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.plan import (
    LocalAnnotationPlan,
    file_digest,
    load_execution_plan,
)
from src.tasks.base.annotation.resources import ResourceManager, ResourceRequest
from src.tasks.base.annotation.workers.contracts import WorkRequest
from src.tasks.base.annotation.workers.ledger import WorkerLedger
from tests.support.tasks.local_annotation import local_configuration


def test_concurrent_local_resubmission_registers_one_workset(tmp_path: Path) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    data = local_configuration(layout)
    barrier = Barrier(2)

    def register(_):
        barrier.wait(timeout=10)
        return configure_local_videos(layout, data)

    with ThreadPoolExecutor(max_workers=2) as pool:
        references = list(pool.map(register, range(2)))
    assert references[0] == references[1]
    with LocalInputLedger(layout).registry.transaction() as db:
        assert db.execute("SELECT count(*) FROM campaigns").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM worker_tasks").fetchone()[0] == 2
        assert (
            db.execute("SELECT count(*) FROM local_annotation_inputs").fetchone()[0]
            == 2
        )


@pytest.mark.parametrize(
    "boundary", ["clip", "request", "local_commit", "worker_commit", "result"]
)
def test_local_registration_recovers_published_inputs_without_resetting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    data = local_configuration(layout)

    with monkeypatch.context() as patch:
        if boundary in {"clip", "request"}:
            name = (
                "prepare_frame_range" if boundary == "clip" else "local_video_request"
            )
            original = getattr(local_videos, name)

            def interrupt(*args, **kwargs):
                original(*args, **kwargs)
                raise RuntimeError("interrupted after publication")

            patch.setattr(local_videos, name, interrupt)
        elif boundary in {"local_commit", "worker_commit"}:
            register = WorkerLedger.register

            def interrupt_worker(self, *args, **kwargs):
                if boundary == "worker_commit":
                    register(self, *args, **kwargs)
                raise RuntimeError("interrupted after publication")

            patch.setattr(WorkerLedger, "register", interrupt_worker)
        else:
            write = local_videos.write_document

            def interrupt_result(layout, path, document):
                reference = write(layout, path, document)
                if path.name == "result.json":
                    raise RuntimeError("interrupted after publication")
                return reference

            patch.setattr(local_videos, "write_document", interrupt_result)
        with pytest.raises(RuntimeError, match="interrupted after publication"):
            configure_local_videos(layout, data)
    frozen = {
        p: (p.stat().st_ino, p.read_bytes())
        for p in layout.root.rglob("*")
        if p.is_file()
        and p.suffix not in {".lock", ".sqlite3", ".sqlite3-wal", ".sqlite3-shm"}
    }
    first = configure_local_videos(layout, data)
    assert configure_local_videos(layout, data) == first
    for path, identity in frozen.items():
        assert (path.stat().st_ino, path.read_bytes()) == identity
    receipt = read_document(layout, first, LocalVideoConfigurationReceipt)
    plan = load_execution_plan(layout.artifact_path(receipt.plan))
    assert isinstance(plan, LocalAnnotationPlan)
    assert len(requests(layout, plan, receipt.plan)) == 2
    with WorkerLedger(layout).registry.transaction() as db:
        assert db.execute("SELECT count(*) FROM worker_tasks").fetchone()[0] == 2
        assert (
            db.execute("SELECT count(*) FROM local_annotation_inputs").fetchone()[0]
            == 2
        )
        assert db.execute("SELECT count(*) FROM worker_attempts").fetchone()[0] == 0


@pytest.mark.parametrize(
    "change",
    [
        "camera",
        "pts",
        "source_bytes",
        "group",
        "target",
        "publication",
        "code",
        "campaign",
    ],
)
def test_local_registration_rejects_reusing_a_different_plan_or_input_version(
    tmp_path: Path,
    change: str,
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    data = local_configuration(layout)
    first = configure_local_videos(layout, data)
    receipt = read_document(layout, first, LocalVideoConfigurationReceipt)
    contents = {
        ref: layout.artifact_path(ref).read_bytes()
        for entry in receipt.registrations
        for ref in (entry.context, entry.request)
    }
    video = data.videos[0]
    expected = video.annotation
    if change == "camera":
        expected = expected.model_copy(
            update={"source": expected.source.model_copy(update={"camera_id": "cam1"})}
        )
    elif change == "pts":
        expected = expected.model_copy(
            update={
                "frames": (
                    expected.frames[0].model_copy(
                        update={"pts": expected.frames[0].pts + 1}
                    ),
                    *expected.frames[1:],
                )
            }
        )
    elif change == "source_bytes":
        path = Path(video.path)
        shutil.copyfile(data.videos[1].path, path)
        timeline = probe_video(path)
        # Same five source frames and same selected frame count/PTS; different pixels.
        expected = expected.model_copy(
            update={
                "source": expected.source.model_copy(
                    update={
                        "media_sha256": file_digest(path),
                        "size_bytes": path.stat().st_size,
                        "width": timeline.width,
                        "height": timeline.height,
                        "time_base": str(timeline.time_base),
                        "nominal_fps": str(timeline.rate),
                    }
                )
            }
        )
    elif change == "group":
        expected = expected.model_copy(
            update={
                "source": expected.source.model_copy(
                    update={"recording_id": "recording-b"}
                )
            }
        )
    elif change == "target":
        expected = expected.model_copy(
            update={
                "target": expected.target.model_copy(
                    update={"track_id": "another-ball"}
                )
            }
        )
    changed = data.model_copy(
        update={
            "videos": (
                video.model_copy(update={"annotation": expected}),
                data.videos[1],
            )
        }
    )
    if change == "publication":
        changed = data.model_copy(update={"publish_dataset": True})
    elif change == "code":
        changed = data.model_copy(
            update={
                "campaign": data.campaign.model_copy(update={"code_commit": "c" * 40})
            }
        )
    elif change == "campaign":
        changed = data.model_copy(
            update={
                "campaign": data.campaign.model_copy(
                    update={"campaign_id": "another-campaign"}
                )
            }
        )
    with pytest.raises(ValueError, match="immutable plan|frame/PTS|different plan"):
        configure_local_videos(layout, changed)
    if change == "campaign":
        assert not layout.path("campaigns", "another-campaign").exists()
    assert layout.artifact_path(first).exists()
    for reference, payload in contents.items():
        assert layout.artifact_path(reference).read_bytes() == payload
    with LocalInputLedger(layout).registry.transaction() as db:
        assert db.execute("SELECT count(*) FROM worker_tasks").fetchone()[0] == 2


def test_blank_context_rejects_mixed_camera_images_and_prediction_context(
    tmp_path: Path,
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    data = local_configuration(layout)
    receipt = read_document(
        layout, configure_local_videos(layout, data), LocalVideoConfigurationReceipt
    )
    first, second = (
        read_document(layout, e.context, BlankAnnotationContext)
        for e in receipt.registrations
    )
    mixed = write_document(
        layout,
        layout.path("campaigns", "local-fixture", "mixed-context.json"),
        first.model_copy(update={"task_input": second.task_input}),
    )
    with pytest.raises(ValueError, match="different camera/input/frame/PTS"):
        read_local_context(layout, mixed, receipt.plan)
    with pytest.raises(ValueError):
        ReviewContext.model_validate_json(first.canonical_bytes())
    for entry in receipt.registrations:
        request = read_document(layout, entry.request, WorkRequest)
        assert request.stage == "review"
        assert request.adapter == "ball_video_annotation"


def test_blank_plan_uses_shared_resource_requests_without_gpu_queue(
    tmp_path: Path,
) -> None:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    data = local_configuration(layout)
    receipt = read_document(
        layout, configure_local_videos(layout, data), LocalVideoConfigurationReceipt
    )
    resources = ResourceManager(layout, receipt.scope)
    request = ResourceRequest(
        request_id="one-camera-render",
        plan=receipt.plan,
        stage="render",
        pss_bytes=16 * 1024**2,
        storage_bytes=1024,
        gpu_seconds=0,
    )
    resources.submit(request)
    resources.submit(request)
    assert resources._row(request)["state"] == "pending"
    with resources.registry.transaction() as db:
        assert db.execute("SELECT count(*) FROM resource_requests").fetchone()[0] == 1
    assert not (layout.repository / ".training_queue").exists()
