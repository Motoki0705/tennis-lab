"""CPU fault injection at the renderer/Drive boundary, using real packed stores."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.synthetic_data_generation.dataset.court.scale import publication
from src.synthetic_data_generation.dataset.court.scale.core import (
    GeometryRecipe,
    QualityPolicy,
    TrajectoryMethod,
)
from src.synthetic_data_generation.dataset.court.scale.durability import (
    DurableGeneration,
    fetch_completed_dataset,
    restore_inputs,
)
from src.synthetic_data_generation.dataset.court.scale.planning import (
    build_candidate_plan,
)
from src.synthetic_data_generation.dataset.court.scale.recipes import ProductionRecipe
from src.synthetic_data_generation.rendering.nht import NHTRenderArrays
from src.utils.data.image_record_store import file_sha256


class Remote:
    def __init__(self, root):
        self.root = root
        self.uploaded = []
        self.fail_cache_ack = False
        self.fail_failure_ack = False
        self.downloaded = []

    def exists(self, relative):
        return (self.root / relative).exists()

    def put(self, local, relative):
        if self.fail_cache_ack and relative.startswith("receipts/caches/"):
            raise RuntimeError("Drive acknowledgement failed")
        if self.fail_failure_ack and relative.startswith("receipts/failures/"):
            raise RuntimeError("Failed-staging acknowledgement failed")
        destination = self.root / relative
        if destination.exists():
            if destination.read_bytes() != local.read_bytes():
                raise ValueError("immutable remote conflict")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, destination)
        self.uploaded.append(relative)
        return {
            "path": relative,
            "id": f"fixture-{file_sha256(local)}",
            "verified": True,
        }

    def verify(self, local, relative):
        if (self.root / relative).read_bytes() != local.read_bytes():
            raise ValueError("immutable remote conflict")
        return {
            "path": relative,
            "id": f"fixture-{file_sha256(local)}",
            "verified": True,
        }

    def get(self, relative, local, *, missing_ok=False):
        assert not local.exists()
        if missing_ok and not self.exists(relative):
            return False
        shutil.copyfile(self.root / relative, local)
        self.downloaded.append(relative)
        return True


class Telemetry:
    def __init__(self):
        self.active = False
        self.release_count = 0

    def begin(self):
        assert not self.active
        self.active = True

    def end(self):
        assert self.active
        self.active = False
        return {
            "sampled_peak_bytes": 123456,
            "source": "cpu_test_double",
            "peak_is_lower_bound": True,
        }

    def released(self):
        assert not self.active
        self.release_count += 1
        return {
            "schema": "court_scale_renderer_release_v1",
            "renderer_context_released": True,
            "compute_processes_after": [],
        }


class Renderer:
    def __init__(self, *, fail_call=None):
        self.calls = 0
        self.ids = []
        self.fail_call = fail_call

    def __call__(self, scene, records):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("renderer crashed")
        for record in records:
            self.ids.append(record["sample_id"])
            h, w = record["height"], record["width"]
            yield (
                record["sample_id"],
                NHTRenderArrays(
                    rgb=np.full((h, w, 3), 0.5, dtype=np.float32),
                    alpha=np.ones((h, w, 1), dtype=np.float32),
                    depth=np.ones((h, w, 1), dtype=np.float32),
                ),
            )


@pytest.fixture(params=["comparison", "production"])
def frozen(tmp_path, captured_cameras, multi_court_layout, monkeypatch, request):
    monkeypatch.setattr(
        publication.AlignmentResult,
        "from_dict",
        lambda raw: SimpleNamespace(layout=multi_court_layout),
    )
    production = None
    if request.param == "production":
        production = ProductionRecipe.from_dict(
            publication.read_json(
                Path(
                    "src/synthetic_data_generation/configs/dataset/court/scale_stratified_counter_production_v1.json"
                )
            )
        )
        production = replace(
            production,
            geometry=replace(production.geometry, width=80, height=48, base_step_m=8),
        )
    candidate = build_candidate_plan(
        scene_id="fixture",
        method=TrajectoryMethod.STRATIFIED_COUNTER,
        recipe=production.geometry
        if production
        else GeometryRecipe(width=80, height=48, base_step_m=8),
        seed=24,
        trajectory_count=3,
        density_level=1,
        cameras=captured_cameras,
        layout=multi_court_layout,
    )
    root = tmp_path / "vm"
    path = publication.freeze_request(
        root,
        plans=[candidate],
        assets=[{"scene_id": "fixture", "alignment": {"fixture": True}}],
        quality=production.quality if production else QualityPolicy(),
        renderer={"schema": "fixture_renderer_v1"},
        production_recipe=production,
    )
    return root, path, Remote(tmp_path / "drive")


def test_vm_loss_after_ack_restores_and_never_redraws_completed_cache(frozen, tmp_path):
    root, request, remote = frozen
    before = Renderer()
    telemetry = Telemetry()
    run = DurableGeneration(root, request, remote, telemetry, stop_after_caches=1)
    assert run.run(before) is None
    assert before.ids
    assert telemetry.release_count == 1
    assert not remote.exists("receipts/completed.json")
    assert not list(root.glob("*.dataset.json"))
    first_metrics = {p.name: p.read_bytes() for p in root.glob("cache-metrics/*.json")}

    # Recreate only the request closure from Drive in an empty VM directory.
    recovered = tmp_path / "new-vm"
    restored_request = restore_inputs(
        recovered, remote, request_sha256=file_sha256(request)
    )
    after = Renderer()
    resumed = DurableGeneration(recovered, restored_request, remote, Telemetry())
    version_path = resumed.run(after)
    assert version_path is not None
    assert set(before.ids).isdisjoint(after.ids)
    assert resumed.events[0]["reused"] and resumed.events[0]["restored_from_drive"]
    input_receipt = publication.read_json(remote.root / "receipts/request.json")
    assert remote.downloaded.count(input_receipt["archive_path"]) == 1
    assert all(
        (recovered / "cache-metrics" / name).read_bytes() == raw
        for name, raw in first_metrics.items()
    )
    version = publication.load_dataset_version(version_path, verify_images=True)
    assert version["scenes"][0]["counts"]["render_cache_reused"] == len(before.ids)
    assert version["scenes"][0]["counts"]["failed"] == 0

    fetched = fetch_completed_dataset(
        tmp_path / "consumer", remote, request_sha256=file_sha256(request)
    )
    assert fetched.name == version_path.name
    assert publication.load_dataset_version(fetched, verify_images=True) == version
    report = publication.read_json(
        fetched.parent / f"reports/{file_sha256(request)}.json"
    )
    scene = report["scenes"][0]
    assert scene["counts_ref"] == {
        "path": version_path.name,
        "json_pointer": "/scenes/0/counts",
    }
    assert (
        scene["rejection_reasons_ref"]["json_pointer"] == "/scenes/0/rejection_reasons"
    )
    assert scene["gpu_sampled_peak_bytes"] == 123456
    assert scene["jpeg_shard_bytes"] == sum(
        p.stat().st_size for p in fetched.parent.glob("stores/*/samples/shards/*.bin")
    )
    assert scene["index_bytes"] == sum(
        p.stat().st_size for p in fetched.parent.glob("stores/*/samples/index.npz")
    )
    assert scene["generation_seconds"] == sum(
        g["generation_seconds"] for g in scene["groups"]
    )
    assert report["renderer_release"]["renderer_context_released"] is True
    assert all("qa_accepted" not in g for g in scene["groups"])


def test_renderer_failure_needs_explicit_retry_and_preserves_failure(frozen):
    root, request, remote = frozen
    failed = Renderer(fail_call=2)
    with pytest.raises(RuntimeError, match="renderer crashed"):
        DurableGeneration(root, request, remote, Telemetry()).run(failed)
    staging = list(root.glob(".staging/*/failure.json"))
    assert len(staging) == 1
    key = staging[0].parent.name
    failure_bytes = staging[0].read_bytes()
    assert not remote.exists(f"receipts/caches/{key}.json")
    assert not remote.exists("receipts/completed.json")
    assert not list(root.glob("*.dataset.json"))
    assert publication.read_json(staging[0])["pending_ids"]

    untouched = Renderer()
    with pytest.raises(RuntimeError, match="--retry-staging"):
        DurableGeneration(root, request, remote, Telemetry()).run(untouched)
    assert not untouched.ids
    assert staging[0].read_bytes() == failure_bytes

    after = Renderer()
    run = DurableGeneration(root, request, remote, Telemetry(), retry_staging=True)
    result = run.run(after)
    assert result is not None
    assert set(failed.ids).isdisjoint(after.ids)
    assert not staging[0].exists()
    archived = list(root.glob(f"failed-attempts/*/{key}/failure.json"))
    assert len(archived) == 1 and archived[0].read_bytes() == failure_bytes
    assert (
        publication.read_json(archived[0].with_name("retry.json"))["action"]
        == "archived_before_explicit_retry"
    )
    assert len(list(remote.root.glob(f"receipts/failures/*-{key}.json"))) == 1


def test_drive_failure_blocks_next_cache_and_retries_local_completion(frozen):
    root, request, remote = frozen
    remote.fail_cache_ack = True
    first = Renderer()
    with pytest.raises(RuntimeError, match="Drive acknowledgement failed"):
        DurableGeneration(root, request, remote, Telemetry()).run(first)
    assert first.calls == 1
    assert len(list(root.glob("stores/*/completion.json"))) == 1
    assert not remote.exists("receipts/completed.json")
    assert not list(root.glob("*.dataset.json"))
    remote.fail_cache_ack = False
    second = Renderer()
    result = DurableGeneration(root, request, remote, Telemetry()).run(second)
    assert result is not None
    assert set(first.ids).isdisjoint(second.ids)


def test_corrupted_drive_cache_is_rejected_before_any_further_render(frozen, tmp_path):
    root, request, remote = frozen
    DurableGeneration(root, request, remote, Telemetry(), stop_after_caches=1).run(
        Renderer()
    )
    receipt_path = next(remote.root.glob("receipts/caches/*.json"))
    receipt = publication.read_json(receipt_path)
    archive = remote.root / receipt["archive_path"]
    with archive.open("r+b") as stream:
        stream.write(b"corruption")
    new_root = tmp_path / "new-vm"
    restored = restore_inputs(new_root, remote, request_sha256=file_sha256(request))
    renderer = Renderer()
    with pytest.raises(ValueError, match="Drive archive size/hash mismatch"):
        DurableGeneration(new_root, restored, remote, Telemetry()).run(renderer)
    assert not renderer.ids
    assert not list(new_root.glob("stores/*/completion.json"))
    assert not remote.exists("receipts/completed.json")


def test_uncommitted_remote_object_is_never_used_as_a_cache(frozen, tmp_path):
    root, request, remote = frozen
    remote.fail_cache_ack = True
    first = Renderer()
    with pytest.raises(RuntimeError, match="acknowledgement"):
        DurableGeneration(root, request, remote, Telemetry()).run(first)
    assert list(remote.root.glob("objects/*.tar"))
    remote.fail_cache_ack = False
    new_root = tmp_path / "new-vm"
    restored = restore_inputs(new_root, remote, request_sha256=file_sha256(request))
    second = Renderer()
    result = DurableGeneration(new_root, restored, remote, Telemetry()).run(second)
    assert result is not None
    assert set(first.ids) <= set(second.ids)
    counts = publication.load_dataset_version(result)["scenes"][0]["counts"]
    assert counts["render_cache_reused"] == 0


def test_fetch_requires_completed_receipt_and_pinned_request(frozen, tmp_path):
    root, request, remote = frozen
    DurableGeneration(root, request, remote, Telemetry(), stop_after_caches=1).run(
        Renderer()
    )
    with pytest.raises(ValueError, match="no completed dataset receipt"):
        fetch_completed_dataset(
            tmp_path / "consumer", remote, request_sha256=file_sha256(request)
        )
    with pytest.raises(ValueError, match="identity disagrees"):
        restore_inputs(tmp_path / "wrong", remote, request_sha256="0" * 64)


def test_release_failure_blocks_final_drive_receipt_and_records_failure(frozen):
    root, request, remote = frozen

    class Unreleased(Telemetry):
        def released(self):
            raise RuntimeError("renderer context still present")

    with pytest.raises(RuntimeError, match="cleanup/release could not be verified"):
        DurableGeneration(root, request, remote, Unreleased()).run(Renderer())
    assert not remote.exists("receipts/completed.json")
    attempt = publication.read_json(next(root.glob("attempts/*.json")))
    assert attempt["outcome"] == "failed"
    assert attempt["renderer_release"]["renderer_context_released"] is False
    assert "renderer context still present" in attempt["error"]


def test_conflicting_local_cache_bytes_are_preserved_and_refused(frozen):
    root, request, remote = frozen
    DurableGeneration(root, request, remote, Telemetry(), stop_after_caches=1).run(
        Renderer()
    )
    completion = next(root.glob("stores/*/completion.json"))
    completion.write_text("corruption")
    renderer = Renderer()
    with pytest.raises(ValueError, match="Local bytes conflict"):
        DurableGeneration(root, request, remote, Telemetry()).run(renderer)
    assert completion.read_text() == "corruption"
    assert not renderer.ids


def test_retry_archive_save_finishes_before_drawing_after_upload_failure(frozen):
    root, request, remote = frozen
    with pytest.raises(RuntimeError, match="renderer crashed"):
        DurableGeneration(root, request, remote, Telemetry()).run(Renderer(fail_call=2))
    remote.fail_failure_ack = True
    before = Renderer()
    with pytest.raises(RuntimeError, match="Failed-staging acknowledgement"):
        DurableGeneration(root, request, remote, Telemetry(), retry_staging=True).run(
            before
        )
    assert not before.ids
    archive = next(root.glob("failed-attempts/*/*/failure.json"))
    failure = archive.read_bytes()
    remote.fail_failure_ack = False
    after = Renderer()
    assert DurableGeneration(root, request, remote, Telemetry()).run(after) is not None
    assert after.ids
    assert archive.read_bytes() == failure
    assert len(list(remote.root.glob("receipts/failures/*.json"))) == 1
