from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.tasks.annotation.storage_comparison import open_store
from src.tasks.base.annotation.artifacts import read_document, write_document
from src.tasks.base.annotation.contracts import ArtifactRef, content_digest
from src.tasks.base.annotation.paths import AnnotationLayout
from src.tasks.base.annotation.storage.comparison.contracts import (
    CapturedStore,
    ComparisonRequest,
    ReadLimits,
    StoreSpec,
)
from src.tasks.base.annotation.storage.comparison.engine import (
    ConvertedRecords,
    compare,
)
from src.tasks.base.annotation.storage.comparison.sources import capture
from src.tasks.base.annotation.storage.contracts import Inventory
from src.utils.data.image_record_store import ImageRecordWriter, encode_rgb_jpeg


@dataclass
class Fixture:
    layout: AnnotationLayout
    root: Path
    specification: ArtifactRef
    captured: ArtifactRef
    request: ComparisonRequest

    def run(self, name: str = "first", **changes: Any):
        request = self.request.model_copy(update=changes)
        request_ref = write_document(
            self.layout, self.layout.path("campaigns", "inputs", name + ".json"), request
        )
        return compare(self.layout, request_ref, self.output(name), open_store)

    def output(self, name: str = "first") -> Path:
        result: Path = self.layout.path("campaigns", "fixture", "jobs", name, "comparison")
        return result


def make_fixture(tmp_path: Path, adapter: str = "image_records") -> Fixture:
    layout = AnnotationLayout.testing(tmp_path / "annotation")
    root = tmp_path / "data" / "fixture"
    descriptor = None
    if adapter == "image_records":
        with ImageRecordWriter(root, shard_bytes=900) as writer:
            for index in range(3):
                rgb: NDArray[np.uint8] = np.zeros((32, 48, 3), np.uint8)
                rgb[:] = (200 - index * 15, 10, 40)
                writer.append(encode_rgb_jpeg(rgb), {"id": str(index), "absent": index == 0, "mask": [index, 1]})
            descriptor = writer.finish({"schema": "fixture.v1", "sources": ["one"]})
    elif adapter == "ball_frames":
        from tests.support.tasks.ball_detection.store import (
            ball,
            frame,
            write_store_clip,
        )

        write_store_clip(root, "train/fixture", [frame(0), frame(1, ball()), frame(2, ball("unresolved", None))])
        with np.load(root / "index.npz") as archive:
            arrays = dict(archive)
        arrays.update(teacher_heatmap=np.asarray([1, 2, 0], np.uint8), teacher_coordinate=np.asarray([0, 1, 0], np.uint8))
        np.savez(root / "index.npz", **arrays)
        metadata = json.loads((root / "metadata.json").read_bytes())
        metadata["schema_version"] = "ball_detection_frames.v2"
        (root / "metadata.json").write_text(json.dumps(metadata))
    elif adapter == "player_frames":
        from src.tasks.player_detection.generate_dataset.builder import build_dataset
        from tests.support.tasks.player_detection.chat_root import make_synthetic_root

        root = build_dataset(make_synthetic_root(tmp_path).config)
    else:
        raise AssertionError(adapter)
    spec = StoreSpec.model_validate_json(json.dumps({
        "adapter": adapter, "root": {"role": "data", "path": root.relative_to(tmp_path / "data").as_posix()}, "descriptor": descriptor,
    }))
    specification = write_document(layout, layout.path("campaigns", "inputs", "source.json"), spec)
    limits = ReadLimits(maximum_entries=10000, maximum_source_bytes=1024**2, maximum_frames=100, maximum_pixels_per_frame=65536)
    captured = write_document(layout, layout.path("campaigns", "inputs", "capture.json"), capture(layout, specification, limits, open_store))
    request = ComparisonRequest(source=captured, rows=(2, 0, 1), attempts=2, passes=2, limits=limits, maximum_written_bytes=8 * 1024**2, converted_shard_bytes=900)
    return Fixture(layout, root, specification, captured, request)


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    return make_fixture(tmp_path)


def hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("adapter", ["image_records", "ball_frames", "player_frames"])
def test_formats_copies_and_caches_preserve_same_inputs(tmp_path: Path, adapter: str) -> None:
    fixture = make_fixture(tmp_path, adapter)
    before = hashes(fixture.root)
    report = fixture.run()
    assert len(report.policies) == 12 and report.adopted_policy is None
    assert [row.row for row in report.rows] == [2, 0, 1]
    assert hashes(fixture.root) == before
    spec = read_document(fixture.layout, fixture.specification, StoreSpec)
    native = open_store(spec, fixture.root)
    converted = ConvertedRecords(fixture.output() / "converted")
    assert len(converted) == len(native)
    assert content_digest(native.metadata) == content_digest(converted.metadata)
    for row in range(len(native)):
        assert native.jpeg(row) == converted.jpeg(row)
        assert content_digest(native.labels(row)) == content_digest(converted.labels(row))
    labels: dict[str, Any] = dict(native.labels(0))
    if adapter == "ball_frames":
        assert labels["teacher_masks"]["heatmap"]["hex"] == "01"
    if adapter == "player_frames":
        assert labels["frame"]["confirmed_empty"]["hex"] == "01"
        assert labels["instances"]["bbox_xyxy"]["shape"] == [0, 4]
    for policy in report.policies:
        passes = [p for attempt in policy.attempts for p in attempt.passes]
        expected_decodes = {"decode": 12, "private_rgb": 6, "shared_rgb": 3}[policy.cache]
        assert sum(p.decoded_frames for p in passes) == expected_decodes
        assert sum(p.cache_hits for p in passes) == 12 - expected_decodes
        measured = read_document(fixture.layout, policy.inventory, Inventory)
        unique = {(e.device, e.inode): e for e in measured.entries}
        assert measured.allocated_bytes == sum(e.allocated_bytes for e in unique.values())
        if policy.shards == "share":
            shard_entries = [e for e in measured.entries if e.path.endswith(".bin")]
            assert len(shard_entries) == 3 * len({(e.device, e.inode) for e in shard_entries})
            assert measured.hardlinked_files >= len(shard_entries)
        else:
            copied = [e for e in measured.entries if "/policies/" in e.path and e.path.endswith(".bin")]
            assert len({(e.device, e.inode) for e in copied}) == len(copied)
    assert report.reserved_written_bytes <= fixture.request.maximum_written_bytes


@pytest.mark.parametrize("fault", ["bytes", "new_file", "symlink"])
def test_source_changes_fail_before_any_output(fixture: Fixture, fault: str) -> None:
    if fault == "bytes":
        path = next((fixture.root / "shards").glob("*.bin"))
        original = path.read_bytes()
        path.write_bytes(original[:-1] + b"x")
    elif fault == "new_file":
        (fixture.root / "unexpected.txt").write_text("another writer")
    else:
        (fixture.root / "external").symlink_to(fixture.layout.repository)
    with pytest.raises(ValueError, match="hash changed|file set changed|symlinks"):
        fixture.run()
    assert not fixture.output().exists()


@pytest.mark.parametrize("limit,value,pattern", [
    ("maximum_entries", 1, "entry budget"),
    ("maximum_source_bytes", 1, "byte limit"),
    ("maximum_frames", 1, "frame count"),
    ("maximum_pixels_per_frame", 1, "pixel limit"),
])
def test_inspection_and_comparison_are_bounded(fixture: Fixture, limit: str, value: int, pattern: str) -> None:
    limits = fixture.request.limits.model_copy(update={limit: value})
    with pytest.raises(ValueError, match=pattern):
        fixture.run(limits=limits)
    assert not (fixture.output() / "policies").exists()


@pytest.mark.parametrize("rows", [(0, 0), (3,)])
def test_read_sample_cannot_duplicate_or_escape_input(fixture: Fixture, rows: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="rows must be unique"):
        fixture.run(rows=rows)
    assert not fixture.output().exists()


def test_write_budget_preserves_failed_attempt_and_requires_fresh_owner(fixture: Fixture) -> None:
    before = hashes(fixture.root)
    with pytest.raises(ValueError, match="write budget"):
        fixture.run(maximum_written_bytes=1)
    assert fixture.output().is_dir() and hashes(fixture.root) == before
    retry = write_document(fixture.layout, fixture.layout.path("campaigns", "inputs", "retry.json"), fixture.request)
    with pytest.raises(FileExistsError):
        compare(fixture.layout, retry, fixture.output(), open_store)
    assert len(fixture.run("fresh").policies) == 12


@pytest.mark.parametrize("fault", ["labels", "rgb"])
def test_conversion_never_accepts_changed_teachers_or_channel_order(fixture: Fixture, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    if fault == "labels":
        monkeypatch.setattr(ConvertedRecords, "labels", lambda self, row: {})
    else:
        original = ConvertedRecords.rgb
        monkeypatch.setattr(ConvertedRecords, "rgb", lambda self, row: original(self, row)[:, :, ::-1])
    with pytest.raises(ValueError, match="teachers|RGB differs"):
        fixture.run()


def test_cache_corruption_is_an_error_without_decode_fallback(fixture: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    original = np.load

    def load(path: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(path, *args, **kwargs)
        if str(path).endswith(".npy"):
            result[0, 0, 0] ^= 1
        return result

    monkeypatch.setattr(np, "load", load)
    with pytest.raises(ValueError, match="cache or decoded RGB differs"):
        fixture.run()


def test_unsupported_hardlink_is_not_silently_replaced_by_copy(fixture: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    def reject(*args: Any, **kwargs: Any) -> None:
        raise OSError("fixture cannot link")

    monkeypatch.setattr(os, "link", reject)
    with pytest.raises(OSError, match="cannot link"):
        fixture.run()


def test_captured_reference_cannot_mark_metadata_as_shared_shard(fixture: Fixture) -> None:
    captured = read_document(fixture.layout, fixture.captured, CapturedStore)
    forged = captured.model_copy(update={"files": tuple(f.model_copy(update={"shared_shard": True}) for f in captured.files)})
    forged_ref = write_document(fixture.layout, fixture.layout.path("campaigns", "inputs", "forged.json"), forged)
    with pytest.raises(ValueError, match="shard sharing"):
        fixture.run(source=forged_ref)


def test_output_owner_cannot_redirect_into_another_directory(fixture: Fixture) -> None:
    outside = fixture.layout.repository / "unowned"
    outside.mkdir()
    output = fixture.output()
    output.parent.parent.mkdir(parents=True)
    output.parent.symlink_to(outside, target_is_directory=True)
    request = write_document(fixture.layout, fixture.layout.path("campaigns", "inputs", "redirect.json"), fixture.request)
    with pytest.raises(ValueError, match="symlinks"):
        compare(fixture.layout, request, output, open_store)
    assert not list(outside.iterdir())


def test_operations_require_stage_owner_and_reuse_completed_receipts(fixture: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.tasks.annotation.contracts import (
        CompareStorageOperation,
        InspectStorageOperation,
        OperationInput,
        OperationReceipt,
    )
    from src.tasks.annotation.operations import execute
    from tests.support.tasks.ball_detection.annotation import Environment

    env = Environment(fixture.layout.root)
    inspect = OperationInput(plan=env.plan_ref, operation=InspectStorageOperation(specification=fixture.specification, limits=fixture.request.limits))
    inspect_ref = write_document(fixture.layout, fixture.layout.path("campaigns", env.plan.campaign_id, "jobs", "inspect", "operation.json"), inspect)
    with pytest.raises(ValueError, match="resource-accounted stage"):
        execute(fixture.layout, inspect_ref)
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", inspect_ref.digest)
    receipt = read_document(fixture.layout, execute(fixture.layout, inspect_ref), OperationReceipt)
    request = write_document(fixture.layout, fixture.layout.path("campaigns", "inputs", "operation-comparison.json"), fixture.request.model_copy(update={"source": receipt.results[0]}))
    operation = OperationInput(plan=env.plan_ref, operation=CompareStorageOperation(request=request))
    op_ref = write_document(fixture.layout, fixture.layout.path("campaigns", env.plan.campaign_id, "jobs", "compare", "operation.json"), operation)
    monkeypatch.setenv("TENNIS_ANNOTATION_STAGE_OPERATION", op_ref.digest)
    complete = execute(fixture.layout, op_ref)
    assert read_document(fixture.layout, complete, OperationReceipt).status == "complete"
    assert execute(fixture.layout, op_ref) == complete
