"""Tests for compact cross-domain synthetic dataset runtime contracts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.runtime import (
    PERFORMANCE_SCHEMA,
    BackgroundArrays,
    ChunkReader,
    ChunkWriter,
    DatasetPerformanceBudget,
    DatasetPerformanceMetrics,
    FinalDatasetAssembler,
    ForegroundDelta,
    ForegroundDeltaBatch,
    RenderSampleKey,
    RenderSession,
    SharedBackgroundStore,
    discard_working_directory,
    load_performance_metrics,
    materialize_logical_sample,
    sparse_delta_from_composite,
    write_performance_metrics,
)
from src.synthetic_data_generation.rendering.nht.contracts import (
    NHTRenderArrays,
    NHTRenderRecord,
    NHTRenderResult,
)


def _background(camera_id: str = "camera-z") -> BackgroundArrays:
    return BackgroundArrays(
        camera_id=camera_id,
        rgb=np.full((2, 3, 3), 0.25, dtype=np.float32),
        alpha=np.ones((2, 3, 1), dtype=np.float32),
        depth=np.full((2, 3, 1), 4.0, dtype=np.float32),
    )


def _delta(frame: int, camera_id: str, pixel: int) -> ForegroundDelta:
    return ForegroundDelta(
        key=RenderSampleKey(frame, camera_id),
        pixel_indices=np.asarray([pixel], dtype=np.int32),
        rgb=np.asarray([[1.0, 0.5, 0.0]], dtype=np.float32),
        alpha=np.asarray([1.0], dtype=np.float32),
        depth=np.asarray([2.0], dtype=np.float32),
        instance_ids=np.asarray([7], dtype=np.int32),
    )


def test_sparse_delta_round_trip_materializes_exact_logical_sample() -> None:
    background = _background()
    rgb = np.array(background.rgb, copy=True)
    alpha = np.array(background.alpha, copy=True)
    depth = np.array(background.depth, copy=True)
    instances: NDArray[np.int32] = np.zeros((2, 3), dtype=np.int32)
    rgb[1, 1] = (1.0, 0.5, 0.0)
    alpha[1, 1] = 0.75
    depth[1, 1] = 2.0
    instances[1, 1] = 7

    delta = sparse_delta_from_composite(
        key=RenderSampleKey(4, "camera-z"),
        background=background,
        rgb=rgb,
        alpha=alpha,
        depth=depth,
        instance_ids=instances,
    )
    materialized = materialize_logical_sample(background, delta)

    assert delta.pixel_indices.tolist() == [4]
    assert delta.visible_instance_counts == {7: 1}
    np.testing.assert_array_equal(materialized.rgb, rgb)
    np.testing.assert_array_equal(materialized.alpha, alpha)
    np.testing.assert_array_equal(materialized.depth, depth)
    np.testing.assert_array_equal(materialized.instance_ids, instances)


def test_compact_chunks_preserve_configured_camera_order_and_exact_coverage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chunks"
    writer = ChunkWriter(
        root,
        attempt_token="attempt-1",
        camera_ids=("camera-z", "camera-a"),
        width=3,
        height=2,
    )
    deltas = tuple(
        _delta(frame, camera_id, frame + camera_index)
        for frame in range(2)
        for camera_index, camera_id in enumerate(("camera-z", "camera-a"))
    )
    batch = ForegroundDeltaBatch(
        chunk_id="chunk-000000",
        deltas=deltas,
        metadata=tuple({"ordinal": index} for index in range(len(deltas))),
    )

    reader = writer.write(batch)
    reopened = reader.validate(expected_attempt_token="attempt-1")
    assembled = FinalDatasetAssembler(
        frame_count=2,
        camera_ids=("camera-z", "camera-a"),
        attempt_token="attempt-1",
    ).validate((reader,))

    assert reopened.keys == tuple(delta.key for delta in deltas)
    assert [record["ordinal"] for record in reader.metadata()] == [0, 1, 2, 3]
    assert [delta.key for delta in reader.deltas()] == list(reopened.keys)
    assert assembled == (reopened,)


def test_chunk_writer_rejects_lexical_order_when_profile_order_differs(
    tmp_path: Path,
) -> None:
    writer = ChunkWriter(
        tmp_path / "chunks",
        attempt_token="attempt-1",
        camera_ids=("camera-z", "camera-a"),
        width=3,
        height=2,
    )
    batch = ForegroundDeltaBatch(
        chunk_id="chunk-000000",
        deltas=(_delta(0, "camera-a", 0), _delta(0, "camera-z", 1)),
        metadata=({}, {}),
    )

    with pytest.raises(ValueError, match="configured-camera order"):
        writer.write(batch)


def test_chunk_recovery_rejects_incomplete_and_stale_attempts(tmp_path: Path) -> None:
    root = tmp_path / "chunks"
    incomplete = root / "chunk-incomplete"
    incomplete.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="chunk.json"):
        ChunkReader(incomplete).validate()

    writer = ChunkWriter(
        root,
        attempt_token="attempt-old",
        camera_ids=("camera-0",),
        width=3,
        height=2,
    )
    reader = writer.write(
        ForegroundDeltaBatch(
            chunk_id="chunk-complete",
            deltas=(_delta(0, "camera-0", 0),),
            metadata=({},),
        )
    )
    complete = reader.validate(expected_attempt_token="attempt-old")
    with pytest.raises(ValueError, match="another stage attempt"):
        ChunkReader(complete.directory).validate(
            expected_attempt_token="attempt-new"
        )


def test_shared_background_store_loads_each_camera_once(tmp_path: Path) -> None:
    render_root = tmp_path / "nht-result"
    render_root.mkdir()
    records = tuple(
        _render_record(render_root, camera_id)
        for camera_id in ("camera-0", "camera-1")
    )
    rendered = NHTRenderResult(
        scene_id="scene-1",
        output_directory=render_root,
        records=records,
    )
    session = RenderSession(
        domain="blcs",
        attempt_token="attempt-1",
        execution_device="cuda:0",
    )
    session.note_nht_invocation()
    store = session.create_background_store(
        "trajectory-0",
        tmp_path / "background-store",
        rendered=rendered,
        nht_scene_units_per_metre=2.0,
        expected_camera_ids=("camera-0", "camera-1"),
    )

    assert isinstance(store, SharedBackgroundStore)
    assert session.nht_invocations == 1
    assert session.background_cache_misses == 2
    first = session.background("trajectory-0", "camera-0")
    second = session.background("trajectory-0", "camera-0")
    assert first is second
    assert session.background_cache_misses == 2
    np.testing.assert_allclose(first.depth, 2.0)


def test_performance_metrics_parse_strictly_and_enforce_budget() -> None:
    raw: dict[str, object] = {
        "schema": PERFORMANCE_SCHEMA,
        "domain": "plcs",
        "wall_seconds": 5.0,
        "cpu_seconds": 2.0,
        "peak_rss_bytes": 100,
        "execution_device": "cuda:0",
        "cuda_peak_bytes": 50,
        "nht_invocations": 1,
        "background_cache_misses": 6,
        "complete_array_scans": 24,
        "generated_bytes": 200,
        "published_bytes": 100,
        "dense_reference_bytes": 200,
        "frame_count": 4,
        "camera_count": 6,
        "sample_count": 24,
    }
    metrics = DatasetPerformanceMetrics.from_dict(raw)
    budget = DatasetPerformanceBudget(
        maximum_wall_seconds=10.0,
        maximum_published_bytes=100,
        maximum_published_fraction_of_dense_reference=0.5,
        maximum_nht_invocations=1,
        maximum_background_cache_misses=6,
        maximum_complete_array_scans_per_sample=1,
        maximum_batch_frames=32,
        execution_device="cuda:0",
        require_cuda=True,
    )
    metrics.validate_budget(budget)

    raw["domain"] = 7
    with pytest.raises((TypeError, ValueError), match="domain"):
        DatasetPerformanceMetrics.from_dict(raw)
    with pytest.raises(ValueError, match="maximum_wall_seconds"):
        metrics.validate_budget(
            DatasetPerformanceBudget(
                maximum_wall_seconds=4.0,
                maximum_published_bytes=100,
                maximum_published_fraction_of_dense_reference=0.5,
                maximum_nht_invocations=1,
                maximum_background_cache_misses=6,
                maximum_complete_array_scans_per_sample=1,
                maximum_batch_frames=32,
                execution_device="cuda:0",
                require_cuda=True,
            )
        )


def test_performance_evidence_round_trip_is_budget_gated(tmp_path: Path) -> None:
    metrics = DatasetPerformanceMetrics(
        domain="blcs",
        wall_seconds=5.0,
        cpu_seconds=2.0,
        peak_rss_bytes=100,
        execution_device="cuda:0",
        cuda_peak_bytes=50,
        nht_invocations=1,
        background_cache_misses=2,
        complete_array_scans=4,
        generated_bytes=200,
        published_bytes=100,
        dense_reference_bytes=500,
        frame_count=2,
        camera_count=2,
        sample_count=4,
    )
    budget = DatasetPerformanceBudget(
        maximum_wall_seconds=10.0,
        maximum_published_bytes=100,
        maximum_published_fraction_of_dense_reference=0.2,
        maximum_nht_invocations=1,
        maximum_background_cache_misses=2,
        maximum_complete_array_scans_per_sample=1,
        maximum_batch_frames=32,
        execution_device="cuda:0",
        require_cuda=True,
    )
    path = tmp_path / "diagnostics" / "performance.json"

    write_performance_metrics(path, metrics=metrics, budget=budget)

    assert load_performance_metrics(path, budget=budget) == metrics
    with pytest.raises(FileExistsError, match="already exists"):
        write_performance_metrics(path, metrics=metrics, budget=budget)


def test_discard_working_directory_is_owner_bounded(tmp_path: Path) -> None:
    owner = tmp_path / "stage"
    working = owner / "working"
    working.mkdir(parents=True)
    (working / "partial.bin").write_bytes(b"partial")

    discard_working_directory(working, owner=owner)

    assert not working.exists()
    with pytest.raises(ValueError, match="outside its stage owner"):
        discard_working_directory(tmp_path, owner=owner)


def test_background_manifest_rejects_escape_reference(tmp_path: Path) -> None:
    root = tmp_path / "background-store"
    root.mkdir()
    (tmp_path / "outside.npy").write_bytes(b"outside")
    (root / "backgrounds.json").write_text(
        json.dumps(
            {
                "schema": "shared_render_background_store_v1",
                "scene_id": "scene-1",
                "depth_coordinate_space": "metric_scene_metres",
                "records": [
                    {
                        "camera_id": "camera-0",
                        "width": 2,
                        "height": 2,
                        "rgb": "../outside.npy",
                        "alpha": "../outside.npy",
                        "depth": "../outside.npy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contained relative path"):
        SharedBackgroundStore(root)


def _render_record(root: Path, camera_id: str) -> NHTRenderRecord:
    camera_root = root / camera_id
    camera_root.mkdir()
    rgb = camera_root / "rgb.npy"
    alpha = camera_root / "alpha.npy"
    depth = camera_root / "depth.npy"
    preview = camera_root / "preview.png"
    np.save(rgb, np.full((2, 3, 3), 0.25, dtype=np.float32))
    np.save(alpha, np.ones((2, 3, 1), dtype=np.float32))
    np.save(depth, np.full((2, 3, 1), 4.0, dtype=np.float32))
    preview.write_bytes(b"unused-by-store")
    record = NHTRenderRecord(
        camera_id=camera_id,
        request_source="arbitrary",
        width=3,
        height=2,
        rgb_path=rgb,
        rgb_preview_path=preview,
        alpha_path=alpha,
        alpha_preview_path=preview,
        depth_path=depth,
    )
    record._bind_arrays(
        NHTRenderArrays(
            rgb=np.load(rgb, allow_pickle=False),
            alpha=np.load(alpha, allow_pickle=False),
            depth=np.load(depth, allow_pickle=False),
        )
    )
    return record
