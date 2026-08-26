"""Tests for the joint NHT result to compact BLCS dataset boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.synthetic_data_generation.dataset.blcs.rendering.nht import (
    _foreground_batch_from_composed_record,
)
from src.synthetic_data_generation.dataset.blcs.timeline import build_blcs_plans
from src.synthetic_data_generation.rendering.nht import NHTComposedChunkRecord


def test_joint_sparse_chunk_converts_nht_depth_and_preserves_sample_identity(
    tmp_path: Path,
    two_court_layout,
    default_camera_profile,
    blcs_assets,
    blcs_trajectory_factory,
) -> None:
    plan = build_blcs_plans(
        (blcs_trajectory_factory("trajectory-0", frame_count=2),),
        dataset_scene_id="B00",
        layout=two_court_layout,
        camera_config=default_camera_profile,
        assets=blcs_assets,
        seed=9,
        chunk_size_frames=1,
    )[0]
    camera_ids = tuple(
        sampled.scene_camera.camera_id for sampled in plan.camera_rig.cameras[:2]
    )
    arrays_path = tmp_path / "composed.npz"
    np.savez(
        arrays_path,
        frame_indices=np.asarray([0, 0], dtype=np.int64),
        camera_indices=np.asarray([0, 1], dtype=np.int32),
        offsets=np.asarray([0, 1, 2], dtype=np.int64),
        pixel_indices=np.asarray([1, 2], dtype=np.int32),
        rgb=np.asarray(
            [[0.72, 0.92, 0.08], [0.92, 0.95, 0.80]], dtype=np.float32
        ),
        alpha=np.asarray([0.97, 0.91], dtype=np.float32),
        depth=np.asarray([4.0, 6.0], dtype=np.float32),
        instance_ids=np.asarray([1, 1], dtype=np.int32),
    )
    record = NHTComposedChunkRecord(
        chunk_id="chunk-000000",
        frame_indices=(0,),
        camera_ids=camera_ids,
        sample_count=2,
        pixel_count=2,
        arrays_path=arrays_path,
        width=32,
        height=24,
        object_count=1,
    )

    batch = _foreground_batch_from_composed_record(
        plan=plan,
        record=record,
        nht_scene_units_per_metre=2.0,
    )

    assert batch.chunk_id == "chunk-000000"
    assert tuple(delta.key.camera_id for delta in batch.deltas) == camera_ids
    assert tuple(delta.key.frame_index for delta in batch.deltas) == (0, 0)
    np.testing.assert_allclose(batch.deltas[0].depth, [2.0])
    np.testing.assert_allclose(batch.deltas[1].depth, [3.0])
    assert batch.deltas[0].instance_ids.tolist() == [1]
    assert batch.metadata[0]["source_frame_index"] == 0
    assert batch.metadata[0]["semantic_arrays"]["rendered_visible"] == [True]
