"""Logical-reader tests for compact PLCS storage."""

import json
from pathlib import Path

import numpy as np

from src.synthetic_data_generation.dataset.plcs.assembler import PLCS_DATASET_SCHEMA
from src.synthetic_data_generation.dataset.plcs.validation import (
    PLCSCompactDatasetReader,
)
from src.synthetic_data_generation.dataset.runtime import (
    BACKGROUND_STORE_SCHEMA,
    ChunkWriter,
    ForegroundDelta,
    ForegroundDeltaBatch,
    RenderSampleKey,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera


def test_logical_reader_reconstructs_shared_background_plus_delta(
    tmp_path: Path,
) -> None:
    root = tmp_path / "plcs"
    background = root / "backgrounds" / "camera-0"
    background.mkdir(parents=True)
    np.save(background / "rgb.npy", np.zeros((2, 3, 3), dtype=np.float32))
    np.save(background / "alpha.npy", np.ones((2, 3, 1), dtype=np.float32))
    np.save(background / "depth-metric.npy", np.full((2, 3, 1), 5.0, dtype=np.float32))
    (root / "backgrounds" / "backgrounds.json").write_text(
        json.dumps(
            {
                "schema": BACKGROUND_STORE_SCHEMA,
                "scene_id": "B00",
                "depth_coordinate_space": "metric_scene_metres",
                "records": [
                    {
                        "camera_id": "camera-0",
                        "width": 3,
                        "height": 2,
                        "rgb": "camera-0/rgb.npy",
                        "alpha": "camera-0/alpha.npy",
                        "depth": "camera-0/depth-metric.npy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    writer = ChunkWriter(
        root / "chunks",
        attempt_token="B00-plcs",
        camera_ids=("camera-0",),
        width=3,
        height=2,
    )
    written = writer.write(
        ForegroundDeltaBatch(
            chunk_id="chunk-000000",
            deltas=(
                ForegroundDelta(
                    key=RenderSampleKey(0, "camera-0"),
                    pixel_indices=np.asarray([4], dtype=np.int32),
                    rgb=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                    alpha=np.asarray([1.0], dtype=np.float32),
                    depth=np.asarray([2.0], dtype=np.float32),
                    instance_ids=np.asarray([3], dtype=np.int32),
                ),
            ),
            metadata=({},),
        )
    )
    camera = SceneCamera(
        camera_id="camera-0",
        source_frame_index=0,
        width=3,
        height=2,
        intrinsics=(2.0, 0.0, 1.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.identity(),
        image_path="request-only",
    )
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "schema": PLCS_DATASET_SCHEMA,
                "scene_id": "B00",
                "domain": "plcs",
                "frame_inventory": {},
                "target_courts": [],
                "metadata": {"cameras": [{"camera": camera.to_dict()}]},
                "diagnostics": [],
                "storage": {
                    "layout": "shared-background-plus-foreground-delta",
                    "background_store": "backgrounds",
                    "chunks": [str(written.directory.relative_to(root))],
                    "attempt_token": "B00-plcs",
                    "sample_order": "global-frame-then-configured-camera",
                },
            }
        ),
        encoding="utf-8",
    )

    sample = PLCSCompactDatasetReader(root).logical_sample(0, "camera-0")

    assert sample.instance_ids[1, 1] == 3
    np.testing.assert_array_equal(sample.rgb[1, 1], (1.0, 0.0, 0.0))
    assert sample.depth[1, 1, 0] == 2.0
    assert sample.depth[0, 0, 0] == 5.0
