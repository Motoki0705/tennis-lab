"""Logical-reader tests for compact PLCS storage."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.synthetic_data_generation.dataset.plcs.assembler import PLCS_DATASET_SCHEMA
from src.synthetic_data_generation.dataset.plcs.coordinates import (
    PLCS_COORDINATE_CONTRACT,
    PLCSSourceSupportPlane,
)
from src.synthetic_data_generation.dataset.plcs.validation import (
    PLCSCompactDatasetReader,
    validate_plcs_dataset,
)
from src.synthetic_data_generation.dataset.runtime import (
    BACKGROUND_STORE_SCHEMA,
    ChunkWriter,
    ForegroundDelta,
    ForegroundDeltaBatch,
    RenderSampleKey,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)


def _write_contract_gate_manifest(root: Path, *, mutation: str) -> None:
    for directory in ("backgrounds", "scenes", "diagnostics"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    contract: object = court_coordinate_normalization_metadata()
    if mutation == "malformed":
        contract = "isotropic_half_length"
    elif mutation in {"unknown", "mismatched"}:
        assert isinstance(contract, dict)
        contract = deepcopy(contract)
        if mutation == "unknown":
            contract["identity"] = "anisotropic"
        else:
            contract["scale_xyz_m"] = [5.485, 11.885, 1.07]
    metadata = {
        "coordinate_contract": PLCS_COORDINATE_CONTRACT.to_dict(),
        "court_coordinate_normalization": contract,
        "seed": 0,
        "logical_scene_count": 1,
        "aggregate_global_frame_count": 1,
        "aggregate_source_frame_count": 1,
        "required_motion_categories": ["running"],
        "accepted_court_instance_ids": ["court-001"],
        "logical_scenes": [],
    }
    if mutation == "missing":
        del metadata["court_coordinate_normalization"]
    manifest = {
        "schema": PLCS_DATASET_SCHEMA,
        "scene_id": "B00",
        "domain": "plcs",
        "frame_inventory": {
            "source": 1,
            "planned": 1,
            "rendered": 1,
            "labelled": 1,
            "first_frame": 0,
            "last_frame": 0,
        },
        "target_courts": [],
        "metadata": metadata,
        "diagnostics": [],
        "storage": {},
    }
    (root / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("surface", ["reader", "validator"])
@pytest.mark.parametrize("mutation", ["missing", "malformed", "unknown", "mismatched"])
def test_compact_plcs_load_surfaces_reject_invalid_normalization_before_arrays(
    tmp_path: Path,
    surface: str,
    mutation: str,
) -> None:
    root = tmp_path / ".transactions" / "plcs_dataset" / "snapshot"
    _write_contract_gate_manifest(root, mutation=mutation)

    with pytest.raises(ValueError, match="incompatible|unknown|mismatched"):
        if surface == "reader":
            PLCSCompactDatasetReader(root)
        else:
            validate_plcs_dataset(root)


def test_logical_reader_reconstructs_shared_background_plus_delta(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets" / "plcs"
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
        root / "scenes" / "B00" / "chunks",
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
    supervision = root / "scenes" / "B00" / "supervision.npz"
    np.savez(
        supervision,
        human_kp=np.zeros((1, 1, 1, 17, 2), dtype=np.float32),
        human_vis=np.zeros((1, 1, 1, 17), dtype=np.bool_),
        court_kp=np.zeros((1, 1, 20, 2), dtype=np.float32),
        court_vis=np.zeros((1, 1, 20), dtype=np.bool_),
        human_mask=np.ones((1, 1, 1), dtype=np.bool_),
        position=np.zeros((1, 1, 3), dtype=np.float32),
        position_court_m=np.zeros((1, 1, 3), dtype=np.float32),
        rotation=np.asarray([[[1.0, 0.0]]], dtype=np.float32),
        present=np.ones((1, 1), dtype=np.bool_),
        human_kp_3d=np.zeros((1, 1, 17, 3), dtype=np.float32),
        canonical_pose_3d=np.zeros((1, 1, 52, 3), dtype=np.float32),
    )
    manifest_path = root / "dataset.json"
    manifest = {
        "schema": PLCS_DATASET_SCHEMA,
        "scene_id": "B00",
        "domain": "plcs",
        "frame_inventory": {
            "source": 1,
            "planned": 1,
            "rendered": 1,
            "labelled": 1,
            "first_frame": 0,
            "last_frame": 0,
        },
        "target_courts": [],
        "metadata": {
            "coordinate_contract": PLCS_COORDINATE_CONTRACT.to_dict(),
            "court_coordinate_normalization": (
                court_coordinate_normalization_metadata()
            ),
            "logical_scenes": [
                {
                    "scene_id": "B00",
                    "split": "train",
                    "frame_inventory": {
                        "source": 1,
                        "planned": 1,
                        "rendered": 1,
                        "labelled": 1,
                        "first_frame": 0,
                        "last_frame": 0,
                    },
                    "tracks": [
                        {
                            "object_id": "player-001",
                            "instance_id": 1,
                            "asset_id": "avatar-001",
                            "support_plane": (
                                PLCSSourceSupportPlane.from_surface_minimum(
                                    initial_root_translation_z_m=0.0,
                                    support_local_z_m=0.0,
                                ).to_dict()
                            ),
                            "start_frame": 0,
                            "stop_frame": 1,
                            "anchor_position_court_m": [0.0, 0.0, 0.0],
                            "yaw_radians": 0.0,
                        }
                    ],
                    "cameras": [
                        {
                            "slot_id": "camera-0",
                            "court_local_center_m": [0.0, 0.0, 1.0],
                            "court_local_look_at_m": [0.0, 1.0, 1.0],
                            "hfov_degrees": 60.0,
                            "camera": camera.to_dict(),
                        }
                    ],
                }
            ],
        },
        "diagnostics": [],
        "storage": {
            "layout": "shared-background-plus-per-scene-foreground-delta",
            "background_store": "backgrounds",
            "scenes": [
                {
                    "scene_id": "B00",
                    "chunks": [str(written.directory.relative_to(root))],
                    "attempt_token": "B00-plcs",
                    "sample_order": "scene-frame-then-configured-camera",
                    "supervision": "scenes/B00/supervision.npz",
                    "camera_ids": ["camera-0"],
                    "object_ids": ["player-001"],
                }
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "diagnostics").mkdir()

    reader = PLCSCompactDatasetReader(root)
    sample = reader.logical_sample("B00", 0, "camera-0")
    all_views = reader.materialize_all_views("B00")

    assert sample.instance_ids[1, 1] == 3
    np.testing.assert_array_equal(sample.rgb[1, 1], (1.0, 0.0, 0.0))
    assert sample.depth[1, 1, 0] == 2.0
    assert sample.depth[0, 0, 0] == 5.0
    assert all_views.index.camera_ids == ("camera-0",)
    assert all_views.supervision.human_kp.shape == (1, 1, 1, 17, 2)

    manifest["schema"] = "tennis_plcs_compact_dataset_v4"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported PLCS dataset schema"):
        PLCSCompactDatasetReader(root)
