from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.synthetic_data_generation.dataset.court.contracts import (
    DatasetSplit,
    PlannedCourtSample,
)
from src.synthetic_data_generation.dataset.court.rendering.nht import (
    _discard_stale_shard,
)
from src.synthetic_data_generation.dataset.court.schema import (
    CourtDatasetSchemaVersion,
)
from src.synthetic_data_generation.dataset.court.shards import (
    CourtRenderedSample,
    StaleCourtShardError,
    inspect_rendered_sample,
    load_attempt_local_shard,
    validate_rendered_sample,
    write_attempt_shard_marker,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera


def _sample() -> PlannedCourtSample:
    camera = SceneCamera(
        camera_id="court-sample-000000",
        source_frame_index=0,
        width=4,
        height=3,
        intrinsics=(4.0, 0.0, 1.5, 0.0, 4.0, 1.0, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.identity(),
        image_path="generated/court-sample-000000.png",
    )
    return PlannedCourtSample(
        sample_index=0,
        sample_id=camera.camera_id,
        trajectory_group_id="group-a",
        trajectory_id="trajectory-a",
        view_id="view-a",
        trajectory_frame_index=0,
        split=DatasetSplit.TRAIN,
        shard_id="shard-000",
        camera_center_scene_m=(0.0, 0.0, 0.0),
        camera=camera,
    )


def _rendered(root: Path, sample: PlannedCourtSample) -> CourtRenderedSample:
    camera_root = root / sample.sample_id
    camera_root.mkdir(parents=True)
    np.save(camera_root / "rgb.npy", np.zeros((3, 4, 3), dtype=np.float32))
    np.save(camera_root / "alpha.npy", np.ones((3, 4, 1), dtype=np.float32))
    np.save(camera_root / "depth.npy", np.ones((3, 4, 1), dtype=np.float32))
    Image.new("RGB", (4, 3)).save(camera_root / "rgb.png")
    Image.new("L", (4, 3)).save(camera_root / "alpha.png")
    return CourtRenderedSample(
        sample=sample,
        rgb_path=camera_root / "rgb.npy",
        rgb_preview_path=camera_root / "rgb.png",
        alpha_path=camera_root / "alpha.npy",
        alpha_preview_path=camera_root / "alpha.png",
        depth_path=camera_root / "depth.npy",
    )


def test_shard_reuse_is_attempt_local_and_semantically_validated(
    tmp_path: Path,
) -> None:
    sample = _sample()
    shard_root = tmp_path / "shard-000"
    _rendered(shard_root, sample)
    write_attempt_shard_marker(
        shard_root,
        attempt_token="attempt-a",
        shard_id="shard-000",
        samples=(sample,),
    )
    reused = load_attempt_local_shard(
        shard_root,
        attempt_token="attempt-a",
        shard_id="shard-000",
        samples=(sample,),
    )
    assert reused is not None and reused[0].sample == sample
    with pytest.raises(StaleCourtShardError):
        load_attempt_local_shard(
            shard_root,
            attempt_token="attempt-b",
            shard_id="shard-000",
            samples=(sample,),
        )


def test_v2_shard_marker_binds_dataset_schema_and_rejects_mixed_reuse(
    tmp_path: Path,
) -> None:
    sample = _sample()
    shard_root = tmp_path / "shard-000"
    _rendered(shard_root, sample)
    marker = write_attempt_shard_marker(
        shard_root,
        attempt_token="attempt-v2",
        shard_id="shard-000",
        samples=(sample,),
        schema_version=CourtDatasetSchemaVersion.V2,
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))

    assert payload == {
        "schema": "court_render_shard_attempt_v2",
        "attempt_token": "attempt-v2",
        "shard_id": "shard-000",
        "trajectory_group_ids": ["group-a"],
        "sample_ids": [sample.sample_id],
        "dataset_schema": "canonical_court_dataset_v2",
    }
    assert (
        load_attempt_local_shard(
            shard_root,
            attempt_token="attempt-v2",
            shard_id="shard-000",
            samples=(sample,),
            schema_version=CourtDatasetSchemaVersion.V2,
        )
        is not None
    )
    with pytest.raises(ValueError, match="schema"):
        load_attempt_local_shard(
            shard_root,
            attempt_token="attempt-v2",
            shard_id="shard-000",
            samples=(sample,),
            schema_version=CourtDatasetSchemaVersion.V1,
        )

    payload["dataset_schema"] = "canonical_court_dataset_v1"
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset schema"):
        load_attempt_local_shard(
            shard_root,
            attempt_token="attempt-v2",
            shard_id="shard-000",
            samples=(sample,),
            schema_version=CourtDatasetSchemaVersion.V2,
        )


def test_renderer_array_shape_and_dtype_fail_closed(tmp_path: Path) -> None:
    sample = _sample()
    rendered = _rendered(tmp_path / "shard-000", sample)
    np.save(rendered.depth_path, np.ones((3, 4, 1), dtype=np.float64))
    with pytest.raises(TypeError, match="float32"):
        validate_rendered_sample(rendered)


def test_shard_inspection_reads_headers_without_repeating_value_scan(
    tmp_path: Path,
) -> None:
    rendered = _rendered(tmp_path / "shard-000", _sample())
    rgb = np.load(rendered.rgb_path, allow_pickle=False)
    rgb[0, 0, 0] = np.nan
    np.save(rendered.rgb_path, rgb, allow_pickle=False)

    inspect_rendered_sample(rendered)

    with pytest.raises(ValueError, match="non-finite"):
        validate_rendered_sample(rendered)


def test_rendered_sample_inventory_cannot_retain_dense_nht_arrays() -> None:
    field_names = {field.name for field in fields(CourtRenderedSample)}

    assert "validated_arrays" not in field_names
    assert {
        "rgb_path",
        "rgb_preview_path",
        "alpha_path",
        "alpha_preview_path",
        "depth_path",
    } <= field_names


def test_stale_shard_recovery_is_attempt_root_bounded(tmp_path: Path) -> None:
    attempt_root = tmp_path / "attempt"
    stale = attempt_root / "renders" / "shard-000"
    stale.mkdir(parents=True)
    (stale / "partial.bin").write_bytes(b"partial")

    _discard_stale_shard(stale, attempt_root=attempt_root)

    assert not stale.exists()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="outside the attempt root"):
        _discard_stale_shard(outside, attempt_root=attempt_root)
