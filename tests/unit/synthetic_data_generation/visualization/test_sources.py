"""Selection and source-order tests for canonical visualization readers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

import src.synthetic_data_generation.visualization.sources as sources_module
from src.synthetic_data_generation.dataset.blcs.contracts import BLCSSampleRecord
from src.synthetic_data_generation.dataset.court.contracts import (
    SupportModelSummary,
    TrajectorySupportPolicy,
)
from src.synthetic_data_generation.dataset.court.occupancy_artifact import (
    COURT_V4_SUPPORT_OCCUPANCY_CELLS_PATH,
    COURT_V4_SUPPORT_OCCUPANCY_METADATA_PATH,
    build_court_v4_support_occupancy_snapshot,
    occupancy_cells_content_digest,
    write_court_v4_support_occupancy,
)
from src.synthetic_data_generation.dataset.plcs.assembler import PLCS_DATASET_SCHEMA
from src.synthetic_data_generation.dataset.runtime import (
    ChunkWriter,
    ForegroundDelta,
    ForegroundDeltaBatch,
    RenderSampleKey,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.synthetic_data_generation.visualization.contracts import CourtOverlayMode
from src.synthetic_data_generation.visualization.sources import (
    BLCSVisualizationSource,
    CourtVisualizationSource,
    PLCSVisualizationSource,
)
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)


def _write_court_fixture(
    root: Path,
    *,
    indices: tuple[int, ...],
    dataset_schema: str = "canonical_court_dataset_v1",
    label_schema: str | None = "canonical_court_sample_v1",
) -> None:
    records = []
    for sample_index, frame_index in enumerate(indices):
        directory = root / "samples" / f"sample-{sample_index}"
        directory.mkdir(parents=True)
        rgb: NDArray[np.float32] = np.full(
            (32, 48, 3),
            fill_value=sample_index / 10.0,
            dtype=np.float32,
        )
        np.save(directory / "rgb.npy", rgb, allow_pickle=False)
        projection: dict[str, object] = {"courts": []}
        labels = {
            "sample_id": f"sample-{sample_index}",
            "view_id": "view-0",
            "trajectory_frame_index": frame_index,
            "projection": projection,
        }
        if label_schema is not None:
            labels["schema"] = label_schema
        (directory / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
        records.append(
            {
                "sample_id": f"sample-{sample_index}",
                "trajectory_id": "orbit-0",
                "view_id": "view-0",
                "trajectory_frame_index": frame_index,
                "width": 48,
                "height": 32,
                "rgb": f"samples/sample-{sample_index}/rgb.npy",
                "labels": f"samples/sample-{sample_index}/labels.json",
                "projection": projection,
            }
        )
    payload = {
        "schema": dataset_schema,
        "scene_id": "scene-0",
        "trajectory_groups": [
            {
                "trajectory": {"trajectory_id": "orbit-0"},
                "views": [{"view_id": "view-0"}],
                "sample_count": 2,
            }
        ],
        "samples": records,
    }
    (root / "dataset.json").write_text(json.dumps(payload), encoding="utf-8")


def _v4_support_policy() -> TrajectorySupportPolicy:
    return TrajectorySupportPolicy(
        decision_id="unit-support-v1",
        support_radius_m=0.8,
        endpoint_radius_m=0.6,
        maximum_camera_link_distance_m=1.5,
        maximum_source_frame_gap=1,
        occupancy_voxel_size_m=0.5,
        minimum_points_per_voxel=1,
        obstacle_inflation_m=0.5,
        camera_ball_clearance_m=0.05,
        camera_capsule_clearance_m=0.04,
        sweep_step_m=0.1,
        boundary_epsilon_m=1.0e-6,
        minimum_captured_cameras=2,
        minimum_public_points=1,
        maximum_capsule_index_cells=10_000,
        maximum_occupancy_cells=10_000,
        minimum_cycle_frame_span=8,
        maximum_cycle_frame_span=16,
        maximum_cycle_closure_distance_m=1.1,
        maximum_constructive_cycle_count=24,
        cycle_smoothing_distance_m=0.03,
    )


def _write_v4_aabb_fixture(root: Path) -> None:
    _write_court_fixture(
        root,
        indices=(0, 1),
        dataset_schema="canonical_court_dataset_v4",
        label_schema="canonical_court_sample_v4",
    )
    policy = _v4_support_policy()
    support_digest = "a" * 64
    summary = SupportModelSummary(
        input_digest=support_digest,
        coordinate_space="metric_scene_metres",
        captured_camera_count=2,
        public_point_count=1,
        density_qualified_voxel_count=1,
        raw_inflated_occupancy_cell_count=2,
        inflated_occupancy_cell_count=1,
        camera_ball_carved_cell_count=1,
        camera_capsule_carved_cell_count=0,
        captured_camera_occupied_count=0,
        endpoint_ball_count=2,
        capsule_count=1,
        skipped_gap_link_count=0,
        skipped_obstacle_link_count=0,
        capsule_index_cell_count=1,
    )
    camera = SceneCamera(
        camera_id="camera-0",
        source_frame_index=0,
        width=48,
        height=32,
        intrinsics=(40.0, 0.0, 23.5, 0.0, 40.0, 15.5, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.from_matrix(np.eye(4, dtype=np.float64)),
        image_path="request-only",
    )
    manifest_path = root / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["profile"] = "unit"
    manifest["metrics"] = {"support_input_digest": support_digest}
    for record in manifest["samples"]:
        directory = root / str(record["rgb"]).rsplit("/", 1)[0]
        alpha: NDArray[np.float32] = np.ones((32, 48, 1), dtype=np.float32)
        depth: NDArray[np.float32] = np.full(
            (32, 48, 1),
            10.0,
            dtype=np.float32,
        )
        np.save(directory / "alpha.npy", alpha, allow_pickle=False)
        np.save(directory / "depth.npy", depth, allow_pickle=False)
        record.update(
            {
                "alpha": f"{record['rgb'].rsplit('/', 1)[0]}/alpha.npy",
                "depth": f"{record['rgb'].rsplit('/', 1)[0]}/depth.npy",
                "depth_coordinate_space": "metric_scene_metres",
                "camera": camera.to_dict(),
                "safety_support_input_digest": support_digest,
            }
        )
        label_path = root / str(record["labels"])
        label = json.loads(label_path.read_text(encoding="utf-8"))
        label["camera"] = camera.to_dict()
        label["safety_support_input_digest"] = support_digest
        label_path.write_text(json.dumps(label), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    diagnostics = root / "diagnostics"
    diagnostics.mkdir()
    snapshot = build_court_v4_support_occupancy_snapshot(
        np.asarray(((0, 0, 4),), dtype=np.int64),
        voxel_size_m=policy.occupancy_voxel_size_m,
        support_input_digest=support_digest,
        policy_decision_id=policy.decision_id,
    )
    (diagnostics / "trajectory-plan.json").write_text(
        json.dumps(
            {
                "schema": "canonical_court_safe_path_plan_v4",
                "support_policy": policy.to_dict(),
                "support_summary": summary.to_dict(),
                "support_occupancy_identity": snapshot.identity.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    write_court_v4_support_occupancy(
        diagnostics,
        snapshot=snapshot,
        scene_id="scene-0",
        profile="unit",
    )


def test_court_source_streams_selected_trajectory_in_exact_frame_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_court_fixture(tmp_path, indices=(0, 1))
    monkeypatch.setattr(
        sources_module, "validate_court_dataset", lambda *args, **kwargs: None
    )

    source = CourtVisualizationSource(tmp_path, trajectory_id="orbit-0")
    frames = tuple(source.frames())

    assert tuple(frame.trajectory_frame_index for frame in frames) == (0, 1)
    assert tuple(frame.sample_id for frame in frames) == ("sample-0", "sample-1")
    assert frames[1].rgb[0, 0, 0] == pytest.approx(0.1)


def test_court_source_fails_closed_on_unknown_id_or_reordered_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_court_fixture(tmp_path, indices=(1, 0))
    monkeypatch.setattr(
        sources_module, "validate_court_dataset", lambda *args, **kwargs: None
    )

    with pytest.raises(KeyError, match="Unknown Court trajectory_id"):
        CourtVisualizationSource(tmp_path, trajectory_id="missing")
    with pytest.raises(ValueError, match="source-frame ordering"):
        CourtVisualizationSource(tmp_path, trajectory_id="orbit-0")


@pytest.mark.parametrize(
    "label_schema",
    [
        None,
        "canonical_court_sample_v1",
        "canonical_court_sample_v2",
        "canonical_court_sample_v3",
    ],
)
def test_v2_court_source_rejects_missing_or_mixed_sample_schema_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label_schema: str | None,
) -> None:
    _write_court_fixture(
        tmp_path,
        indices=(0, 1),
        dataset_schema="canonical_court_dataset_v2",
        label_schema=label_schema,
    )
    monkeypatch.setattr(
        sources_module, "validate_court_dataset", lambda *args, **kwargs: None
    )
    source = CourtVisualizationSource(tmp_path, trajectory_id="orbit-0")

    if label_schema == "canonical_court_sample_v2":
        assert tuple(source.frames())
    else:
        with pytest.raises(ValueError, match="labels schema changed"):
            tuple(source.frames())


def test_court_source_rejects_unknown_dataset_schema_without_shape_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_court_fixture(
        tmp_path,
        indices=(0, 1),
        dataset_schema="canonical_court_dataset_v5",
        label_schema="canonical_court_sample_v2",
    )
    monkeypatch.setattr(
        sources_module, "validate_court_dataset", lambda *args, **kwargs: None
    )

    with pytest.raises(
        ValueError,
        match=r"^Unknown Court dataset schema: 'canonical_court_dataset_v5'\.$",
    ):
        CourtVisualizationSource(tmp_path, trajectory_id="orbit-0")


def test_court_source_dispatches_explicit_v4_sample_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_court_fixture(
        tmp_path,
        indices=(0, 1),
        dataset_schema="canonical_court_dataset_v4",
        label_schema="canonical_court_sample_v4",
    )
    monkeypatch.setattr(
        sources_module, "validate_court_dataset", lambda *args, **kwargs: None
    )

    source = CourtVisualizationSource(tmp_path, trajectory_id="orbit-0")

    assert tuple(frame.schema_version.value for frame in source.frames()) == (
        "v4",
        "v4",
    )


def test_v4_aabb_source_loads_metric_arrays_camera_and_bound_exact_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_v4_aabb_fixture(tmp_path)
    monkeypatch.setattr(
        sources_module, "validate_court_dataset", lambda *args, **kwargs: None
    )

    source = CourtVisualizationSource(
        tmp_path,
        trajectory_id="orbit-0",
        overlay_mode=CourtOverlayMode.TRAJECTORY_SUPPORT_AABB,
        maximum_occupancy_cells=1,
    )
    frames = tuple(source.frames())

    assert source.support_occupancy is not None
    assert source.support_occupancy.cells.tolist() == [[0, 0, 4]]
    assert source.support_occupancy.support_input_digest == "a" * 64
    assert all(frame.camera is not None for frame in frames)
    assert all(frame.alpha is not None for frame in frames)
    assert all(frame.depth_metric_m is not None for frame in frames)
    assert frames[0].depth_metric_m is not None
    assert float(frames[0].depth_metric_m[0, 0, 0]) == 10.0


def test_aabb_source_rejects_non_v4_missing_or_mismatched_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sources_module, "validate_court_dataset", lambda *args, **kwargs: None
    )
    v3_root = tmp_path / "v3"
    _write_court_fixture(
        v3_root,
        indices=(0, 1),
        dataset_schema="canonical_court_dataset_v3",
        label_schema="canonical_court_sample_v3",
    )
    with pytest.raises(ValueError, match="requires a Court V4 dataset"):
        CourtVisualizationSource(
            v3_root,
            trajectory_id="orbit-0",
            overlay_mode=CourtOverlayMode.TRAJECTORY_SUPPORT_AABB,
        )

    missing_root = tmp_path / "missing"
    _write_court_fixture(
        missing_root,
        indices=(0, 1),
        dataset_schema="canonical_court_dataset_v4",
        label_schema="canonical_court_sample_v4",
    )
    with pytest.raises((FileNotFoundError, ValueError)):
        CourtVisualizationSource(
            missing_root,
            trajectory_id="orbit-0",
            overlay_mode=CourtOverlayMode.TRAJECTORY_SUPPORT_AABB,
        )

    mismatched_root = tmp_path / "mismatched"
    _write_v4_aabb_fixture(mismatched_root)
    metadata_path = mismatched_root / COURT_V4_SUPPORT_OCCUPANCY_METADATA_PATH
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["support_input_digest"] = "b" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="support_input_digest binding disagrees"):
        CourtVisualizationSource(
            mismatched_root,
            trajectory_id="orbit-0",
            overlay_mode=CourtOverlayMode.TRAJECTORY_SUPPORT_AABB,
        )


def test_v4_semantic_source_does_not_require_aabb_payload_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_court_fixture(
        tmp_path,
        indices=(0, 1),
        dataset_schema="canonical_court_dataset_v4",
        label_schema="canonical_court_sample_v4",
    )
    monkeypatch.setattr(
        sources_module, "validate_court_dataset", lambda *args, **kwargs: None
    )

    frames = tuple(CourtVisualizationSource(tmp_path, trajectory_id="orbit-0").frames())

    assert all(frame.alpha is None for frame in frames)
    assert all(frame.depth_metric_m is None for frame in frames)
    assert all(frame.camera is None for frame in frames)


def test_aabb_source_rejects_replaced_cells_with_recomputed_artifact_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_v4_aabb_fixture(tmp_path)
    monkeypatch.setattr(
        sources_module,
        "validate_court_dataset",
        lambda *args, **kwargs: None,
    )
    cells_path = tmp_path / COURT_V4_SUPPORT_OCCUPANCY_CELLS_PATH
    cells = np.load(cells_path, allow_pickle=False)
    cells[0, 0] -= 1
    np.save(cells_path, cells, allow_pickle=False)
    metadata_path = tmp_path / COURT_V4_SUPPORT_OCCUPANCY_METADATA_PATH
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["content_digest"] = occupancy_cells_content_digest(cells)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="content_digest binding disagrees"):
        CourtVisualizationSource(
            tmp_path,
            trajectory_id="orbit-0",
            overlay_mode=CourtOverlayMode.TRAJECTORY_SUPPORT_AABB,
        )


def test_blcs_stream_rejects_chunk_replaced_by_a_foreign_attempt(
    tmp_path: Path,
) -> None:
    writer = ChunkWriter(
        tmp_path / "chunks",
        attempt_token="foreign-attempt",
        camera_ids=("camera-0",),
        width=2,
        height=2,
    )
    chunk = writer.write(
        ForegroundDeltaBatch(
            chunk_id="chunk-000000",
            deltas=(
                ForegroundDelta(
                    key=RenderSampleKey(0, "camera-0"),
                    pixel_indices=np.asarray([0], dtype=np.int32),
                    rgb=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                    alpha=np.asarray([1.0], dtype=np.float32),
                    depth=np.asarray([1.0], dtype=np.float32),
                    instance_ids=np.asarray([1], dtype=np.int32),
                ),
            ),
            metadata=({},),
        )
    )
    source = object.__new__(BLCSVisualizationSource)
    source.root = tmp_path
    source.logical_scene_id = "trajectory-0"
    source.camera_id = "camera-0"
    source._attempt_token = "selected-attempt"
    source._records = (
        BLCSSampleRecord(
            trajectory_id="trajectory-0",
            split="train",
            global_frame_index=0,
            source_frame_index=0,
            chunk_index=0,
            camera_id="camera-0",
            background_store="backgrounds",
            foreground_chunk=chunk.directory.relative_to(tmp_path).as_posix(),
            chunk_sample_index=0,
        ),
    )

    with pytest.raises(ValueError, match="another stage attempt"):
        next(source.frames())


def test_plcs_visualization_rejects_v4_before_reading_any_payload(
    tmp_path: Path,
) -> None:
    for directory in ("backgrounds", "scenes", "diagnostics"):
        (tmp_path / directory).mkdir()
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "schema": "tennis_plcs_compact_dataset_v4",
                "scene_id": "B00",
                "domain": "plcs",
                "frame_inventory": {},
                "target_courts": [],
                "metadata": {},
                "diagnostics": [],
                "storage": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported canonical compact PLCS"):
        PLCSVisualizationSource(
            tmp_path,
            logical_scene_id="B00",
            camera_id="camera-0",
        )


@pytest.mark.parametrize("mutation", ["missing", "malformed", "unknown", "mismatched"])
def test_plcs_visualization_rejects_invalid_normalization_before_payloads(
    tmp_path: Path,
    mutation: str,
) -> None:
    for directory in ("backgrounds", "scenes", "diagnostics"):
        (tmp_path / directory).mkdir()
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
        "coordinate_contract": {},
        "court_coordinate_normalization": contract,
        "seed": 0,
        "logical_scene_count": 1,
        "aggregate_global_frame_count": 1,
        "aggregate_source_frame_count": 1,
        "required_motion_categories": [],
        "accepted_court_instance_ids": [],
        "logical_scenes": [],
    }
    if mutation == "missing":
        del metadata["court_coordinate_normalization"]
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "schema": PLCS_DATASET_SCHEMA,
                "scene_id": "B00",
                "domain": "plcs",
                "frame_inventory": {},
                "target_courts": [],
                "metadata": metadata,
                "diagnostics": [],
                "storage": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incompatible|unknown|mismatched"):
        PLCSVisualizationSource(
            tmp_path,
            logical_scene_id="B00",
            camera_id="camera-0",
        )
