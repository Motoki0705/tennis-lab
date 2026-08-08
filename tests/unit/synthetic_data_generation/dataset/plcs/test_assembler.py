"""Aggregate exact-inventory tests for complete PLCS logical scenes."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.plcs.assembler import (
    PLCSSceneAssemblyInput,
    PLCSSupervisionArrays,
    assemble_plcs_dataset,
    build_frame_label,
)
from src.synthetic_data_generation.dataset.plcs.diagnostics import (
    write_plcs_diagnostics,
)
from src.synthetic_data_generation.dataset.plcs.handler import (
    _write_performance_metrics,
)
from src.synthetic_data_generation.dataset.plcs.production import PLCSProductionMode
from src.synthetic_data_generation.dataset.plcs.timeline import (
    PLCSLogicalScene,
    PLCSObjectTrack,
    PLCSSceneInventory,
    build_global_timeline,
)
from src.synthetic_data_generation.dataset.runtime import (
    BACKGROUND_STORE_SCHEMA,
    ChunkWriter,
    DatasetPerformanceMetrics,
    ForegroundDelta,
    ForegroundDeltaBatch,
    RenderSampleKey,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.tasks.base.generate_dataset.camera_profiles import (
    SampledCamera,
    SampledCameraRig,
)
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import PLCSMotionClip
from src.utils.projection.camera_projector import make_look_at_camera


def _tracks(tmp_path: Path) -> tuple[PLCSObjectTrack, ...]:
    tracks = []
    for index, category in enumerate(("running", "walking", "general"), start=1):
        path = tmp_path / f"{category}.npz"
        path.touch()
        clip = PLCSMotionClip.from_amass_arrays(
            source_path=path,
            category=category,
            gender="neutral",
            fps=30.0,
            poses=np.zeros((2, 156), dtype=np.float64),
            trans=np.zeros((2, 3), dtype=np.float64),
            betas=np.zeros(16, dtype=np.float64),
        )
        tracks.append(
            PLCSObjectTrack(
                object_id=f"player-{index:03d}",
                instance_id=index,
                asset_id=f"avatar-{index:03d}",
                clip=clip,
                start_frame=0,
                anchor_position_court_m=(0.0, 0.0, 0.0),
                yaw_radians=0.0,
            )
        )
    return tuple(tracks)


def _scene(
    scene_id: str,
    *,
    court_index: int,
    tracks: tuple[PLCSObjectTrack, ...],
    production_mode: PLCSProductionMode = (
        PLCSProductionMode.MULTI_OBJECT_GLOBAL_TIMELINE
    ),
) -> PLCSLogicalScene:
    return PLCSLogicalScene(
        split="train",
        timeline=build_global_timeline(
            scene_id=scene_id,
            production_mode=production_mode,
            target_court=TargetCourtBinding(
                court_instance_id=f"court-{court_index:03d}",
                candidate_id=f"candidate-{court_index:03d}",
                scene_from_court=RigidTransform.identity(),
                selection_seed=7,
            ),
            tracks=tracks,
        ),
    )


def _rig(scene: PLCSLogicalScene) -> SampledCameraRig:
    court_id = scene.timeline.target_court.court_instance_id
    cameras = []
    for index, y_value in enumerate((-4.0, 4.0)):
        slot_id = f"camera-{index}"
        center = (0.0, y_value, 3.0)
        look_at = (0.0, 0.0, 1.0)
        local = make_look_at_camera(
            center,
            look_at=look_at,
            image_size=(3, 2),
            hfov_deg=60.0,
        )
        camera_to_court = np.eye(4, dtype=np.float64)
        camera_to_court[:3, :3] = local.R.detach().cpu().numpy().T
        camera_to_court[:3, 3] = local.C.detach().cpu().numpy()
        camera = SceneCamera(
            camera_id=f"{court_id}-{slot_id}",
            source_frame_index=0,
            width=3,
            height=2,
            intrinsics=(
                local.f,
                0.0,
                local.cx,
                0.0,
                local.f,
                local.cy,
                0.0,
                0.0,
                1.0,
            ),
            camera_to_scene=RigidTransform.from_matrix(camera_to_court),
            image_path=f"generated/broadcast/{slot_id}.png",
        )
        cameras.append(
            SampledCamera(
                slot_id=slot_id,
                court_local_center_m=center,
                court_local_look_at_m=look_at,
                hfov_degrees=60.0,
                scene_camera=camera,
            )
        )
    return SampledCameraRig(
        profile="broadcast",
        seed=7,
        court_instance_id=court_id,
        cameras=tuple(cameras),
    )


def _scene_input(
    staging: Path,
    scene: PLCSLogicalScene,
    *,
    omit_last_frame: bool = False,
    storage_chunk_size: int | None = None,
) -> PLCSSceneAssemblyInput:
    timeline = scene.timeline
    rig = _rig(scene)
    attempt_token = f"B00-plcs-{timeline.scene_id}"
    writer = ChunkWriter(
        staging / "scenes" / timeline.scene_id / "chunks",
        attempt_token=attempt_token,
        camera_ids=tuple(camera.scene_camera.camera_id for camera in rig.cameras),
        width=3,
        height=2,
    )
    stop_frame = timeline.frame_count - int(omit_last_frame)
    chunk_size = storage_chunk_size or max(stop_frame, 1)
    readers = []
    for chunk_start in range(0, stop_frame, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, stop_frame)
        frame_indices = range(chunk_start, chunk_stop)
        deltas = tuple(
            ForegroundDelta(
                key=RenderSampleKey(
                    frame_index,
                    camera.scene_camera.camera_id,
                ),
                pixel_indices=np.empty(0, dtype=np.int32),
                rgb=np.empty((0, 3), dtype=np.float32),
                alpha=np.empty(0, dtype=np.float32),
                depth=np.empty(0, dtype=np.float32),
                instance_ids=np.empty(0, dtype=np.int32),
            )
            for frame_index in frame_indices
            for camera in rig.cameras
        )
        labels = tuple(
            build_frame_label(
                timeline=timeline,
                rig=rig,
                frame_index=frame_index,
                camera_index=camera_index,
                visibility={},
                seed=7,
            )
            for frame_index in frame_indices
            for camera_index in range(len(rig.cameras))
        )
        readers.append(
            writer.write(
                ForegroundDeltaBatch(
                    chunk_id=f"chunk-{chunk_start:06d}",
                    deltas=deltas,
                    metadata=labels,
                )
            )
        )
    frame_count = timeline.frame_count
    camera_count = len(rig.cameras)
    object_count = len(timeline.tracks)
    present: NDArray[np.bool_] = np.asarray(
        [
            [entry.present for entry in frame.entries]
            for frame in timeline.frames
        ],
        dtype=np.bool_,
    )
    rotation: NDArray[np.float32] = np.zeros(
        (frame_count, object_count, 2), dtype=np.float32
    )
    rotation[..., 0] = 1.0
    return PLCSSceneAssemblyInput(
        timeline=timeline,
        split=scene.split,
        rig=rig,
        chunk_readers=tuple(readers),
        attempt_token=attempt_token,
        supervision=PLCSSupervisionArrays(
            human_kp=np.zeros(
                (frame_count, camera_count, object_count, 17, 2), dtype=np.float32
            ),
            human_vis=np.zeros(
                (frame_count, camera_count, object_count, 17), dtype=np.bool_
            ),
            court_kp=np.zeros((frame_count, camera_count, 20, 2), dtype=np.float32),
            court_vis=np.zeros((frame_count, camera_count, 20), dtype=np.bool_),
            human_mask=np.broadcast_to(
                present[:, None, :], (frame_count, camera_count, object_count)
            ).copy(),
            position=np.zeros((frame_count, object_count, 3), dtype=np.float32),
            position_court_m=np.zeros((frame_count, object_count, 3), dtype=np.float32),
            rotation=rotation,
            present=present,
            human_kp_3d=np.zeros((frame_count, object_count, 17, 3), dtype=np.float32),
            canonical_pose_3d=np.zeros(
                (frame_count, object_count, 52, 3), dtype=np.float32
            ),
        ),
    )


def _inventory(tmp_path: Path) -> PLCSSceneInventory:
    tracks = _tracks(tmp_path)
    return PLCSSceneInventory(
        dataset_scene_id="B00",
        scenes=(
            _scene("B00", court_index=1, tracks=tracks),
            _scene("B00-plcs-002", court_index=2, tracks=tracks),
        ),
        accepted_court_instance_ids=("court-001", "court-002"),
        required_motion_categories=frozenset({"running", "walking", "general"}),
    )


def test_assembler_publishes_exact_aggregate_scene_inventory(tmp_path: Path) -> None:
    staging = tmp_path / ".transactions" / "plcs_dataset" / "snapshot"
    (staging / "backgrounds").mkdir(parents=True)
    inventory = _inventory(tmp_path)
    inputs = tuple(_scene_input(staging, scene) for scene in inventory.scenes)

    result = assemble_plcs_dataset(
        staging_directory=staging,
        inventory=inventory,
        scene_inputs=inputs,
        chunk_size=2,
        diagnostics=("diagnostics/example.json",),
        seed=7,
    )

    assert result.manifest.frame_inventory.source_count == 4
    assert result.manifest.frame_inventory.planned_indices == (0, 1, 2, 3)
    assert result.sample_count == 8
    assert result.chunk_count == 2
    assert [scene.continuity.frame_count for scene in result.scenes] == [2, 2]
    assert result.manifest.metadata["aggregate_source_frame_count"] == 12
    assert [binding.court_instance_id for binding in result.manifest.target_courts] == [
        "court-001",
        "court-002",
    ]


def test_assembler_rejects_incomplete_per_scene_global_timeline(tmp_path: Path) -> None:
    staging = tmp_path / ".transactions" / "plcs_dataset" / "snapshot"
    (staging / "backgrounds").mkdir(parents=True)
    inventory = _inventory(tmp_path)
    first, second = inventory.scenes

    with pytest.raises(ValueError, match="missing=.*1"):
        assemble_plcs_dataset(
            staging_directory=staging,
            inventory=inventory,
            scene_inputs=(
                _scene_input(staging, first),
                _scene_input(staging, second, omit_last_frame=True),
            ),
            chunk_size=2,
            diagnostics=("diagnostics/example.json",),
            seed=7,
        )


@dataclass(frozen=True)
class _Surface:
    gaussian_count: int = 8


@dataclass(frozen=True)
class _Articulation:
    def to_dict(self) -> dict[str, object]:
        return {
            "frame_count": 2,
            "category": "general",
            "non_root_pose_range_radians": 0.1,
            "gaussian_nonrigid_residual_m": 0.01,
            "region_displacement_m": {"arms": 0.01, "legs": 0.01, "torso": 0.01},
            "deformed_frame_indices": [0, 1],
        }


@dataclass(frozen=True)
class _Avatar:
    surface_asset: _Surface = _Surface()
    articulation: _Articulation = _Articulation()


def _write_background_store(
    staging: Path,
    scene_inputs: tuple[PLCSSceneAssemblyInput, ...],
) -> int:
    records = []
    for value in scene_inputs:
        for sampled in value.rig.cameras:
            camera = sampled.scene_camera
            camera_root = staging / "backgrounds" / camera.camera_id
            camera_root.mkdir(parents=True)
            np.save(camera_root / "rgb.npy", np.zeros((2, 3, 3), dtype=np.float32))
            np.save(camera_root / "alpha.npy", np.ones((2, 3, 1), dtype=np.float32))
            np.save(
                camera_root / "depth-metric.npy",
                np.full((2, 3, 1), 5.0, dtype=np.float32),
            )
            records.append(
                {
                    "camera_id": camera.camera_id,
                    "width": 3,
                    "height": 2,
                    "rgb": f"{camera.camera_id}/rgb.npy",
                    "alpha": f"{camera.camera_id}/alpha.npy",
                    "depth": f"{camera.camera_id}/depth-metric.npy",
                }
            )
    (staging / "backgrounds" / "backgrounds.json").write_text(
        json.dumps(
            {
                "schema": BACKGROUND_STORE_SCHEMA,
                "scene_id": "B00",
                "depth_coordinate_space": "metric_scene_metres",
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    return len(records)


def test_complete_multiscene_dataset_passes_strict_publication_validation(
    tmp_path: Path,
) -> None:
    from src.synthetic_data_generation.dataset.plcs.validation import (
        validate_plcs_dataset,
    )

    staging = tmp_path / ".transactions" / "plcs_dataset" / "snapshot"
    (staging / "backgrounds").mkdir(parents=True)
    inventory = _inventory(tmp_path)
    inputs = tuple(_scene_input(staging, scene) for scene in inventory.scenes)
    camera_count = _write_background_store(staging, inputs)
    diagnostic_paths = write_plcs_diagnostics(
        staging_directory=staging,
        inventory=inventory,
        rigs={value.timeline.scene_id: value.rig for value in inputs},
        avatars={
            track.object_id: _Avatar() for track in inventory.scenes[0].timeline.tracks
        },
        clip_load_count=3,
        model_load_count=1,
        execution_device="test-cpu-oracle",
        allow_test_cpu_oracle=True,
    )
    assembly = assemble_plcs_dataset(
        staging_directory=staging,
        inventory=inventory,
        scene_inputs=inputs,
        chunk_size=2,
        diagnostics=(*diagnostic_paths, "diagnostics/performance.json"),
        seed=7,
    )
    _write_performance_metrics(
        staging,
        DatasetPerformanceMetrics(
            domain="plcs",
            wall_seconds=0.1,
            cpu_seconds=0.1,
            peak_rss_bytes=1,
            execution_device="test-cpu-oracle",
            cuda_peak_bytes=0,
            nht_invocations=1,
            background_cache_misses=camera_count,
            complete_array_scans=assembly.sample_count,
            generated_bytes=0,
            published_bytes=0,
            dense_reference_bytes=1_000_000,
            frame_count=inventory.aggregate_global_frame_count,
            camera_count=camera_count,
            sample_count=assembly.sample_count,
        ),
    )

    result = validate_plcs_dataset(staging)

    assert result["logical_scene_count"] == 2
    assert result["frame_count"] == 4
    assert result["camera_count"] == 4
    assert result["camera_count_per_scene"] == 2
    assert result["sample_count"] == 8


def test_single_object_all_frames_cross_chunks_and_reopen_compact_v4(
    tmp_path: Path,
) -> None:
    from src.synthetic_data_generation.dataset.plcs.validation import (
        PLCSCompactDatasetReader,
        validate_plcs_dataset,
    )

    source_path = tmp_path / "running-single.npz"
    source_path.touch()
    frame_count = 5
    clip = PLCSMotionClip.from_amass_arrays(
        source_path=source_path,
        category="running",
        gender="neutral",
        fps=30.0,
        poses=np.zeros((frame_count, 156), dtype=np.float64),
        trans=np.zeros((frame_count, 3), dtype=np.float64),
        betas=np.zeros(16, dtype=np.float64),
    )
    tracks = (
        PLCSObjectTrack(
            object_id="player-001",
            instance_id=1,
            asset_id="avatar-001",
            clip=clip,
            start_frame=0,
            anchor_position_court_m=(0.0, 0.0, 0.0),
            yaw_radians=0.0,
        ),
    )
    inventory = PLCSSceneInventory(
        dataset_scene_id="B00",
        scenes=tuple(
            _scene(
                scene_id,
                court_index=court_index,
                tracks=tracks,
                production_mode=PLCSProductionMode.SINGLE_OBJECT,
            )
            for court_index, scene_id in enumerate(
                ("B00", "B00-plcs-002"), start=1
            )
        ),
        accepted_court_instance_ids=("court-001", "court-002"),
        required_motion_categories=frozenset({"running"}),
    )
    staging = tmp_path / ".transactions" / "plcs_dataset" / "snapshot"
    (staging / "backgrounds").mkdir(parents=True)
    inputs = tuple(
        _scene_input(staging, scene, storage_chunk_size=2)
        for scene in inventory.scenes
    )
    camera_count = _write_background_store(staging, inputs)
    diagnostic_paths = write_plcs_diagnostics(
        staging_directory=staging,
        inventory=inventory,
        rigs={value.timeline.scene_id: value.rig for value in inputs},
        avatars={"player-001": _Avatar()},
        clip_load_count=1,
        model_load_count=1,
        execution_device="test-cpu-oracle",
        allow_test_cpu_oracle=True,
    )
    assembly = assemble_plcs_dataset(
        staging_directory=staging,
        inventory=inventory,
        scene_inputs=inputs,
        chunk_size=2,
        diagnostics=(*diagnostic_paths, "diagnostics/performance.json"),
        seed=7,
    )
    _write_performance_metrics(
        staging,
        DatasetPerformanceMetrics(
            domain="plcs",
            wall_seconds=0.1,
            cpu_seconds=0.1,
            peak_rss_bytes=1,
            execution_device="test-cpu-oracle",
            cuda_peak_bytes=0,
            nht_invocations=1,
            background_cache_misses=camera_count,
            complete_array_scans=assembly.sample_count,
            generated_bytes=0,
            published_bytes=0,
            dense_reference_bytes=1_000_000,
            frame_count=inventory.aggregate_global_frame_count,
            camera_count=camera_count,
            sample_count=assembly.sample_count,
        ),
    )

    result = validate_plcs_dataset(staging)
    reader = PLCSCompactDatasetReader(staging)
    reopened = reader.materialize_all_views("B00")
    boundary = reader.logical_sample("B00", 2, "court-001-camera-0")
    payload = json.loads((staging / "dataset.json").read_text(encoding="utf-8"))

    assert assembly.manifest.schema == "tennis_plcs_compact_dataset_v4"
    assert result["frame_count"] == 2 * frame_count
    assert result["sample_count"] == 2 * frame_count * 2
    assert assembly.chunk_count == 6
    assert [scene.continuity.chunk_count for scene in assembly.scenes] == [3, 3]
    assert reopened.index.object_ids == ("player-001",)
    assert reopened.supervision.present[:, 0].tolist() == [True] * frame_count
    assert boundary.instance_ids.shape == (2, 3)
    assert payload["frame_inventory"] == {
        "first_frame": 0,
        "labelled": 2 * frame_count,
        "last_frame": 2 * frame_count - 1,
        "planned": 2 * frame_count,
        "rendered": 2 * frame_count,
        "source": 2 * frame_count,
    }
    assert all(
        scene["mode"] == "single"
        and scene["frame_inventory"]
        == {
            "first_frame": 0,
            "labelled": frame_count,
            "last_frame": frame_count - 1,
            "planned": frame_count,
            "rendered": frame_count,
            "source": frame_count,
        }
        for scene in payload["metadata"]["logical_scenes"]
    )


def test_persisted_validator_rejects_mode_cardinality_mismatch(tmp_path: Path) -> None:
    from src.synthetic_data_generation.dataset.plcs.validation import (
        validate_plcs_dataset,
    )

    staging = tmp_path / ".transactions" / "plcs_dataset" / "snapshot"
    (staging / "backgrounds").mkdir(parents=True)
    inventory = _inventory(tmp_path)
    inputs = tuple(_scene_input(staging, scene) for scene in inventory.scenes)
    camera_count = _write_background_store(staging, inputs)
    diagnostic_paths = write_plcs_diagnostics(
        staging_directory=staging,
        inventory=inventory,
        rigs={value.timeline.scene_id: value.rig for value in inputs},
        avatars={
            track.object_id: _Avatar()
            for track in inventory.scenes[0].timeline.tracks
        },
        clip_load_count=3,
        model_load_count=1,
        execution_device="test-cpu-oracle",
        allow_test_cpu_oracle=True,
    )
    assembly = assemble_plcs_dataset(
        staging_directory=staging,
        inventory=inventory,
        scene_inputs=inputs,
        chunk_size=2,
        diagnostics=(*diagnostic_paths, "diagnostics/performance.json"),
        seed=7,
    )
    _write_performance_metrics(
        staging,
        DatasetPerformanceMetrics(
            domain="plcs",
            wall_seconds=0.1,
            cpu_seconds=0.1,
            peak_rss_bytes=1,
            execution_device="test-cpu-oracle",
            cuda_peak_bytes=0,
            nht_invocations=1,
            background_cache_misses=camera_count,
            complete_array_scans=assembly.sample_count,
            generated_bytes=0,
            published_bytes=0,
            dense_reference_bytes=1_000_000,
            frame_count=inventory.aggregate_global_frame_count,
            camera_count=camera_count,
            sample_count=assembly.sample_count,
        ),
    )
    manifest_path = staging / "dataset.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["metadata"]["logical_scenes"][0]["mode"] = "single"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="single-object mode requires exactly one"):
        validate_plcs_dataset(staging)
