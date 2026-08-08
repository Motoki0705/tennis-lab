"""Tests for complete single/multi PLCS global timelines."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.plcs.assembler import build_frame_label
from src.synthetic_data_generation.dataset.plcs.production import PLCSProductionMode
from src.synthetic_data_generation.dataset.plcs.timeline import (
    PLCSLogicalScene,
    PLCSObjectTrack,
    PLCSSceneInventory,
    build_global_timeline,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.tasks.base.generate_dataset.camera_profiles import (
    SampledCamera,
    SampledCameraRig,
)
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import PLCSMotionClip


def _clip(
    tmp_path: Path,
    name: str,
    frame_count: int,
    *,
    category: str = "general",
) -> PLCSMotionClip:
    poses: NDArray[np.float64] = np.zeros((frame_count, 156), dtype=np.float64)
    poses[:, 3] = np.linspace(0.0, 0.3, frame_count)
    trans: NDArray[np.float64] = np.zeros((frame_count, 3), dtype=np.float64)
    trans[:, 0] = np.linspace(0.0, 0.5, frame_count)
    return PLCSMotionClip.from_amass_arrays(
        source_path=tmp_path / f"{name}.npz",
        category=category,
        gender="neutral",
        fps=30.0,
        poses=poses,
        trans=trans,
        betas=np.zeros(16, dtype=np.float64),
    )


def _binding(index: int = 1) -> TargetCourtBinding:
    return TargetCourtBinding(
        court_instance_id=f"court-{index:03d}",
        candidate_id=f"candidate-{index:03d}",
        scene_from_court=RigidTransform.identity(),
        selection_seed=7,
    )


def test_single_timeline_is_exact_source_interval(tmp_path: Path) -> None:
    clip = _clip(tmp_path, "single", 5)
    track = PLCSObjectTrack(
        object_id="player-001",
        instance_id=1,
        asset_id="avatar-001",
        clip=clip,
        start_frame=0,
        anchor_position_court_m=(0.0, 0.0, 0.0),
        yaw_radians=0.0,
    )
    timeline = build_global_timeline(
        scene_id="B00",
        production_mode=PLCSProductionMode.SINGLE_OBJECT,
        target_court=_binding(),
        tracks=(track,),
    )

    assert timeline.mode == "single"
    assert timeline.frame_count == clip.frame_count
    assert [frame.entries[0].source_frame_index for frame in timeline.frames] == list(
        range(clip.frame_count)
    )
    assert all(frame.entries[0].present for frame in timeline.frames)


def test_multi_timeline_keeps_global_interval_presence_and_source_mapping(
    tmp_path: Path,
) -> None:
    first = PLCSObjectTrack(
        object_id="player-001",
        instance_id=1,
        asset_id="avatar-001",
        clip=_clip(tmp_path, "first", 4, category="running"),
        start_frame=0,
        anchor_position_court_m=(-1.0, -5.0, 0.0),
        yaw_radians=0.0,
    )
    second = PLCSObjectTrack(
        object_id="player-002",
        instance_id=2,
        asset_id="avatar-002",
        clip=_clip(tmp_path, "second", 5, category="walking"),
        start_frame=2,
        anchor_position_court_m=(1.0, 5.0, 0.0),
        yaw_radians=np.pi,
    )
    third = PLCSObjectTrack(
        object_id="player-003",
        instance_id=3,
        asset_id="avatar-003",
        clip=_clip(tmp_path, "third", 3, category="general"),
        start_frame=1,
        anchor_position_court_m=(0.0, 0.0, 0.0),
        yaw_radians=np.pi / 2.0,
    )
    timeline = build_global_timeline(
        scene_id="B00",
        production_mode=PLCSProductionMode.MULTI_OBJECT_GLOBAL_TIMELINE,
        target_court=_binding(),
        tracks=(first, second, third),
    )

    assert timeline.mode == "multi"
    assert timeline.frame_count == 7
    assert [frame.entries[0].source_frame_index for frame in timeline.frames] == [
        0,
        1,
        2,
        3,
        None,
        None,
        None,
    ]
    assert [frame.entries[1].source_frame_index for frame in timeline.frames] == [
        None,
        None,
        0,
        1,
        2,
        3,
        4,
    ]
    final_second_transform = timeline.frames[-1].entries[1].scene_from_asset
    assert final_second_transform is not None
    np.testing.assert_allclose(
        final_second_transform.rigid.matrix()[:3, 3],
        (0.5, 5.0, 0.0),
        atol=1.0e-12,
        rtol=0.0,
    )

    camera = SceneCamera(
        camera_id="camera-0",
        source_frame_index=0,
        width=8,
        height=8,
        intrinsics=(8.0, 0.0, 3.5, 0.0, 8.0, 3.5, 0.0, 0.0, 1.0),
        camera_to_scene=RigidTransform.identity(),
        image_path="request-only",
    )
    rig = SampledCameraRig(
        profile="default",
        seed=7,
        court_instance_id="court-001",
        cameras=(
            SampledCamera(
                slot_id="camera-0",
                court_local_center_m=(0.0, 0.0, 1.0),
                court_local_look_at_m=(0.0, 1.0, 1.0),
                hfov_degrees=60.0,
                scene_camera=camera,
            ),
        ),
    )

    label = build_frame_label(
        timeline=timeline,
        rig=rig,
        frame_index=0,
        camera_index=0,
        visibility={},
        seed=7,
    )
    objects = label["objects"]
    assert isinstance(objects, list)
    assert objects[0]["present"] is True
    assert objects[0]["visible_pixel_count"] == 0
    assert objects[1]["present"] is False
    assert objects[1]["visible_pixel_count"] == 0


def _complete_tracks(tmp_path: Path) -> tuple[PLCSObjectTrack, ...]:
    return tuple(
        PLCSObjectTrack(
            object_id=f"player-{index:03d}",
            instance_id=index,
            asset_id=f"avatar-{index:03d}",
            clip=_clip(
                tmp_path,
                category,
                frame_count,
                category=category,
            ),
            start_frame=start,
            anchor_position_court_m=(0.0, 0.0, 0.0),
            yaw_radians=0.0,
        )
        for index, (category, frame_count, start) in enumerate(
            (("running", 4, 0), ("walking", 5, 2), ("general", 3, 2)),
            start=1,
        )
    )


def _logical_scene(
    scene_id: str,
    *,
    split: str,
    binding_index: int,
    tracks: tuple[PLCSObjectTrack, ...],
) -> PLCSLogicalScene:
    return PLCSLogicalScene(
        split=split,
        timeline=build_global_timeline(
            scene_id=scene_id,
            production_mode=PLCSProductionMode.MULTI_OBJECT_GLOBAL_TIMELINE,
            target_court=_binding(binding_index),
            tracks=tracks,
        ),
    )


def test_scene_inventory_keeps_each_global_timeline_intact_and_uses_every_court(
    tmp_path: Path,
) -> None:
    tracks = _complete_tracks(tmp_path)
    inventory = PLCSSceneInventory(
        dataset_scene_id="B00",
        scenes=(
            _logical_scene("B00", split="train", binding_index=1, tracks=tracks),
            _logical_scene(
                "B00-plcs-002", split="train", binding_index=2, tracks=tracks
            ),
        ),
        accepted_court_instance_ids=("court-001", "court-002"),
        required_motion_categories=frozenset({"running", "walking", "general"}),
    )

    assert inventory.scene_count == 2
    assert inventory.aggregate_global_frame_count == 14
    assert inventory.aggregate_source_frame_count == 24
    assert [
        scene.timeline.target_court.court_instance_id for scene in inventory.scenes
    ] == ["court-001", "court-002"]
    for scene in inventory.scenes:
        assert scene.timeline.frame_count == 7
        assert {track.clip.category.value for track in scene.timeline.tracks} == {
            "running",
            "walking",
            "general",
        }
        for track_index, track in enumerate(scene.timeline.tracks):
            assert [
                frame.entries[track_index].source_frame_index
                for frame in scene.timeline.frames
                if frame.entries[track_index].present
            ] == list(range(track.clip.frame_count))


def test_scene_inventory_rejects_too_few_scenes_for_accepted_courts(
    tmp_path: Path,
) -> None:
    tracks = _complete_tracks(tmp_path)

    with pytest.raises(ValueError, match="cannot cover every accepted court"):
        PLCSSceneInventory(
            dataset_scene_id="B00",
            scenes=(
                _logical_scene("B00", split="train", binding_index=1, tracks=tracks),
            ),
            accepted_court_instance_ids=("court-001", "court-002"),
            required_motion_categories=frozenset({"running", "walking", "general"}),
        )


def test_scene_inventory_rejects_missing_accepted_court(tmp_path: Path) -> None:
    tracks = _complete_tracks(tmp_path)

    with pytest.raises(ValueError, match="do not use every accepted court"):
        PLCSSceneInventory(
            dataset_scene_id="B00",
            scenes=(
                _logical_scene("B00", split="train", binding_index=1, tracks=tracks),
                _logical_scene(
                    "B00-plcs-002",
                    split="train",
                    binding_index=1,
                    tracks=tracks,
                ),
            ),
            accepted_court_instance_ids=("court-001", "court-002"),
            required_motion_categories=frozenset({"running", "walking", "general"}),
        )


def test_scene_inventory_rejects_court_count_imbalance(tmp_path: Path) -> None:
    tracks = _complete_tracks(tmp_path)

    with pytest.raises(ValueError, match="count difference exceeds one"):
        PLCSSceneInventory(
            dataset_scene_id="B00",
            scenes=(
                _logical_scene("B00", split="train", binding_index=1, tracks=tracks),
                _logical_scene(
                    "B00-plcs-002",
                    split="train",
                    binding_index=1,
                    tracks=tracks,
                ),
                _logical_scene(
                    "B00-plcs-003",
                    split="train",
                    binding_index=1,
                    tracks=tracks,
                ),
                _logical_scene(
                    "B00-plcs-004",
                    split="train",
                    binding_index=2,
                    tracks=tracks,
                ),
            ),
            accepted_court_instance_ids=("court-001", "court-002"),
            required_motion_categories=frozenset({"running", "walking", "general"}),
        )


def test_scene_inventory_rejects_per_split_court_imbalance(tmp_path: Path) -> None:
    tracks = _complete_tracks(tmp_path)

    with pytest.raises(ValueError, match="imbalanced in 'train'"):
        PLCSSceneInventory(
            dataset_scene_id="B00",
            scenes=(
                _logical_scene("B00", split="train", binding_index=1, tracks=tracks),
                _logical_scene(
                    "B00-plcs-002",
                    split="train",
                    binding_index=1,
                    tracks=tracks,
                ),
                _logical_scene(
                    "B00-plcs-003",
                    split="validation",
                    binding_index=2,
                    tracks=tracks,
                ),
                _logical_scene(
                    "B00-plcs-004",
                    split="validation",
                    binding_index=2,
                    tracks=tracks,
                ),
            ),
            accepted_court_instance_ids=("court-001", "court-002"),
            required_motion_categories=frozenset({"running", "walking", "general"}),
        )
