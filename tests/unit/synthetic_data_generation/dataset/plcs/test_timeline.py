"""Tests for complete single/multi PLCS global timelines."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.plcs.assembler import build_frame_label
from src.synthetic_data_generation.dataset.plcs.timeline import (
    PLCSObjectTrack,
    build_global_timeline,
)
from src.synthetic_data_generation.scene_contract import RigidTransform, SceneCamera
from src.tasks.base.generate_dataset.camera_profiles import (
    SampledCamera,
    SampledCameraRig,
)
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import PLCSMotionClip


def _clip(tmp_path: Path, name: str, frame_count: int) -> PLCSMotionClip:
    poses: NDArray[np.float64] = np.zeros((frame_count, 156), dtype=np.float64)
    poses[:, 3] = np.linspace(0.0, 0.3, frame_count)
    trans: NDArray[np.float64] = np.zeros((frame_count, 3), dtype=np.float64)
    trans[:, 0] = np.linspace(0.0, 0.5, frame_count)
    return PLCSMotionClip.from_amass_arrays(
        source_path=tmp_path / f"{name}.npz",
        category="general",
        gender="neutral",
        fps=30.0,
        poses=poses,
        trans=trans,
        betas=np.zeros(16, dtype=np.float64),
    )


def _binding() -> TargetCourtBinding:
    return TargetCourtBinding(
        court_instance_id="court-001",
        candidate_id="candidate-001",
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
        clip=_clip(tmp_path, "first", 4),
        start_frame=0,
        anchor_position_court_m=(-1.0, -5.0, 0.0),
        yaw_radians=0.0,
    )
    second = PLCSObjectTrack(
        object_id="player-002",
        instance_id=2,
        asset_id="avatar-002",
        clip=_clip(tmp_path, "second", 5),
        start_frame=2,
        anchor_position_court_m=(1.0, 5.0, 0.0),
        yaw_radians=np.pi,
    )
    timeline = build_global_timeline(
        scene_id="B00",
        target_court=_binding(),
        tracks=(first, second),
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
