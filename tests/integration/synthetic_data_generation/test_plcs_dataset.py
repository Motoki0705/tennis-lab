"""Real full-clip PLCS semantic composition integration."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.synthetic_data_generation.composition.gaussians import (
    assign_instance_id,
    compose_foreground_frame_gaussians,
)
from src.synthetic_data_generation.dataset.continuity import (
    TimelineFrameRecord,
    validate_frame_continuity,
)
from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.plcs.composition import (
    AvatarAppearance,
    prepare_avatar,
)
from src.synthetic_data_generation.dataset.plcs.coordinates import (
    SMPLH_SURFACE_VERTEX_COUNT,
    PLCSSourceSupportPlane,
)
from src.synthetic_data_generation.dataset.plcs.production import PLCSProductionMode
from src.synthetic_data_generation.dataset.plcs.smplh import (
    SMPLHDeviceClip,
    SMPLHDeviceModel,
    SMPLHModelData,
    load_smplh_model,
    pose_smplh_surface_batch,
    upload_motion_clip,
    upload_smplh_model,
)
from src.synthetic_data_generation.dataset.plcs.timeline import (
    PLCSLogicalScene,
    PLCSObjectTrack,
    PLCSSceneInventory,
    build_global_timeline,
)
from src.synthetic_data_generation.scene_contract import RigidTransform
from src.tasks.plcs.generate_dataset.sampling.motion_source import (
    ACCADMotionLibrary,
    MotionCategory,
    PLCSMotionClip,
    load_amass_motion_clip,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ACCAD_ROOT = _PROJECT_ROOT / "data" / "ACCAD"
_ACCAD = _ACCAD_ROOT / "Male1Running_c3d" / "Run C25 - quick side step right_poses.npz"
_SMPLH = _PROJECT_ROOT / "data" / "smplh"


class _RealSupportEvidenceEvaluator:
    """Reuse CUDA models while deriving exact support for each real clip."""

    def __init__(self, *, device: torch.device) -> None:
        self.device = device
        self._models: dict[str, tuple[SMPLHModelData, SMPLHDeviceModel]] = {}

    def prepare(
        self,
        clip: PLCSMotionClip,
    ) -> tuple[
        SMPLHModelData,
        SMPLHDeviceModel,
        SMPLHDeviceClip,
        PLCSSourceSupportPlane,
    ]:
        if clip.gender not in self._models:
            model = load_smplh_model(_SMPLH, gender=clip.gender)
            self._models[clip.gender] = (
                model,
                upload_smplh_model(model, device=self.device),
            )
        model, device_model = self._models[clip.gender]
        device_clip = upload_motion_clip(clip, model, device=self.device)
        frame_zero_surface = pose_smplh_surface_batch(
            device_model,
            device_clip,
            source_frame_indices=(0,),
        )
        assert frame_zero_surface.shape == (1, SMPLH_SURFACE_VERTEX_COUNT, 3)
        support_plane = PLCSSourceSupportPlane.from_surface_minimum(
            initial_root_translation_z_m=float(clip.root_translation_m[0, 2]),
            support_local_z_m=float(frame_zero_surface[0, :, 2].amin().item()),
        )
        return model, device_model, device_clip, support_plane


@pytest.fixture(scope="module")
def real_support_evaluator() -> _RealSupportEvidenceEvaluator:
    assert _SMPLH.is_dir(), "Licensed SMPL-H assets are required."
    assert torch.cuda.is_available(), "Real SMPL-H support evidence requires CUDA."
    return _RealSupportEvidenceEvaluator(device=torch.device("cuda:0"))


def test_continuity_rejects_a_missing_terminal_track_camera_label() -> None:
    records = tuple(
        TimelineFrameRecord(
            frame_index=frame,
            chunk_index=frame // 2,
            track_id=track,
            present=True,
            source_frame_index=frame,
            camera_id=camera,
            label_id=f"{track}-{camera}-{frame}",
            court_instance_id="court-0",
        )
        for track in ("player-001", "player-002")
        for camera in ("camera-0", "camera-1")
        for frame in range(5)
        if (track, camera, frame) != ("player-002", "camera-1", 4)
    )

    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_frame_continuity(records, frame_count=5)


@pytest.mark.parametrize(
    ("missing_track", "missing_camera", "missing_frame"),
    [
        ("player-001", "camera-1", 4),
        ("player-002", "camera-0", 0),
    ],
)
def test_continuity_requires_explicit_terminal_absence_for_each_track_camera(
    missing_track: str,
    missing_camera: str,
    missing_frame: int,
) -> None:
    source_frames = {
        "player-001": (0, 1, 2, None, None),
        "player-002": (None, None, 0, 1, 2),
    }
    complete = tuple(
        TimelineFrameRecord(
            frame_index=frame,
            chunk_index=frame // 2,
            track_id=track,
            present=source_frame is not None,
            source_frame_index=source_frame,
            camera_id=camera,
            label_id=f"{track}-{camera}-{frame}",
            court_instance_id="court-0",
        )
        for track, mappings in source_frames.items()
        for camera in ("camera-0", "camera-1")
        for frame, source_frame in enumerate(mappings)
    )

    report = validate_frame_continuity(complete, frame_count=5)

    assert report.record_count == 2 * 2 * 5
    incomplete = tuple(
        record
        for record in complete
        if (record.track_id, record.camera_id, record.frame_index)
        != (missing_track, missing_camera, missing_frame)
    )
    with pytest.raises(ValueError, match="Track/camera timeline coverage mismatch"):
        validate_frame_continuity(incomplete, frame_count=5)


@pytest.mark.local_data
@pytest.mark.cuda
def test_full_real_accad_timeline_is_articulated_and_composable(
    real_support_evaluator: _RealSupportEvidenceEvaluator,
) -> None:
    assert _ACCAD.is_file(), "Licensed ACCAD assets are required."
    device = real_support_evaluator.device
    clip = load_amass_motion_clip(_ACCAD, category="running")
    model, device_model, device_clip, support_plane = real_support_evaluator.prepare(
        clip
    )
    appearance = AvatarAppearance(
        features=torch.full((64, 3), 0.5, dtype=torch.float32, device=device),
        appearance_model="rgb",
        appearance_space="linear_rgb",
    )
    avatar = prepare_avatar(
        asset_id="avatar-001",
        clip=clip,
        model=model,
        device_model=device_model,
        device_clip=device_clip,
        appearance=appearance,
        gaussian_count=64,
        seed=23,
    )
    timeline = build_global_timeline(
        scene_id="B00",
        production_mode=PLCSProductionMode.SINGLE_OBJECT,
        target_court=TargetCourtBinding(
            court_instance_id="court-001",
            candidate_id="candidate-001",
            scene_from_court=RigidTransform.identity(),
            selection_seed=23,
        ),
        tracks=(
            PLCSObjectTrack(
                object_id="player-001",
                instance_id=1,
                asset_id="avatar-001",
                clip=clip,
                support_plane=support_plane,
                start_frame=0,
                anchor_position_court_m=(0.0, 0.0, 0.0),
                yaw_radians=0.0,
            ),
        ),
    )
    composition = timeline.to_foreground_composition(
        assets=(avatar.semantic_asset,),
    )

    assert timeline.frame_count == clip.frame_count
    assert avatar.articulation.frame_count == clip.frame_count
    assert avatar.articulation.gaussian_nonrigid_residual_m > 0.01
    tensors = avatar.frame_tensors_batch((0, clip.frame_count - 1))
    assert all(
        value.gaussians.means.device.type == "cuda" for value in tensors.values()
    )
    assert all(value.joints_m.shape == (52, 3) for value in tensors.values())
    first = compose_foreground_frame_gaussians(
        composition,
        frame_index=0,
        object_tensors={"player-001": assign_instance_id(tensors[0].gaussians, 1)},
    )
    last = compose_foreground_frame_gaussians(
        composition,
        frame_index=clip.frame_count - 1,
        object_tensors={
            "player-001": assign_instance_id(tensors[clip.frame_count - 1].gaussians, 1)
        },
    )
    assert first.gaussian_count == 64 == last.gaussian_count
    assert set(first.instance_ids.tolist()) == {1}
    assert set(last.instance_ids.tolist()) == {1}


@pytest.mark.local_data
@pytest.mark.cuda
def test_real_accad_inventory_is_repeated_wholly_per_balanced_logical_scene(
    real_support_evaluator: _RealSupportEvidenceEvaluator,
) -> None:
    assert _ACCAD_ROOT.is_dir(), "Licensed ACCAD assets are required."
    library = ACCADMotionLibrary.from_root(_ACCAD_ROOT)
    requests = (
        (MotionCategory.RUNNING, 0, (-2.0, -5.0, 0.0), 0.0),
        (MotionCategory.WALKING, 120, (2.0, 5.0, 0.0), np.pi),
        (MotionCategory.GENERAL, 240, (0.0, 0.0, 0.0), np.pi / 2.0),
    )
    tracks = []
    for index, (category, start, anchor, yaw) in enumerate(requests):
        clip = library.select(category, seed=695 + index)
        _model, _device_model, _device_clip, support_plane = (
            real_support_evaluator.prepare(clip)
        )
        tracks.append(
            PLCSObjectTrack(
                object_id=f"player-{index + 1:03d}",
                instance_id=index + 1,
                asset_id=f"avatar-{index + 1:03d}",
                clip=clip,
                support_plane=support_plane,
                start_frame=start,
                anchor_position_court_m=anchor,
                yaw_radians=yaw,
            )
        )
    track_inventory = tuple(tracks)
    second_transform = np.eye(4, dtype=np.float64)
    second_transform[0, 3] = 30.0
    scenes = tuple(
        PLCSLogicalScene(
            split="train",
            timeline=build_global_timeline(
                scene_id=scene_id,
                production_mode=(PLCSProductionMode.MULTI_OBJECT_GLOBAL_TIMELINE),
                target_court=TargetCourtBinding(
                    court_instance_id=f"court-{index:03d}",
                    candidate_id=f"candidate-{index:03d}",
                    scene_from_court=(
                        RigidTransform.identity()
                        if index == 0
                        else RigidTransform.from_matrix(second_transform)
                    ),
                    selection_seed=695,
                ),
                tracks=track_inventory,
            ),
        )
        for index, scene_id in enumerate(("B00", "B00-plcs-002"))
    )

    inventory = PLCSSceneInventory(
        dataset_scene_id="B00",
        scenes=scenes,
        accepted_court_instance_ids=("court-000", "court-001"),
        required_motion_categories=frozenset({"running", "walking", "general"}),
    )

    assert inventory.scene_count == 2
    assert inventory.aggregate_global_frame_count == 2 * scenes[0].timeline.frame_count
    assert inventory.aggregate_source_frame_count == 2 * sum(
        track.clip.frame_count for track in track_inventory
    )
    for scene in inventory.scenes:
        assert tuple(
            track.clip.source_path for track in scene.timeline.tracks
        ) == tuple(track.clip.source_path for track in track_inventory)
        assert tuple(frame.frame_index for frame in scene.timeline.frames) == tuple(
            range(scene.timeline.frame_count)
        )
