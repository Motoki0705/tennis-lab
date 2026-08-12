"""Licensed real-data regression for the PLCS AMASS/SMPL-H coordinate contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.plcs.composition import (
    transform_asset_points,
)
from src.synthetic_data_generation.dataset.plcs.coordinates import (
    PLCS_SUPPORT_PLACEMENT_TOLERANCE_M,
    SMPLH_SURFACE_VERTEX_COUNT,
    PLCSSourceSupportPlane,
)
from src.synthetic_data_generation.dataset.plcs.production import PLCSProductionMode
from src.synthetic_data_generation.dataset.plcs.smplh import (
    SMPLHDeviceModel,
    SMPLHModelData,
    build_smplh_surface_asset,
    load_smplh_model,
    pose_smplh_surface_batch,
    skin_gaussian_batch,
    upload_gaussian_asset,
    upload_motion_clip,
    upload_smplh_model,
)
from src.synthetic_data_generation.dataset.plcs.timeline import (
    PLCSObjectTrack,
    build_global_timeline,
)
from src.synthetic_data_generation.scene_contract import RigidTransform
from src.tasks.plcs.generate_dataset.sampling.motion_source import PLCSMotionClip

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SMPLH_ROOT = _PROJECT_ROOT / "data" / "smplh"


@dataclass(frozen=True, slots=True)
class _RealClipCase:
    name: str
    category: str
    gender: str
    archive: Path
    frame_indices: tuple[int, ...]


_STANDING = _RealClipCase(
    name="standing",
    category="general",
    gender="female",
    archive=(_PROJECT_ROOT / "data/ACCAD/Female1General_c3d/A1 - Stand_poses.npz"),
    frame_indices=(0, 180, 359),
)
_UPRIGHT_REGRESSIONS = (
    _RealClipCase(
        name="running",
        category="running",
        gender="male",
        archive=(
            _PROJECT_ROOT
            / "data/ACCAD/Male1Running_c3d/Run C25 - quick side step right_poses.npz"
        ),
        frame_indices=(0, 199, 397),
    ),
    _RealClipCase(
        name="walking",
        category="walking",
        gender="male",
        archive=(
            _PROJECT_ROOT
            / "data/ACCAD/Male1Walking_c3d/Walk B16 - Walk turn change_poses.npz"
        ),
        frame_indices=(0, 300, 599),
    ),
    _RealClipCase(
        name="general",
        category="general",
        gender="male",
        archive=(
            _PROJECT_ROOT
            / "data/ACCAD/Male1General_c3d/General A3 - Swing Arms While Stand_poses.npz"
        ),
        frame_indices=(0, 344, 687),
    ),
)


def _require_local_coordinate_assets(cases: tuple[_RealClipCase, ...]) -> None:
    missing = [str(case.archive) for case in cases if not case.archive.is_file()]
    missing.extend(
        str(_SMPLH_ROOT / case.gender / "model.npz")
        for case in cases
        if not (_SMPLH_ROOT / case.gender / "model.npz").is_file()
    )
    if missing or not torch.cuda.is_available():
        pytest.skip(
            "Licensed fixed ACCAD/SMPL-H assets or CUDA are unavailable: "
            + ", ".join(missing)
        )


def _load_clip(case: _RealClipCase) -> PLCSMotionClip:
    with np.load(case.archive, allow_pickle=False) as archive:
        assert str(archive["gender"].item()) == case.gender
        stop = max(case.frame_indices) + 1
        clip = PLCSMotionClip.from_amass_arrays(
            source_path=case.archive,
            category=case.category,
            gender=case.gender,
            fps=float(archive["mocap_framerate"].item()),
            poses=archive["poses"][:stop],
            trans=archive["trans"][:stop],
            betas=archive["betas"],
        )
    assert max(case.frame_indices) < clip.frame_count
    return clip


def _models(gender: str) -> tuple[SMPLHModelData, SMPLHDeviceModel]:
    model = load_smplh_model(_SMPLH_ROOT, gender=gender)
    assert model.gender == gender
    return model, upload_smplh_model(model, device="cuda:0")


def _placed_joint_vectors(
    case: _RealClipCase,
    *,
    yaw_radians: float,
) -> tuple[np.ndarray, PLCSObjectTrack, torch.Tensor]:
    clip = _load_clip(case)
    model, device_model = _models(case.gender)
    device_clip = upload_motion_clip(clip, model, device="cuda:0")
    surface_asset = build_smplh_surface_asset(
        model,
        clip,
        gaussian_count=32,
        seed=743,
    )
    device_asset = upload_gaussian_asset(surface_asset, device="cuda:0")
    posed = skin_gaussian_batch(
        device_model,
        device_clip,
        device_asset,
        source_frame_indices=case.frame_indices,
    )
    frame_zero_surface = pose_smplh_surface_batch(
        device_model,
        device_clip,
        source_frame_indices=(0,),
    )
    assert frame_zero_surface.shape == (1, SMPLH_SURFACE_VERTEX_COUNT, 3)
    local_min_z = float(frame_zero_surface[0, :, 2].amin().item())
    support = PLCSSourceSupportPlane.from_surface_minimum(
        initial_root_translation_z_m=float(clip.root_translation_m[0, 2]),
        support_local_z_m=local_min_z,
    )
    track = PLCSObjectTrack(
        object_id="player-001",
        instance_id=1,
        asset_id="avatar-001",
        clip=clip,
        support_plane=support,
        start_frame=0,
        anchor_position_court_m=(0.0, 0.0, 0.0),
        yaw_radians=yaw_radians,
    )
    timeline = build_global_timeline(
        scene_id="B00",
        production_mode=PLCSProductionMode.SINGLE_OBJECT,
        target_court=TargetCourtBinding(
            court_instance_id="court-001",
            candidate_id="candidate-001",
            scene_from_court=RigidTransform.identity(),
            selection_seed=743,
        ),
        tracks=(track,),
    )
    placed_vectors = []
    for batch_index, source_frame_index in enumerate(case.frame_indices):
        transform = timeline.frames[source_frame_index].entries[0].scene_from_asset
        assert transform is not None
        placed_joints = transform_asset_points(
            posed.joints_m[batch_index],
            transform=transform,
        )
        placed_vectors.append((placed_joints[15] - placed_joints[0]).cpu().numpy())
    return np.stack(placed_vectors), track, frame_zero_surface


@pytest.mark.local_data
def test_fixed_standing_clip_is_court_z_up_and_full_surface_supported() -> None:
    _require_local_coordinate_assets((_STANDING,))

    vectors, track, frame_zero_surface = _placed_joint_vectors(
        _STANDING,
        yaw_radians=0.63,
    )

    assert np.all(vectors[:, 2] > 0.0)
    assert np.all(vectors[:, 2] > np.max(np.abs(vectors[:, :2]), axis=1))
    timeline = build_global_timeline(
        scene_id="B00",
        production_mode=PLCSProductionMode.SINGLE_OBJECT,
        target_court=TargetCourtBinding(
            court_instance_id="court-001",
            candidate_id="candidate-001",
            scene_from_court=RigidTransform.identity(),
            selection_seed=743,
        ),
        tracks=(track,),
    )
    transform = timeline.frames[0].entries[0].scene_from_asset
    assert transform is not None
    placed_surface = transform_asset_points(
        frame_zero_surface[0],
        transform=transform,
    )
    assert float(placed_surface[:, 2].amin().abs().item()) <= (
        PLCS_SUPPORT_PLACEMENT_TOLERANCE_M
    )
    expected_yaw = np.asarray(
        (
            (np.cos(0.63), -np.sin(0.63), 0.0),
            (np.sin(0.63), np.cos(0.63), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        np.asarray(transform.rotation).reshape(3, 3),
        expected_yaw,
        atol=1.0e-12,
        rtol=0.0,
    )


@pytest.mark.local_data
@pytest.mark.parametrize("case", _UPRIGHT_REGRESSIONS, ids=lambda case: case.name)
def test_fixed_accad_category_clips_remain_upright_after_yaw_only_placement(
    case: _RealClipCase,
) -> None:
    _require_local_coordinate_assets((case,))

    vectors, _track, _surface = _placed_joint_vectors(
        case,
        yaw_radians=-0.41,
    )

    assert np.all(vectors[:, 2] > 0.0)
    assert np.all(vectors[:, 2] > np.max(np.abs(vectors[:, :2]), axis=1))
