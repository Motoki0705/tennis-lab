"""Real full-clip PLCS semantic composition integration."""

from pathlib import Path

import pytest
import torch

from src.synthetic_data_generation.composition.gaussians import (
    assign_instance_id,
    compose_foreground_frame_gaussians,
)
from src.synthetic_data_generation.dataset.contracts import TargetCourtBinding
from src.synthetic_data_generation.dataset.plcs.composition import (
    AvatarAppearance,
    prepare_avatar,
)
from src.synthetic_data_generation.dataset.plcs.smplh import (
    load_smplh_model,
    upload_motion_clip,
    upload_smplh_model,
)
from src.synthetic_data_generation.dataset.plcs.timeline import (
    PLCSObjectTrack,
    build_global_timeline,
)
from src.synthetic_data_generation.scene_contract import RigidTransform
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import (
    load_amass_motion_clip,
)

_ACCAD = Path(
    "/home/kamimura/projects/tennis-lab/data/ACCAD/"
    "Male1Running_c3d/Run C25 - quick side step right_poses.npz"
)
_SMPLH = Path("/home/kamimura/projects/tennis-lab/data/smplh")


@pytest.mark.local_data
def test_full_real_accad_timeline_is_articulated_and_composable() -> None:
    if not _ACCAD.is_file() or not _SMPLH.is_dir():
        pytest.skip("Licensed ACCAD/SMPL-H assets are unavailable.")
    if not torch.cuda.is_available():
        pytest.skip("PLCS production composition requires CUDA.")
    device = torch.device("cuda:0")
    clip = load_amass_motion_clip(_ACCAD, category="running")
    model = load_smplh_model(_SMPLH, gender=clip.gender)
    device_model = upload_smplh_model(model, device=device)
    device_clip = upload_motion_clip(clip, model, device=device)
    appearance = AvatarAppearance(
        features=torch.full(
            (64, 3), 0.5, dtype=torch.float32, device=device
        ),
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
    assert all(value.means.device.type == "cuda" for value in tensors.values())
    first = compose_foreground_frame_gaussians(
        composition,
        frame_index=0,
        object_tensors={
            "player-001": assign_instance_id(tensors[0], 1)
        },
    )
    last = compose_foreground_frame_gaussians(
        composition,
        frame_index=clip.frame_count - 1,
        object_tensors={
            "player-001": assign_instance_id(tensors[clip.frame_count - 1], 1)
        },
    )
    assert first.gaussian_count == 64 == last.gaussian_count
    assert set(first.instance_ids.tolist()) == {1}
    assert set(last.instance_ids.tolist()) == {1}
