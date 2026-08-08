"""Bounded validation against the real local ACCAD and SMPL-H assets."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.synthetic_data_generation.dataset.plcs.smplh import (
    build_smplh_surface_asset,
    load_smplh_model,
    skin_gaussian_batch,
    upload_gaussian_asset,
    upload_motion_clip,
    upload_smplh_model,
)
from src.tasks.plcs.generate_dataset.sampling.motion_sampler import PLCSMotionClip

_ACCAD = Path(
    "/home/kamimura/projects/tennis-lab/data/ACCAD/"
    "Male1Running_c3d/Run C25 - quick side step right_poses.npz"
)
_SMPLH = Path("/home/kamimura/projects/tennis-lab/data/smplh")


@pytest.mark.local_data
def test_real_smplh_model_applies_gaussian_lbs_to_every_bounded_source_frame() -> None:
    if not _ACCAD.is_file() or not _SMPLH.is_dir() or not torch.cuda.is_available():
        pytest.skip("Licensed ACCAD/SMPL-H assets or CUDA are unavailable.")
    with np.load(_ACCAD, allow_pickle=False) as archive:
        clip = PLCSMotionClip.from_amass_arrays(
            source_path=_ACCAD,
            category="running",
            gender=str(archive["gender"].item()),
            fps=float(archive["mocap_framerate"].item()),
            poses=archive["poses"][:4],
            trans=archive["trans"][:4],
            betas=archive["betas"],
        )
    model = load_smplh_model(_SMPLH, gender=clip.gender)
    asset = build_smplh_surface_asset(
        model,
        clip,
        gaussian_count=32,
        seed=11,
    )
    device_model = upload_smplh_model(model, device="cuda:0")
    device_clip = upload_motion_clip(clip, model, device="cuda:0")
    device_asset = upload_gaussian_asset(asset, device="cuda:0")
    first = skin_gaussian_batch(
        device_model,
        device_clip,
        device_asset,
        source_frame_indices=(0, 1),
    )
    second = skin_gaussian_batch(
        device_model,
        device_clip,
        device_asset,
        source_frame_indices=(2, 3),
    )

    assert first.means_m.shape == (2, 32, 3)
    assert second.means_m.shape == (2, 32, 3)
    assert first.joints_m.shape == (2, 52, 3)
    all_means = torch.cat((first.means_m, second.means_m), dim=0)
    assert (
        float(
            torch.max(all_means.max(dim=0).values - all_means.min(dim=0).values).item()
        )
        > 1.0e-5
    )
    assert all_means.device.type == "cuda"
