"""Fixed-width BLCS observation packing and collation tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from src.tasks.blcs.data.observation_candidates import (
    build_fixed_lifecycle_assignment,
    pack_observation_candidates,
)
from src.tasks.blcs.data.tracking_dataset import (
    BLCSTrackingDataset,
    collate_blcs_tracking_batch,
)

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


def test_observations_pack_all_views_to_exact_width_with_lifecycle_reuse() -> None:
    presence = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [False, False, True],
            [False, False, True],
        ]
    )
    visible = presence.unsqueeze(0).expand(2, -1, -1).clone()
    visible[1, 0, 1] = False
    uv = torch.arange(2 * 4 * 3 * 2, dtype=torch.float32).reshape(2, 4, 3, 2)

    packed = pack_observation_candidates(
        ball_uv=uv,
        ball_visible=visible,
        physical_presence=presence,
        num_slots=2,
        min_reuse_gap_frames=0,
        randomize_slots=False,
    )

    assert packed.uv.shape == (2, 4, 2, 2)
    assert packed.visible.shape == packed.candidate_mask.shape == (2, 4, 2)
    torch.testing.assert_close(packed.candidate_mask[0], packed.candidate_mask[1])
    torch.testing.assert_close(packed.gt_index[0], packed.gt_index[1])
    assert packed.gt_index[0, 0, 0] == 0
    assert packed.gt_index[0, 2, 0] == 2
    assert packed.candidate_mask[1, 0, 1]
    assert not packed.visible[1, 0, 1]


def test_concurrent_candidate_overflow_is_rejected_without_truncation() -> None:
    presence = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="cannot be packed"):
        pack_observation_candidates(
            ball_uv=torch.rand(1, 2, 3, 2),
            ball_visible=presence.unsqueeze(0),
            physical_presence=presence,
            num_slots=2,
            min_reuse_gap_frames=0,
            randomize_slots=False,
        )


def test_training_assignments_use_independent_torch_draws_and_eval_is_stable() -> None:
    presence = torch.ones(2, 4, dtype=torch.bool)
    torch.manual_seed(753)
    target = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
        randomize_slots=True,
    )
    observation = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
        randomize_slots=True,
    )
    assert not torch.equal(target.track_to_slot, observation.track_to_slot)

    first_eval = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
        randomize_slots=False,
    )
    second_eval = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
        randomize_slots=False,
    )
    torch.testing.assert_close(first_eval.track_to_slot, second_eval.track_to_slot)


def test_dataset_packs_more_physical_tracks_than_q_with_independent_assignments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the persisted-scene boundary, not only the packing helper."""
    scene = tmp_path / "scenes" / "scene_000000"
    scene.mkdir(parents=True)
    (scene / "meta.json").write_text(
        json.dumps({"num_frames": 4}), encoding="utf-8"
    )
    (scene / "scalars.json").write_text(
        json.dumps({"num_cameras": 2}), encoding="utf-8"
    )
    (tmp_path / "train.txt").write_text("scene_000000\n", encoding="utf-8")

    presence = np.asarray(
        [
            [True, True, False],
            [True, False, False],
            [False, False, True],
            [False, False, True],
        ],
        dtype=np.bool_,
    )
    np.save(scene / "ball_present.npy", presence)
    np.save(scene / "ball_pos_norm.npy", np.zeros((4, 3, 3), dtype=np.float32))
    np.save(scene / "ball_vel_world.npy", np.zeros((4, 3, 3), dtype=np.float32))
    for camera_index in range(2):
        uv = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2) / 100.0
        visible = presence.copy()
        if camera_index == 1:
            visible[0, 1] = False
        np.save(scene / f"cam_{camera_index}_ball_uv.npy", uv)
        np.save(scene / f"cam_{camera_index}_ball_visible.npy", visible)
        np.save(
            scene / f"cam_{camera_index}_court_kp_uv.npy",
            np.zeros((14, 2), dtype=np.float32),
        )
        np.save(
            scene / f"cam_{camera_index}_court_kp_visible.npy",
            np.ones(14, dtype=np.bool_),
        )

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_tracking")
    config.model.num_queries = 2
    config.data.seq_len_range = [4, 4]
    config.data.num_views_range = [2, 2]
    config.data.camera_mode = "first"
    config.data.lifecycle.min_reuse_gap_frames = 0
    config.data.augmentation.enabled = False

    permutations = iter((torch.tensor([0, 1]), torch.tensor([1, 0])))

    def next_permutation(
        width: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        del generator
        assert width == 2
        return next(permutations).to(device=device)

    monkeypatch.setattr(torch, "randperm", next_permutation)
    sample = BLCSTrackingDataset(
        scene_dir=tmp_path,
        split_file="train.txt",
        config=config,
        augment=True,
    )[0]

    assert sample["ball_uv"].shape == (2, 4, 2, 2)
    assert sample["candidate_mask"].shape == (2, 4, 2)
    torch.testing.assert_close(
        sample["candidate_mask"][0], sample["candidate_mask"][1]
    )
    assert set(sample["candidate_gt_index"].unique().tolist()) == {-1, 0, 1, 2}
    assert set(sample["target_instance_id"].unique().tolist()) == {-1, 0, 1, 2}
    assert not torch.equal(
        sample["candidate_gt_index"][0], sample["target_instance_id"]
    )
    assert sample["candidate_mask"][1, 0, 0]
    assert not sample["ball_visible"][1, 0, 0]


def _sample(*, views: int, frames: int, queries: int) -> dict[str, torch.Tensor]:
    candidate_shape = (views, frames, queries)
    return {
        "scene_format_version": torch.tensor(3),
        "ball_uv": torch.zeros(*candidate_shape, 2),
        "ball_visible": torch.zeros(candidate_shape, dtype=torch.bool),
        "candidate_mask": torch.zeros(candidate_shape, dtype=torch.bool),
        "court_kp": torch.zeros(views, frames, 14, 2),
        "court_vis": torch.zeros(views, frames, 14, dtype=torch.bool),
        "frame_mask": torch.ones(frames, dtype=torch.bool),
        "view_mask": torch.ones(views, dtype=torch.bool),
        "target_position": torch.zeros(frames, queries, 3),
        "target_velocity": torch.zeros(frames, queries, 3),
        "target_presence": torch.zeros(frames, queries, dtype=torch.bool),
        "target_instance_id": torch.full((frames, queries), -1),
        "target_slot_mask": torch.zeros(queries, dtype=torch.bool),
        "clean_ball_uv": torch.zeros(*candidate_shape, 2),
        "clean_ball_visible": torch.zeros(candidate_shape, dtype=torch.bool),
        "candidate_gt_index": torch.full(candidate_shape, -1),
    }


def test_collate_pads_view_and_time_but_never_candidate_axis() -> None:
    result = collate_blcs_tracking_batch(
        [_sample(views=1, frames=2, queries=4), _sample(views=2, frames=3, queries=4)]
    )
    assert result["ball_uv"].shape == (2, 2, 3, 4, 2)
    assert result["candidate_mask"].shape == (2, 2, 3, 4)

    with pytest.raises(ValueError, match="exact candidate width"):
        collate_blcs_tracking_batch(
            [_sample(views=1, frames=2, queries=3), _sample(views=1, frames=2, queries=4)]
        )
