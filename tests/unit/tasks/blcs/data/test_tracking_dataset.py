"""Fixed-width BLCS observation packing and collation tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.tasks.base.data.lifecycle_slots import build_fixed_lifecycle_assignment
from src.tasks.base.generate_dataset import (
    CourtReferenceFrameError,
    CourtReferenceFrameProvenance,
    build_court_view_record,
    build_physical_court_provenance,
    build_reference_frame_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.data.observation_candidates import (
    pack_observation_candidates,
)
from src.tasks.blcs.data.tracking_dataset import (
    BLCSTrackingDataset,
    collate_blcs_tracking_batch,
)
from src.utils.configuration import MissingConfigurationKeyError
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


def test_dataset_rejects_missing_court_keypoint_selector(tmp_path: Path) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_tracking")
    with open_dict(config):
        del config["court_keypoints"]

    with pytest.raises(
        MissingConfigurationKeyError,
        match=r"configuration\.court_keypoints",
    ):
        BLCSTrackingDataset(
            scene_dir=tmp_path,
            split_file="train.txt",
            config=config,
            augment=False,
        )


def test_observations_pack_all_views_to_exact_width_with_lifecycle_reuse() -> None:
    presence = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [False, False, True],
            [False, False, True],
        ]
    )
    vis = presence.unsqueeze(0).expand(2, -1, -1).clone()
    vis[1, 0, 1] = False
    uv = torch.arange(2 * 4 * 3 * 2, dtype=torch.float32).reshape(2, 4, 3, 2)

    packed = pack_observation_candidates(
        ball_uv=uv,
        ball_vis=vis,
        physical_presence=presence,
        num_slots=2,
        min_reuse_gap_frames=0,
        randomize_slots=False,
    )

    assert packed.uv.shape == (2, 4, 2, 2)
    assert packed.vis.shape == (2, 4, 2)
    torch.testing.assert_close(packed.gt_index[0], packed.gt_index[1])
    assert packed.gt_index[0, 0, 0] == 0
    assert packed.gt_index[0, 2, 0] == 2
    assert not packed.vis[1, 0, 1]


def test_concurrent_candidate_overflow_is_rejected_without_truncation() -> None:
    presence = torch.ones(2, 3, dtype=torch.bool)

    with pytest.raises(ValueError, match="cannot be packed"):
        pack_observation_candidates(
            ball_uv=torch.rand(1, 2, 3, 2),
            ball_vis=presence.unsqueeze(0),
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
        generator=None,
    )
    observation = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
        randomize_slots=True,
        generator=None,
    )
    assert not torch.equal(target.track_to_slot, observation.track_to_slot)

    first_eval = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
        randomize_slots=False,
        generator=None,
    )
    second_eval = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
        randomize_slots=False,
        generator=None,
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
        json.dumps(
            {
                "num_frames": 4,
                "court_coordinate_normalization": (
                    court_coordinate_normalization_metadata()
                ),
            }
        ),
        encoding="utf-8",
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
    np.save(scene / "ball_vel_norm.npy", np.zeros((4, 3, 3), dtype=np.float32))
    for camera_index in range(2):
        uv = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2) / 100.0
        vis = presence.copy()
        if camera_index == 1:
            vis[0, 1] = False
        np.save(scene / f"cam_{camera_index}_ball_uv.npy", uv)
        np.save(scene / f"cam_{camera_index}_ball_vis.npy", vis)
        np.save(
            scene / f"cam_{camera_index}_court_kp_uv.npy",
            np.zeros((14, 2), dtype=np.float32),
        )
        np.save(
            scene / f"cam_{camera_index}_court_kp_vis.npy",
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
    assert sample["ball_vis"].shape == (2, 4, 2)
    assert set(sample["candidate_gt_index"].unique().tolist()) == {-1, 0, 1, 2}
    assert set(sample["target_instance_id"].unique().tolist()) == {-1, 0, 1, 2}
    assert not torch.equal(
        sample["candidate_gt_index"][0], sample["target_instance_id"]
    )
    assert not sample["ball_vis"][1, 0, 0]
    assert not sample["padding_mask"].any()


def _sample(*, views: int, frames: int, queries: int) -> dict[str, object]:
    candidate_shape = (views, frames, queries)
    return {
        "scene_format_version": torch.tensor(4),
        "ball_uv": torch.zeros(*candidate_shape, 2),
        "ball_vis": torch.zeros(candidate_shape, dtype=torch.bool),
        "court_kp": torch.zeros(views, frames, 14, 2),
        "court_vis": torch.zeros(views, frames, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(views, frames, dtype=torch.bool),
        "target_position": torch.zeros(frames, queries, 3),
        "target_velocity": torch.zeros(frames, queries, 3),
        "target_presence": torch.zeros(frames, queries, dtype=torch.bool),
        "target_instance_id": torch.full((frames, queries), -1),
        "target_slot_mask": torch.zeros(queries, dtype=torch.bool),
        "clean_ball_uv": torch.zeros(*candidate_shape, 2),
        "clean_ball_vis": torch.zeros(candidate_shape, dtype=torch.bool),
        "candidate_gt_index": torch.full(candidate_shape, -1),
        "court_reference_provenance": build_physical_court_provenance(),
    }


def test_collate_pads_view_and_time_but_never_candidate_axis() -> None:
    result = collate_blcs_tracking_batch(
        [_sample(views=1, frames=2, queries=4), _sample(views=2, frames=3, queries=4)]
    )
    assert result["ball_uv"].shape == (2, 2, 3, 4, 2)
    assert result["padding_mask"].shape == (2, 2, 3)
    assert result["padding_mask"][0, 0, 2]
    assert result["padding_mask"][0, 1].all()

    with pytest.raises(ValueError, match="exact candidate width"):
        collate_blcs_tracking_batch(
            [
                _sample(views=1, frames=2, queries=3),
                _sample(views=1, frames=2, queries=4),
            ]
        )


@pytest.mark.parametrize(
    "invalid_provenance",
    [None, {"contract_id": "unknown_courtkp20_contract"}, object()],
)
def test_collate_rejects_missing_or_non_record_provenance_before_padding(
    monkeypatch: pytest.MonkeyPatch,
    invalid_provenance: object,
) -> None:
    sample = _sample(views=1, frames=2, queries=4)
    if invalid_provenance is None:
        sample.pop("court_reference_provenance")
    else:
        sample["court_reference_provenance"] = invalid_provenance

    def _forbid_padding(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Tensor padding ran before provenance validation.")

    monkeypatch.setattr(
        "src.tasks.blcs.data.tracking_dataset.pad_and_stack_tracking_batch",
        _forbid_padding,
    )
    with pytest.raises(
        CourtReferenceFrameError,
        match="must provide a validated CourtReferenceFrameProvenance",
    ):
        collate_blcs_tracking_batch([sample])


def _camera_view_provenance(
    *,
    reference_camera_id: str,
) -> CourtReferenceFrameProvenance:
    contract = resolve_court_keypoint_contract("camera_view_v2")
    views = (
        build_court_view_record(
            camera_id="camera_0",
            camera_center_court_m=(2.0, -12.0, 4.0),
            contract=contract,
        ),
        build_court_view_record(
            camera_id="camera_1",
            camera_center_court_m=(-3.0, 12.0, 5.0),
            contract=contract,
        ),
    )
    return build_reference_frame_provenance(
        views,
        reference_camera_id=reference_camera_id,
    )


def test_collate_rejects_mixed_contracts_before_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physical = _sample(views=1, frames=2, queries=4)
    camera_view = _sample(views=1, frames=2, queries=4)
    camera_view["court_reference_provenance"] = _camera_view_provenance(
        reference_camera_id="camera_0"
    )

    def _forbid_padding(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Tensor padding ran before provenance validation.")

    monkeypatch.setattr(
        "src.tasks.blcs.data.tracking_dataset.pad_and_stack_tracking_batch",
        _forbid_padding,
    )
    with pytest.raises(
        CourtReferenceFrameError, match="cannot mix CourtKP20 contracts"
    ):
        collate_blcs_tracking_batch([physical, camera_view])


def test_collate_allows_same_v2_contract_with_distinct_reference_cameras() -> None:
    first = _sample(views=1, frames=2, queries=4)
    second = _sample(views=1, frames=2, queries=4)
    first_provenance = _camera_view_provenance(reference_camera_id="camera_0")
    second_provenance = _camera_view_provenance(reference_camera_id="camera_1")
    first["court_reference_provenance"] = first_provenance
    second["court_reference_provenance"] = second_provenance

    result = collate_blcs_tracking_batch([first, second])

    assert result["court_reference_provenance"] == (
        first_provenance,
        second_provenance,
    )
