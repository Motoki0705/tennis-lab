"""Fixed-width BLCS observation packing and collation tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

import src.tasks.blcs.data.tracking_dataset as tracking_dataset_module
from src.tasks.base.data.lifecycle_slots import build_fixed_lifecycle_assignment
from src.tasks.base.data.observation_tracking import (
    ObservationTrackingConfig,
    TrackedObservations,
    TrackingCapacityError,
)
from src.tasks.base.generate_dataset import (
    CourtReferenceFrameError,
    CourtReferenceFrameProvenance,
    build_court_view_record,
    build_physical_court_provenance,
    build_reference_frame_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.blcs.data.observation_candidates import (
    PhysicalObservationCandidates,
    build_physical_observation_candidates,
)
from src.tasks.blcs.data.tracking_augmentation import (
    BLCSTrackingDetectionAugmentation,
)
from src.tasks.blcs.data.tracking_dataset import (
    BLCSTrackingDataset,
    collate_blcs_tracking_batch,
)
from src.tasks.blcs.model_io import TrackQueryModelIOAdapter
from src.utils.configuration import MissingConfigurationKeyError
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
)

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()
_AUGMENTATION_CONFIG = _CONFIG_DIR / "data/_augmentation.yaml"


def _false_positive_only_augmentation_config() -> DictConfig:
    config = OmegaConf.load(_AUGMENTATION_CONFIG).augmentation
    if not isinstance(config, DictConfig):
        raise AssertionError("BLCS augmentation config must be a mapping.")
    config.enabled = True
    for block_name in (
        "uv_scale",
        "gaussian_noise",
        "visibility_dropout",
        "temporal_jitter",
        "burst_dropout",
        "false_positive",
        "edge_degradation",
        "speed_conditioned",
    ):
        config[block_name].enabled = False
    config.false_positive.enabled = True
    config.false_positive.prob = 1.0
    config.false_positive.prob_absent = 1.0
    config.false_positive.prob_after_dropout = 1.0
    config.false_positive.after_dropout_window = 1
    return config


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


def test_dataset_rejects_missing_num_queries_before_augmentation_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def initialize_without_queries(dataset: object, **kwargs: object) -> None:
        del kwargs
        dataset.hydra_cfg = {"data": {"augmentation": {}}}  # type: ignore[attr-defined]
        dataset.num_queries = None  # type: ignore[attr-defined]
        dataset.pack_to_query_slots = True  # type: ignore[attr-defined]
        dataset.observation_tracking_config = _association_config()  # type: ignore[attr-defined]

    def forbid_augmentation_construction(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("augmentation was constructed before Q validation")

    monkeypatch.setattr(
        tracking_dataset_module,
        "parse_court_keypoint_contract",
        lambda config: object(),
    )
    monkeypatch.setattr(
        tracking_dataset_module,
        "blcs_track_query_reference_contract_document",
        lambda config, contract: {},
    )
    monkeypatch.setattr(
        tracking_dataset_module,
        "validate_blcs_dataset_court_keypoints",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        tracking_dataset_module,
        "court_views_by_scene",
        lambda dataset: {},
    )
    monkeypatch.setattr(
        tracking_dataset_module.CanonicalTrackingDataset,
        "__init__",
        initialize_without_queries,
    )
    monkeypatch.setattr(
        tracking_dataset_module,
        "BLCSTrackingDetectionAugmentation",
        forbid_augmentation_construction,
    )

    with pytest.raises(ValueError, match=r"requires model\.num_queries"):
        BLCSTrackingDataset(
            scene_dir=tmp_path,
            split_file=tmp_path / "train.txt",
            config={},
            augment=False,
        )


def _association_config(
    *,
    max_distance: float = 0.04,
    max_missed_frames: int = 2,
    min_reuse_gap_frames: int = 4,
) -> ObservationTrackingConfig:
    return ObservationTrackingConfig(
        max_distance=max_distance,
        max_missed_frames=max_missed_frames,
        min_reuse_gap_frames=min_reuse_gap_frames,
        use_velocity_prediction=True,
        min_common_keypoints=1,
        cost_reduction="mean",
        overflow_policy="error",
    )


def test_physical_observations_remain_unpacked_before_tracking() -> None:
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

    candidates = build_physical_observation_candidates(
        ball_uv=uv,
        ball_vis=vis,
        physical_presence=presence,
    )

    assert candidates.uv.shape == (2, 4, 3, 2)
    assert candidates.vis.shape == (2, 4, 3)
    assert candidates.gt_index[0, 0].tolist() == [0, 1, -1]
    assert candidates.gt_index[1, 0].tolist() == [0, -1, -1]
    assert not candidates.vis[1, 0, 1]
    assert not candidates.uv[1, 0, 1].any()


def _internal_sample(
    *,
    uv: torch.Tensor,
    vis: torch.Tensor,
    gt_index: torch.Tensor | None = None,
    camera_indices: tuple[int, ...] | None = None,
    target_queries: int = 2,
) -> dict[str, object]:
    views, frames, carriers = vis.shape
    if gt_index is None:
        ids = torch.arange(carriers).view(1, 1, carriers)
        gt_index = torch.where(vis, ids.expand_as(vis), -1)
    return {
        "scene_format_version": torch.tensor(4),
        "_physical_ball_uv": uv,
        "_physical_ball_vis": vis,
        "_physical_gt_index": gt_index,
        "_selected_camera_indices": camera_indices or tuple(range(views)),
        "court_kp": torch.zeros(views, frames, 14, 2),
        "court_vis": torch.ones(views, frames, 14, dtype=torch.bool),
        "padding_mask": torch.zeros(views, frames, dtype=torch.bool),
        "target_position": torch.zeros(frames, target_queries, 3),
        "target_velocity": torch.zeros(frames, target_queries, 3),
        "target_presence": torch.zeros(frames, target_queries, dtype=torch.bool),
        "target_instance_id": torch.full((frames, target_queries), -1),
        "target_slot_mask": torch.zeros(target_queries, dtype=torch.bool),
        "court_reference_provenance": build_physical_court_provenance(),
        "selected_camera_ids": tuple(f"camera_{index}" for index in range(views)),
    }


def _bare_dataset(
    *,
    num_queries: int,
    augment: bool,
    association: ObservationTrackingConfig | None = None,
) -> BLCSTrackingDataset:
    dataset = object.__new__(BLCSTrackingDataset)
    dataset.num_queries = num_queries
    dataset.augment = augment
    dataset.observation_tracking_config = association or _association_config()
    return dataset


def test_augment_sample_corrupts_before_tracking_and_returns_exact_q(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uv = torch.tensor([[[[0.10, 0.20]], [[0.12, 0.22]]]])
    vis = torch.ones(1, 2, 1, dtype=torch.bool)
    sample = _internal_sample(uv=uv, vis=vis, camera_indices=(7,))
    dataset = _bare_dataset(num_queries=2, augment=True)
    events: list[str] = []

    class _Noise:
        def __call__(
            self,
            detections: PhysicalObservationCandidates,
            *,
            court_kp: torch.Tensor,
            court_vis: torch.Tensor,
        ) -> PhysicalObservationCandidates:
            del court_kp, court_vis
            events.append("noise")
            return PhysicalObservationCandidates(
                uv=detections.uv + 0.01,
                vis=detections.vis,
                gt_index=detections.gt_index,
            )

    dataset.tracking_augmentation = cast(
        BLCSTrackingDetectionAugmentation,
        _Noise(),
    )
    real_tracker = tracking_dataset_module.track_multiview_observations

    def _tracking_spy(
        values: torch.Tensor,
        visibility: torch.Tensor,
        *,
        num_slots: int,
        config: ObservationTrackingConfig,
        camera_indices: Sequence[int] | None = None,
        debug_provenance: torch.Tensor | None = None,
    ) -> TrackedObservations:
        events.append("track")
        torch.testing.assert_close(values[..., 0, :], uv + 0.01)
        assert camera_indices == (7,)
        return real_tracker(
            values,
            visibility,
            num_slots=num_slots,
            config=config,
            camera_indices=camera_indices,
            debug_provenance=debug_provenance,
        )

    monkeypatch.setattr(
        tracking_dataset_module, "track_multiview_observations", _tracking_spy
    )

    result = dataset.augment_sample(sample)

    assert events == ["noise", "track"]
    assert result["ball_uv"].shape == (1, 2, 2, 2)
    assert result["ball_vis"].shape == (1, 2, 2)
    torch.testing.assert_close(result["ball_uv"][0, :, 0], uv[0, :, 0] + 0.01)
    torch.testing.assert_close(result["clean_ball_uv"][0, :, 0], uv[0, :, 0])
    assert result["candidate_gt_index"][0, :, 0].tolist() == [0, 0]
    assert not any(key.startswith("_") for key in result)


def test_public_sample_zeroes_only_invisible_court_uv_before_model_call() -> None:
    sample = _internal_sample(
        uv=torch.tensor([[[[0.2, 0.3]]]]),
        vis=torch.ones(1, 1, 1, dtype=torch.bool),
    )
    court_kp = sample["court_kp"]
    court_vis = sample["court_vis"]
    assert isinstance(court_kp, torch.Tensor)
    assert isinstance(court_vis, torch.Tensor)
    court_kp[0, 0, :4] = torch.tensor(
        [
            [0.25, 0.75],
            [0.10, 0.90],
            [-0.25, 1.25],
            [float("nan"), float("nan")],
        ]
    )
    court_vis[0, 0, 1:4] = False

    result = _bare_dataset(num_queries=2, augment=False).augment_sample(sample)

    assert result["ball_uv"].shape == (1, 1, 2, 2)
    assert result["court_kp"].shape == (1, 1, 14, 2)
    torch.testing.assert_close(
        result["court_kp"][0, 0, 0],
        torch.tensor([0.25, 0.75]),
    )
    assert result["court_kp"][0, 0, 1:4].eq(0.0).all()
    assert torch.isfinite(result["court_kp"]).all()

    batch = collate_blcs_tracking_batch([result])
    call = TrackQueryModelIOAdapter(
        num_court_tokens=14,
        num_queries=2,
        presence_threshold=0.5,
    ).build_call(batch)
    ball_uv = call.kwargs["ball_uv"]
    court_kp = call.kwargs["court_kp"]
    assert isinstance(ball_uv, torch.Tensor)
    assert isinstance(court_kp, torch.Tensor)
    assert ball_uv.shape == (1, 1, 1, 2, 2)
    assert court_kp.shape == (1, 1, 1, 14, 2)


def test_model_visible_tracking_is_independent_of_carrier_and_gt_order() -> None:
    first_uv = torch.tensor([[[[0.20, 0.20], [0.80, 0.80]]]])
    second_uv = first_uv.flip(2)
    vis = torch.ones(1, 1, 2, dtype=torch.bool)
    first = _internal_sample(
        uv=first_uv,
        vis=vis,
        gt_index=torch.tensor([[[3, 7]]]),
    )
    second = _internal_sample(
        uv=second_uv,
        vis=vis,
        gt_index=torch.tensor([[[41, 29]]]),
    )
    first["target_position"] = torch.ones(1, 2, 3)
    second["target_position"] = torch.full((1, 2, 3), 9.0)

    dataset = _bare_dataset(num_queries=2, augment=False)
    first_result = dataset.augment_sample(first)
    second_result = dataset.augment_sample(second)

    torch.testing.assert_close(first_result["ball_uv"], second_result["ball_uv"])
    torch.testing.assert_close(first_result["ball_vis"], second_result["ball_vis"])
    assert not torch.equal(
        first_result["candidate_gt_index"], second_result["candidate_gt_index"]
    )
    assert not torch.equal(
        first_result["target_position"], second_result["target_position"]
    )


def test_tracking_slots_are_camera_local() -> None:
    uv = torch.tensor(
        [
            [[[0.10, 0.10], [0.80, 0.80]]],
            [[[0.80, 0.80], [0.10, 0.10]]],
        ]
    )
    vis = torch.ones(2, 1, 2, dtype=torch.bool)
    dataset = _bare_dataset(num_queries=2, augment=False)

    result = dataset.augment_sample(_internal_sample(uv=uv, vis=vis))

    torch.testing.assert_close(result["ball_uv"][0, 0], result["ball_uv"][1, 0])
    assert result["candidate_gt_index"][:, 0, 0].tolist() == [0, 1]


def test_all_detections_dropped_still_returns_zero_filled_exact_q() -> None:
    uv = torch.rand(1, 3, 1, 2)
    vis = torch.zeros(1, 3, 1, dtype=torch.bool)
    dataset = _bare_dataset(num_queries=3, augment=False)

    result = dataset.augment_sample(
        _internal_sample(uv=uv, vis=vis, target_queries=3)
    )

    assert result["ball_uv"].shape == (1, 3, 3, 2)
    assert result["ball_vis"].shape == (1, 3, 3)
    assert not result["ball_uv"].any()
    assert not result["ball_vis"].any()
    assert (result["candidate_gt_index"] == -1).all()
    assert not result["clean_ball_vis"].any()


def test_false_positive_debug_alignment_is_zero_and_invisible() -> None:
    uv = torch.zeros(1, 1, 1, 2)
    vis = torch.zeros(1, 1, 1, dtype=torch.bool)
    dataset = _bare_dataset(num_queries=2, augment=True)

    class _FalsePositive:
        def __call__(
            self,
            detections: PhysicalObservationCandidates,
            *,
            court_kp: torch.Tensor,
            court_vis: torch.Tensor,
        ) -> PhysicalObservationCandidates:
            del court_kp, court_vis
            return PhysicalObservationCandidates(
                uv=torch.full_like(detections.uv, 0.5),
                vis=torch.ones_like(detections.vis),
                gt_index=torch.full_like(detections.gt_index, -1),
            )

    dataset.tracking_augmentation = cast(
        BLCSTrackingDetectionAugmentation,
        _FalsePositive(),
    )
    result = dataset.augment_sample(_internal_sample(uv=uv, vis=vis))

    assert result["ball_vis"][0, 0, 0]
    assert result["candidate_gt_index"][0, 0, 0] == -1
    assert not result["clean_ball_vis"][0, 0, 0]
    assert not result["clean_ball_uv"][0, 0, 0].any()


def test_synthetic_false_positives_are_limited_before_dataset_tracking() -> None:
    torch.manual_seed(43)
    uv = torch.zeros(1, 1, 4, 2)
    vis = torch.zeros(1, 1, 4, dtype=torch.bool)
    dataset = _bare_dataset(num_queries=2, augment=True)
    dataset.tracking_augmentation = BLCSTrackingDetectionAugmentation(
        _false_positive_only_augmentation_config(),
        num_slots=2,
    )

    result = dataset.augment_sample(_internal_sample(uv=uv, vis=vis))

    assert result["ball_uv"].shape == (1, 1, 2, 2)
    assert result["ball_vis"].shape == (1, 1, 2)
    assert result["ball_vis"].sum().item() == 2
    assert result["candidate_gt_index"].eq(-1).all()
    assert not result["clean_ball_vis"].any()


def test_inherited_genuine_overflow_reaches_tracker_without_truncation() -> None:
    uv = torch.tensor([[[[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]]])
    vis = torch.ones(1, 1, 3, dtype=torch.bool)
    dataset = _bare_dataset(num_queries=2, augment=True)
    dataset.tracking_augmentation = BLCSTrackingDetectionAugmentation(
        _false_positive_only_augmentation_config(),
        num_slots=2,
    )

    with pytest.raises(TrackingCapacityError) as exc_info:
        dataset.augment_sample(
            _internal_sample(
                uv=uv,
                vis=vis,
                camera_indices=(11,),
            )
        )

    assert exc_info.value.camera_index == 11
    assert exc_info.value.frame_index == 0
    assert exc_info.value.num_slots == 2


def test_target_lifecycle_assignment_is_separate_and_deterministic() -> None:
    presence = torch.ones(2, 4, dtype=torch.bool)
    first = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
    )
    second = build_fixed_lifecycle_assignment(
        presence,
        num_slots=4,
        min_reuse_gap_frames=0,
    )
    torch.testing.assert_close(first.track_to_slot, second.track_to_slot)
    torch.testing.assert_close(first.target_instance_id, second.target_instance_id)


def test_dataset_tracks_more_physical_lifecycles_than_q_without_random_slots(
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
    config.data.association.max_missed_frames = 0
    config.data.association.min_reuse_gap_frames = 0
    config.data.augmentation.enabled = False
    constructed_num_slots: list[int] = []
    real_augmentation = tracking_dataset_module.BLCSTrackingDetectionAugmentation

    def build_augmentation(
        augmentation_config: DictConfig,
        *,
        num_slots: int,
    ) -> BLCSTrackingDetectionAugmentation:
        constructed_num_slots.append(num_slots)
        return real_augmentation(augmentation_config, num_slots=num_slots)

    monkeypatch.setattr(
        tracking_dataset_module,
        "BLCSTrackingDetectionAugmentation",
        build_augmentation,
    )
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
    assert not sample["ball_vis"][1, 0, 1]
    assert not sample["padding_mask"].any()
    assert constructed_num_slots == [2]


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
        "selected_camera_ids": tuple(f"camera_{index}" for index in range(views)),
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
