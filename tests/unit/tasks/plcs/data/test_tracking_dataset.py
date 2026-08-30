"""PLCS post-corruption observation tracking dataset tests."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict
from torch import Tensor

import src.tasks.plcs.data.tracking_dataset as tracking_dataset_module
from src.tasks.base.data import ObservationTrackingConfig, TrackingCapacityError
from src.tasks.base.data.canonical_tracking import CanonicalTrackingDataset
from src.tasks.base.generate_dataset import (
    build_physical_court_provenance,
    resolve_court_keypoint_contract,
)
from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.data.tracking_augmentation import (
    PLCSTrackingDetectionAugmentation,
)
from src.tasks.plcs.data.tracking_dataset import (
    PLCSTrackingDataset,
    collate_plcs_tracking_batch,
)
from src.tasks.plcs.model_io import PLCSTrackQueryIOAdapter
from src.utils.configuration import (
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.paths import PROJECT_ROOT


def _association_config(**overrides: object) -> ObservationTrackingConfig:
    values: dict[str, object] = {
        "max_distance": 0.08,
        "max_missed_frames": 8,
        "min_reuse_gap_frames": 4,
        "use_velocity_prediction": True,
        "min_common_keypoints": 4,
        "cost_reduction": "median",
        "overflow_policy": "error",
    }
    values.update(overrides)
    return ObservationTrackingConfig.from_mapping(values)


def _dataset(
    *,
    num_queries: int,
    augment: bool = False,
    association: ObservationTrackingConfig | None = None,
    corruption: object | None = None,
) -> PLCSTrackingDataset:
    dataset = object.__new__(PLCSTrackingDataset)
    dataset.num_queries = num_queries
    dataset.augment = augment
    dataset.observation_tracking_config = association or _association_config()
    if corruption is not None:
        dataset.tracking_augmentation = cast(
            "PLCSTrackingDetectionAugmentation",
            corruption,
        )
    return dataset


def _physical_sample(
    values: Tensor,
    visibility: Tensor | None = None,
    provenance: Tensor | None = None,
    *,
    camera_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    if values.ndim != 3:
        raise AssertionError("values must have shape (V,T,D).")
    views, frames, detections = values.shape
    human_kp = values[..., None, None].expand(-1, -1, -1, 17, 2).clone()
    human_vis = (
        torch.ones((views, frames, detections, 17), dtype=torch.bool)
        if visibility is None
        else visibility.clone()
    )
    human_kp[~human_vis] = 0.0
    if provenance is None:
        physical_ids = torch.arange(detections, dtype=torch.long).view(1, 1, -1)
        provenance = physical_ids.expand(views, frames, -1).clone()
        provenance[~human_vis.any(-1)] = -1
    return {
        "scene_format_version": torch.tensor(4),
        "human_kp": human_kp.clone(),
        "human_vis": human_vis.clone(),
        "court_kp": torch.full((views, frames, 14, 2), 0.5),
        "court_vis": torch.ones((views, frames, 14), dtype=torch.bool),
        "padding_mask": torch.zeros((views, frames), dtype=torch.bool),
        "target_position": torch.rand(frames, 2, 3),
        "target_rotation": torch.tensor([1.0, 0.0]).expand(frames, 2, 2).clone(),
        "target_canonical_pose_3d": torch.rand(frames, 2, 17, 3),
        "target_human_kp_3d": torch.rand(frames, 2, 17, 3),
        "target_presence": torch.ones(frames, 2, dtype=torch.bool),
        "target_instance_id": torch.arange(2).expand(frames, 2).clone(),
        "target_slot_mask": torch.ones(2, dtype=torch.bool),
        "clean_human_kp": human_kp.clone(),
        "clean_human_vis": human_vis.clone(),
        "detection_gt_index": provenance.clone(),
        "camera_C": torch.zeros(views, 3),
        "camera_R": torch.eye(3).expand(views, 3, 3).clone(),
        "court_keypoint_metadata": {"contract": "test"},
        "court_reference_provenance": {"frame": "test"},
        "selected_camera_ids": tuple(f"selected_{i}" for i in range(views)),
        "track_query_reference": {"schema": "preserved"},
        "_observation_camera_indices": camera_indices or tuple(range(views)),
    }


class _FixedCorruption:
    def __init__(
        self,
        values: Tensor,
        visibility: Tensor,
        provenance: Tensor | None = None,
    ) -> None:
        self.values = values
        self.visibility = visibility
        self.provenance = provenance
        self.called = False

    def __call__(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        self.called = True
        output = {key: value.clone() for key, value in sample.items()}
        output["human_kp"] = self.values.clone()
        output["human_vis"] = self.visibility.clone()
        if self.provenance is not None:
            output["detection_gt_index"] = self.provenance.clone()
        return output


def test_augment_sample_tracks_post_noise_values_to_exact_query_width() -> None:
    sample = _physical_sample(torch.tensor([[[0.2], [0.2]]]))
    noisy = torch.full((1, 2, 1, 17, 2), 0.7)
    noisy_vis = torch.ones((1, 2, 1, 17), dtype=torch.bool)
    corruption = _FixedCorruption(noisy, noisy_vis)
    dataset = _dataset(num_queries=3, augment=True, corruption=corruption)
    targets = sample["target_position"].clone()

    result = dataset.augment_sample(sample)

    assert corruption.called
    assert result["human_kp"].shape == (1, 2, 3, 17, 2)
    assert result["human_vis"].shape == (1, 2, 3, 17)
    torch.testing.assert_close(result["human_kp"][:, :, 0], noisy[:, :, 0])
    assert result["human_vis"][:, :, 1:].logical_not().all()
    torch.testing.assert_close(result["target_position"], targets)
    assert result["track_query_reference"] == {"schema": "preserved"}


def test_public_sample_zeroes_only_invisible_court_uv_before_model_call() -> None:
    sample = _physical_sample(torch.tensor([[[0.2]]]))
    sample["court_kp"][0, 0, :4] = torch.tensor(
        [
            [0.25, 0.75],
            [0.10, 0.90],
            [-0.25, 1.25],
            [float("nan"), float("nan")],
        ]
    )
    sample["court_vis"][0, 0, 1:4] = False
    sample["court_reference_provenance"] = build_physical_court_provenance()

    result = _dataset(num_queries=3).augment_sample(sample)

    assert result["human_kp"].shape == (1, 1, 3, 17, 2)
    assert result["court_kp"].shape == (1, 1, 14, 2)
    torch.testing.assert_close(
        result["court_kp"][0, 0, 0],
        torch.tensor([0.25, 0.75]),
    )
    assert result["court_kp"][0, 0, 1:4].eq(0.0).all()
    assert torch.isfinite(result["court_kp"]).all()

    batch = collate_plcs_tracking_batch([result])
    call = PLCSTrackQueryIOAdapter(
        model_type=torch.nn.Module,
        num_queries=3,
        num_court_tokens=14,
        num_joints=17,
        court_keypoint_contract=resolve_court_keypoint_contract("physical_v1"),
    ).build_call(batch)
    assert call.kwargs["human_kp"].shape == (1, 1, 1, 3, 17, 2)
    assert call.kwargs["court_kp"].shape == (1, 1, 1, 14, 2)


def test_model_visible_tracking_ignores_debug_identity_and_carrier_order() -> None:
    values = torch.tensor([[[0.2, 0.8], [0.21, 0.79]]])
    first = _physical_sample(
        values,
        provenance=torch.tensor([[[4, 9], [4, 9]]]),
    )
    identity_mutated = _physical_sample(
        values,
        provenance=torch.tensor([[[90, 1], [90, 1]]]),
    )
    identity_mutated["clean_human_kp"].fill_(0.99)
    identity_mutated["target_position"].fill_(0.88)
    carrier_permuted = _physical_sample(
        values.flip(-1),
        provenance=torch.tensor([[[9, 4], [9, 4]]]),
    )
    dataset = _dataset(num_queries=2)

    result = dataset.augment_sample(first)
    mutated_result = dataset.augment_sample(identity_mutated)
    permuted_result = dataset.augment_sample(carrier_permuted)

    torch.testing.assert_close(result["human_kp"], mutated_result["human_kp"])
    torch.testing.assert_close(result["human_vis"], mutated_result["human_vis"])
    torch.testing.assert_close(result["human_kp"], permuted_result["human_kp"])
    torch.testing.assert_close(result["human_vis"], permuted_result["human_vis"])
    assert not torch.equal(
        result["detection_gt_index"], mutated_result["detection_gt_index"]
    )
    assert not torch.equal(result["clean_human_kp"], mutated_result["clean_human_kp"])


def test_partial_common_joints_match_and_disjoint_pose_starts_new_slot() -> None:
    human_kp = torch.zeros(1, 3, 2, 17, 2)
    human_vis = torch.zeros(1, 3, 2, 17, dtype=torch.bool)
    human_kp[0, 0, 0, :8] = 0.20
    human_vis[0, 0, 0, :8] = True
    human_kp[0, 1, 1, 4:12] = 0.21
    human_vis[0, 1, 1, 4:12] = True
    human_kp[0, 2, 0, 12:17] = 0.22
    human_vis[0, 2, 0, 12:17] = True
    sample = _physical_sample(torch.zeros(1, 3, 2), visibility=human_vis)
    sample["human_kp"] = human_kp
    sample["clean_human_kp"] = human_kp.clone()
    sample["detection_gt_index"] = torch.tensor(
        [[[7, -1], [-1, 9], [7, -1]]], dtype=torch.long
    )

    result = _dataset(num_queries=2).augment_sample(sample)

    assert result["human_vis"][0, 0, 0].any()
    assert result["human_vis"][0, 1, 0].any()
    assert result["human_vis"][0, 2, 1].any()
    assert result["detection_gt_index"][0, :, 0].tolist() == [7, 9, -1]
    assert result["detection_gt_index"][0, 2, 1].item() == 7


def test_camera_local_slots_are_independent() -> None:
    values = torch.tensor(
        [
            [[0.2, 0.8]],
            [[0.8, 0.2]],
        ]
    )
    sample = _physical_sample(values, camera_indices=(3, 11))

    result = _dataset(num_queries=2).augment_sample(sample)

    torch.testing.assert_close(result["human_kp"][:, 0, 0], torch.full((2, 17, 2), 0.2))
    assert result["detection_gt_index"][:, 0, 0].tolist() == [0, 1]


def test_false_positive_and_padding_have_no_clean_alignment() -> None:
    visibility = torch.zeros((1, 1, 1, 17), dtype=torch.bool)
    provenance = torch.full((1, 1, 1), -1, dtype=torch.long)
    sample = _physical_sample(
        torch.zeros(1, 1, 1),
        visibility=visibility,
        provenance=provenance,
    )
    false_positive = torch.full((1, 1, 1, 17, 2), 0.4)
    false_positive_vis = torch.ones((1, 1, 1, 17), dtype=torch.bool)
    corruption = _FixedCorruption(
        false_positive,
        false_positive_vis,
        provenance,
    )

    result = _dataset(
        num_queries=2,
        augment=True,
        corruption=corruption,
    ).augment_sample(sample)

    assert result["human_vis"][0, 0, 0].all()
    assert result["detection_gt_index"].eq(-1).all()
    assert result["clean_human_kp"].eq(0).all()
    assert result["clean_human_vis"].logical_not().all()


def test_all_padding_and_complete_dropout_keep_exact_q_and_targets() -> None:
    physical_visibility = torch.ones((1, 2, 1, 17), dtype=torch.bool)
    sample = _physical_sample(
        torch.full((1, 2, 1), 0.3), visibility=physical_visibility
    )
    dropped_values = torch.zeros((1, 2, 1, 17, 2))
    dropped_visibility = torch.zeros((1, 2, 1, 17), dtype=torch.bool)
    dropped_provenance = torch.full((1, 2, 1), -1, dtype=torch.long)
    corruption = _FixedCorruption(
        dropped_values,
        dropped_visibility,
        dropped_provenance,
    )
    targets = sample["target_presence"].clone()

    result = _dataset(
        num_queries=3,
        augment=True,
        corruption=corruption,
    ).augment_sample(sample)

    assert result["human_kp"].shape == (1, 2, 3, 17, 2)
    assert result["human_kp"].eq(0).all()
    assert result["human_vis"].logical_not().all()
    assert result["detection_gt_index"].eq(-1).all()
    torch.testing.assert_close(result["target_presence"], targets)


def test_collation_pads_only_view_and_time_axes_not_fixed_query_width() -> None:
    short = _dataset(num_queries=2).augment_sample(
        _physical_sample(torch.tensor([[[0.2]]]))
    )
    long = _dataset(num_queries=2).augment_sample(
        _physical_sample(
            torch.tensor(
                [
                    [[0.3], [0.31]],
                    [[0.7], [0.71]],
                ]
            )
        )
    )

    result = collate_plcs_tracking_batch([short, long])

    assert result["human_kp"].shape == (2, 2, 2, 2, 17, 2)
    assert result["human_vis"].shape == (2, 2, 2, 2, 17)
    assert result["detection_gt_index"].shape == (2, 2, 2, 2)
    assert result["detection_gt_index"][0, 1].eq(-1).all()
    assert result["padding_mask"][0, 1].all()


def test_inherited_genuine_overflow_reports_selected_source_camera() -> None:
    sample = _physical_sample(
        torch.tensor([[[0.2, 0.8]]]),
        camera_indices=(7,),
    )
    augmentation_config = deepcopy(_training_config("train_tracking").data.augmentation)
    augmentation_config.enabled = False
    augmentation = PLCSTrackingDetectionAugmentation(
        augmentation_config,
        num_slots=1,
    )

    with pytest.raises(TrackingCapacityError, match=r"camera=7, frame=0"):
        _dataset(
            num_queries=1,
            augment=True,
            corruption=augmentation,
        ).augment_sample(sample)


def _training_config(config_name: str) -> DictConfig:
    config_dir = Path(PROJECT_ROOT, "src/tasks/plcs/configs")
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name)


def test_dataset_passes_validated_query_width_to_tracking_augmentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _training_config("train_tracking")
    calls: list[tuple[object, int]] = []

    class _SpyTrackingAugmentation:
        def __init__(self, augmentation: object, *, num_slots: int) -> None:
            calls.append((augmentation, num_slots))

    def fake_base_init(
        dataset: CanonicalTrackingDataset,
        **kwargs: Any,
    ) -> None:
        dataset.hydra_cfg = kwargs["config"]
        dataset.num_queries = 3
        dataset.pack_to_query_slots = True

    monkeypatch.setattr(
        tracking_dataset_module,
        "validate_plcs_dataset_court_keypoints",
        lambda *_args: object(),
    )
    monkeypatch.setattr(CanonicalTrackingDataset, "__init__", fake_base_init)
    monkeypatch.setattr(
        tracking_dataset_module,
        "PLCSTrackingDetectionAugmentation",
        _SpyTrackingAugmentation,
    )

    dataset = PLCSTrackingDataset(
        scene_dir="unused",
        split_file="unused",
        config=config,
    )

    assert dataset.tracking_augmentation is not None
    assert calls == [(config.data.augmentation, 3)]


@pytest.mark.parametrize("num_queries", [None, 0])
def test_dataset_rejects_invalid_query_width_before_augmentation_construction(
    monkeypatch: pytest.MonkeyPatch,
    num_queries: int | None,
) -> None:
    config = _training_config("train_tracking")

    class _UnexpectedTrackingAugmentation:
        def __init__(self, augmentation: object, *, num_slots: int) -> None:
            pytest.fail(
                "Tracking augmentation was constructed before query validation: "
                f"augmentation={augmentation!r}, num_slots={num_slots!r}."
            )

    def fake_base_init(
        dataset: CanonicalTrackingDataset,
        **kwargs: Any,
    ) -> None:
        dataset.hydra_cfg = kwargs["config"]
        dataset.num_queries = num_queries
        dataset.pack_to_query_slots = True

    monkeypatch.setattr(
        tracking_dataset_module,
        "validate_plcs_dataset_court_keypoints",
        lambda *_args: object(),
    )
    monkeypatch.setattr(CanonicalTrackingDataset, "__init__", fake_base_init)
    monkeypatch.setattr(
        tracking_dataset_module,
        "PLCSTrackingDetectionAugmentation",
        _UnexpectedTrackingAugmentation,
    )

    with pytest.raises(ValueError, match="model.num_queries"):
        PLCSTrackingDataset(
            scene_dir="unused",
            split_file="unused",
            config=config,
        )


@pytest.mark.parametrize("config_name", ["train_tracking", "train_tracking_chunked"])
def test_tracking_config_composes_strict_pose_association(config_name: str) -> None:
    runtime = PLCSTrainingConfig.from_config(_training_config(config_name))

    association = cast(
        "Mapping[str, object]",
        runtime.data.values["association"],
    )
    assert dict(association) == {
        "max_distance": 0.08,
        "max_missed_frames": 8,
        "min_reuse_gap_frames": 4,
        "use_velocity_prediction": True,
        "min_common_keypoints": 4,
        "cost_reduction": "median",
        "overflow_policy": "error",
    }
    lifecycle = cast(
        "Mapping[str, object]",
        runtime.data.values["lifecycle"],
    )
    assert "randomize_slots_train" not in lifecycle


def test_tracking_config_rejects_removed_and_unknown_association_keys() -> None:
    legacy = deepcopy(_training_config("train_tracking"))
    with open_dict(legacy.data.lifecycle):
        legacy.data.lifecycle.randomize_slots_train = True
    with pytest.raises(UnknownConfigurationKeyError, match="randomize_slots_train"):
        PLCSTrainingConfig.from_config(legacy)

    unknown = deepcopy(_training_config("train_tracking"))
    with open_dict(unknown.data.association):
        unknown.data.association.legacy_fallback = True
    with pytest.raises(UnknownConfigurationKeyError, match="legacy_fallback"):
        PLCSTrainingConfig.from_config(unknown)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("cost_reduction", "mean", "cost_reduction.*median"),
        ("min_common_keypoints", 3, "min_common_keypoints.*4"),
        ("min_common_keypoints", 18, "min_common_keypoints.*17"),
    ],
)
def test_tracking_config_rejects_non_pose_association_contract(
    key: str,
    value: object,
    message: str,
) -> None:
    config = deepcopy(_training_config("train_tracking"))
    config.data.association[key] = value

    with pytest.raises(SemanticConfigurationError, match=message):
        PLCSTrainingConfig.from_config(config)
