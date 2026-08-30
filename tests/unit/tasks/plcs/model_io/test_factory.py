"""Composition tests for every active PLCS model/profile pair."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest
from hydra import compose, initialize_config_dir
from torch import nn

from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.model_io import (
    PLCSAdapter,
    PLCSInputProfile,
    PLCSTrackQueryIOAdapter,
    PLCSTrackQueryReferenceIOAdapter,
    build_plcs_model_io,
)
from src.tasks.plcs.models.plcs_track_query_ablation_model import (
    PLCSTrackQueryAblationModel,
)
from src.tasks.plcs.models.plcs_track_query_reference_ablation_model import (
    PLCSTrackQueryReferenceAblationModel,
)
from src.tasks.plcs.models.plcs_track_query_reference_model import (
    PLCSTrackQueryReferenceModel,
)
from src.tasks.plcs.training.composition import (
    build_plcs_datamodule,
    build_plcs_lightning_module,
)
from src.utils.paths import PROJECT_ROOT

_SMALL_MODEL = (
    "model.hidden_dim=16",
    "model.num_heads=4",
    "model.ffn_dim=32",
    "model.rope_dim=4",
)
_CONFIG_DIR = PROJECT_ROOT / "src/tasks/plcs/configs"
_TRACKING_SMALL = (
    "model.hidden_dim=16",
    "model.num_heads=4",
    "model.ffn_dim=32",
    "model.rope_dim=4",
    "model.num_stages=4",
    "model.mhc.coefficient_dim=8",
    "model.mhc.sinkhorn_iters=5",
    "model.cswa.compression_ratio=2",
    "model.cswa.window_radius=1",
)


@pytest.mark.parametrize(
    ("config_name", "overrides", "model_name", "profile"),
    [
        (
            "train",
            ("model=frame", "data=singleview_frame", "loss=no_canonical", *_SMALL_MODEL),
            "PLCSModel",
            PLCSInputProfile.FRAME,
        ),
        (
            "train",
            (
                "model=frame",
                "data=singleview_sequence",
                "data.seq_len_range=[1,3]",
                "loss=no_canonical",
                *_SMALL_MODEL,
            ),
            "PLCSModel",
            PLCSInputProfile.SEQUENCE,
        ),
        (
            "train",
            (
                "model=multiview",
                "loss=no_canonical",
                "data.seq_len_range=[1,3]",
                "model.max_seq_len=3",
                *_SMALL_MODEL,
            ),
            "PLCSMultiViewModel",
            PLCSInputProfile.MULTIVIEW,
        ),
        (
            "train",
            (
                "data.seq_len_range=[1,3]",
                "model.max_seq_len=3",
                "model.num_layers=1",
                *_SMALL_MODEL,
            ),
            "PLCSMultiViewAxialModel",
            PLCSInputProfile.MULTIVIEW,
        ),
        (
            "train",
            (
                "model=multiview_axial_split",
                "data.seq_len_range=[1,3]",
                "model.max_seq_len=3",
                "model.num_task_layers=1",
                "model.rot_num_task_layers=1",
                "model.pose_num_task_layers=1",
                *_SMALL_MODEL,
            ),
            "PLCSMultiViewAxialSplitModel",
            PLCSInputProfile.MULTIVIEW,
        ),
        (
            "train",
            (
                "model=multiview_axial_camtoken",
                "data.seq_len_range=[1,3]",
                "model.max_seq_len=3",
                "model.num_layers=1",
                *_SMALL_MODEL,
            ),
            "PLCSMultiViewAxialCamTokenModel",
            PLCSInputProfile.MULTIVIEW,
        ),
        (
            "train_tracking",
            (
                "model.hidden_dim=16",
                "model.num_heads=4",
                "model.ffn_dim=32",
                "model.rope_dim=4",
                "model.num_stages=4",
                "model.mhc.coefficient_dim=8",
                "model.mhc.sinkhorn_iters=5",
                "model.cswa.compression_ratio=2",
                "model.cswa.window_radius=1",
            ),
            "PLCSTrackQueryModel",
            PLCSInputProfile.TRACK_QUERY,
        ),
    ],
)
def test_factory_binds_each_validated_model_profile_once(
    config_name: str,
    overrides: Sequence[str],
    model_name: str,
    profile: PLCSInputProfile,
) -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_DIR)):
        config = compose(config_name=config_name, overrides=list(overrides))
    bound = build_plcs_model_io(PLCSTrainingConfig.from_config(config))
    adapter = cast("PLCSAdapter", bound.adapter)
    assert type(bound.model).__name__ == model_name
    assert adapter.profile is profile
    assert type(bound.model) is adapter.model_type
    if profile is PLCSInputProfile.TRACK_QUERY:
        assert isinstance(adapter, PLCSTrackQueryIOAdapter)
        assert not adapter.predict_canonical_pose
        assert not adapter.reprojection_enabled
        assert bound.model.canonical_pose_head is None


def _tracking_config(condition: str) -> object:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_DIR)):
        return compose(
            config_name="train_tracking",
            overrides=[f"model=track_query_ablation_{condition}", *_TRACKING_SMALL],
        )


@pytest.mark.parametrize("condition", ["a", "b", "c", "d"])
def test_factory_binds_every_ablation_config_to_exact_model_and_adapter(
    condition: str,
) -> None:
    runtime = PLCSTrainingConfig.from_config(_tracking_config(condition))

    binding = build_plcs_model_io(runtime)

    assert type(binding.model) is PLCSTrackQueryAblationModel
    assert type(binding.adapter) is PLCSTrackQueryIOAdapter
    assert binding.adapter.model_type is PLCSTrackQueryAblationModel
    assert binding.adapter.profile is PLCSInputProfile.TRACK_QUERY
    assert not binding.adapter.predict_canonical_pose
    assert not binding.adapter.reprojection_enabled
    assert binding.model.canonical_pose_head is None


def test_ablation_uses_tracking_training_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tracking_config("d")

    class _DataModule:
        def __init__(self, received: object) -> None:
            self.received = received

    class _LightningModule:
        def __init__(self, received: object) -> None:
            self.received = received

    monkeypatch.setattr(
        "src.tasks.plcs.data.tracking_datamodule.PLCSTrackingDataModule",
        _DataModule,
    )
    monkeypatch.setattr(
        "src.tasks.plcs.training.tracking_lightning_module.PLCSTrackingLightningModule",
        _LightningModule,
    )

    datamodule = build_plcs_datamodule(config)
    lightning_module = build_plcs_lightning_module(config)

    assert type(datamodule) is _DataModule
    assert type(lightning_module) is _LightningModule
    assert datamodule.received is config
    assert lightning_module.received is config


@pytest.mark.parametrize(
    ("profile", "model_type", "selector_mode"),
    [
        (
            "track_query_reference",
            PLCSTrackQueryReferenceModel,
            "reference",
        ),
        (
            "track_query_ablation_d_v2_selector",
            PLCSTrackQueryReferenceAblationModel,
            "reference",
        ),
        (
            "track_query_ablation_d_v2_selector_zero",
            PLCSTrackQueryReferenceAblationModel,
            "selector_zero",
        ),
    ],
)
def test_factory_binds_reference_v2_model_and_exact_six_input_adapter(
    profile: str,
    model_type: type[nn.Module],
    selector_mode: str,
) -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_DIR)):
        config = compose(
            config_name="train_tracking",
            overrides=[
                f"model={profile}",
                "court_keypoints=camera_view_v2",
                "model.hidden_dim=24",
                "model.num_heads=4",
                "model.ffn_dim=48",
                "model.rope_dim=6",
                "model.num_stages=4",
                "model.mhc.coefficient_dim=8",
                "model.mhc.sinkhorn_iters=5",
                "model.cswa.compression_ratio=2",
                "model.cswa.window_radius=1",
            ],
        )
    binding = build_plcs_model_io(PLCSTrainingConfig.from_config(config))

    assert type(binding.model) is model_type
    assert type(binding.adapter) is PLCSTrackQueryReferenceIOAdapter
    assert binding.adapter.model_type is model_type
    assert binding.adapter.reference_selector_mode.value == selector_mode
    assert not binding.adapter.predict_canonical_pose
    assert not binding.adapter.reprojection_enabled
    assert binding.model.canonical_pose_head is None


@pytest.mark.parametrize(
    ("reprojection_weight", "expected_reprojection_enabled"),
    [(1.0, True), (0.0, False)],
)
def test_factory_derives_reprojection_contract_from_validated_loss(
    reprojection_weight: float,
    expected_reprojection_enabled: bool,
) -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_DIR)):
        config = compose(
            config_name="train_tracking_pose",
            overrides=[
                *_TRACKING_SMALL,
                f"loss.reprojection_weight={reprojection_weight}",
            ],
        )

    binding = build_plcs_model_io(PLCSTrainingConfig.from_config(config))

    assert isinstance(binding.adapter, PLCSTrackQueryIOAdapter)
    assert binding.adapter.predict_canonical_pose
    assert (
        binding.adapter.reprojection_enabled is expected_reprojection_enabled
    )
    assert binding.model.canonical_pose_head is not None
