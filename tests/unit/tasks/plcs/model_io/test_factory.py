"""Composition tests for every active PLCS model/profile pair."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.model_io import PLCSAdapter, PLCSInputProfile, build_plcs_model_io
from src.utils.paths import PROJECT_ROOT

_SMALL_MODEL = (
    "model.hidden_dim=16",
    "model.num_heads=4",
    "model.ffn_dim=32",
    "model.rope_dim=4",
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
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(config_name=config_name, overrides=list(overrides))
    bound = build_plcs_model_io(PLCSTrainingConfig.from_config(config))
    adapter = cast("PLCSAdapter", bound.adapter)
    assert type(bound.model).__name__ == model_name
    assert adapter.profile is profile
    assert type(bound.model) is adapter.model_type
