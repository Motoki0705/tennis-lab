"""Typed and Hydra composition contracts for Court detection configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

from src.tasks.court_detection.configuration import (
    CourtModelConfig,
    CourtQueryDPTDecoderConfig,
    CourtQueryLossConfig,
    CourtQueryModelConfig,
    CourtQueryProgressiveDecoderConfig,
    CourtTrainingConfig,
    SyntheticCourtSourceConfig,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    SemanticConfigurationError,
)

_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/court_detection/configs"


def _compose(source: str, *overrides: str) -> DictConfig:
    query_selected = any(
        override.startswith("model=query_encoder") for override in overrides
    )
    query_requirements = (
        [
            "data.source.keypoint_court_scope=target_court",
            "data/augmentation=pose_safe",
            "loss=query_pose",
        ]
        if query_selected
        else []
    )
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[
                f"data/source={source}",
                "data/processing=kp",
                *query_requirements,
                *overrides,
            ],
        )


@pytest.mark.parametrize(
    ("source", "schema"),
    [
        ("synthetic_court_v1", "v1"),
        ("synthetic_court_v2", "v2"),
        ("synthetic_court", "v3"),
    ],
)
def test_hydra_explicitly_composes_each_synthetic_schema(
    source: str,
    schema: str,
) -> None:
    runtime = CourtTrainingConfig.from_config(_compose(source))

    assert isinstance(runtime.data.source, SyntheticCourtSourceConfig)
    assert runtime.data.source.schema == schema
    assert runtime.data.source.kind == "synthetic_court"
    assert runtime.data.source.keypoint_court_scope == "all_courts"


@pytest.mark.parametrize(
    ("source", "schema"),
    [("synthetic_court_v2", "v2"), ("synthetic_court", "v3")],
)
def test_hydra_composes_target_court_scope_for_singleton_schemas(
    source: str,
    schema: str,
) -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose(
            source,
            "data.source.keypoint_court_scope=target_court",
        )
    )

    assert isinstance(runtime.data.source, SyntheticCourtSourceConfig)
    assert runtime.data.source.schema == schema
    assert runtime.data.source.keypoint_court_scope == "target_court"


def test_synthetic_schema_cannot_be_omitted_or_guessed() -> None:
    missing = deepcopy(_compose("synthetic_court"))
    with open_dict(missing.data.source):
        del missing.data.source.schema
    with pytest.raises(MissingConfigurationKeyError, match="data.source.schema"):
        CourtTrainingConfig.from_config(missing)

    unknown = deepcopy(_compose("synthetic_court"))
    unknown.data.source.schema = "auto"
    with pytest.raises(
        SemanticConfigurationError,
        match="explicitly 'v1', 'v2', or 'v3'",
    ):
        CourtTrainingConfig.from_config(unknown)


def test_synthetic_keypoint_court_scope_is_required_and_strict() -> None:
    missing = deepcopy(_compose("synthetic_court"))
    with open_dict(missing.data.source):
        del missing.data.source.keypoint_court_scope
    with pytest.raises(
        MissingConfigurationKeyError,
        match="data.source.keypoint_court_scope",
    ):
        CourtTrainingConfig.from_config(missing)

    unknown = deepcopy(_compose("synthetic_court"))
    unknown.data.source.keypoint_court_scope = "primary_court"
    with pytest.raises(
        SemanticConfigurationError,
        match="keypoint_court_scope must be 'all_courts' or 'target_court'",
    ):
        CourtTrainingConfig.from_config(unknown)

    wrong_type = deepcopy(_compose("synthetic_court"))
    wrong_type.data.source.keypoint_court_scope = 1
    with pytest.raises(ConfigurationTypeError, match="keypoint_court_scope"):
        CourtTrainingConfig.from_config(wrong_type)


def test_v1_rejects_target_court_scope_at_typed_configuration_boundary() -> None:
    config = _compose(
        "synthetic_court_v1",
        "data.source.keypoint_court_scope=target_court",
    )

    with pytest.raises(
        SemanticConfigurationError,
        match="target_court.*requires.*schema='v2'.*'v3'",
    ):
        CourtTrainingConfig.from_config(config)


@pytest.mark.parametrize("scene_id", [".", ".."])
def test_synthetic_scene_ids_reject_dot_segments(scene_id: str) -> None:
    config = _compose("synthetic_court")
    config.data.source.scene_ids = [scene_id]

    with pytest.raises(ConfigurationTypeError, match="safe non-empty scene IDs"):
        CourtTrainingConfig.from_config(config)


def test_tennis_default_has_no_validation_as_test_mapping() -> None:
    runtime = CourtTrainingConfig.from_config(_compose("tennis_court_detector"))

    assert runtime.data.source.kind == "tennis_court_detector"
    assert runtime.data.source.split_mapping["test"] is None


def test_tennis_rejects_validation_as_test_mapping() -> None:
    config = _compose("tennis_court_detector")
    config.data.source.split_mapping.test = "val"

    with pytest.raises(SemanticConfigurationError, match="cannot be reused as test"):
        CourtTrainingConfig.from_config(config)


@pytest.mark.parametrize(
    ("model_name", "preset", "hidden_dim", "depth", "decoder_width"),
    [
        ("query_encoder", "tiny", 64, 2, 32),
        ("query_encoder_small", "small", 128, 4, 64),
        ("query_encoder_base", "base", 256, 8, 128),
    ],
)
def test_query_presets_compose_with_monotonic_capacity(
    model_name: str,
    preset: str,
    hidden_dim: int,
    depth: int,
    decoder_width: int,
) -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose("synthetic_court", f"model={model_name}")
    )

    assert isinstance(runtime.model, CourtQueryModelConfig)
    assert runtime.model.preset == preset
    assert runtime.model.task_encoder.hidden_dim == hidden_dim
    assert runtime.model.task_encoder.depth == depth
    assert runtime.model.decoder.width == decoder_width


def test_legacy_default_remains_exactly_hierarchical() -> None:
    runtime = CourtTrainingConfig.from_config(_compose("synthetic_court"))

    assert isinstance(runtime.model, CourtModelConfig)
    assert runtime.model.name == "court_hierarchical"
    assert runtime.model.encoder.name == "default"
    assert runtime.model.decoder.name == "fpn"


def test_query_hydra_axes_can_be_overridden_independently_in_raw_mode() -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose(
            "synthetic_court",
            "model=query_encoder_base",
            "model.preset=raw",
            "model/task_encoder=query_small",
            "model/decoder=query_progressive_tiny",
            "model/heads=query_small",
        )
    )

    assert isinstance(runtime.model, CourtQueryModelConfig)
    assert runtime.model.preset == "raw"
    assert runtime.model.backbone.name == "dinov3"
    assert runtime.model.task_encoder.hidden_dim == 128
    assert isinstance(runtime.model.decoder, CourtQueryProgressiveDecoderConfig)
    assert runtime.model.decoder.width == 32
    assert runtime.model.heads.pose_hidden_dim == 128


@pytest.mark.parametrize("preset", ["tiny", "small", "base"])
@pytest.mark.parametrize("family", ["linear", "progressive", "dpt"])
def test_every_query_preset_decoder_family_composes(
    preset: str,
    family: str,
) -> None:
    model_name = "query_encoder" if preset == "tiny" else f"query_encoder_{preset}"
    runtime = CourtTrainingConfig.from_config(
        _compose(
            "synthetic_court",
            f"model={model_name}",
            f"model/decoder=query_{family}_{preset}",
        )
    )

    assert isinstance(runtime.model, CourtQueryModelConfig)
    assert runtime.model.preset == preset
    assert runtime.model.decoder.family == family


def test_query_preset_rejects_disagreeing_raw_override() -> None:
    config = _compose(
        "synthetic_court",
        "model=query_encoder",
        "model.task_encoder.hidden_dim=128",
    )

    with pytest.raises(SemanticConfigurationError, match="preset.*disagrees"):
        CourtTrainingConfig.from_config(config)


def test_query_config_rejects_invalid_rope_taps_and_decoder_family() -> None:
    bad_rope = _compose(
        "synthetic_court",
        "model=query_encoder",
        "model.preset=raw",
        "model.task_encoder.rope_dim=6",
    )
    with pytest.raises(SemanticConfigurationError, match="rope_dim"):
        CourtTrainingConfig.from_config(bad_rope)

    duplicate_tap = _compose(
        "synthetic_court",
        "model=query_encoder",
        "model.preset=raw",
        "model.task_encoder.tap_indices=[0,0]",
    )
    with pytest.raises(SemanticConfigurationError, match="non-empty.*unique"):
        CourtTrainingConfig.from_config(duplicate_tap)

    unknown_family = _compose(
        "synthetic_court",
        "model=query_encoder",
        "model.preset=raw",
        "model.decoder.family=magic",
    )
    with pytest.raises(SemanticConfigurationError, match="linear, progressive, or dpt"):
        CourtTrainingConfig.from_config(unknown_family)


def test_query_dpt_override_has_strict_multi_tap_contract() -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose(
            "synthetic_court",
            "model=query_encoder_small",
            "model/decoder=query_dpt_small",
        )
    )
    assert isinstance(runtime.model, CourtQueryModelConfig)
    assert isinstance(runtime.model.decoder, CourtQueryDPTDecoderConfig)
    assert runtime.model.decoder.tap_indices == (0, 1, 2, 3)

    config = _compose(
        "synthetic_court",
        "model=query_encoder_small",
        "model.preset=raw",
        "model/decoder=query_dpt_small",
        "model.decoder.fusion_levels=3",
    )
    with pytest.raises(SemanticConfigurationError, match="fusion_levels"):
        CourtTrainingConfig.from_config(config)


def test_query_dense_head_subset_must_match_processing_targets() -> None:
    config = _compose(
        "synthetic_court",
        "model=query_encoder",
        "data/processing=kp_seg",
    )

    with pytest.raises(SemanticConfigurationError, match="dense_targets.*exactly"):
        CourtTrainingConfig.from_config(config)

    matching = _compose(
        "synthetic_court",
        "model=query_encoder",
        "data/processing=kp_seg",
        "model.heads.dense_targets=[kp,seg]",
    )
    runtime = CourtTrainingConfig.from_config(matching)
    assert isinstance(runtime.model, CourtQueryModelConfig)
    assert runtime.model.heads.dense_targets == ("kp", "seg")


def test_query_requires_v3_target_court_singleton_authority() -> None:
    with pytest.raises(SemanticConfigurationError, match="Synthetic Court V3"):
        CourtTrainingConfig.from_config(
            _compose("synthetic_court_v2", "model=query_encoder")
        )

    with pytest.raises(SemanticConfigurationError, match="target_court"):
        CourtTrainingConfig.from_config(
            _compose(
                "synthetic_court",
                "model=query_encoder",
                "data.source.keypoint_court_scope=all_courts",
            )
        )


@pytest.mark.parametrize(
    "override",
    [
        "data.augmentation.preserve_fx_fy=false",
        "data.augmentation.hflip_prob=0.1",
        "data.augmentation.crop_ratio=[0.75,1.333]",
        "data.augmentation.affine_shear=1.0",
        "data.augmentation.perspective_prob=0.1",
    ],
)
def test_pose_unsafe_augmentation_is_rejected_at_typed_boundary(
    override: str,
) -> None:
    config = _compose("synthetic_court", "model=query_encoder", override)

    with pytest.raises(SemanticConfigurationError, match="Pose|pose"):
        CourtTrainingConfig.from_config(config)


def test_query_pose_supervision_and_weights_are_explicit() -> None:
    pose_runtime = CourtTrainingConfig.from_config(
        _compose("synthetic_court", "model=query_encoder")
    )
    dense_runtime = CourtTrainingConfig.from_config(
        _compose(
            "synthetic_court",
            "model=query_encoder",
            "loss=query_dense",
        )
    )

    assert isinstance(pose_runtime.loss, CourtQueryLossConfig)
    assert isinstance(dense_runtime.loss, CourtQueryLossConfig)
    assert pose_runtime.loss.name == "query_pose_v1"
    assert pose_runtime.loss.pose.enabled
    assert dense_runtime.loss.name == "query_dense_v1"
    assert not dense_runtime.loss.pose.enabled


def test_query_rejects_legacy_loss_schema_without_silent_defaults() -> None:
    config = _compose(
        "synthetic_court",
        "model=query_encoder",
        "loss=default",
    )

    with pytest.raises(MissingConfigurationKeyError, match="loss.name|loss.pose"):
        CourtTrainingConfig.from_config(config)
