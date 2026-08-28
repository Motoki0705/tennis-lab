"""Typed and Hydra composition contracts for Court detection configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

from src.tasks.court_detection.configuration import (
    CourtLossConfig,
    CourtModelConfig,
    CourtPoseLossConfig,
    CourtTrainingConfig,
    CourtTransformerEncoderConfig,
    SyntheticCourtSourceConfig,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    SemanticConfigurationError,
)

_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/court_detection/configs"


def _compose(source: str, *overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[f"data/source={source}", "data/processing=kp", *overrides],
        )


def _pose_overrides() -> tuple[str, ...]:
    return (
        "data.source.keypoint_court_scope=target_court",
        "data/augmentation=pose_safe",
        "model/encoder=dinov3",
        "model/transformer_encoder=default",
        "model/decoder=dpt",
        "loss.pose.enabled=true",
        "loss.pose.translation_weight=1.0",
        "loss.pose.rotation_weight=1.0",
        "loss.pose.focal_weight=1.0",
    )


def _pose_only_overrides() -> tuple[str, ...]:
    return (
        "loss=default",
        "loss.kp.weight=0.0",
        "loss.seg.weight=0.0",
        "loss.line.weight=0.0",
        *_pose_overrides(),
        "loss.consistency.enabled=false",
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
        _compose(source, "data.source.keypoint_court_scope=target_court")
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
    assert runtime.data.source.excluded_sample_ids == ("QszoUKyCOHo_600",)


def test_tennis_rejects_validation_as_test_mapping() -> None:
    config = _compose("tennis_court_detector")
    config.data.source.split_mapping.test = "val"

    with pytest.raises(SemanticConfigurationError, match="cannot be reused as test"):
        CourtTrainingConfig.from_config(config)


def test_default_model_is_hierarchical_with_dinov3_transformer_and_dpt() -> None:
    runtime = CourtTrainingConfig.from_config(_compose("synthetic_court"))

    assert isinstance(runtime.model, CourtModelConfig)
    assert runtime.model.name == "court_hierarchical"
    assert runtime.model.encoder.name == "dinov3"
    assert runtime.model.decoder.name == "dpt"
    assert runtime.model.decoder.size == "large"
    assert runtime.model.decoder.channels == 512
    assert runtime.model.transformer_encoder.name == "transformer"
    assert runtime.model.transformer_encoder.enabled
    assert runtime.model.transformer_encoder.depth == 8


def test_transformer_config_group_has_only_none_and_enabled_default_presets() -> None:
    transformer_configs = _CONFIG_DIR / "model" / "transformer_encoder"

    assert {path.name for path in transformer_configs.glob("*.yaml")} == {
        "default.yaml",
        "none.yaml",
    }


def test_dinov3_dpt_can_enable_transformer_refinement() -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose(
            "synthetic_court",
            "data/processing=all",
            "model/encoder=dinov3",
            "model/transformer_encoder=default",
            "model/decoder=dpt",
        )
    )

    assert isinstance(runtime.model, CourtModelConfig)
    assert runtime.model.encoder.name == "dinov3"
    assert runtime.model.decoder.name == "dpt"
    assert isinstance(
        runtime.model.transformer_encoder,
        CourtTransformerEncoderConfig,
    )
    assert runtime.model.transformer_encoder.enabled
    assert runtime.model.transformer_encoder.dim == 768
    assert runtime.model.transformer_encoder.depth == 8


@pytest.mark.parametrize(
    ("preset", "size", "channels"),
    [
        ("dpt", "large", 512),
    ],
)
def test_dpt_size_presets_are_strict_and_regular(
    preset: str,
    size: str,
    channels: int,
) -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose(
            "synthetic_court",
            "model/encoder=dinov3",
            f"model/decoder={preset}",
        )
    )

    assert runtime.model.decoder.name == "dpt"
    assert runtime.model.decoder.size == size
    assert runtime.model.decoder.channels == channels


@pytest.mark.parametrize("depth", [1, 8])
def test_transformer_depth_is_selected_from_config(depth: int) -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose(
            "synthetic_court",
            "model/encoder=dinov3",
            "model/transformer_encoder=default",
            "model/decoder=dpt",
            f"model.transformer_encoder.depth={depth}",
        )
    )

    assert runtime.model.transformer_encoder.depth == depth


def test_dpt_size_rejects_arbitrary_or_mismatched_channels() -> None:
    missing = _compose(
        "synthetic_court",
        "model/encoder=dinov3",
        "model/decoder=dpt",
    )
    with open_dict(missing.model.decoder):
        del missing.model.decoder.size
    with pytest.raises(MissingConfigurationKeyError, match="model.decoder.size"):
        CourtTrainingConfig.from_config(missing)

    mismatched = _compose(
        "synthetic_court",
        "model/encoder=dinov3",
        "model/decoder=dpt",
    )
    mismatched.model.decoder.channels = 65
    with pytest.raises(SemanticConfigurationError, match="strict size preset"):
        CourtTrainingConfig.from_config(mismatched)

    unknown = _compose(
        "synthetic_court",
        "model/encoder=dinov3",
        "model/decoder=dpt",
    )
    unknown.model.decoder.size = "micro"
    with pytest.raises(SemanticConfigurationError, match="tiny, small, base, or large"):
        CourtTrainingConfig.from_config(unknown)


def test_transformer_refinement_requires_dinov3_encoder() -> None:
    config = _compose(
        "synthetic_court",
        "model/encoder=default",
        "model/transformer_encoder=default",
        "model/decoder=fpn",
    )

    with pytest.raises(SemanticConfigurationError, match="DINOv3"):
        CourtTrainingConfig.from_config(config)


def test_transformer_encoder_rejects_inconsistent_head_dimension() -> None:
    config = _compose(
        "synthetic_court",
        "model/encoder=dinov3",
        "model/transformer_encoder=default",
        "model/decoder=dpt",
    )
    config.model.transformer_encoder.head_dim = 32

    with pytest.raises(SemanticConfigurationError, match="head_dim"):
        CourtTrainingConfig.from_config(config)


def test_default_loss_is_dense_only_with_disabled_pose_and_consistency() -> None:
    runtime = CourtTrainingConfig.from_config(_compose("synthetic_court"))

    assert isinstance(runtime.loss, CourtLossConfig)
    assert runtime.loss.dense_weights == {"kp": 1.0, "seg": 1.0, "line": 1.0}
    assert not runtime.loss.pose.enabled
    assert not runtime.loss.consistency.enabled


def test_pose_supervision_requires_explicit_transformer_and_weights() -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose("synthetic_court", *_pose_overrides())
    )

    assert isinstance(runtime.loss, CourtLossConfig)
    assert isinstance(runtime.loss.pose, CourtPoseLossConfig)
    assert runtime.loss.pose.enabled
    assert runtime.model.transformer_encoder.enabled

    no_transformer = _compose(
        "synthetic_court",
        "data.source.keypoint_court_scope=target_court",
        "data/augmentation=pose_safe",
        "model/transformer_encoder=none",
        "loss.pose.enabled=true",
        "loss.pose.translation_weight=1.0",
        "loss.pose.rotation_weight=1.0",
        "loss.pose.focal_weight=1.0",
    )
    with pytest.raises(SemanticConfigurationError, match="transformer_encoder"):
        CourtTrainingConfig.from_config(no_transformer)


def test_default_is_the_only_court_loss_config() -> None:
    loss_configs = _CONFIG_DIR / "loss"

    assert {path.name for path in loss_configs.glob("*.yaml")} == {"default.yaml"}


def test_pose_only_overrides_keep_kp_contract_with_zero_dense_weights() -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose("synthetic_court", *_pose_only_overrides())
    )

    assert tuple(target.kind for target in runtime.data.processing.targets) == ("kp",)
    assert runtime.loss.dense_weights == {"kp": 0.0, "seg": 0.0, "line": 0.0}
    assert runtime.loss.pose.enabled
    assert (
        runtime.loss.pose.translation_weight,
        runtime.loss.pose.rotation_weight,
        runtime.loss.pose.focal_weight,
    ) == (1.0, 1.0, 1.0)
    assert not runtime.loss.consistency.enabled


def test_pose_only_objective_rejects_a_bundle_without_kp() -> None:
    config = _compose(
        "synthetic_court",
        *_pose_only_overrides(),
        "data/processing=seg",
    )

    with pytest.raises(
        SemanticConfigurationError,
        match="pose-only objective requires KP",
    ):
        CourtTrainingConfig.from_config(config)


@pytest.mark.parametrize("kind", ["kp", "seg", "line"])
def test_dense_only_loss_rejects_zero_head_weight(kind: str) -> None:
    config = _compose("synthetic_court")
    config.loss[kind].weight = 0.0

    with pytest.raises(
        SemanticConfigurationError,
        match="Zero dense loss weights require enabled pose supervision",
    ):
        CourtTrainingConfig.from_config(config)


def test_loss_requires_at_least_one_positive_objective_weight() -> None:
    config = _compose("synthetic_court")
    for kind in ("kp", "seg", "line"):
        config.loss[kind].weight = 0.0

    with pytest.raises(
        SemanticConfigurationError,
        match="at least one positive objective weight",
    ):
        CourtTrainingConfig.from_config(config)


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
    config = _compose("synthetic_court", *_pose_overrides(), override)

    with pytest.raises(SemanticConfigurationError, match="Pose|pose"):
        CourtTrainingConfig.from_config(config)


def test_consistency_requires_kp_and_enabled_pose_supervision() -> None:
    config = _compose(
        "synthetic_court",
        *_pose_overrides(),
        "loss.consistency.enabled=true",
        "loss.consistency.weight=1.0",
    )
    runtime = CourtTrainingConfig.from_config(config)
    assert runtime.loss.consistency.enabled

    no_kp = _compose(
        "synthetic_court",
        *_pose_overrides(),
        "data/processing=seg",
        "loss.consistency.enabled=true",
        "loss.consistency.weight=1.0",
    )
    with pytest.raises(SemanticConfigurationError, match="KP"):
        CourtTrainingConfig.from_config(no_kp)
