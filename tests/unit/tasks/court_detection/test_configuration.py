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
from src.tasks.court_detection.target_schemas import (
    LINE_TARGET_SCHEMA,
)
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)

_CONFIG_DIR = Path(__file__).resolve().parents[4] / "src/tasks/court_detection/configs"


def _compose(source: str, *overrides: str) -> DictConfig:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=[
                "loss=default",
                "data/augmentation=default",
                f"data/source={source}",
                "data/processing=kp",
                *overrides,
            ],
        )


def _pose_overrides() -> tuple[str, ...]:
    return (
        "data.source.court_scope=target_court",
        "data/augmentation=pose_safe",
        "loss.pose.enabled=true",
        "loss.pose.translation_weight=1.0",
        "loss.pose.rotation_weight=1.0",
        "loss.pose.focal_weight=1.0",
    )


@pytest.mark.parametrize("removed_value", [None, "court_detection/derived_targets"])
def test_processing_rejects_removed_offline_target_root(
    removed_value: str | None,
) -> None:
    config = _compose("tennis_court_detector")
    with open_dict(config.data.processing):
        config.data.processing.derived_target_root = removed_value
    with pytest.raises(UnknownConfigurationKeyError, match="derived_target_root"):
        CourtTrainingConfig.from_config(config)


def _pose_only_overrides() -> tuple[str, ...]:
    return (
        "loss=default",
        "loss.kp.weight=0.0",
        "loss.seg.weight=0.0",
        "loss.line.weight=0.0",
        "loss.semantic_line.weight=0.0",
        *_pose_overrides(),
        "loss.consistency.enabled=false",
    )


@pytest.mark.parametrize(
    ("source", "schema"),
    [
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
    assert runtime.data.source.court_scope == "target_court"


@pytest.mark.parametrize(
    ("source", "schema"),
    [("synthetic_court", "v3")],
)
def test_hydra_composes_target_court_scope_for_singleton_schemas(
    source: str,
    schema: str,
) -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose(source, "data.source.court_scope=target_court")
    )

    assert isinstance(runtime.data.source, SyntheticCourtSourceConfig)
    assert runtime.data.source.schema == schema
    assert runtime.data.source.court_scope == "target_court"


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
        match="schema must be 'v3'",
    ):
        CourtTrainingConfig.from_config(unknown)


def test_synthetic_court_scope_is_required_and_strict() -> None:
    missing = deepcopy(_compose("synthetic_court"))
    with open_dict(missing.data.source):
        del missing.data.source.court_scope
    with pytest.raises(
        MissingConfigurationKeyError,
        match="data.source.court_scope",
    ):
        CourtTrainingConfig.from_config(missing)

    unknown = deepcopy(_compose("synthetic_court"))
    unknown.data.source.court_scope = "primary_court"
    with pytest.raises(
        SemanticConfigurationError,
        match="court_scope must be 'target_court'",
    ):
        CourtTrainingConfig.from_config(unknown)

    wrong_type = deepcopy(_compose("synthetic_court"))
    wrong_type.data.source.court_scope = 1
    with pytest.raises(ConfigurationTypeError, match="court_scope"):
        CourtTrainingConfig.from_config(wrong_type)


@pytest.mark.parametrize("source", ["synthetic_court"])
def test_current_dense_schemas_reject_all_court_source_scope(source: str) -> None:
    config = _compose(
        source,
        "data/processing=seg_line",
        "data.source.court_scope=all_courts",
    )

    with pytest.raises(
        SemanticConfigurationError,
        match="court_scope must be",
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
    assert runtime.shared.resolver.roots.checkpoint_root.name == "ckpt"
    assert runtime.model.encoder.checkpoint_path == (
        runtime.shared.resolver.roots.checkpoint_root / "dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    )
    assert runtime.model.encoder.repository_path == runtime.shared.resolver.roots.external_asset_root / "dinov3"
    assert runtime.model.decoder.name == "dpt"
    assert runtime.model.decoder.size == "large"
    assert runtime.model.decoder.channels == 512
    assert runtime.model.transformer_encoder.name == "transformer"
    assert runtime.model.transformer_encoder.enabled
    assert runtime.model.transformer_encoder.depth == 8
    assert runtime.model.dense_head.name == "residual"
    assert runtime.model.dense_head.normalization_groups == 32
    assert {
        kind: (branch.hidden_channels, branch.depth)
        for kind, branch in runtime.model.dense_head.branches.items()
    } == {
        "kp": (256, 2),
        "seg": (256, 2),
        "line": (256, 2),
        "semantic_line": (256, 2),
    }


@pytest.mark.parametrize('field', ['resume', 'init_weights'])
def test_existing_training_checkpoint_is_independent_of_pretrained_assets(
    tmp_path: Path, field: str,
) -> None:
    backbone = tmp_path / 'ckpt/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    previous = tmp_path / 'outputs/court_detection/previous/checkpoints/last.ckpt'
    for path, content in ((backbone, b'pretrained backbone'), (previous, b'previous training state')):
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
    config = _compose('synthetic_court', f'paths.project_root={tmp_path}',
        f'run.{field}={{role:artifact,path:court_detection/previous/checkpoints/last.ckpt}}')
    runtime = CourtTrainingConfig.from_config(config)
    assert runtime.model.encoder.checkpoint_path == backbone
    assert getattr(runtime.shared.run, field) == previous
    assert runtime.shared.resolver.roots.checkpoint_root == tmp_path / 'ckpt'
    assert runtime.shared.resolver.roots.artifact_root == tmp_path / 'outputs'
    assert backbone.read_bytes() == b'pretrained backbone'
    assert previous.read_bytes() == b'previous training state'


def test_current_line_schema_uses_wide_physical_target() -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose("synthetic_court", "data/processing=all")
    )

    line = next(
        target for target in runtime.data.processing.targets if target.kind == "line"
    )
    assert line.target_schema == LINE_TARGET_SCHEMA


def test_dinov3_dpt_can_enable_transformer_refinement() -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose(
            "synthetic_court",
            "data/processing=all",
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
            f"model.transformer_encoder.depth={depth}",
        )
    )

    assert runtime.model.transformer_encoder.depth == depth


def test_dpt_size_rejects_arbitrary_or_mismatched_channels() -> None:
    missing = _compose(
        "synthetic_court",
    )
    with open_dict(missing.model.decoder):
        del missing.model.decoder.size
    with pytest.raises(MissingConfigurationKeyError, match="model.decoder.size"):
        CourtTrainingConfig.from_config(missing)

    mismatched = _compose(
        "synthetic_court",
    )
    mismatched.model.decoder.channels = 65
    with pytest.raises(SemanticConfigurationError, match="strict size preset"):
        CourtTrainingConfig.from_config(mismatched)

    unknown = _compose(
        "synthetic_court",
    )
    unknown.model.decoder.size = "micro"
    with pytest.raises(SemanticConfigurationError, match="tiny, small, base, or large"):
        CourtTrainingConfig.from_config(unknown)


def test_transformer_refinement_requires_dinov3_encoder() -> None:
    config = _compose(
        "synthetic_court",
        "model.encoder.name=default",
    )

    with pytest.raises(SemanticConfigurationError, match="encoder.name"):
        CourtTrainingConfig.from_config(config)


def test_transformer_encoder_rejects_inconsistent_head_dimension() -> None:
    config = _compose(
        "synthetic_court",
    )
    config.model.transformer_encoder.head_dim = 32

    with pytest.raises(SemanticConfigurationError, match="head_dim"):
        CourtTrainingConfig.from_config(config)


def test_default_loss_is_dense_only_with_disabled_pose_and_consistency() -> None:
    runtime = CourtTrainingConfig.from_config(_compose("synthetic_court"))

    assert isinstance(runtime.loss, CourtLossConfig)
    assert runtime.loss.dense_weights == {
        "kp": 1.0,
        "seg": 1.0,
        "line": 1.0,
        "semantic_line": 1.0,
    }
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
        "data.source.court_scope=target_court",
        "data/augmentation=pose_safe",
        "model.transformer_encoder.enabled=false",
        "loss.pose.enabled=true",
        "loss.pose.translation_weight=1.0",
        "loss.pose.rotation_weight=1.0",
        "loss.pose.focal_weight=1.0",
    )
    with pytest.raises(SemanticConfigurationError, match="transformer_encoder"):
        CourtTrainingConfig.from_config(no_transformer)


def test_court_loss_presets_are_explicit() -> None:
    loss_configs = _CONFIG_DIR / "loss"

    assert {path.name for path in loss_configs.glob("*.yaml")} == {
        "default.yaml",
        "pose.yaml",
    }


def test_pose_loss_preset_keeps_dense_heads_and_enables_pose() -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose(
            "synthetic_court",
            "loss=pose",
            "data.source.court_scope=target_court",
            "data/augmentation=pose_safe",
        )
    )

    assert runtime.loss.dense_weights == {
        "kp": 1.0,
        "seg": 1.0,
        "line": 1.0,
        "semantic_line": 1.0,
    }
    assert runtime.loss.pose.enabled
    assert (
        runtime.loss.pose.translation_weight,
        runtime.loss.pose.rotation_weight,
        runtime.loss.pose.focal_weight,
    ) == (1.0, 1.0, 1.0)
    assert not runtime.loss.consistency.enabled


def test_pose_frozen_training_keeps_pr867_schedule_without_lora() -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose(
            "synthetic_court",
            "data/processing=all",
            "training=pose_frozen",
        )
    )

    assert runtime.model.encoder.train_mode == "frozen"
    assert runtime.model.encoder.lora is not None
    assert not runtime.model.encoder.lora.enabled
    assert runtime.shared.training.trainer.max_epochs == 20
    assert runtime.shared.training.checkpoint.monitor == "val/loss_direct_pose"


def test_pose_only_overrides_keep_kp_contract_with_zero_dense_weights() -> None:
    runtime = CourtTrainingConfig.from_config(
        _compose("synthetic_court", *_pose_only_overrides())
    )

    assert tuple(target.kind for target in runtime.data.processing.targets) == ("kp",)
    assert runtime.loss.dense_weights == {
        "kp": 0.0,
        "seg": 0.0,
        "line": 0.0,
        "semantic_line": 0.0,
    }
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


@pytest.mark.parametrize("kind", ["kp", "seg", "line", "semantic_line"])
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
    for kind in ("kp", "seg", "line", "semantic_line"):
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


@pytest.mark.parametrize("schema", ["v1", "v2"])
def test_legacy_synthetic_schema_is_rejected(schema: str) -> None:
    config = _compose("synthetic_court")
    config.data.source.schema = schema
    with pytest.raises(SemanticConfigurationError, match="schema must be 'v3'"):
        CourtTrainingConfig.from_config(config)


def test_only_one_model_configuration_is_published() -> None:
    assert [path.name for path in (_CONFIG_DIR / "model").rglob("*.yaml")] == [
        "dinov3_dpt.yaml"
    ]


def test_missing_dense_head_never_restores_legacy_linear_architecture() -> None:
    config = _compose("synthetic_court")
    with open_dict(config.model):
        del config.model.dense_head
    with pytest.raises(MissingConfigurationKeyError, match="dense_head"):
        CourtTrainingConfig.from_config(config)
