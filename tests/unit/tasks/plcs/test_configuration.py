"""PLCS strict configuration contract tests."""

from __future__ import annotations

from copy import deepcopy

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

from src.tasks.plcs.configuration import (
    PLCSAnalysisRuntimeConfig,
    PLCSTrainingConfig,
)
from src.tasks.plcs.generate_dataset.config import PLCSGenerationConfig
from src.utils.configuration import (
    ConfigurationTypeError,
    MissingConfigurationKeyError,
    PathContractError,
    PathRole,
    SemanticConfigurationError,
    UnknownConfigurationKeyError,
)
from src.utils.paths import PROJECT_ROOT


def _training_config() -> DictConfig:
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="train")


def _config(config_name: str) -> DictConfig:
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name=config_name)


def test_training_rejects_legacy_output_root_prefix() -> None:
    config = deepcopy(_training_config())
    with open_dict(config):
        config.run.output_dir = "outputs/legacy"

    with pytest.raises(PathContractError, match="root-prefixed or legacy fragment"):
        PLCSTrainingConfig.from_config(config)


def test_generation_output_is_explicitly_data_root_relative() -> None:
    runtime = PLCSGenerationConfig.from_config(_config("generate_dataset"))

    assert runtime.OUTPUT_ROLE is PathRole.DATA
    assert runtime.output_dir == PROJECT_ROOT / "data/plcs/single_object"


def test_analysis_output_is_explicitly_output_root_relative() -> None:
    runtime = PLCSAnalysisRuntimeConfig.angle_velocity(
        _config("analyze_angle_velocity")
    )

    assert runtime.OUTPUT_ROLE is PathRole.OUTPUT
    assert runtime.output_dir == PROJECT_ROOT / "outputs/plcs/analysis/angle_velocity"
    assert runtime.result_path is not None
    assert runtime.result_path.parent == runtime.output_dir


def test_frame_model_accepts_explicit_sequence_data_profile() -> None:
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name="train",
            overrides=[
                "model=frame",
                "data=singleview_sequence",
                "loss=no_canonical",
            ],
        )

    runtime = PLCSTrainingConfig.from_config(config)
    assert runtime.data.values["mode"] == "sequence"
    assert runtime.data.values["seq_stride"] == 128


def test_tracking_rejects_legacy_invisible_attention_config() -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.model):
        config.model.mask_invisible_observations = True

    with pytest.raises(
        UnknownConfigurationKeyError, match="mask_invisible_observations"
    ):
        PLCSTrainingConfig.from_config(config)


def test_tracking_requires_positive_four_stage_cycle() -> None:
    config = deepcopy(_config("train_tracking"))
    config.model.num_stages = 2

    with pytest.raises(SemanticConfigurationError, match="positive multiple of 4"):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize("nested_key", ["mhc", "cswa"])
def test_tracking_requires_nested_architecture_config(nested_key: str) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.model):
        del config.model[nested_key]

    with pytest.raises(MissingConfigurationKeyError, match=f"model.{nested_key}"):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize("nested_key", ["mhc", "cswa"])
def test_tracking_rejects_unknown_nested_architecture_config(
    nested_key: str,
) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.model[nested_key]):
        config.model[nested_key].legacy_fallback = True

    with pytest.raises(
        UnknownConfigurationKeyError,
        match=f"model.{nested_key}.legacy_fallback",
    ):
        PLCSTrainingConfig.from_config(config)


def test_tracking_rejects_unknown_cswa_backend() -> None:
    config = deepcopy(_config("train_tracking"))
    config.model.cswa.backend = "auto"

    with pytest.raises(SemanticConfigurationError, match="reference.*cuda"):
        PLCSTrainingConfig.from_config(config)


def test_legacy_training_without_fine_tune_mode_defaults_to_all() -> None:
    config = deepcopy(_config("train_tracking_pose"))
    with open_dict(config.training):
        del config.training["fine_tune_mode"]

    runtime = PLCSTrainingConfig.from_config(config)

    assert runtime.fine_tune_mode == "all"


def test_explicit_all_fine_tune_mode_preserves_default_training() -> None:
    config = _config("train_tracking_pose")

    runtime = PLCSTrainingConfig.from_config(config)

    assert config.training.fine_tune_mode == "all"
    assert runtime.fine_tune_mode == "all"


def test_presence_head_fine_tune_mode_requires_init_weights() -> None:
    config = deepcopy(_config("train_tracking_pose"))
    config.training.fine_tune_mode = "presence_head"

    with pytest.raises(
        SemanticConfigurationError,
        match=r"presence_head.*requires run\.init_weights",
    ):
        PLCSTrainingConfig.from_config(config)


def test_presence_head_fine_tune_mode_accepts_weight_initialization() -> None:
    config = deepcopy(_config("train_tracking_pose"))
    config.training.fine_tune_mode = "presence_head"
    config.run.init_weights = "source.ckpt"

    runtime = PLCSTrainingConfig.from_config(config)

    assert runtime.fine_tune_mode == "presence_head"
    assert runtime.shared.run.init_weights is not None


def test_presence_head_fine_tune_mode_rejects_resume() -> None:
    config = deepcopy(_config("train_tracking_pose"))
    config.training.fine_tune_mode = "presence_head"
    config.run.resume = "source.ckpt"

    with pytest.raises(
        SemanticConfigurationError,
        match=r"presence_head.*forbids run\.resume",
    ):
        PLCSTrainingConfig.from_config(config)


def test_presence_head_keeps_resume_and_init_mutually_exclusive() -> None:
    config = deepcopy(_config("train_tracking_pose"))
    config.training.fine_tune_mode = "presence_head"
    config.run.resume = "resume.ckpt"
    config.run.init_weights = "source.ckpt"

    with pytest.raises(
        SemanticConfigurationError,
        match=r"run\.resume and run\.init_weights are mutually exclusive",
    ):
        PLCSTrainingConfig.from_config(config)


def test_unknown_fine_tune_mode_is_rejected() -> None:
    config = deepcopy(_config("train_tracking_pose"))
    config.training.fine_tune_mode = "presence_only"

    with pytest.raises(
        SemanticConfigurationError,
        match="training.fine_tune_mode must be one of",
    ):
        PLCSTrainingConfig.from_config(config)


def test_wrong_type_fine_tune_mode_is_rejected() -> None:
    config = deepcopy(_config("train_tracking_pose"))
    config.training.fine_tune_mode = 1

    with pytest.raises(
        ConfigurationTypeError,
        match="training.fine_tune_mode",
    ):
        PLCSTrainingConfig.from_config(config)


def test_presence_head_fine_tune_rejects_model_without_presence_head() -> None:
    config = deepcopy(_training_config())
    config.training.fine_tune_mode = "presence_head"
    config.run.init_weights = "source.ckpt"

    with pytest.raises(
        SemanticConfigurationError,
        match=r"track-query model.*independent presence_head",
    ):
        PLCSTrainingConfig.from_config(config)


def test_tracking_legacy_loss_contract_remains_valid() -> None:
    config = _config("train_tracking")
    runtime = PLCSTrainingConfig.from_config(config)

    assert "predict_canonical_pose" not in runtime.model.values
    assert "cardinality_weight" not in config.loss
    assert "cardinality_nll_weight" not in config.loss
    assert "presence_hard_negative_weight" not in config.loss
    assert "presence_hard_negative_gamma" not in config.loss
    assert "presence_pairwise_weight" not in config.loss
    assert "presence_pairwise_margin" not in config.loss
    assert config.loss.presence_inactive_weight == 0.25
    assert config.loss.match_presence_inactive_weight == 0.25
    assert not runtime.tracking_reprojection_enabled


@pytest.mark.parametrize(
    ("reprojection_weight", "expected_enabled"),
    [(0.0, False), (1.0, True)],
)
def test_tracking_runtime_derives_reprojection_contract_from_validated_loss(
    reprojection_weight: float,
    expected_enabled: bool,
) -> None:
    config = deepcopy(_config("train_tracking_pose"))
    config.loss.reprojection_weight = reprojection_weight

    runtime = PLCSTrainingConfig.from_config(config)

    assert runtime.tracking_reprojection_enabled is expected_enabled


@pytest.mark.parametrize("config_name", ["train_tracking", "train_tracking_pose"])
def test_tracking_accepts_complete_legacy_loss_mapping_without_split_weight(
    config_name: str,
) -> None:
    config = deepcopy(_config(config_name))
    with open_dict(config.loss):
        del config.loss["match_presence_inactive_weight"]

    PLCSTrainingConfig.from_config(config)


def test_tracking_rejects_partial_legacy_loss_mapping() -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        del config.loss["match_presence_inactive_weight"]
        del config.loss["match_presence_weight"]

    with pytest.raises(
        MissingConfigurationKeyError,
        match="loss.match_presence_weight",
    ):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan")])
def test_tracking_rejects_invalid_matching_inactive_weight(value: float) -> None:
    config = deepcopy(_config("train_tracking"))
    config.loss.match_presence_inactive_weight = value

    with pytest.raises(
        SemanticConfigurationError,
        match="loss.match_presence_inactive_weight",
    ):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize("value", [0.0, 0.75])
def test_tracking_accepts_optional_nonnegative_cardinality_weight(value: float) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss.cardinality_weight = value

    PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_tracking_rejects_invalid_cardinality_weight(value: float) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss.cardinality_weight = value

    with pytest.raises(
        SemanticConfigurationError,
        match="loss.cardinality_weight",
    ):
        PLCSTrainingConfig.from_config(config)


def test_tracking_rejects_overflowing_integer_cardinality_weight() -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss.cardinality_weight = 10**400

    with pytest.raises(
        SemanticConfigurationError,
        match="loss.cardinality_weight must be finite",
    ):
        PLCSTrainingConfig.from_config(config)


def test_tracking_rejects_wrong_type_cardinality_weight() -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss.cardinality_weight = "invalid"

    with pytest.raises(
        ConfigurationTypeError,
        match=r"loss.cardinality_weight: expected float \| int, got str",
    ):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize("value", [0.0, 0.75])
def test_tracking_accepts_optional_nonnegative_cardinality_nll_weight(
    value: float,
) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss.cardinality_nll_weight = value

    PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_tracking_rejects_invalid_cardinality_nll_weight(value: float) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss.cardinality_nll_weight = value

    with pytest.raises(
        SemanticConfigurationError,
        match="loss.cardinality_nll_weight",
    ):
        PLCSTrainingConfig.from_config(config)


def test_tracking_rejects_overflowing_integer_cardinality_nll_weight() -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss.cardinality_nll_weight = 10**400

    with pytest.raises(
        SemanticConfigurationError,
        match="loss.cardinality_nll_weight must be finite",
    ):
        PLCSTrainingConfig.from_config(config)


def test_tracking_rejects_wrong_type_cardinality_nll_weight() -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss.cardinality_nll_weight = "invalid"

    with pytest.raises(
        ConfigurationTypeError,
        match=r"loss.cardinality_nll_weight: expected float \| int, got str",
    ):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("presence_hard_negative_weight", 0.0),
        ("presence_hard_negative_weight", 0.75),
        ("presence_hard_negative_gamma", 0.0),
        ("presence_hard_negative_gamma", 2.0),
    ],
)
def test_tracking_accepts_nonnegative_hard_negative_settings(
    field: str,
    value: float,
) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss[field] = value

    PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize(
    "field",
    ["presence_hard_negative_weight", "presence_hard_negative_gamma"],
)
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_tracking_rejects_invalid_hard_negative_settings(
    field: str,
    value: float,
) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss[field] = value

    with pytest.raises(
        SemanticConfigurationError,
        match=f"loss.{field}",
    ):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize(
    "field",
    ["presence_hard_negative_weight", "presence_hard_negative_gamma"],
)
def test_tracking_rejects_overflowing_integer_hard_negative_settings(
    field: str,
) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss[field] = 10**400

    with pytest.raises(
        SemanticConfigurationError,
        match=rf"loss.{field} must be finite",
    ):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize(
    "field",
    ["presence_hard_negative_weight", "presence_hard_negative_gamma"],
)
def test_tracking_rejects_wrong_type_hard_negative_settings(field: str) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss[field] = "invalid"

    with pytest.raises(
        ConfigurationTypeError,
        match=rf"loss.{field}: expected float \| int, got str",
    ):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("presence_pairwise_weight", 0.0),
        ("presence_pairwise_weight", 0.75),
        ("presence_pairwise_margin", 0.0),
        ("presence_pairwise_margin", 0.5),
    ],
)
def test_tracking_accepts_nonnegative_pairwise_settings(
    field: str,
    value: float,
) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss[field] = value

    PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize(
    "field",
    ["presence_pairwise_weight", "presence_pairwise_margin"],
)
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_tracking_rejects_invalid_pairwise_settings(
    field: str,
    value: float,
) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss[field] = value

    with pytest.raises(
        SemanticConfigurationError,
        match=f"loss.{field}",
    ):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize(
    "field",
    ["presence_pairwise_weight", "presence_pairwise_margin"],
)
def test_tracking_rejects_overflowing_integer_pairwise_settings(field: str) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss[field] = 10**400

    with pytest.raises(
        SemanticConfigurationError,
        match=rf"loss.{field} must be finite",
    ):
        PLCSTrainingConfig.from_config(config)


@pytest.mark.parametrize(
    "field",
    ["presence_pairwise_weight", "presence_pairwise_margin"],
)
def test_tracking_rejects_wrong_type_pairwise_settings(field: str) -> None:
    config = deepcopy(_config("train_tracking"))
    with open_dict(config.loss):
        config.loss[field] = "invalid"

    with pytest.raises(
        ConfigurationTypeError,
        match=rf"loss.{field}: expected float \| int, got str",
    ):
        PLCSTrainingConfig.from_config(config)


def test_tracking_pose_loss_requires_canonical_pose_model_output() -> None:
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name="train_tracking",
            overrides=["loss=tracking_all_outputs_beta01_reprojection"],
        )

    with pytest.raises(
        SemanticConfigurationError,
        match="model.predict_canonical_pose=true",
    ):
        PLCSTrainingConfig.from_config(config)


def test_tracking_pose_loss_rejects_partial_standard_contract() -> None:
    config_dir = PROJECT_ROOT / "src/tasks/plcs/configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name="train_tracking",
            overrides=["loss=tracking_all_outputs_beta01_reprojection"],
        )
    with open_dict(config.loss):
        del config.loss["canonical_pose_smooth_l1_beta"]

    with pytest.raises(
        MissingConfigurationKeyError,
        match="loss.canonical_pose_smooth_l1_beta",
    ):
        PLCSTrainingConfig.from_config(config)
