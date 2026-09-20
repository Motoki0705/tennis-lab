"""Historical recipes stay explicit, and the v2 fit cannot mix camera perturbations."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.tasks.base.triangulation_residual.configuration import validate_config
from src.utils.models.components.ffn_layers import SUPPORTED_FFN_TYPES


def recipe(task="blcs", version=2, overrides=None):
    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(
        config_dir=str(root / f"src/tasks/{task}/configs"), version_base="1.3"
    ):
        return compose(
            config_name="train_triangulation_residual"
            + ("_v2" if version == 2 else ""),
            overrides=overrides or [],
        )


@pytest.mark.parametrize("task", ["plcs", "blcs"])
def test_v2_recipe_has_six_candidates_and_three_evaluation_views(task):
    config = validate_config(recipe(task))
    assert config.v2 is not None
    assert config.v2.camera_preset == "four_corners_front_pair"
    assert config.data.max_views == 6
    assert config.v2.evaluation_views == 3
    assert config.model.name.endswith("_v2")


@pytest.mark.parametrize(
    "key,value",
    [
        ("camera_center_std_m", 0.1),
        ("camera_rotation_std_deg", 0.1),
        ("camera_focal_log_std", 0.1),
    ],
)
def test_v2_rejects_independent_calibration_perturbation(key, value):
    config = recipe()
    config.corruption[key] = value
    with pytest.raises(ValueError, match="independent camera"):
        validate_config(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("camera_preset", "broadcast"),
        ("evaluation_views", 7),
        ("loss_mode", "relative_gt_normalized"),
        ("persistent_min_seconds", 2.0),
    ],
)
def test_v2_rejects_ambiguous_or_invalid_policy(key, value):
    config = recipe()
    config.v2[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize("task", ["plcs", "blcs"])
@pytest.mark.parametrize("version", [1, 2])
def test_default_features_stay_raw_and_cli_can_opt_in(task, version):
    default = validate_config(recipe(task, version))
    assert default.model.ffn_type == "swiglu"
    assert default.features.residual_encoding == "raw"
    assert default.features.residual_scale == 1.0
    encoded = validate_config(
        recipe(
            task,
            version,
            [
                "features.residual_encoding=asinh",
                "features.residual_scale=0.01",
            ],
        )
    )
    assert encoded.features.residual_encoding == "asinh"
    assert encoded.features.residual_scale == 0.01


@pytest.mark.parametrize("ffn_type", sorted(SUPPORTED_FFN_TYPES))
def test_ffn_registry_is_accepted_by_residual_config(ffn_type):
    config = recipe(overrides=[f"model.ffn_type={ffn_type}"])
    assert validate_config(config).model.ffn_type == ffn_type


@pytest.mark.parametrize("field", ["features", "ffn_type", "residual_scale"])
def test_training_never_defaults_missing_checkpoint_migration_fields(field):
    config = recipe()
    section = (
        config
        if field == "features"
        else config.model
        if field == "ffn_type"
        else config.features
    )
    with open_dict(section):
        del section[field]
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ffn_type", "unknown"),
        ("residual_encoding", "log"),
        ("residual_scale", 0.0),
        ("residual_scale", -1.0),
        ("residual_scale", float("nan")),
        ("residual_scale", float("inf")),
        ("residual_scale", True),
        ("residual_scale", "0.01"),
    ],
)
def test_feature_and_ffn_configuration_rejects_invalid_values(field, value):
    config = recipe()
    config.features.residual_encoding = "asinh"
    (config.model if field == "ffn_type" else config.features)[field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_raw_encoding_rejects_unused_scale():
    config = recipe()
    config.features.residual_scale = 0.01
    with pytest.raises(ValueError, match="raw residual encoding"):
        validate_config(config)
