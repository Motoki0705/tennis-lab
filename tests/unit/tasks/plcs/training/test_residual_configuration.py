"""The package-by-layer residual recipe has one explicit PLCS contract."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.tasks.plcs.configuration import validate_residual_config
from src.utils.models.components.ffn_layers import SUPPORTED_FFN_TYPES


def _recipe(overrides: list[str] | None = None):
    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(
        config_dir=str(root / "src/tasks/plcs/configs"), version_base="1.3"
    ):
        return compose(
            config_name="train_triangulation_residual", overrides=overrides or []
        )


def test_recipe_exposes_the_fixed_plcs_contract() -> None:
    config = validate_residual_config(_recipe())
    assert config.joints == 17
    assert config.root_indices == (11, 12)
    assert config.model.name == "plcs_triangulation_residual"
    assert config.data.max_views == 6
    assert config.augmentation.evaluation_views == 3


@pytest.mark.parametrize("ffn_type", sorted(SUPPORTED_FFN_TYPES))
def test_registered_ffn_types_are_accepted(ffn_type: str) -> None:
    assert (
        validate_residual_config(_recipe([f"model.ffn_type={ffn_type}"])).model.ffn_type
        == ffn_type
    )


@pytest.mark.parametrize("section", ["task", "features", "v2", "corruption"])
def test_removed_top_level_compatibility_sections_are_rejected(section: str) -> None:
    config = _recipe()
    with open_dict(config):
        config[section] = {}
    with pytest.raises(ValueError, match="config keys differ"):
        validate_residual_config(config)


def test_unknown_or_ill_typed_model_values_are_rejected() -> None:
    config = _recipe()
    config.model.ffn_type = "unknown"
    with pytest.raises(ValueError, match="Unsupported ffn_type"):
        validate_residual_config(config)
    config = _recipe()
    config.model.hidden_dim = True
    with pytest.raises(ValueError, match="must have type"):
        validate_residual_config(config)
