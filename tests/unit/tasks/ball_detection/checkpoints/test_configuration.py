from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import pytest

from src.tasks.ball_detection.checkpoints.configuration import (
    checkpoint_model_config,
    checkpoint_training_recipe,
)
from src.tasks.ball_detection.checkpoints.coordinates import query_checkpoint_config
from src.tasks.ball_detection.checkpoints.pretraining import pretraining_config
from src.tasks.ball_detection.preprocessing import RGBToMDD
from src.tasks.ball_detection.preprocessing.rgb_mdd import RGBToRGBMDD
from tests.unit.tasks.ball_detection.models.mdd.test_cnn import small_config


def historical_checkpoint(schema: str) -> dict[str, Any]:
    config = asdict(small_config())
    config.pop("input_channels")
    if schema == "mdd_dpt_pretraining.v1":
        config.pop("encoder_variant")
        config.pop("temporal_mixing")
    contract = RGBToMDD(a=.3, b=.1).input_contract()
    return dict(schema=schema, stage="query_posttraining", model_config=config,
                recipe=dict(model=deepcopy(config), input_contract=contract,
                            image_decode={}, parent=dict(input_contract=deepcopy(contract), image_decode={})))


@pytest.mark.parametrize("schema", [
    "mdd_dpt_pretraining.v1", "mdd_dpt_pretraining.v2", "mdd_query_posttraining.v1",
])
def test_historical_schema_and_complete_contract_identify_two_channels_without_mutation(schema: str) -> None:
    saved = historical_checkpoint(schema)
    original = deepcopy(saved)
    if schema.startswith("mdd_dpt"):
        config = pretraining_config(saved)
    else:
        config, transform = query_checkpoint_config(saved)
        assert (transform.a, transform.b) == (.3, .1)
    assert config == small_config()
    assert checkpoint_training_recipe(saved, config)["model"] == asdict(config)
    assert saved == original


@pytest.mark.parametrize("schema", ["mdd_dpt_pretraining.v2", "mdd_query_posttraining.v1"])
@pytest.mark.parametrize("damage", ["missing", "partial", "unknown", "normalization", "rgb_mdd", "architecture"])
def test_missing_channels_cannot_be_inferred_without_the_exact_historical_contract(schema: str, damage: str) -> None:
    saved = historical_checkpoint(schema)
    recipe = saved["recipe"]
    if damage == "missing":
        recipe.pop("input_contract")
    elif damage == "partial":
        recipe["input_contract"].pop("luminance")
    elif damage == "unknown":
        recipe["input_contract"]["schema"] = "unknown.v1"
    elif damage == "normalization":
        recipe["input_contract"]["normalization"] = "imagenet"
    elif damage == "rgb_mdd":
        recipe["input_contract"] = RGBToRGBMDD().input_contract()
    else:
        saved["model_config"].pop("dim")
    with pytest.raises(ValueError, match="input contract|configuration"):
        checkpoint_model_config(saved)


def test_schema_recipe_and_explicit_channels_must_agree() -> None:
    saved = historical_checkpoint("unknown.v1")
    with pytest.raises(ValueError, match="schema"):
        checkpoint_model_config(saved)
    saved["schema"] = "mdd_query_posttraining.v1"
    saved["model_config"]["input_channels"] = 5
    saved["recipe"]["model"]["input_channels"] = 5
    with pytest.raises(ValueError, match="input contract"):
        query_checkpoint_config(saved)
    saved["recipe"]["input_contract"] = RGBToRGBMDD().input_contract()
    saved["recipe"]["parent"]["input_contract"] = deepcopy(saved["recipe"]["input_contract"])
    config, _ = query_checkpoint_config(saved)
    assert config == replace(small_config(), input_channels=5)
    saved["recipe"]["model"].pop("input_channels")
    with pytest.raises(ValueError, match="configuration differs"):
        query_checkpoint_config(saved)
    with pytest.raises(ValueError, match="configuration differs"):
        checkpoint_training_recipe(saved, config)
