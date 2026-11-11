"""CA denotes the query-only speed baseline, despite sharing the CNN prefix."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from protocol import model_config, recipe

from src.tasks.ball_detection.models.mdd.config import MDDMHCQueryConfig


def test_ca_speed_baseline_uses_the_saved_query_recipe():
    config = model_config("CA")
    assert type(config) is MDDMHCQueryConfig
    assert (config.frames, config.dim, config.layers, config.num_streams) == (
        64,
        512,
        6,
        4,
    )
    assert recipe("CA")["model"]["frames"] == 64


def test_unknown_architecture_condition_is_not_reinterpreted_as_t2():
    with pytest.raises(ValueError, match="Unknown fixed architecture"):
        model_config("C4")
