from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.plcs.configuration import PLCSModelConfig

_CONFIG_DIR = Path("src/tasks/plcs/configs").resolve()


@pytest.mark.parametrize(
    (
        "model_name",
        "hidden_dim",
        "num_heads",
        "num_stages",
        "ffn_dim",
        "rope_dim",
        "dropout",
    ),
    [
        ("track_query", 64, 4, 4, 128, 16, 0.0),
        ("track_query_base", 512, 8, 8, 1408, 64, 0.1),
    ],
)
def test_track_query_size_configs_compose_and_validate(
    model_name: str,
    hidden_dim: int,
    num_heads: int,
    num_stages: int,
    ffn_dim: int,
    rope_dim: int,
    dropout: float,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(
            config_name="train_tracking_chunked",
            overrides=[f"model={model_name}"],
        )

    assert config.model.name == "plcs_track_query"
    assert config.model.hidden_dim == hidden_dim
    assert config.model.num_heads == num_heads
    assert config.model.num_stages == num_stages
    assert config.model.ffn_dim == ffn_dim
    assert config.model.rope_dim == rope_dim
    assert config.model.dropout == dropout
    assert config.model.mhc.coefficient_dim == 64
    assert config.model.cswa.compression_ratio == 4

    parsed = PLCSModelConfig.from_mapping(config.model)
    assert parsed.name == "plcs_track_query"
    assert parsed.integer("hidden_dim") == hidden_dim
    assert parsed.integer("num_stages") == num_stages
