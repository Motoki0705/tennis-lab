from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


@pytest.mark.parametrize(
    (
        "model_name",
        "hidden_dim",
        "num_heads",
        "num_stages",
        "ffn_dim",
        "rope_dim",
    ),
    [
        ("track_query_small", 256, 4, 8, 704, 64),
        ("track_query_base", 512, 8, 8, 1408, 64),
        ("track_query_large", 512, 8, 12, 1408, 64),
        ("track_query_xlarge", 1024, 8, 12, 2752, 128),
    ],
)
def test_track_query_size_configs_compose(
    model_name: str,
    hidden_dim: int,
    num_heads: int,
    num_stages: int,
    ffn_dim: int,
    rope_dim: int,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_tracking", overrides=[f"model={model_name}"])

    assert config.model.name == "blcs_track_query"
    assert config.model.hidden_dim == hidden_dim
    assert config.model.num_heads == num_heads
    assert config.model.num_stages == num_stages
    assert config.model.ffn_dim == ffn_dim
    assert config.model.rope_dim == rope_dim
