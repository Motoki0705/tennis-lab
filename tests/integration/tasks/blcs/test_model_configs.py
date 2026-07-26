from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


@pytest.mark.parametrize(
    ("model_name", "hidden_dim", "num_heads", "num_stages"),
    [
        ("track_query_small", 256, 4, 8),
        ("track_query_base", 512, 8, 8),
        ("track_query_large", 512, 8, 12),
        ("track_query_xlarge", 1024, 8, 12),
    ],
)
def test_track_query_size_configs_compose(
    model_name: str,
    hidden_dim: int,
    num_heads: int,
    num_stages: int,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train_tracking", overrides=[f"model={model_name}"])

    assert config.model.name == "blcs_track_query"
    assert config.model.hidden_dim == hidden_dim
    assert config.model.num_heads == num_heads
    assert config.model.num_stages == num_stages
    assert config.model.ffn_dim is None
    assert config.model.rope_dim is None
