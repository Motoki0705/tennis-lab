from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tasks.blcs.configuration import (
    AxialModelConfig,
    parse_model_config,
)

_CONFIG_DIR = Path("src/tasks/blcs/configs").resolve()


@pytest.mark.parametrize(
    ("model_name", "num_layers"),
    [
        ("multiview_axial_small", 8),
        ("multiview_axial_base", 8),
        ("multiview_axial_large", 12),
        ("multiview_axial_xlarge", 12),
    ],
)
def test_axial_configs_preserve_local_temporal_attention_contract(
    model_name: str,
    num_layers: int,
) -> None:
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.3"):
        config = compose(config_name="train", overrides=[f"model={model_name}"])

    parsed = parse_model_config(config)

    assert isinstance(parsed, AxialModelConfig)
    assert parsed.time_window_radius == 16
    assert parsed.time_global_stage_mask == (False,) * num_layers
