"""CA denotes the query-only speed baseline, despite sharing the CNN prefix."""

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import protocol
from protocol import model_config, recipe

from src.tasks.ball_detection.models.mdd.config import (
    MDDHeatmapConfig,
    MDDMHCQueryConfig,
)


def test_ca_speed_baseline_uses_the_saved_query_recipe(tmp_path, monkeypatch):
    # Two explicit saved recipes detect CA being routed through the CNN prefix.
    # Keep the regression runnable without the original Colab filesystem.
    configs = Path("src/tasks/ball_detection/configs/model")
    query = replace(
        MDDMHCQueryConfig.load(configs / "mdd_mhc_query.yaml"),
        dim=512,
        heads=8,
        layers=6,
        ffn_dim=1408,
        dropout=0.07,
    )
    dpt = MDDHeatmapConfig.load(configs / "mdd_dpt_heatmap.yaml")
    for name, model in (("common-dpt-T32", dpt), ("query-d512-l6-T64", query)):
        path = tmp_path / name / "run.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"recipe": {"model": asdict(model)}}))
    monkeypatch.setattr(protocol, "INPUTS", tmp_path)
    config = model_config("CA")
    assert type(config) is MDDMHCQueryConfig
    assert (config.frames, config.dim, config.layers, config.num_streams) == (
        64,
        512,
        6,
        4,
    )
    assert asdict(config) == asdict(query)
    assert recipe("CA")["model"]["dropout"] == 0.07


def test_unknown_architecture_condition_is_not_reinterpreted_as_t2():
    with pytest.raises(ValueError, match="Unknown fixed architecture"):
        model_config("C4")
