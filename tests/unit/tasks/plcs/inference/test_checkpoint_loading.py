"""Inference composition excludes historical training-only schema changes."""

import copy

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.plcs.configuration import PLCSTrainingConfig
from src.tasks.plcs.inference.checkpoint import load_plcs_pair
from src.tasks.plcs.model_io.factory import build_plcs_model_io
from src.utils.paths import PROJECT_ROOT
from src.utils.schema.court_normalization import add_court_coordinate_normalization


def test_inference_loads_exact_weights_without_training_config():
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tasks/plcs/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="train",
            overrides=[
                "model.hidden_dim=16",
                "model.num_heads=4",
                "model.ffn_dim=32",
                "model.rope_dim=4",
                "model.num_layers=1",
            ],
        )
    runtime = PLCSTrainingConfig.from_config(cfg)
    original = build_plcs_model_io(runtime)
    raw = OmegaConf.to_container(cfg, resolve=True)
    raw.pop("run")
    raw.pop("training")
    raw["data"].pop("augmentation")
    checkpoint = {
        "state_dict": {f"model.{k}": v for k, v in original.model.state_dict().items()}
    }
    checkpoint["state_dict"]["criterion.obsolete_buffer"] = torch.ones(1)
    add_court_coordinate_normalization(checkpoint, artifact="test")
    restored = load_plcs_pair(checkpoint, raw, runtime.court_keypoint_contract)
    for key, value in original.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[key], value)
    invalid = copy.deepcopy(checkpoint)
    invalid["state_dict"]["model.unexpected"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_plcs_pair(invalid, raw, runtime.court_keypoint_contract)
    checkpoint.pop("court_coordinate_normalization")
    with pytest.raises(ValueError, match="normalization"):
        load_plcs_pair(checkpoint, raw, runtime.court_keypoint_contract)
