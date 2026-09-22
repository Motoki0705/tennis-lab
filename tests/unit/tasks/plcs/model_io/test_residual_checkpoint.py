"""Strict schema-3 loading and explicit selected legacy conversion."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.plcs.configuration import validate_residual_config
from src.tasks.plcs.model_io.residual_checkpoint import (
    checkpoint_contract,
    validate_residual_checkpoint,
)
from src.tasks.plcs.scripts.migrate_residual_checkpoint import convert_checkpoint
from src.tasks.plcs.training.residual_lightning_module import ResidualLightningModule
from src.utils.schema.court_normalization import court_coordinate_normalization_metadata

_CORRUPTION_FIELDS = {
    "observation_sigma_px",
    "temporal_sigma_px",
    "view_bias_px",
    "confidence_noise",
    "outlier_probability",
    "outlier_sigma_px",
    "dropout_probability",
    "burst_probability",
    "burst_max_frames",
    "time_shift_probability",
    "time_shift_max_frames",
    "radial_std",
    "focal_scale_min",
    "focal_scale_max",
    "clean_probability",
    "hard_probability",
}


def _recipe():
    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(
        config_dir=str(root / "src/tasks/plcs/configs"), version_base="1.3"
    ):
        config = compose(config_name="train_triangulation_residual")
    config.model.hidden_dim = 32
    config.model.num_heads = 4
    config.model.ffn_dim = 64
    config.model.num_layers = 1
    return config


def _historical_checkpoint() -> tuple[dict, ResidualLightningModule]:
    config = _recipe()
    parsed = validate_residual_config(config)
    module = ResidualLightningModule(config).eval()
    with torch.no_grad():
        for head in (module.model.root_head, module.model.relative_head):
            head[-1].weight.normal_(std=0.01)
    new = OmegaConf.to_container(config, resolve=True)
    assert isinstance(new, dict)
    old = deepcopy(new)
    augmentation = old.pop("augmentation")
    assert isinstance(augmentation, dict)
    old["task"] = "plcs"
    old["model"]["name"] = "plcs_triangulation_residual_v2"
    old["features"] = {"residual_encoding": "raw", "residual_scale": 1.0}
    old["corruption"] = {key: augmentation[key] for key in _CORRUPTION_FIELDS} | {
        "camera_rotation_std_deg": 0.0,
        "camera_center_std_m": 0.0,
        "camera_focal_log_std": 0.0,
        "camera_principal_std_px": 0.0,
    }
    old["v2"] = {
        key: value
        for key, value in augmentation.items()
        if key not in _CORRUPTION_FIELDS
    } | {
        "loss_mode": "legacy",
        "regret_weight": 0.2,
        "regret_tolerance_m": 0.002,
        "balanced_world_weight": 0.25,
    }
    marker = checkpoint_contract(parsed)
    marker["family"] = "plcs_triangulation_residual_v2"
    marker["schema_version"] = 2
    return (
        {
            "hyper_parameters": {"config": old},
            "state_dict": module.state_dict(),
            "geometric_residual_contract": marker,
            "court_coordinate_normalization": court_coordinate_normalization_metadata(),
        },
        module,
    )


def test_selected_v2_raw_checkpoint_converts_without_mutating_or_changing_forward() -> (
    None
):
    source, before = _historical_checkpoint()
    original = deepcopy(source)
    converted = convert_checkpoint(source)
    assert source["hyper_parameters"] == original["hyper_parameters"]
    assert (
        source["geometric_residual_contract"] == original["geometric_residual_contract"]
    )
    assert converted["state_dict"] is source["state_dict"]
    assert converted["geometric_residual_contract"]["schema_version"] == 3
    assert converted["geometric_residual_migration"]["weights_only"] is True
    after = ResidualLightningModule(
        OmegaConf.create(converted["hyper_parameters"]["config"])
    ).eval()
    after.load_state_dict(converted["state_dict"], strict=True)
    features = torch.randn(2, 3, 8, after.model.input_dim)
    valid = torch.ones(2, 3, 8, dtype=torch.bool)
    positions = torch.arange(8)[None].expand(2, -1).float()
    with torch.inference_mode():
        first = before(features, valid, positions)
        second = after(features, valid, positions)
    for key in first:
        assert first[key].numpy().tobytes() == second[key].numpy().tobytes()


@pytest.mark.parametrize("change", ["blcs", "asinh", "balanced", "schema3"])
def test_conversion_rejects_unselected_historical_semantics(change: str) -> None:
    source, _ = _historical_checkpoint()
    config = source["hyper_parameters"]["config"]
    if change == "blcs":
        config["task"] = "blcs"
    elif change == "asinh":
        config["features"] = {
            "residual_encoding": "asinh",
            "residual_scale": 0.01,
        }
    elif change == "balanced":
        config["v2"]["loss_mode"] = "balanced_regret"
    else:
        source["geometric_residual_contract"]["schema_version"] = 3
    with pytest.raises(ValueError):
        convert_checkpoint(source)


def test_runtime_validator_rejects_old_or_tampered_contracts() -> None:
    source, _ = _historical_checkpoint()
    with pytest.raises(ValueError, match="schema 3"):
        validate_residual_checkpoint(source)
    converted = convert_checkpoint(source)
    converted["geometric_residual_contract"]["root_definition"] = "smpl_translation"
    with pytest.raises(ValueError, match="semantics"):
        validate_residual_checkpoint(converted)
