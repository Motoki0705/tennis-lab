"""Checkpoint migration is explicit, non-mutating, and specific to known semantics."""

import warnings
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tasks.base.triangulation_residual.training import (
    ResidualLightningModule,
    migrate_legacy_checkpoint,
)


def checkpoint_fixture(
    task="blcs", version=1, *, legacy=False, encoding="raw", ffn="swiglu"
):
    root = Path(__file__).resolve().parents[5]
    with initialize_config_dir(
        config_dir=str(root / f"src/tasks/{task}/configs"), version_base="1.3"
    ):
        config = compose(
            config_name="train_triangulation_residual"
            + ("_v2" if version == 2 else ""),
            overrides=[
                "model.hidden_dim=32",
                "model.num_heads=4",
                "model.ffn_dim=64",
                "model.num_layers=1",
                "model.dropout=0",
                f"model.ffn_type={ffn}",
                f"features.residual_encoding={encoding}",
                f"features.residual_scale={0.01 if encoding == 'asinh' else 1.0}",
            ],
        )
    module = ResidualLightningModule(config).eval()
    with torch.no_grad():
        for head in (module.model.root_head, module.model.relative_head):
            if head is not None:
                head[-1].weight.normal_(std=0.01)
    checkpoint = {
        "hyper_parameters": {"config": OmegaConf.to_container(config, resolve=True)},
        "state_dict": module.state_dict(),
    }
    module.on_save_checkpoint(checkpoint)
    if legacy:
        marker = checkpoint["geometric_residual_contract"]
        marker["schema_version"] = 1
        del marker["ffn_type"], marker["features"]
        embedded = checkpoint["hyper_parameters"]["config"]
        del embedded["model"]["ffn_type"], embedded["features"]
    return config, module, checkpoint


@pytest.mark.parametrize("task", ["plcs", "blcs"])
@pytest.mark.parametrize("version", [1, 2])
def test_known_legacy_migration_preserves_state_and_bit_exact_forward(task, version):
    _, before, checkpoint = checkpoint_fixture(task, version, legacy=True)
    original_config = deepcopy(checkpoint["hyper_parameters"]["config"])
    original_marker = deepcopy(checkpoint["geometric_residual_contract"])
    with pytest.warns(UserWarning, match="Migrating legacy geometric residual"):
        migrated = migrate_legacy_checkpoint(checkpoint)
    assert checkpoint["hyper_parameters"]["config"] == original_config
    assert checkpoint["geometric_residual_contract"] == original_marker
    assert migrated is not checkpoint
    assert migrated["state_dict"] is checkpoint["state_dict"]
    provenance = migrated["geometric_residual_migration"]
    assert provenance["source_contract"] == original_marker
    assert (
        provenance["source_contract"] is not checkpoint["geometric_residual_contract"]
    )
    assert provenance["target_schema_version"] == 2
    after = ResidualLightningModule(
        OmegaConf.create(migrated["hyper_parameters"]["config"])
    ).eval()
    with warnings.catch_warnings(record=True) as emitted:
        after.on_load_checkpoint(migrated)
    assert not emitted
    after.load_state_dict(migrated["state_dict"], strict=True)
    assert after.residual_config.model.ffn_type == "swiglu"
    assert after.residual_config.features.residual_encoding == "raw"
    features = torch.randn(2, 3, 8, after.model.input_dim)
    valid = torch.ones(2, 3, 8, dtype=torch.bool)
    valid[0, 2] = False
    positions = torch.arange(8)[None].expand(2, -1).float()
    with torch.inference_mode():
        first, second = (
            before(features, valid, positions),
            after(features, valid, positions),
        )
    for key in first:
        assert first[key].numpy().tobytes() == second[key].numpy().tobytes()
    with pytest.warns(UserWarning, match="Migrating legacy"):
        after.on_load_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 0),
        ("schema_version", 3),
        ("schema_version", True),
        ("family", "plcs_position_yaw"),
        ("family", "blcs_triangulation_residual_v3"),
        ("task", "plcs"),
        ("root_definition", "smpl_translation"),
        ("output_unit", "millimetre"),
        ("feature_dim", 80),
        ("root_indices", [1]),
        ("relative_axes", "yaw_canonical"),
        ("coordinate_frame", "smpl"),
        ("unknown", True),
    ],
)
def test_legacy_migration_rejects_unknown_or_changed_contracts(field, value):
    _, module, checkpoint = checkpoint_fixture(legacy=True)
    checkpoint["geometric_residual_contract"][field] = value
    with pytest.raises(ValueError, match="semantics"):
        migrate_legacy_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="semantics"):
        module.on_load_checkpoint(checkpoint)


@pytest.mark.parametrize("encoding,ffn", [("asinh", "swiglu"), ("raw", "mlp")])
@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_cannot_be_loaded_into_conditioned_or_alternative_ffn_module(
    encoding, ffn, legacy
):
    _, _, checkpoint = checkpoint_fixture(legacy=legacy)
    _, different, _ = checkpoint_fixture(encoding=encoding, ffn=ffn)
    expectation = (
        pytest.warns(UserWarning, match="Migrating legacy") if legacy else nullcontext()
    )
    with expectation, pytest.raises(ValueError, match="semantics"):
        different.on_load_checkpoint(checkpoint)


@pytest.mark.parametrize("encoding,ffn", [("asinh", "swiglu"), ("raw", "mlp")])
def test_schema_one_marker_cannot_hide_new_config_semantics(encoding, ffn):
    _, _, checkpoint = checkpoint_fixture(encoding=encoding, ffn=ffn)
    marker = checkpoint["geometric_residual_contract"]
    marker["schema_version"] = 1
    del marker["features"], marker["ffn_type"]
    with pytest.raises(ValueError, match="SwiGLU/raw"):
        migrate_legacy_checkpoint(checkpoint)


@pytest.mark.parametrize("encoding,ffn", [("raw", "swiglu"), ("asinh", "mlp")])
def test_schema_two_round_trip_preserves_explicit_feature_and_ffn_contract(
    encoding, ffn
):
    _, module, checkpoint = checkpoint_fixture(encoding=encoding, ffn=ffn)
    with warnings.catch_warnings(record=True) as emitted:
        validated = migrate_legacy_checkpoint(checkpoint)
        module.on_load_checkpoint(validated)
    assert not emitted
    assert validated["geometric_residual_contract"]["schema_version"] == 2
    assert validated["geometric_residual_contract"]["ffn_type"] == ffn
    assert (
        validated["geometric_residual_contract"]["features"]["residual_encoding"]
        == encoding
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing_features",
        "missing_ffn",
        "config_encoding",
        "config_ffn",
        "marker_scale",
        "missing_marker",
        "missing_config",
    ],
)
def test_schema_two_requires_complete_matching_embedded_configuration(change):
    _, module, checkpoint = checkpoint_fixture()
    config = checkpoint["hyper_parameters"]["config"]
    if change == "missing_features":
        del config["features"]
    elif change == "missing_ffn":
        del config["model"]["ffn_type"]
    elif change == "config_encoding":
        config["features"] = {"residual_encoding": "asinh", "residual_scale": 0.01}
    elif change == "config_ffn":
        config["model"]["ffn_type"] = "mlp"
    elif change == "marker_scale":
        checkpoint["geometric_residual_contract"]["features"]["residual_scale"] = 0.1
    elif change == "missing_marker":
        del checkpoint["geometric_residual_contract"]
    else:
        del checkpoint["hyper_parameters"]["config"]
    with pytest.raises(ValueError):
        module.on_load_checkpoint(checkpoint)
