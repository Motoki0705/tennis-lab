"""Independent checkpoint contracts and strict task training composition."""

from pathlib import Path
from typing import Any

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.tasks.plcs.association_configuration import validate_person_config
from src.tasks.plcs.model_io.person_association import (
    MODEL_CONTRACTS,
    REID_MODEL,
    SIDE_MODEL,
    validate_person_checkpoint,
)
from src.tasks.plcs.training.association_lightning_module import (
    PLCSAssociationLightningModule,
)
from src.tasks.plcs.training.composition import (
    build_plcs_datamodule,
    build_plcs_lightning_module,
)


def configuration(mode="reid"):
    directory = Path(__file__).parents[5] / "src/tasks/plcs/configs"
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        return compose(config_name=f"train_{mode}", overrides=["model.hidden_dim=32", "model.ffn_dim=64", "model.num_heads=4", "model.rope_dim=8", "model.num_stages=2"])


@pytest.mark.parametrize("mode,name", [("reid", REID_MODEL), ("court_side", SIDE_MODEL)])
def test_task_composition_and_checkpoint_model_ownership(mode, name):
    cfg = configuration(mode)
    validate_person_config(cfg)
    module = build_plcs_lightning_module(cfg)
    assert isinstance(module, PLCSAssociationLightningModule)
    assert build_plcs_datamodule(cfg).__class__.__module__.startswith("src.tasks.plcs.")
    checkpoint: dict[str, Any] = {}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint["person_association_contract"] == MODEL_CONTRACTS[name]
    validate_person_checkpoint(checkpoint, model_name=name)
    with pytest.raises(ValueError, match="mismatch"):
        validate_person_checkpoint(checkpoint, model_name=SIDE_MODEL if name == REID_MODEL else REID_MODEL)
    with pytest.raises(ValueError, match="retraining"):
        validate_person_checkpoint({"association_contract": "camera_local_global_mha_mhc_association_v2"}, model_name=name)
    assert module.compilation_targets() == {"model": module.model}


def test_training_rejects_unknown_keys_and_unsupported_mcmc():
    cfg = configuration()
    with open_dict(cfg.training):
        cfg.training.unused_switch = True
    with pytest.raises(ValueError, match="unused_switch"):
        validate_person_config(cfg)
    cfg = configuration()
    cfg.training.mcmc.enabled = True
    with pytest.raises(ValueError, match="MCMC"):
        validate_person_config(cfg)


def test_reid_forward_excludes_reference_and_all_identity_metadata():
    module = PLCSAssociationLightningModule(configuration())
    batch = dict(human_kp=torch.rand(1, 2, 2, 4, 17, 2), human_vis=torch.ones(1, 2, 2, 4, 17, dtype=torch.bool),
        court_kp=torch.rand(1, 2, 2, 14, 2), court_vis=torch.ones(1, 2, 2, 14, dtype=torch.bool),
        padding_mask=torch.zeros(1, 2, 2, dtype=torch.bool), reference_view_index=torch.tensor([0]),
        track_person_id=torch.arange(4)[None, None].expand(1, 2, 4), local_track_ids=torch.ones(1, 2, 4, dtype=torch.int64),
        side_target=torch.ones(1, 2, dtype=torch.bool))
    call = module.model_io.build_call(batch)
    assert set(call.kwargs) == {"human_kp", "human_vis", "court_kp", "court_vis", "padding_mask"}
