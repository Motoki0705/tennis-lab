"""Removing the auxiliary head preserves embeddings and rejects implicit migration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from src.tasks.plcs.association_configuration import validate_person_config
from src.tasks.plcs.inference.person_predictor import PlayerReIDPredictor
from src.tasks.plcs.model_io.person_association import (
    MODEL_CONTRACTS,
    REID_MODEL,
    PersonInferencePolicy,
    PersonObservationRequest,
)
from src.tasks.plcs.model_io.reid_checkpoint import export_headless_reid_checkpoint
from src.tasks.plcs.training.association_lightning_module import (
    PLCSAssociationLightningModule,
)


def configuration():
    directory = Path(__file__).parents[5] / "src/tasks/plcs/configs"
    with initialize_config_dir(config_dir=str(directory), version_base="1.3"):
        return compose(config_name="train_reid", overrides=["model.hidden_dim=32", "model.ffn_dim=64", "model.num_heads=4", "model.rope_dim=8", "model.num_stages=2"])


def legacy_checkpoint(tmp_path):
    module = PLCSAssociationLightningModule(configuration()).eval()
    config = deepcopy(module.config)
    with open_dict(config):
        config.loss.player_weight = .1
        config.metrics.player_threshold = .5
        config.data.augmentation.false_track_probability = .3
    state = {key: value.clone() for key, value in module.state_dict().items()}
    state["model.player_head.weight"] = torch.randn(1, 32)
    state["model.player_head.bias"] = torch.tensor([-100.])
    checkpoint: dict[str, Any] = {
        "state_dict": state, "hyper_parameters": {"config": config},
        "pytorch-lightning_version": pl.__version__, "epoch": 41, "global_step": 2100,
        "person_association_model": REID_MODEL, "person_association_contract": "plcs_fixed_track_reid_v1",
        "optimizer_states": [{"old_head_state": True}], "lr_schedulers": [{}],
    }
    source = tmp_path / "legacy.ckpt"
    torch.save(checkpoint, source)
    return module, source, checkpoint


def test_explicit_export_preserves_every_retained_tensor_and_embedding(tmp_path):
    module, source, legacy = legacy_checkpoint(tmp_path)
    original = source.read_bytes()
    output = tmp_path / "headless.ckpt"
    export_headless_reid_checkpoint(source, output)
    converted = torch.load(output, map_location="cpu", weights_only=False)
    assert source.read_bytes() == original
    assert converted["person_association_contract"] == MODEL_CONTRACTS[REID_MODEL]
    assert "optimizer_states" not in converted and "lr_schedulers" not in converted
    assert converted["weights_only_export"]
    assert set(converted["state_dict"]) == set(module.state_dict())
    for name, value in converted["state_dict"].items():
        assert torch.equal(value, legacy["state_dict"][name])
    assert "player_weight" in legacy["hyper_parameters"]["config"].loss
    cfg = converted["hyper_parameters"]["config"]
    assert "player_weight" not in cfg.loss and "player_threshold" not in cfg.metrics
    assert "false_track_probability" not in cfg.data.augmentation
    restored = PlayerReIDPredictor.load(output)
    assert not any("player_head" in name for name in restored.module.model.state_dict())
    batch = dict(human_kp=torch.rand(1, 2, 5, 4, 17, 2), human_vis=torch.ones(1, 2, 5, 4, 17, dtype=torch.bool),
        court_kp=torch.rand(1, 2, 5, 14, 2), court_vis=torch.ones(1, 2, 5, 14, dtype=torch.bool),
        padding_mask=torch.zeros(1, 2, 5, dtype=torch.bool))
    with torch.no_grad():
        expected = module.model_io.run(batch)
        actual = restored.predict(batch)
    assert set(actual) == {"track_embedding", "track_valid"}
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    batch["human_vis"][:, :, :, -1] = False
    request = PersonObservationRequest(batch["human_kp"][0], batch["human_vis"][0],
        torch.tensor([[1, 2, 3, 4], [20, 30, 40, 50]]), batch["court_kp"][0], batch["court_vis"][0],
        ("a", "b"), "a", torch.arange(5))
    result = restored.predict_observations(request, policy=PersonInferencePolicy(min_frames=5, max_frames=5, padded_views=2))
    assert result.raw_track_ids[:, :3].ge(0).all()
    assert result.raw_track_ids[:, 3].eq(-1).all()
    assert not hasattr(result, "player_probability")
    with pytest.raises(FileExistsError):
        export_headless_reid_checkpoint(source, output)


def test_legacy_load_is_rejected_and_export_cannot_resume_old_optimizer(tmp_path):
    module, source, legacy = legacy_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        module.on_load_checkpoint(legacy)
    output = export_headless_reid_checkpoint(source, tmp_path / "headless.ckpt")
    converted = torch.load(output, map_location="cpu", weights_only=False)
    module.config.run.resume = str(output)
    with pytest.raises(ValueError, match="optimizer resume"):
        module.on_load_checkpoint(converted)


@pytest.mark.parametrize("section,key,value", [
    ("loss", "player_weight", .1), ("metrics", "player_threshold", .5),
    ("data.augmentation", "false_track_probability", .3),
])
def test_deleted_training_controls_are_rejected(section, key, value):
    config = configuration()
    target = config
    for field in section.split("."):
        target = target[field]
    with open_dict(target):
        target[key] = value
    with pytest.raises(ValueError, match=key):
        validate_person_config(config)


def test_export_rejects_incomplete_or_unexpected_parameters(tmp_path):
    _, source, legacy = legacy_checkpoint(tmp_path)
    del legacy["state_dict"]["model.player_head.bias"]
    torch.save(legacy, source)
    with pytest.raises(ValueError, match="both auxiliary-head"):
        export_headless_reid_checkpoint(source, tmp_path / "missing.ckpt")
    legacy["state_dict"]["model.player_head.bias"] = torch.zeros(1)
    legacy["state_dict"]["model.unexpected"] = torch.zeros(1)
    torch.save(legacy, source)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        export_headless_reid_checkpoint(source, tmp_path / "extra.ckpt")
    assert not (tmp_path / "extra.ckpt").exists()
