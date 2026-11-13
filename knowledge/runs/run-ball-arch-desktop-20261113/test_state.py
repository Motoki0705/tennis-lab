"""One-time migration preserves scientific tensors and rejects implicit changes."""
# ruff: noqa: E402
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from rebuild_dataset import sha, write
from state import (
    HISTORICAL_STORE,
    migrate_resume,
    migrate_t1,
    relocate_checkpoint_data,
    relocate_manifest,
    same,
)

from src.tasks.ball_detection.checkpoints.provenance import source_identity
from src.tasks.ball_detection.checkpoints.temporal import CHECKPOINT_SCHEMA
from src.tasks.ball_detection.model_io.mdd_temporal import (
    build_temporal_model,
    temporal_model_payload,
)
from src.tasks.ball_detection.models.mdd.multiscale_config import (
    MDDMultiScaleQueryConfig,
)


def multiscale_config(**overrides):
    raw = dict(stem_channels=(4, 4, 8, 8), mixed_channels=(8, 16), residual_blocks=(0, 1, 1, 1, 1, 1),
               decoder_channels=8, dim=64, heads=1, layers=2, ffn_dim=192, dropout=.1, rope_base=10000.,
               activation_checkpointing=False, encoder_variant="x3d_csn", temporal_mixing="depthwise3d",
               input_channels=5, frames=64, frame_chunk_size=None, num_streams=4, mhc_coefficient_dim=8,
               mhc_sinkhorn_iters=3, mhc_eps=1e-6, mhc_residual_identity_bias=4., mhc_update_scale_init=.5,
               spatial_layers=("M", "C"), msda_levels=(16, 32, 64), msda_points=4, msda_chunk_size=4, msda_im2col_step=64)
    raw.update(overrides)
    return MDDMultiScaleQueryConfig(**raw)


def test_relocation_changes_only_the_explicit_snapshot_directory(tmp_path):
    original = dict(ball_store=dict(directory=HISTORICAL_STORE, hashes=dict(dataset="original")), owner=[1, 4, 9])
    localized = relocate_manifest(original, tmp_path)
    assert original["ball_store"]["directory"] == HISTORICAL_STORE
    localized["ball_store"]["directory"] = HISTORICAL_STORE
    assert localized == original
    with pytest.raises(ValueError, match="Unexpected historical"):
        relocate_manifest(dict(ball_store=dict(directory="different snapshot")), tmp_path)


def test_checkpoint_metadata_aliases_are_preserved_without_mutating_the_original(tmp_path):
    data = dict(snapshot=dict(directory=HISTORICAL_STORE, hashes=dict(dataset="pinned")))
    checkpoint = dict(recipe=dict(data=data), model_artifact=dict(training=dict(data=data)))
    result = relocate_checkpoint_data(checkpoint, tmp_path)
    assert checkpoint["recipe"]["data"]["snapshot"]["directory"] == HISTORICAL_STORE
    assert result["recipe"]["data"] is result["model_artifact"]["training"]["data"]
    assert result["recipe"]["data"]["snapshot"] == dict(directory=str(tmp_path), hashes=dict(dataset="pinned"))


def test_migrated_checkpoint_preserves_model_adamw_rng_sampler_and_committed_log(tmp_path):
    config = multiscale_config(frames=64)
    model = build_temporal_model(config)
    optimizer = torch.optim.AdamW(model.parameters())
    # Genuine nonempty AdamW state, including unequal parameter moments.
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, .001 * (index + 1))
    optimizer.step()
    artifact = temporal_model_payload(model)
    artifact["model_config"].pop("checkpoint_scope")
    artifact["training"] = dict(data=dict(snapshot=dict(directory=HISTORICAL_STORE)))
    original = tmp_path / "original/T1-q06000-q12000/checkpoint-00005500-cddcef20"
    original.mkdir(parents=True)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    fields = {}
    for name, field in (("train", "manifest_sha256"), ("evaluation", "evaluation_manifest_sha256")):
        old = dict(ball_store=dict(directory=HISTORICAL_STORE), arbitrary=[3, 2, 7])
        write(original / (name + "_manifest.json"), old)
        write(prepared / (name + "_T64.json"), relocate_manifest(old, tmp_path / "dataset"))
        fields[field] = sha(original / (name + "_manifest.json"))
    recipe = dict(model=artifact["model_config"], data=dict(snapshot=dict(directory=HISTORICAL_STORE)), **fields)
    saved = dict(schema=CHECKPOINT_SCHEMA, recipe=recipe, code=source_identity(),
                 updates_completed=5500, sample_position=21500, optimizer=optimizer.state_dict(),
                 optimizer_parameter_names=[[n for n, _ in model.named_parameters()]],
                 model_artifact=artifact, rng=dict(torch=torch.get_rng_state(), python=(2, 7)),
                 schedule=dict(updates_completed=5500, schedule_offset=6000), train_log=dict(bytes=7, sha256="pinned"))
    target = original / "checkpoint-update-00005500.pt"
    torch.save(saved, target)
    write(original / "latest.json", dict(update=5500, checkpoint=target.name, sha256=sha(target)))
    write(original / "run.json", dict(recipe=recipe, code=saved["code"]))
    (original / "train.jsonl").write_bytes(b"pinned\n")
    enabled = multiscale_config(frames=64, activation_checkpointing=True, checkpoint_scope="encoder_msda")
    output = tmp_path / "production/T1-q06000-q12000"
    checkpoint = migrate_t1(tmp_path, output, enabled)
    actual = torch.load(checkpoint, weights_only=True)
    for key in ("optimizer", "optimizer_parameter_names", "rng", "sample_position", "schedule", "updates_completed", "train_log"):
        assert same(saved[key], actual[key]), key
    assert same(saved["model_artifact"]["state_dict"], actual["model_artifact"]["state_dict"])
    assert (output / "train.jsonl").read_bytes() == b"pinned\n"
    assert actual["recipe"]["model"]["activation_checkpointing"] is True
    assert actual["recipe"]["model"]["checkpoint_scope"] == "encoder_msda"
    assert not saved["recipe"]["model"]["activation_checkpointing"]
    with pytest.raises(ValueError, match="overwrite"):
        migrate_t1(tmp_path, output, enabled)
    other_policy = multiscale_config(frames=64, activation_checkpointing=True, checkpoint_scope="msda")
    with pytest.raises(ValueError, match="policy changed"):
        migrate_resume(tmp_path, output, tmp_path / "illegal-policy", other_policy, expected_update=5500)
    recovered_root = tmp_path / "new-host"
    for name in ("train", "evaluation"):
        raw = json.loads((prepared / (name + "_T64.json")).read_text())
        raw["ball_store"]["directory"] = str((recovered_root / "dataset").resolve())
        write(recovered_root / "prepared" / (name + "_T64.json"), raw)
    recovered = migrate_resume(recovered_root, output, recovered_root / "production/t1", enabled, expected_update=5500)
    recovered_state = torch.load(recovered, weights_only=True)
    assert same(actual["optimizer"], recovered_state["optimizer"])
    assert same(actual["rng"], recovered_state["rng"])
    assert same(actual["model_artifact"]["state_dict"], recovered_state["model_artifact"]["state_dict"])
