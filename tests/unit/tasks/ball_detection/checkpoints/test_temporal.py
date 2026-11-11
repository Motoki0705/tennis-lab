"""Encoder-only initialization has a narrower contract than full-model transfer."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.checkpoints.provenance import source_identity
from src.tasks.ball_detection.checkpoints.temporal import (
    RUN_SCHEMA,
    initialize_finetuning,
)
from src.tasks.ball_detection.model_io.mdd_temporal import (
    build_temporal_model,
    temporal_model_payload,
)
from src.tasks.ball_detection.models.mdd import MDDTemporalQueryConfig
from src.utils.checksum import dual_sha256
from tests.support.tasks.ball_detection.temporal import small_mhc_model, small_model


def save_artifact(path: Path, config, *, phase: str = "pretrain") -> dict:
    artifact: dict = temporal_model_payload(build_temporal_model(config))
    artifact["training"] = dict(schema=RUN_SCHEMA, phase=phase, update=2,
                                sample_position=2, cumulative_updates=2, cumulative_input_frames=256)
    torch.save(artifact, path)
    return artifact


@pytest.mark.parametrize("frames", [32, 64])
def test_dpt_to_mhc_validates_only_encoder_and_transfers_only_encoder(tmp_path: Path, frames: int) -> None:
    source = replace(small_model("dpt"), encoder_variant="x3d_csn", temporal_mixing="depthwise3d",
                     residual_blocks=(0, 1, 1, 1, 1, 1), layers=2)
    path = tmp_path / "dpt.pt"
    artifact = save_artifact(path, source)
    config = small_mhc_model(frames=frames, decoder_channels=16)
    assert (source.dim, source.heads, source.ffn_dim, source.layers) != (config.dim, config.heads, config.ffn_dim, config.layers)
    model, receipt = initialize_finetuning(path, expected_sha256=dual_sha256(path), parent_update=2, config=config)
    assert receipt["mode"] == "encoder" and receipt["saved_frames"] == 32 and receipt["target_frames"] == frames
    assert receipt["transferred_keys"] == sorted(name for name in artifact["state_dict"] if name.startswith("encoder."))
    assert any(name.startswith("blocks.") for name in receipt["new_keys"])
    assert not hasattr(model, "decoder")
    for name in receipt["transferred_keys"]:
        torch.testing.assert_close(model.state_dict()[name], artifact["state_dict"][name], rtol=0, atol=0)
    for changes in ({"mixed_channels": (8, 24)}, {"residual_blocks": (0, 2, 1, 1, 1, 1)}):
        with pytest.raises(ValueError, match="encoder topology"):
            initialize_finetuning(path, expected_sha256=dual_sha256(path), parent_update=2, config=replace(config, **changes))
    with pytest.raises(ValueError, match="fresh T32 or T64"):
        initialize_finetuning(path, expected_sha256=dual_sha256(path), parent_update=2, config=replace(config, frames=128))


def test_dpt_to_single_query_transfers_encoder_but_full_mhc_transfer_stays_strict(tmp_path: Path) -> None:
    source = small_model("dpt")
    path = tmp_path / "old.pt"
    save_artifact(path, source)
    target = small_model("query", frames=64)
    assert type(target) is MDDTemporalQueryConfig
    for config in (target, replace(target, frames=32, layers=2)):
        model, receipt = initialize_finetuning(path, expected_sha256=dual_sha256(path), parent_update=2, config=config)
        assert receipt["mode"] == "encoder"
        assert all(name.startswith("encoder.") for name in receipt["transferred_keys"])
        assert model.config.frames == config.frames
    with pytest.raises(ValueError, match="encoder topology"):
        initialize_finetuning(path, expected_sha256=dual_sha256(path), parent_update=2,
                              config=replace(target, residual_blocks=(0, 1, 0, 0, 0, 0)))
    save_artifact(path, source, phase="finetune")
    with pytest.raises(ValueError, match="common T32 pretraining"):
        initialize_finetuning(path, expected_sha256=dual_sha256(path), parent_update=2, config=target)
    mhc = small_mhc_model()
    path = tmp_path / "mhc.pt"
    save_artifact(path, mhc, phase="finetune")
    with pytest.raises(ValueError, match="every parameter topology"):
        initialize_finetuning(path, expected_sha256=dual_sha256(path), parent_update=2,
                              config=replace(mhc, frames=128, mhc_eps=1e-5))


def test_source_identity_records_the_shared_mhc_implementation() -> None:
    code = source_identity()
    relative = "src/utils/models/components/mhc.py"
    expected = hashlib.sha256((Path(code["directory"]) / relative).read_bytes()).hexdigest()
    assert code["source_sha256"][relative] == expected
