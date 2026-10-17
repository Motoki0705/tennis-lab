from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from src.tasks.ball_detection.checkpoints.coordinates import (
    restore_coordinate_model,
    save_coordinate_checkpoint,
)
from src.tasks.ball_detection.checkpoints.pretraining import (
    pretraining_config,
    save_pretraining,
    transfer_encoder,
)
from src.tasks.ball_detection.model_io.predict import (
    predict_coordinates,
    predict_dpt_heatmaps,
)
from src.tasks.ball_detection.models.mdd import (
    DeepMDDQueryDetector,
    MDDDPTDetector,
    MDDPretrainConfig,
)


def config(channels: int = 5) -> MDDPretrainConfig:
    return MDDPretrainConfig(
        stem_channels=(4, 4, 8, 8), mixed_channels=(8, 16), residual_blocks=(0,) * 6,
        decoder_channels=8, dim=16, heads=2, layers=1, ffn_dim=64, dropout=0.,
        rope_base=10000., activation_checkpointing=False, encoder_variant="convnext_v2",
        temporal_mixing="factorized", input_channels=channels,
    )


@pytest.mark.parametrize("kind", ["dpt", "query"])
def test_five_channel_stem_gets_gradients_in_a_full_cpu_graph(kind: str) -> None:
    torch.set_num_threads(2)
    model = MDDDPTDetector(config()) if kind == "dpt" else DeepMDDQueryDetector(config())
    rgb = torch.randint(256, (1, 32, 3, 33, 49), dtype=torch.uint8)
    args = (rgb,) if kind == "dpt" else (rgb, torch.arange(32)[None].float() / 30)
    expected = model(*args)
    model.compile(backend="eager", fullgraph=True, dynamic=False)
    actual = model(*args)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().mean().backward()
    gradient = model.encoder.stages[0][0][0].weight.grad
    assert gradient is not None and (gradient.abs().sum((0, 2, 3)) > 0).all()


@pytest.mark.parametrize("channels", [2, 5])
def test_dpt_save_encoder_transfer_query_restore_and_model_io(tmp_path: Path, channels: int) -> None:
    torch.set_num_threads(2)
    source = MDDDPTDetector(config(channels)).eval()
    optimizer = torch.optim.AdamW(source.parameters())
    path = tmp_path / "dpt.pt"
    recipe = dict(input_contract=source.mdd.input_contract(), image_decode={})
    save_pretraining(path, source, optimizer, epoch=0, step=0, recipe=recipe, code={}, report={}, best=0., best_epoch=0)
    saved = torch.load(path, weights_only=True)
    assert pretraining_config(saved) == source.config
    restored_dpt = MDDDPTDetector(pretraining_config(saved)).eval()
    restored_dpt.load_state_dict(saved["state_dict"], strict=True)
    rgb = torch.randint(256, (1, 32, 3, 33, 49), dtype=torch.uint8)
    times = torch.arange(32)[None].float() / 30
    with torch.no_grad():
        torch.testing.assert_close(predict_dpt_heatmaps(source, rgb), restored_dpt(rgb), atol=0, rtol=0)
    target = DeepMDDQueryDetector(config(channels)).eval()
    parent = transfer_encoder(path, target)
    for name, value in source.encoder.state_dict().items():
        torch.testing.assert_close(value, target.encoder.state_dict()[name], atol=0, rtol=0)
    recipe.update(parent=parent, model=asdict(target.config), manifest_sha256="fixture")
    query_path = tmp_path / "query.pt"
    save_coordinate_checkpoint(dict(schema="mdd_query_posttraining.v1", stage="query_posttraining",
        model_config=asdict(target.config), recipe=recipe, state_dict=target.state_dict()), query_path)
    loaded, _, _ = restore_coordinate_model(torch.load(query_path, weights_only=True))
    if channels == 2:
        historical = torch.load(query_path, weights_only=True)
        historical["model_config"].pop("input_channels")
        historical["recipe"]["model"].pop("input_channels")
        legacy_model, _, _ = restore_coordinate_model(historical)
        assert legacy_model.config == target.config
    loaded.eval()
    batch = dict(rgb=rgb, timestamps=times)
    with torch.no_grad():
        torch.testing.assert_close(predict_coordinates(loaded, batch, torch.device("cpu")), target(rgb, times), rtol=0, atol=0)
    with pytest.raises(ValueError, match="input_channels"):
        transfer_encoder(path, DeepMDDQueryDetector(config(7 - channels)))


def test_config_and_contract_cannot_disagree_or_reinterpret_old_weights(tmp_path: Path) -> None:
    source = MDDDPTDetector(config(2))
    saved: dict[str, Any] = dict(schema="mdd_dpt_pretraining.v2", model_config=asdict(source.config),
                 recipe=dict(input_contract=source.mdd.input_contract()))
    # Even an otherwise valid 2ch contract cannot fill in missing configuration.
    saved["model_config"].pop("input_channels")
    with pytest.raises(ValueError, match="Incomplete pretraining CNN configuration"):
        pretraining_config(saved)
    saved["model_config"]["input_channels"] = 5
    with pytest.raises(ValueError, match="input contract"):
        pretraining_config(saved)
    target = MDDDPTDetector(config())
    with pytest.raises(RuntimeError, match="size mismatch"):
        target.load_state_dict(source.state_dict(), strict=True)
    for value in (3, True, 5.0):
        with pytest.raises(ValueError, match="input_channels"):
            replace(config(), input_channels=value)


def test_encoder_and_query_boundary_reject_wrong_channel_or_time_contract() -> None:
    model = DeepMDDQueryDetector(config())
    with pytest.raises(RuntimeError, match="channels"):
        model.encoder(torch.zeros(1, 2, 32, 8, 8))
    rgb = torch.zeros(1, 32, 3, 8, 8, dtype=torch.uint8)
    for times in (torch.zeros(1, 32), torch.full((1, 32), float("nan"))):
        with pytest.raises(ValueError, match="timestamps|Timestamps"):
            predict_coordinates(model, dict(rgb=rgb, timestamps=times), torch.device("cpu"))
