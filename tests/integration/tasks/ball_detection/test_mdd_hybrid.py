from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import yaml
from torch import Tensor, nn

from src.tasks.ball_detection.checkpoints.hybrid import initialize_hybrid_from_query
from src.tasks.ball_detection.model_io.contracts import BallPrediction
from src.tasks.ball_detection.model_io.mdd_hybrid import (
    MDDHybridAdapter,
    build_mdd_hybrid_detector,
)
from src.tasks.ball_detection.model_io.mdd_query import MDDQueryInput
from src.tasks.ball_detection.model_io.predict import predict_coordinates
from src.tasks.ball_detection.models.mdd import (
    DeepMDDHybridDetector,
    DeepMDDQueryDetector,
    MDDDPTDetector,
    MDDHybridConfig,
    MDDHybridOutput,
    MDDPretrainConfig,
)
from src.tasks.ball_detection.training.dpt_pretrain.objective import heatmap_objective
from src.utils.data.heatmaps import generate_gaussian_heatmaps
from src.utils.paths import PROJECT_ROOT


def config() -> MDDHybridConfig:
    return MDDHybridConfig(
        stem_channels=(4, 4, 8, 8), mixed_channels=(8, 16), residual_blocks=(0,) * 6,
        decoder_channels=8, dim=16, heads=2, layers=1, ffn_dim=64, dropout=0.,
        rope_base=10000., activation_checkpointing=False, encoder_variant="convnext_v2",
        temporal_mixing="factorized", input_channels=5,
    )


def inputs(height: int = 65, width: int = 129) -> MDDQueryInput:
    return MDDQueryInput(torch.randint(256, (1, 32, 3, height, width), dtype=torch.uint8),
                         torch.arange(32)[None].float() / 30)


def test_query_broadcast_and_all_four_encoder_levels_reach_dpt() -> None:
    torch.set_num_threads(2)
    model = DeepMDDHybridDetector(config()).eval()
    source = inputs()
    captured: dict[str, Any] = {}

    def encoder_hook(module: nn.Module, args: tuple, output: tuple[Tensor, ...]) -> None:
        captured["encoder"] = output

    def query_hook(module: nn.Module, args: tuple, output: Tensor) -> None:
        captured["query"] = output

    def decoder_hook(module: nn.Module, args: tuple) -> None:
        captured["dpt"] = args[0]

    handles = [model.encoder.register_forward_hook(encoder_hook), model.blocks[-1].register_forward_hook(query_hook),
               model.decoder.register_forward_pre_hook(decoder_hook)]
    output = model(source.rgb, source.timestamps)
    for handle in handles:
        handle.remove()
    features = captured["encoder"]
    assert [f.shape[-2:] for f in features] == [(9, 17), (5, 9), (3, 5), (2, 3)]
    for level in range(3):
        torch.testing.assert_close(captured["dpt"][level], features[level].flatten(0, 1), rtol=0, atol=0)
    deepest = captured["dpt"][-1].reshape(1, 32, 32, 2, 3)
    torch.testing.assert_close(deepest[:, :, :16], features[-1], rtol=0, atol=0)
    torch.testing.assert_close(deepest[:, :, 16:], captured["query"][..., None, None].expand(-1, -1, -1, 2, 3), rtol=0, atol=0)
    assert model.decoder.encoder_channels == (8, 8, 8, 32)
    assert output.heatmap_logits.shape == (1, 32, 17, 33)
    assert output.presence_logits.shape == (1, 32)
    assert not hasattr(model, "head")  # No coordinate-regression head survives.


def test_query_changes_heatmaps_and_heatmap_loss_reaches_query_and_encoder() -> None:
    torch.set_num_threads(2)
    model = DeepMDDHybridDetector(config()).eval()
    source = inputs(33, 65)
    first = model(source.rgb, source.timestamps)
    with torch.no_grad():
        model.ball_query.add_(torch.linspace(-2, 2, model.config.dim))
    second = model(source.rgb, source.timestamps)
    assert not torch.allclose(first.heatmap_logits, second.heatmap_logits)
    assert not torch.allclose(first.presence_logits, second.presence_logits)
    second.heatmap_logits.square().mean().backward()
    for parameter in (model.ball_query, model.projection.weight,
                      model.blocks[0].temporal.qkv.weight, model.encoder.stages[0][0][0].weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model(source.rgb, source.timestamps).presence_logits.sum().backward()
    assert model.presence_head[1].weight.grad.abs().sum() > 0
    assert model.ball_query.grad.abs().sum() > 0


def test_absent_heatmap_and_presence_can_be_supervised_independently() -> None:
    torch.set_num_threads(2)
    model = DeepMDDHybridDetector(config())
    source = inputs(33, 49)
    out = model(source.rgb, source.timestamps)
    out.heatmap_logits.retain_grad()
    out.presence_logits.retain_grad()
    known = torch.ones(1, 32, dtype=torch.bool)
    known[:, 5] = False  # Unknown is not absence.
    batch = dict(uv=torch.full((1, 32, 2), float("nan")), position_valid=torch.zeros_like(known), heatmap_valid=known)
    loss, _ = heatmap_objective(out.heatmap_logits, batch)
    loss.backward(retain_graph=True)
    assert out.heatmap_logits.grad[:, 5].abs().sum() == 0
    assert (out.heatmap_logits.grad[:, known[0]] > 0).all()
    assert out.presence_logits.grad is None  # No training-time gate on the dense logits.
    presence_loss = torch.nn.functional.binary_cross_entropy_with_logits(out.presence_logits[known], torch.zeros_like(out.presence_logits[known]))
    presence_loss.backward()
    gradient = cast(Tensor, out.presence_logits.grad)
    assert gradient[:, 5].abs().sum() == 0
    assert (gradient[known] > 0).all()


@pytest.mark.parametrize("subpixel", [True, False])
def test_prediction_uses_native_grid_and_never_emits_absent_coordinates(subpixel: bool) -> None:
    uv = torch.tensor([.333, .621]).expand(1, 32, 2).clone()
    uv[:, 0], uv[:, 2] = 0., 1.
    heatmaps = generate_gaussian_heatmaps((16, 32), uv, .1).clamp(1e-6, 1 - 1e-6)
    presence = torch.full((1, 32), 9.)
    presence[:, 1::2] = -9.
    adapter = MDDHybridAdapter(config(), presence_threshold=.5, subpixel=subpixel)
    decoded = adapter.decode_output(MDDHybridOutput(torch.logit(heatmaps), presence))
    assert isinstance(decoded, BallPrediction)
    assert decoded.coords.device.type == "cpu" and decoded.heatmaps.dtype == torch.float32
    torch.testing.assert_close(decoded.heatmaps, heatmaps)
    assert decoded.present[0].tolist() == [i % 2 == 0 for i in range(32)]
    assert torch.isnan(decoded.coords[~decoded.present]).all()
    assert (decoded.confidence[~decoded.present] == 0).all()
    assert not decoded.candidates.valid[~decoded.present].any()
    assert not decoded.candidates.coords[~decoded.present].any()
    assert not decoded.candidates.scores[~decoded.present].any()
    assert not decoded.candidates.patch_valid[~decoded.present].any()
    assert not decoded.candidates.patches[~decoded.present].any()
    if subpixel:
        torch.testing.assert_close(decoded.coords[decoded.present], uv[decoded.present], atol=1e-6, rtol=0)
    else:
        expected = (uv * torch.tensor([31., 15.])).round() / torch.tensor([31., 15.])
        torch.testing.assert_close(decoded.coords[decoded.present], expected[decoded.present])
    # /4 grid uses endpoint normalization, not a naive multiply-by-four.
    pixels = decoded.coords[decoded.present] * torch.tensor([127., 63.])
    assert torch.isfinite(pixels).all() and len(pixels) == 16
    if subpixel:
        torch.testing.assert_close(pixels, uv[decoded.present] * torch.tensor([127., 63.]), atol=1e-4, rtol=0)


def test_bound_hybrid_graph_and_coordinate_only_consumer_rejection() -> None:
    torch.set_num_threads(2)
    pair = build_mdd_hybrid_detector(config(), presence_threshold=.75)
    model = pair.model.eval()
    source = inputs(33, 49)
    model.compile(backend="eager", fullgraph=True, dynamic=False)
    output = pair.run(source)
    assert output.heatmaps.shape == (1, 32, 9, 13) and output.present.shape == (1, 32)
    with pytest.raises(ValueError, match="hybrid presence"):
        predict_coordinates(model, dict(rgb=source.rgb, timestamps=source.timestamps), torch.device("cpu"))


def query_checkpoint(path: Path, *, cfg: MDDPretrainConfig, parent: Path) -> DeepMDDQueryDetector:
    model = DeepMDDQueryDetector(cfg)
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            p.fill_((i + 1) / 100)
    contract = model.mdd.input_contract()
    recipe = dict(model=asdict(cfg), input_contract=contract, image_decode={}, manifest_sha256="fixture",
                  parent=dict(checkpoint=str(parent), input_contract=contract, image_decode={}))
    torch.save(dict(schema="mdd_query_posttraining.v1", stage="query_posttraining", recipe=recipe,
                    model_config=asdict(cfg), state_dict=model.state_dict(), global_step=321), path)
    return model


def test_transfer_only_reads_stage_two_and_leaves_all_new_parameters_initialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config()
    dpt = MDDDPTDetector(MDDPretrainConfig(**asdict(cfg)))
    with torch.no_grad():
        for p in dpt.parameters():
            p.fill_(99.)
    stage_one = tmp_path / "stage-one.pt"
    torch.save(dpt.state_dict(), stage_one)
    path = tmp_path / "stage-two.pt"
    source = query_checkpoint(path, cfg=MDDPretrainConfig(**asdict(cfg)), parent=stage_one)
    torch.manual_seed(52)
    fresh = DeepMDDHybridDetector(cfg)
    load = torch.load
    opened = []

    def tracked_load(filename: Any, **kwargs: Any) -> Any:
        opened.append(filename)
        assert filename == path
        return load(filename, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    torch.manual_seed(52)
    target, receipt = initialize_hybrid_from_query(path, cfg)
    assert opened == [path]
    assert receipt["transferred_modules"] == ["encoder", "projection", "spatial_position", "ball_query", "blocks"]
    assert receipt["initialized_modules"] == ["decoder", "heatmap_head", "presence_head"]
    assert receipt["discarded_modules"] == ["head"] and receipt["global_step"] == 321
    for name in receipt["transferred_parameters"]:
        torch.testing.assert_close(target.state_dict()[name], source.state_dict()[name], rtol=0, atol=0)
    for name in receipt["initialized_parameters"]:
        torch.testing.assert_close(target.state_dict()[name], fresh.state_dict()[name], rtol=0, atol=0)
    assert set(receipt["transferred_parameters"]) | set(receipt["initialized_parameters"]) == set(target.state_dict())
    assert all(p.requires_grad for p in target.parameters())


@pytest.mark.parametrize("violation", ["two_channels", "historical_two_channels", "missing_channels", "contract", "heads", "missing_weight", "extra_dpt", "shape", "stage_one"])
def test_transfer_rejects_incompatible_sources(tmp_path: Path, violation: str) -> None:
    cfg = MDDPretrainConfig(**asdict(config()))
    if violation in {"two_channels", "historical_two_channels"}:
        cfg = replace(cfg, input_channels=2)
    if violation == "heads":
        cfg = replace(cfg, heads=4)
    path = tmp_path / "query.pt"
    query_checkpoint(path, cfg=cfg, parent=tmp_path / "unused-parent.pt")
    saved = torch.load(path, weights_only=True)
    if violation in {"historical_two_channels", "missing_channels"}:
        saved["model_config"].pop("input_channels")
        saved["recipe"]["model"].pop("input_channels")
    elif violation == "contract":
        saved["recipe"]["input_contract"]["rgb_normalization"] = "imagenet"
    elif violation == "missing_weight":
        del saved["state_dict"]["projection.weight"]
    elif violation == "extra_dpt":
        saved["state_dict"]["decoder.projections.0.0.weight"] = torch.ones(1)
    elif violation == "shape":
        saved["state_dict"]["projection.weight"] = torch.ones(1)
    elif violation == "stage_one":
        saved.update(schema="mdd_dpt_pretraining.v2", stage="heatmap_pretraining")
    torch.save(saved, path)
    with pytest.raises(ValueError, match="contract|mismatch|checkpoint|exactly"):
        initialize_hybrid_from_query(path, config())


@pytest.mark.parametrize("threshold", [0., 1., float("nan"), float("inf"), True])
def test_presence_threshold_must_be_explicit_and_valid(threshold: float) -> None:
    with pytest.raises(ValueError, match="presence_threshold"):
        MDDHybridAdapter(config(), presence_threshold=threshold)


@pytest.mark.parametrize("violation", ["frames", "presence_shape", "nan", "dtype"])
def test_rejects_invalid_hybrid_logits(violation: str) -> None:
    heatmap = torch.zeros(1, 32, 4, 4)
    presence = torch.zeros(1, 32)
    if violation == "frames":
        heatmap = heatmap[:, :16]
    elif violation == "presence_shape":
        presence = presence[..., None]
    elif violation == "nan":
        presence[:, 0] = float("nan")
    elif violation == "dtype":
        heatmap = heatmap.long()
    with pytest.raises(ValueError, match="Hybrid"):
        MDDHybridAdapter(config(), presence_threshold=.5).decode_output(MDDHybridOutput(heatmap, presence))


def test_hybrid_config_is_explicit_and_separate_from_pretraining(tmp_path: Path) -> None:
    shipped = PROJECT_ROOT / "src/tasks/ball_detection/configs/model/mdd_hybrid.yaml"
    cfg = MDDHybridConfig.load(shipped)
    assert cfg.input_channels == 5 and cfg.frames == 32
    with pytest.raises(ValueError):
        replace(cfg, input_channels=2)
    with pytest.raises(ValueError):
        replace(cfg, decoder_channels=1)
    with pytest.raises(ValueError, match="explicit"):
        DeepMDDHybridDetector(cast(Any, MDDPretrainConfig(**asdict(cfg))))
    raw = dict(name="mdd_hybrid", **asdict(cfg))
    del raw["input_channels"]
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="every field"):
        MDDHybridConfig.load(path)
