from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from src.tasks.ball_detection.model_io.mdd_heatmap import (
    DPT_HEATMAP_KIND,
    HEATMAP_MODEL_SCHEMA,
    HYBRID_HEATMAP_KIND,
    MDDHeatmapAdapter,
    build_mdd_heatmap_detector,
    heatmap_model_payload,
    load_mdd_heatmap_detector,
    restore_heatmap_model,
)
from src.tasks.ball_detection.model_io.mdd_hybrid import BallHybridPrediction
from src.tasks.ball_detection.model_io.mdd_query import MDDQueryInput
from src.tasks.ball_detection.model_io.predict import predict_dpt_heatmaps
from src.tasks.ball_detection.models.mdd import (
    DeepMDDHeatmapHybridDetector,
    DeepMDDHybridDetector,
    MDDDPTDetector,
    MDDHeatmapConfig,
    MDDHeatmapHybridConfig,
    MDDHybridConfig,
)
from src.tasks.ball_detection.preprocessing.rgb_mdd import RGBToRGBMDD
from src.tasks.ball_detection.training.heatmap_objective import heatmap_objective
from tests.integration.tasks.ball_detection.test_rgb_mdd_contract import (
    config as legacy_config,
)


def config(kind: str, *, frames: int, frame_chunk_size: int | None, **kwargs: Any) -> MDDHeatmapConfig:
    cls = MDDHeatmapHybridConfig if kind == "hybrid" else MDDHeatmapConfig
    return cls(**dict(asdict(legacy_config()), frames=frames, frame_chunk_size=frame_chunk_size, **kwargs))


def inputs(frames: int, batch: int = 1, height: int = 17, width: int = 25) -> MDDQueryInput:
    # Nonuniform real PTS exercise RoPE without synthesizing fixed-FPS timing.
    times = torch.linspace(.01, .05, frames).cumsum(0)[None].expand(batch, -1).contiguous()
    return MDDQueryInput(torch.randint(256, (batch, frames, 3, height, width), dtype=torch.uint8), times)


@pytest.mark.parametrize("kind", ["dpt", "hybrid"])
def test_same_all_weights_load_strictly_and_forward_at_each_temporal_length(kind: str) -> None:
    torch.set_num_threads(1)
    pair = build_mdd_heatmap_detector(config(kind, frames=32, frame_chunk_size=None))
    state = pair.model.state_dict()
    assert not any("presence" in name for name in state)
    for frames in (1, 3, 32, 64, 128, 256):
        target = build_mdd_heatmap_detector(config(kind, frames=frames, frame_chunk_size=None))
        result = target.model.load_state_dict(state, strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        source = inputs(frames)
        target.model.eval()
        with torch.no_grad():
            call = target.adapter.build_call(source)
            logits = target.model(*call.args)
            prediction = target.adapter.decode_output(logits)
        assert logits.shape == prediction.heatmaps.shape == (1, frames, 5, 7)
        assert prediction.coords.shape == (1, frames, 2)
        assert not isinstance(prediction, BallHybridPrediction)
        assert not hasattr(target.model, "presence_head") and not hasattr(prediction, "present")
        assert target.model.mdd.input_contract()["frames"] == frames
        if kind == "dpt":
            torch.testing.assert_close(predict_dpt_heatmaps(target.model, source.rgb), logits, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["dpt", "hybrid"])
@pytest.mark.parametrize("mixing", ["dense3d", "factorized"])
@pytest.mark.parametrize("chunk_size,checkpointing", [(None, True), (3, False), (3, True)])
def test_memory_controls_preserve_full_model_outputs_and_heatmap_gradients(kind: str, mixing: str, chunk_size: int | None, checkpointing: bool) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = config(kind, frames=9, frame_chunk_size=None, temporal_mixing=mixing, dropout=.2)
    full = build_mdd_heatmap_detector(cfg)
    optimized = build_mdd_heatmap_detector(replace(cfg, frame_chunk_size=chunk_size, activation_checkpointing=checkpointing))
    optimized.model.load_state_dict(full.model.state_dict(), strict=True)
    source = inputs(9, batch=2, height=33, width=49)
    mask = torch.zeros(2, 9, dtype=torch.bool)
    mask[:, 3:6] = True
    position = mask.clone()
    position[:, 5] = False  # Confirmed negative, supervised by the same heatmap loss.
    labels = dict(uv=torch.full((2, 9, 2), .45).masked_fill(~position[..., None], float("nan")), position_valid=position, heatmap_valid=mask)
    results: list[Tensor] = []
    for pair in (full, optimized):
        pair.model.train()
        call = pair.adapter.build_call(source)
        torch.manual_seed(51)  # Checkpoint recomputation must preserve query dropout RNG.
        logits = pair.model(*call.args)
        logits.retain_grad()
        loss, _ = heatmap_objective(logits, labels)
        loss.backward()
        assert logits.grad[~mask].abs().sum() == 0
        results.append(logits.detach())
    torch.testing.assert_close(results[0], results[1], rtol=2e-5, atol=2e-6)
    for (name, first), (other, second) in zip(full.model.named_parameters(), optimized.model.named_parameters(), strict=True):
        assert name == other
        if first.grad is None:
            assert second.grad is None  # DPT's retained unused deepest skip.
        else:
            assert second.grad is not None
            torch.testing.assert_close(first.grad, second.grad, rtol=3e-4, atol=2e-6, msg=name)
    for parameter in (optimized.model.encoder.temporal[0].conv.weight, optimized.model.encoder.stages[0][0][0].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    if isinstance(optimized.model, DeepMDDHeatmapHybridDetector):
        assert optimized.model.ball_query.grad.abs().sum() > 0
        assert optimized.model.blocks[0].temporal.qkv.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["dpt", "hybrid"])
def test_model_artifact_roundtrip_keeps_explicit_length_contract_and_all_weights(tmp_path: Path, kind: str) -> None:
    torch.set_num_threads(1)
    pair = build_mdd_heatmap_detector(config(kind, frames=64, frame_chunk_size=11))
    saved = heatmap_model_payload(pair.model)
    assert saved["schema"] == HEATMAP_MODEL_SCHEMA
    assert saved["model_kind"] == (HYBRID_HEATMAP_KIND if kind == "hybrid" else DPT_HEATMAP_KIND)
    assert saved["model_config"]["frames"] == saved["input_contract"]["frames"] == 64
    path = tmp_path / "heatmap.pt"
    torch.save(saved, path)
    loaded = load_mdd_heatmap_detector(path)
    assert loaded.model.config == pair.model.config
    pair.model.eval()
    source = inputs(64)
    with torch.no_grad():
        expected, actual = pair.run(source), loaded.run(source)
    torch.testing.assert_close(actual.heatmaps, expected.heatmaps, rtol=0, atol=0)
    for key, value in pair.model.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("damage", ["missing_frames", "missing_chunk_size", "changed_frames", "changed_contract", "old_schema", "old_kind", "wrong_architecture", "extra_presence", "missing_weight"])
def test_artifacts_never_guess_length_or_reinterpret_presence_models(damage: str) -> None:
    saved = deepcopy(heatmap_model_payload(build_mdd_heatmap_detector(config("hybrid", frames=64, frame_chunk_size=None)).model))
    if damage == "missing_frames":
        saved["model_config"].pop("frames")
    elif damage == "missing_chunk_size":
        saved["model_config"].pop("frame_chunk_size")
    elif damage == "changed_frames":
        saved["model_config"]["frames"] = 128
    elif damage == "changed_contract":
        saved["input_contract"]["frames"] = 32
    elif damage == "old_schema":
        saved["schema"] = "mdd_hybrid_training.v1"
    elif damage == "old_kind":
        saved["model_kind"] = "mdd_hybrid"
    elif damage == "wrong_architecture":
        saved["model_kind"] = DPT_HEATMAP_KIND
    elif damage == "extra_presence":
        saved["state_dict"]["presence_head.1.weight"] = torch.zeros(1, 16)
    else:
        saved["state_dict"].pop("ball_query")
    with pytest.raises((ValueError, RuntimeError)):
        restore_heatmap_model(saved)


def test_legacy_presence_hybrid_state_and_configuration_stay_separate() -> None:
    torch.set_num_threads(1)
    cfg = MDDHybridConfig(**asdict(legacy_config()))
    legacy = DeepMDDHybridDetector(cfg)
    assert "frames" not in asdict(cfg) and "frame_chunk_size" not in asdict(cfg)
    assert legacy.mdd.input_contract() == RGBToRGBMDD().input_contract()
    with pytest.raises(ValueError, match="explicit"):
        heatmap_model_payload(legacy)
    with pytest.raises(ValueError, match="explicit"):
        MDDHeatmapAdapter(cast(Any, cfg))
    pair = build_mdd_heatmap_detector(config("hybrid", frames=32, frame_chunk_size=None))
    with pytest.raises(RuntimeError, match="presence_head"):
        pair.model.load_state_dict(legacy.state_dict(), strict=True)


@pytest.mark.parametrize("kind", ["dpt", "hybrid"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_typed_heatmap_call_compiles_with_memory_controls(kind: str, checkpointing: bool) -> None:
    torch.set_num_threads(1)
    pair = build_mdd_heatmap_detector(config(kind, frames=7, frame_chunk_size=3, activation_checkpointing=checkpointing))
    source = inputs(7)
    call = pair.adapter.build_call(source)
    expected = pair.model(*call.args)
    pair.model.compile(backend="eager", fullgraph=True, dynamic=False)
    actual = pair.model(*call.args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert next(pair.model.encoder.parameters()).grad is not None


def test_io_rejects_wrong_rgb_length_timestamps_model_pair_and_output() -> None:
    pair = build_mdd_heatmap_detector(config("hybrid", frames=64, frame_chunk_size=None))
    source = inputs(64)
    with pytest.raises(ValueError, match="64 frames"):
        pair.adapter.build_call(inputs(32))
    with pytest.raises(ValueError, match="increase"):
        pair.adapter.build_call(MDDQueryInput(source.rgb, torch.zeros_like(source.timestamps)))
    with pytest.raises(ValueError, match="mismatch"):
        pair.adapter.validate_model_pair(MDDDPTDetector(config("dpt", frames=64, frame_chunk_size=None)))
    for out in (torch.zeros(1, 32, 5, 7), torch.zeros(1, 64, 5, 7, dtype=torch.long), torch.full((1, 64, 5, 7), float("nan"))):
        with pytest.raises(ValueError, match="64"):
            pair.adapter.decode_output(out)
    # Weak heatmaps still expose raw coordinates and scores, with no presence gate.
    decoded = pair.adapter.decode_output(torch.full((1, 64, 5, 7), -9.))
    assert torch.isfinite(decoded.coords).all() and (decoded.confidence > 0).all()
