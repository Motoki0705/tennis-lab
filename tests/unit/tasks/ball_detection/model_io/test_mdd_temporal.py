"""New query artifacts are explicit; original T is validated before transfer."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.tasks.ball_detection.model_io.mdd_temporal import (
    MHC_QUERY_SCHEMA,
    QUERY_SCHEMA,
    build_temporal_model,
    restore_temporal_model,
    temporal_forward,
    temporal_model_payload,
)
from src.tasks.ball_detection.models.mdd import (
    DeepMDDMHCQueryDetector,
    DeepMDDQueryDetector,
)
from tests.support.tasks.ball_detection.temporal import small_mhc_model


def test_mhc_saved_model_roundtrip_and_strict_t32_t64_t128_state_transfer(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(95)
    config = small_mhc_model(frames=32, dropout=0.0)
    model = build_temporal_model(config).eval()
    rgb = torch.randint(256, (1, 32, 3, 17, 25), dtype=torch.uint8)
    times = torch.arange(32)[None].float() / 30
    expected = temporal_forward(model, rgb, times)
    path = tmp_path / "model.pt"
    torch.save(temporal_model_payload(model), path)
    saved = torch.load(path, weights_only=True)
    assert saved["schema"] == MHC_QUERY_SCHEMA and saved["model_kind"] == "mdd_mhc_query"
    restored = restore_temporal_model(saved).eval()
    assert isinstance(restored, DeepMDDMHCQueryDetector) and isinstance(restored, DeepMDDQueryDetector)
    assert restored.config == config
    torch.testing.assert_close(temporal_forward(restored, rgb, times), expected, rtol=0, atol=0)
    for frames in (64, 128):
        target = build_temporal_model(replace(config, frames=frames))
        target.load_state_dict(restored.state_dict(), strict=True)
        assert target.config.frames == frames
        assert target.mdd.input_contract()["frames"] == frames
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    saved["model_config"]["frames"] = 64
    with pytest.raises(ValueError, match="contract"):
        restore_temporal_model(saved)
    with pytest.raises(ValueError):
        temporal_forward(restored, rgb[:, :16], times[:, :16])


@pytest.mark.parametrize("violation", ["old_schema", "old_kind", "missing_field", "unknown_field", "missing_state", "wrong_shape"])
def test_mhc_artifacts_never_silently_load_as_an_old_model(violation: str) -> None:
    saved = temporal_model_payload(build_temporal_model(small_mhc_model()))
    if violation == "old_schema":
        saved["schema"] = QUERY_SCHEMA
    elif violation == "old_kind":
        saved["model_kind"] = "mdd_temporal_query"
    elif violation == "missing_field":
        saved["model_config"].pop("mhc_eps")
    elif violation == "unknown_field":
        saved["model_config"]["unrecognized"] = 1
    elif violation == "missing_state":
        saved["state_dict"].pop("ball_query")
    else:
        saved["state_dict"]["ball_query"] = torch.zeros(1, 64)
    with pytest.raises((ValueError, RuntimeError)):
        restore_temporal_model(saved)


def test_mhc_configuration_cannot_instantiate_a_legacy_single_query_detector() -> None:
    with pytest.raises(ValueError, match="mHC"):
        DeepMDDQueryDetector(small_mhc_model())
