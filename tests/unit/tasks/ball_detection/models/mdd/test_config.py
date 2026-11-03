from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.tasks.ball_detection.models.mdd import MDDHybridConfig, MDDPretrainConfig
from src.utils.paths import PROJECT_ROOT


@pytest.mark.parametrize("name,channels", [
    ("mdd_dpt_pretrain", 2), ("mdd_dpt_convnext_v2", 2),
    ("mdd_dpt_fasternet", 2), ("mdd_dpt_rgb_mdd", 5),
])
def test_shipped_configs_require_explicit_input_channels(
    tmp_path: Path, name: str, channels: int,
) -> None:
    path = PROJECT_ROOT / "src/tasks/ball_detection/configs/model" / f"{name}.yaml"
    config = MDDPretrainConfig.load(path)
    assert config.input_channels == channels
    raw = asdict(config)
    assert config.frames == 32 and config.frame_chunk_size is None
    assert "frames" not in raw and "frame_chunk_size" not in raw
    raw.pop("input_channels")
    with pytest.raises(TypeError, match="input_channels"):
        MDDPretrainConfig(**raw)
    missing = tmp_path / "missing.yaml"
    missing.write_text(yaml.safe_dump(dict(name="mdd_dpt_pretrain", **raw)))
    with pytest.raises(ValueError, match="every field"):
        MDDPretrainConfig.load(missing)


def test_hybrid_requires_explicit_input_channels(tmp_path: Path) -> None:
    path = PROJECT_ROOT / "src/tasks/ball_detection/configs/model/mdd_hybrid.yaml"
    config = MDDHybridConfig.load(path)
    assert config.input_channels == 5
    raw = asdict(config)
    assert config.frames == 32 and config.frame_chunk_size is None
    assert "frames" not in raw and "frame_chunk_size" not in raw
    raw.pop("input_channels")
    with pytest.raises(TypeError, match="input_channels"):
        MDDHybridConfig(**raw)
    missing = tmp_path / "missing.yaml"
    missing.write_text(yaml.safe_dump(dict(name="mdd_hybrid", **raw)))
    with pytest.raises(ValueError, match="every field"):
        MDDHybridConfig.load(missing)


@pytest.mark.parametrize("filename,config_type", [
    ("mdd_dpt_heatmap.yaml", "dpt"), ("mdd_heatmap_hybrid.yaml", "hybrid"),
    ("mdd_temporal_query.yaml", "query"),
])
def test_temporal_configs_keep_widths_and_require_explicit_saved_fields(tmp_path: Path, filename: str, config_type: str) -> None:
    from src.tasks.ball_detection.models.mdd import (
        MDDHeatmapConfig,
        MDDHeatmapHybridConfig,
        MDDTemporalQueryConfig,
    )

    cls = {"dpt": MDDHeatmapConfig, "hybrid": MDDHeatmapHybridConfig, "query": MDDTemporalQueryConfig}[config_type]
    root = PROJECT_ROOT / "src/tasks/ball_detection/configs/model"
    config = cls.load(root / filename)
    legacy = MDDPretrainConfig.load(root / "mdd_dpt_rgb_mdd.yaml")
    raw = asdict(config)
    assert raw.pop("frames") == 32 and raw.pop("frame_chunk_size") is None
    assert raw == asdict(legacy)
    assert config.dim == 256 and config.layers == 4 and config.heads == 8 and config.ffn_dim == 704
    assert config.decoder_channels == 128 and config.encoder_variant == "convnext_v2" and config.temporal_mixing == "factorized"
    assert not config.activation_checkpointing
    for missing in asdict(config):
        constructor_fields = asdict(config)
        constructor_fields.pop(missing)
        with pytest.raises(TypeError, match=missing):
            cls(**constructor_fields)
        raw = yaml.safe_load((root / filename).read_text())
        raw.pop(missing)
        path = tmp_path / "incomplete.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="every field"):
            cls.load(path)


@pytest.mark.parametrize("field,value", [
    ("frames", 0), ("frames", -1), ("frames", 2.5), ("frames", True),
    ("frame_chunk_size", 0), ("frame_chunk_size", -1), ("frame_chunk_size", 2.5), ("frame_chunk_size", True),
    ("activation_checkpointing", 1), ("input_channels", 2), ("decoder_channels", 1),
])
def test_temporal_heatmap_configs_reject_invalid_memory_and_length_settings(field: str, value: object) -> None:
    from dataclasses import replace

    from src.tasks.ball_detection.models.mdd import MDDHeatmapConfig

    config = MDDHeatmapConfig.load(PROJECT_ROOT / "src/tasks/ball_detection/configs/model/mdd_dpt_heatmap.yaml")
    changes: dict[str, Any] = {field: value}
    with pytest.raises(ValueError):
        replace(config, **changes)


def test_shipped_mhc_config_is_explicit_and_every_field_is_required(tmp_path: Path) -> None:
    from src.tasks.ball_detection.models.mdd import MDDMHCQueryConfig
    from src.tasks.ball_detection.training.temporal.configuration import (
        load_model_config,
    )

    path = PROJECT_ROOT / "src/tasks/ball_detection/configs/model/mdd_mhc_query.yaml"
    config = MDDMHCQueryConfig.load(path)
    assert load_model_config(path, "query") == config
    assert (*config.stem_channels, *config.mixed_channels) == (16, 32, 64, 128, 256, 512)
    assert config.residual_blocks == (0, 1, 2, 3, 3, 2)
    assert config.frames == 64 and config.heads * 64 == config.dim
    assert config.dropout == .1 and config.rope_base == 10000.
    assert config.ffn_dim == 704 and config.num_streams == 4
    assert (config.mhc_coefficient_dim, config.mhc_sinkhorn_iters, config.mhc_eps,
            config.mhc_residual_identity_bias, config.mhc_update_scale_init) == (64, 20, 1e-6, 4., 0.)
    assert config.frame_chunk_size is None and not config.activation_checkpointing
    for missing in asdict(config):
        fields = asdict(config)
        fields.pop(missing)
        with pytest.raises(TypeError, match=missing):
            MDDMHCQueryConfig(**fields)
        raw = dict(name="mdd_mhc_query", **fields)
        incomplete = tmp_path / "incomplete.yaml"
        incomplete.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="field"):
            load_model_config(incomplete, "query")
    with pytest.raises(ValueError, match="explicitly selected"):
        load_model_config(path, "dpt")
    for name in ([], {}):
        malformed = tmp_path / "malformed.yaml"
        malformed.write_text(yaml.safe_dump(dict(name=name, **asdict(config))))
        with pytest.raises(ValueError, match="explicitly selected"):
            load_model_config(malformed, "query")


@pytest.mark.parametrize("fields", [
    {"heads": 2}, {"num_streams": 3}, {"encoder_variant": "convnext_v2", "temporal_mixing": "factorized"},
    {"temporal_mixing": "dense3d"}, {"residual_blocks": (1, 1, 1, 1, 1, 1)},
    {"mhc_eps": float("nan")}, {"mhc_eps": 0.}, {"mhc_sinkhorn_iters": 0},
    {"mhc_coefficient_dim": True}, {"mhc_update_scale_init": float("inf")},
    {"mhc_residual_identity_bias": -1.}, {"ffn_dim": 128},
])
def test_invalid_mhc_query_topology_or_numerics_are_rejected(fields: dict[str, Any]) -> None:
    from tests.support.tasks.ball_detection.temporal import small_mhc_model

    with pytest.raises(ValueError):
        small_mhc_model(**fields)
