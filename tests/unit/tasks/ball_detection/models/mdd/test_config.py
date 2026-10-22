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
    raw.pop("input_channels")
    with pytest.raises(TypeError, match="input_channels"):
        MDDHybridConfig(**raw)
    missing = tmp_path / "missing.yaml"
    missing.write_text(yaml.safe_dump(dict(name="mdd_hybrid", **raw)))
    with pytest.raises(ValueError, match="every field"):
        MDDHybridConfig.load(missing)


@pytest.mark.parametrize("filename,config_type", [
    ("mdd_dpt_heatmap.yaml", "dpt"), ("mdd_heatmap_hybrid.yaml", "hybrid"),
])
def test_temporal_heatmap_configs_keep_widths_and_require_explicit_saved_fields(tmp_path: Path, filename: str, config_type: str) -> None:
    from src.tasks.ball_detection.models.mdd import (
        MDDHeatmapConfig,
        MDDHeatmapHybridConfig,
    )

    cls = MDDHeatmapConfig if config_type == "dpt" else MDDHeatmapHybridConfig
    root = PROJECT_ROOT / "src/tasks/ball_detection/configs/model"
    config = cls.load(root / filename)
    legacy = MDDPretrainConfig.load(root / "mdd_dpt_rgb_mdd.yaml")
    raw = asdict(config)
    assert raw.pop("frames") == 32 and raw.pop("frame_chunk_size") is None
    assert raw == asdict(legacy)
    assert config.dim == 256 and config.layers == 4 and config.heads == 8 and config.ffn_dim == 704
    assert config.decoder_channels == 128 and config.encoder_variant == "convnext_v2" and config.temporal_mixing == "factorized"
    assert not config.activation_checkpointing
    for missing in ("frames", "frame_chunk_size", "input_channels"):
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
