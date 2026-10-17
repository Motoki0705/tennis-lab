from dataclasses import asdict
from pathlib import Path

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
