"""Temporary roots for typed configuration tests; no model/video is decoded."""

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from src.utils.paths import PROJECT_ROOT


def compose_scene_config(root: Path, *overrides: str) -> DictConfig:
    root = root.resolve()
    config_path = root / "third_party/nht/configs/production.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("schema: nht_pipeline_config_v1\n", encoding="utf-8")
    source = root / "data/synthetic_data_generation/raw/B00.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"configuration fixture; never decoded")
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(PROJECT_ROOT / "src/synthetic_data_generation/configs"),
    ):
        return compose(
            config_name="run_scene_pipeline",
            overrides=[f"roots.project_root={root.as_posix()}", *overrides],
        )
