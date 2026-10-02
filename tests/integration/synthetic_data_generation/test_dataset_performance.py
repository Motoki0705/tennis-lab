"""Cross-domain integration for the canonical dataset performance contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from src.synthetic_data_generation.configuration import ScenePipelineConfiguration
from src.synthetic_data_generation.pipeline import (
    SceneWorkspace,
    StageName,
)
from src.synthetic_data_generation.pipeline.application import build_stage_registry
from src.synthetic_data_generation.pipeline.publication import StagePublisher
from src.utils.paths import PROJECT_ROOT

_CONFIG_ROOT = PROJECT_ROOT / "src/synthetic_data_generation/configs"


def _runtime(tmp_path: Path) -> ScenePipelineConfiguration:
    data_root = tmp_path / "data"
    external_root = tmp_path / "third_party"
    data_root.mkdir()
    source_video = data_root / "synthetic_data_generation/raw/B00.mp4"
    source_video.parent.mkdir(parents=True)
    source_video.write_bytes(b"integration fixture")
    accad_root = data_root / "ACCAD"
    accad_root.mkdir()
    for category in ("running", "walking", "general"):
        np.savez(
            accad_root / f"{category}_poses.npz",
            poses=np.zeros((2, 156), dtype=np.float32),
            trans=np.zeros((2, 3), dtype=np.float32),
            betas=np.zeros(10, dtype=np.float32),
            gender=np.asarray("neutral"),
            mocap_framerate=np.asarray(30.0),
        )
    (data_root / "smplh").mkdir()
    checkpoint = (
        external_root
        / "dinov3/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"integration fixture")
    nht_config = external_root / "nht/configs/production.yaml"
    nht_config.parent.mkdir(parents=True)
    nht_config.write_text("schema: nht_pipeline_config_v1\n", encoding="utf-8")
    with initialize_config_dir(version_base="1.3", config_dir=str(_CONFIG_ROOT)):
        config = compose(
            config_name="run_scene_pipeline",
            overrides=[
                f"roots.data_root={data_root.as_posix()}",
                f"roots.external_asset_root={external_root.as_posix()}",
            ],
        )
    return ScenePipelineConfiguration.from_config(config)


def test_stale_partial_dataset_attempts_are_discarded_for_all_domains(
    tmp_path,
) -> None:
    workspace = SceneWorkspace(scene_id="B00", root=tmp_path / "B00")
    registry = build_stage_registry(_runtime(tmp_path))

    for stage in (StageName.COURT_DATASET,):
        publisher = StagePublisher(workspace, registry.definition(stage))
        publisher.staging.mkdir(parents=True)
        (publisher.staging / "partial.bin").write_bytes(b"partial")

        prepared = publisher.prepare()

        assert prepared.is_dir()
        assert not any(prepared.iterdir())
        assert not (publisher.owner / "dataset.json").exists()
        publisher.abandon()
        assert not publisher.staging.exists()
