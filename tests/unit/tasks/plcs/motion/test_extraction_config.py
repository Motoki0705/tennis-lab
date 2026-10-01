"""Dataset portability and strict Hydra-owned extraction settings."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from src.tasks.plcs.motion.extraction_config import ExtractionConfig
from src.tasks.plcs.motion.gvhmr_extraction import load_model_runtime
from src.tasks.plcs.motion.reproducibility import publish_run

ROOT = Path(__file__).parents[5]


def test_gvhmr_and_dino_use_the_same_explicit_checkpoint_root(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / 'published'
    dino = checkpoint_root / 'dino/checkpoint0029_4scale_swin.pth'
    dino.parent.mkdir(parents=True)
    dino.touch()
    runtime = load_model_runtime(ROOT / 'src/submodules/configs/demo_gvhmr.yaml',
        repository_root=tmp_path, checkpoint_root=checkpoint_root)
    assert runtime.assets.dino_checkpoint == dino
    assert runtime.assets.gvhmr_checkpoint == checkpoint_root / 'gvhmr/gvhmr_siga24_release.ckpt'
    assert runtime.assets.body_models_dir == checkpoint_root / 'body_models'


def test_dataset_selection_is_required() -> None:
    with (
        initialize_config_dir(
            config_dir=str(ROOT / "src/tasks/plcs/configs"), version_base="1.3"
        ),
        pytest.raises(ConfigCompositionException),
    ):
        compose(config_name="extract_gvhmr_motions")


def test_other_dataset_and_runtime_override_without_script_changes(
    tmp_path: Path,
) -> None:
    (tmp_path / "input").mkdir()
    selection = tmp_path / "selection.yaml"
    selection.write_text(
        "schema_version: plcs_gvhmr_selection_v1\ndataset_id: other\ncameras:\n  - camera_id: side\n    player_role: player\n    footpoint_polygon_normalized: [[0,0], [1,0], [1,1]]\n"
    )
    with initialize_config_dir(
        config_dir=str(ROOT / "src/tasks/plcs/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="extract_gvhmr_motions", overrides=["dataset=meiji_3cam"]
        )
    cfg.paths.project_root = str(tmp_path)
    cfg.paths.data_root = str(tmp_path)
    cfg.dataset.name = "other"
    cfg.dataset.root = "input"
    cfg.dataset.selection_config = "selection.yaml"
    cfg.run.output_dir = "motions"
    cfg.run.camera_ids = ["side"]
    cfg.models.runtime_overrides = {"static_cam": False}
    with patch("src.tasks.plcs.motion.extraction_config.load_model_runtime") as load:
        validated = ExtractionConfig.from_config(cfg)
        assert validated.dataset_root == tmp_path / "input"
        assert validated.selection.dataset_id == "other"
        assert load.call_args.kwargs["runtime_overrides"] == {"static_cam": False}
        assert load.call_args.kwargs["checkpoint_root"] == tmp_path / "ckpt"
        assert load.call_args.kwargs["dino_checkpoint"] == tmp_path / "ckpt/dino/checkpoint0029_4scale_swin.pth"
        publish_run(
            tmp_path / "snapshot", {"sha256": "fixture", "config": validated.resolved}
        )
        replay = ExtractionConfig.from_config(
            OmegaConf.load(tmp_path / "snapshot/config.yaml")
        )
        assert replay.dataset_root == validated.dataset_root
        assert replay.selection == validated.selection
        cfg.run.seed = True
        with pytest.raises((TypeError, ValueError)):
            ExtractionConfig.from_config(cfg)


def test_accad_profile_is_named_explicitly() -> None:
    config_dir = ROOT / "src/tasks/plcs/configs"
    assert not (config_dir / "motion_sources/default.yaml").exists()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(
            config_name="generate_dataset", overrides=["motion_sources=accad"]
        )
    assert set(OmegaConf.to_container(cfg.motion_sources)) == {
        "running",
        "walking",
        "general",
    }
