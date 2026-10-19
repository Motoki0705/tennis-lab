"""Pipeline and visualization share canonical checkpoint asset roots."""

from pathlib import Path

import pytest

from src.submodules.smpl_assets import SMPL_ASSET_SOURCES
from src.tennis_scene.configuration import (
    PipelineRuntimeConfig,
    parse_visualization_config,
)
from src.tennis_scene.pipeline.contracts import ClipSource, SourceVideo
from src.tennis_scene.pipeline.definition import (
    enabled_model_assets,
    standard_definition,
)
from tests.unit.tennis_scene.test_configuration import _composed, _pipeline_config


def test_people_weights_and_body_models_follow_checkpoint_root(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / 'published'
    cfg = PipelineRuntimeConfig.from_config(
        _pipeline_config(tmp_path, f'paths.checkpoint_root={checkpoint_root}', f'paths.external_asset_root={tmp_path / "source"}'),
        bind_inputs=False,
    )
    expected = {
        'dino_checkpoint': 'dino/checkpoint0029_4scale_swin.pth',
        'vitpose_checkpoint': 'vitpose/vitpose-h-multi-coco.pth',
        'hmr2_checkpoint': 'hmr2/epoch=10-step=25000.ckpt',
        'gvhmr_checkpoint': 'gvhmr/gvhmr_siga24_release.ckpt',
        'body_models_dir': 'body_models',
    }
    for name, relative in expected.items():
        assert getattr(cfg.people, name) == checkpoint_root / relative
    assert cfg.people.dino_repository == tmp_path / 'source/DINO'
    assert cfg.tracking_encoder_weights == cfg.association_encoder_weights
    assert cfg.tracking_encoder_weights.is_relative_to(checkpoint_root)
    assert cfg.aflink_checkpoint.is_relative_to(checkpoint_root)
    for name, source in SMPL_ASSET_SOURCES.items():
        assert getattr(cfg.people.smpl_assets, name) == checkpoint_root / source.checkpoint_path


def test_pipeline_and_visualization_share_body_model_and_smpl_regressor(tmp_path: Path) -> None:
    pipeline = PipelineRuntimeConfig.from_config(_pipeline_config(tmp_path), bind_inputs=False)
    with _composed('visualization', [f'paths.project_root={tmp_path}']) as value:
        visualization = parse_visualization_config(value)
    expected = tmp_path / 'ckpt/body_models/smplh/neutral/model.npz'
    assert pipeline.people.body_models_dir / 'smplh/neutral/model.npz' == visualization.smpl_faces_path == expected
    assert pipeline.people.smpl_assets.smpl_neutral_joint_regressor == visualization.smpl_joint_regressor_path
    assert pipeline.people.body_models_dir == tmp_path / 'ckpt/body_models'


@pytest.mark.parametrize("missing_name", SMPL_ASSET_SOURCES)
def test_standard_pipeline_reports_acquisition_before_model_loading(tmp_path: Path, missing_name: str) -> None:
    cfg = PipelineRuntimeConfig.from_config(_pipeline_config(tmp_path, "device=cpu"), bind_inputs=False)
    missing = getattr(cfg.people.smpl_assets, missing_name)
    for path in enabled_model_assets(cfg).values():
        assert path.is_relative_to(tmp_path)
        if path != missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    source = ClipSource("test", (SourceVideo("cam0", tmp_path / "input.mp4", "test", 1, 30.0, 640, 480),))
    with pytest.raises(FileNotFoundError) as error:
        standard_definition(cfg, source, code_identity="test")
    assert str(missing) in str(error.value)
    assert SMPL_ASSET_SOURCES[missing_name].source_url in str(error.value)
