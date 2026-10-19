"""Pipeline and visualization share canonical checkpoint asset roots."""

from pathlib import Path

from src.tennis_scene.configuration import (
    PipelineRuntimeConfig,
    parse_visualization_config,
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


def test_pipeline_and_visualization_share_body_model_and_bundled_regressor(tmp_path: Path) -> None:
    pipeline = PipelineRuntimeConfig.from_config(_pipeline_config(tmp_path), bind_inputs=False)
    with _composed('visualization', [f'paths.project_root={tmp_path}']) as value:
        visualization = parse_visualization_config(value)
    expected = tmp_path / 'ckpt/body_models/smplh/neutral/model.npz'
    assert pipeline.people.body_models_dir / 'smplh/neutral/model.npz' == visualization.smpl_faces_path == expected
    assert pipeline.people.bundled_assets.smpl_neutral_joint_regressor == visualization.smpl_joint_regressor_path
    assert pipeline.people.body_models_dir == tmp_path / 'ckpt/body_models'
