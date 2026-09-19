"""Strict import and BLCS preparation recipes retain curated values."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.tennis_scene.dataset_pipeline.legacy import copy_legacy_broadcast
from src.tennis_scene.dataset_pipeline.preparation import (
    BLCSPreparationConfig,
    BroadcastImportConfig,
    prepare_blcs,
)
from src.utils.paths import PROJECT_ROOT


def test_broadcast_recipe_and_unknown_keys() -> None:
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        cfg = compose(config_name="import_broadcast_ball")
    runtime = BroadcastImportConfig.from_config(cfg)
    assert len(runtime.video_groups) == 9
    assert runtime.video_groups["tennis_clip_source/clip_006"] == "broadcast_shanghai"
    assert runtime.isolated_jump == 0.12 and runtime.neighbor_distance == 0.06
    OmegaConf.set_struct(cfg, False)
    cfg.unknown = True
    with pytest.raises(Exception, match="unknown|Unknown"):
        BroadcastImportConfig.from_config(cfg)


def test_blcs_recipe_explicit_geometry_and_split() -> None:
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="prepare_blcs_real_dataset",
            overrides=["destination=blcs/prepared", "synthetic_source=blcs/synthetic"],
        )
    runtime = BLCSPreparationConfig.from_config(cfg)
    assert runtime.source_recipe == "build_slcs_dataset"
    assert runtime.geometry.max_abs_xy_m == (15.0, 28.0)
    assert runtime.geometry.height_range_m == (-0.1, 12.0)
    assert runtime.destination == PROJECT_ROOT / "data/blcs/prepared"
    cfg.video_splits.video_002 = "train"
    with pytest.raises(ValueError, match="assignments"):
        BLCSPreparationConfig.from_config(cfg)


def test_duplicate_yaml_group_key_is_rejected(tmp_path: Path) -> None:
    recipe = tmp_path / "duplicate.yaml"
    recipe.write_text("video_groups:\n  a/b: c\n  a/b: d\n")
    with pytest.raises(Exception, match="duplicate"):
        OmegaConf.load(recipe)


@pytest.mark.parametrize("clips", [["a/clip", "a/clip"], ["a/clip", "b/clip"]])
def test_import_rejects_duplicate_source_or_destination_ids(
    tmp_path: Path, clips: list[str]
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "dataset.json").write_text(
        json.dumps(
            {
                "version": 1,
                "created_at": "test",
                "updated_at": "test",
                "clips": [{"clip_id": name} for name in clips],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        copy_legacy_broadcast(
            source,
            tmp_path / "target",
            dataset_id="curated",
            video_groups={name: "same_venue" for name in clips},
            isolated_jump=0.12,
            neighbor_distance=0.06,
        )
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize("initialized", [False, True])
def test_explicit_blcs_source_recipe_reaches_export(
    initialized: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = Mock()
    monkeypatch.setattr(
        "src.tennis_scene.dataset_pipeline.blcs_training.export_blcs_training", export
    )
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="prepare_blcs_real_dataset",
            overrides=[
                "destination=blcs/prepared",
                "synthetic_source=blcs/synthetic",
                "paths.external_asset_root=/explicit/assets",
                'source_overrides=["sample_stride=3"]',
            ],
        )
        runtime = BLCSPreparationConfig.from_config(cfg)
        if initialized:
            prepare_blcs(runtime)
    if not initialized:
        prepare_blcs(runtime)
    exported = export.call_args.args[0]
    assert exported.sample_stride == 3
    assert exported.paths.external_asset_root == "/explicit/assets"
    assert export.call_args.kwargs["video_splits"] == {
        "video_000": "train",
        "video_001": "val",
        "video_002": "test",
    }
