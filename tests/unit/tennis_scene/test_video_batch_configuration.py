"""Strict video-entry configuration shares the dataset path and override authority."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.tennis_scene.configuration import parse_video_batch_config
from src.tennis_scene.pipeline.batching import BatchConfig
from src.tennis_scene.scripts.generate_dataset import _compose_pipeline_config
from src.utils.paths import PROJECT_ROOT


@pytest.mark.parametrize("name,expected", [("pipeline", "conv_next_unet"), ("pipeline_x3d_dpt", "x3d_csn_dpt")])
def test_explicit_pipeline_preset_and_subset(tmp_path: Path, name: str, expected: str) -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs")):
        cfg = compose(config_name="run_video_batch", overrides=[f"paths.data_root={tmp_path}",
            "dataset_directory=dataset", "video_id=video_000", f"pipeline_config={name}",
            "clip_ids=[video_000/clip_037]", "batching.enabled=true", "batching.sizes.dino=1",
            "batching.clips_per_pass=3", "record_cuda=true", "record_cuda_synchronize=false",
            "record_sample_interval=0.25", 'pipeline_overrides=["event_detection.method=rule"]'])
        parsed = parse_video_batch_config(cfg)
        _, pipeline = _compose_pipeline_config(parsed.generation, config_name=parsed.pipeline_config)
    assert parsed.video_id == "video_000" and parsed.generation.clip_ids == ("video_000/clip_037",)
    assert parsed.generation.dataset_directory == tmp_path / "dataset"
    assert parsed.batching.enabled and parsed.batching.sizes.dino == 1
    assert parsed.batching.max_working_bytes == BatchConfig().max_working_bytes
    assert parsed.batching.clips_per_pass == 3 and parsed.record_cuda
    assert not parsed.record_cuda_synchronize and parsed.record_sample_interval == .25
    assert pipeline.ball_detection is not None and pipeline.ball_detection.model_type == expected


@pytest.mark.parametrize("override,message", [
    ("+typo=true", "typo"), ("video_id=../bad", "video_id"),
    ("pipeline_config=unknown", "pipeline_config"),
    ('pipeline_overrides=["paths.data_root=elsewhere"]', "path authority"),
    ("clip_ids=[video_000/clip_000,video_000/clip_000]", "unique"),
    ("+batching.sizes.typo=2", "typo"), ("batching.sizes.hmr2=0", "sizes.hmr2"),
    ("batching.max_pending_items=1", "cover every configured batch"),
    ("batching.clips_per_pass=-1", "clips_per_pass"),
    ("record_sample_interval=0", "finite and positive"),
])
def test_batch_config_rejects_unknown_fields_and_invalid_authority(override: str, message: str) -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs")):
        cfg = compose(config_name="run_video_batch", overrides=["dataset_directory=dataset", "video_id=video_000", override])
        with pytest.raises((ValueError, TypeError), match=message):
            parse_video_batch_config(cfg)
