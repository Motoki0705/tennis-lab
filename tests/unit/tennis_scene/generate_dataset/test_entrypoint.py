"""Exercise the dataset CLI's DATA-role video boundary through publication."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir

from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline import TennisSceneOrchestrator
from src.tennis_scene.schema import SceneResult
from src.tennis_scene.scripts.generate_dataset import main
from src.utils.configuration import PathRole
from src.utils.paths import PROJECT_ROOT


def test_dataset_cli_passes_data_videos_with_separate_artifact_root(
    structured_dataset: Path,
    valid_scene_result: SceneResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[PathRole] = []

    def create(runtime: PipelineRuntimeConfig) -> SimpleNamespace:
        def run(
            *,
            video_paths: Sequence[Path],
            video_role: PathRole,
            camera_ids: Sequence[str],
            store_root: Path,
            clip_id: str,
            max_frames: int | None,
        ) -> SceneResult:
            del camera_ids, max_frames
            # Structured clips always own their store; it is passed explicitly.
            assert store_root == video_paths[0].parents[1] / "annotations" / "tennis_scene"
            assert clip_id
            for path in video_paths:
                runtime.resolver.validate(video_role, path)
            calls.append(video_role)
            return valid_scene_result

        return SimpleNamespace(run=run, publication_identity=lambda: {"test": "fixed"}, last_receipt={})

    monkeypatch.setattr(TennisSceneOrchestrator, "from_runtime_config", create)
    with initialize_config_dir(
        config_dir=str(PROJECT_ROOT / "src/tennis_scene/configs"), version_base="1.3"
    ):
        cfg = compose(
            config_name="generate_dataset",
            overrides=[
                f"paths.data_root={structured_dataset.parent}",
                f"paths.artifact_root={structured_dataset.parent / 'artifacts'}",
                "dataset_directory=dataset",
                'pipeline_overrides=[]',
            ],
        )
        assert inspect.unwrap(main)(cfg) == 0
    assert calls == [PathRole.DATA]
    assert (
        structured_dataset
        / "videos/video_000/clips/clip_000/annotations/tennis_scene/annotation.json"
    ).is_file()


def test_real_hydra_cli_fails_when_a_clip_fails(
    structured_dataset: Path,
) -> None:
    harness = """
import runpy
from types import SimpleNamespace
from unittest.mock import patch
from src.tennis_scene.pipeline import TennisSceneOrchestrator

def fail(**kwargs):
    raise ValueError("deliberate clip failure")

with patch.object(TennisSceneOrchestrator, "from_runtime_config",
                  return_value=SimpleNamespace(run=fail, publication_identity=lambda: {"test": "fixed"}, last_receipt={})):
    runpy.run_module("src.tennis_scene.scripts.generate_dataset", run_name="__main__")
"""
    completed = subprocess.run(
        [
            sys.executable, "-c", harness,
            f"paths.data_root={structured_dataset.parent}",
            f"paths.output_root={structured_dataset.parent / 'logs'}",
            "dataset_directory=dataset",
            'pipeline_overrides=[]',
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "deliberate clip failure" in completed.stdout + completed.stderr
