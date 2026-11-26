"""Real Hydra entry, pinned shared court and model-free deterministic scene stages."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration.tennis_scene.test_video_court import CLIPS, shared_fixture


@pytest.mark.parametrize('missing', [False, True])
def test_single_clip_cli_loads_pinned_shared_geometry_or_records_failure(tmp_path, monkeypatch, missing):
    pipeline, _, _, _, _, artifact = shared_fixture(tmp_path, monkeypatch)
    digest = 'b' * 64 if missing else artifact.digest
    harness = '''
import importlib.util
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import pytest
from src.tennis_scene.pipeline import TennisSceneOrchestrator

fixture_path = Path.cwd() / 'tests/unit/tennis_scene/pipeline/test_auto_pipeline.py'
spec = importlib.util.spec_from_file_location('shared_cli_fixture', fixture_path)
fixture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixture
spec.loader.exec_module(fixture)
root = Path(sys.argv.pop(1))
patches = pytest.MonkeyPatch()
base, _, _ = fixture.setup_pipeline(root, patches)
components = {name: stage for name, stage in base.components.items() if not name.startswith('court_detection/')}
def construct(config):
    return TennisSceneOrchestrator(config, components=components)
with patch.object(TennisSceneOrchestrator, 'from_runtime_config', side_effect=construct):
    runpy.run_module('src.tennis_scene.scripts.run_pipeline', run_name='__main__')
'''
    root = Path(__file__).parents[3]
    command = [sys.executable, '-c', harness, str(tmp_path), 'device=cpu', 'gvhmr.enabled=false', 'event_detection.method=rule',
               f'paths.data_root={tmp_path}', f'paths.artifact_root={tmp_path}', f'paths.output_root={tmp_path}',
               f'paths.checkpoint_root={tmp_path / "ckpt"}', f'paths.external_asset_root={tmp_path / "third_party"}',
               'video_paths=[cam0.mp4,cam1.mp4,cam2.mp4]', 'camera_ids=[cam0,cam1,cam2]', f'clip_id={CLIPS[0]}',
               'output_directory=cli-output', 'cache.directory=cli-cache', 'court_kp.mode=video_shared',
               'court_kp.video_shared.action=load', 'court_kp.video_shared.directory=shared',
               'court_kp.video_shared.dataset_id=fixture', 'court_kp.video_shared.video_id=video_000',
               f'court_kp.video_shared.artifact_digest={digest}']
    completed = subprocess.run(command, cwd=root, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'HYDRA_FULL_ERROR': '1'},
                               capture_output=True, text=True, timeout=60, check=False)
    assert completed.returncode == (1 if missing else 0), completed.stdout + completed.stderr
    receipts = list((tmp_path / 'cli-cache').glob('*/run.json'))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    if missing:
        assert receipt['status'] == 'failed' and receipt['error_type'] == 'FileNotFoundError'
        assert not (tmp_path / 'cli-output/tennis_clip.npz').exists()
    else:
        assert receipt['video_court']['artifact_digest'] == artifact.digest
        assert (tmp_path / 'cli-output/tennis_clip.npz').is_file()
