"""Real CLI exits and failure records when clip publication cannot finish."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.utils.paths import PROJECT_ROOT

_HARNESS = r"""
import runpy
import sys
import types
from pathlib import Path

# Bind the checkout's tests namespace even if the shared venv has another tests package.
checkout = Path.cwd()
namespace = types.ModuleType('tests')
namespace.__path__ = [str(checkout / 'tests')]
sys.modules['tests'] = namespace
import pytest
from src.tennis_scene.pipeline import TennisSceneOrchestrator
from tests.support.tennis_scene.entrypoint_comparison import prepare_fixture

root, entry = Path(sys.argv[1]), sys.argv[2]
root.mkdir()
with pytest.MonkeyPatch.context() as patch:
    fixture = prepare_fixture(checkout, root, patch)
    patch.setattr(TennisSceneOrchestrator, 'from_runtime_config', lambda _: fixture['pipeline'])
    overrides = [f'paths.data_root={root}', f'paths.artifact_root={root}', f'paths.output_root={root / "logs"}',
                 f'paths.checkpoint_root={root / "ckpt"}', f'paths.external_asset_root={root / "third_party"}']
    if entry == 'single':
        def fail_export(*args, **kwargs):
            raise OSError('scene export write failure')
        patch.setattr('src.tennis_scene.pipeline.storage.scene_export.save_scene_result', fail_export)
        overrides += ['video_paths=[' + ','.join(str(p.relative_to(root)) for p in fixture['paths']) + ']',
                      'camera_ids=[cam0,cam1,cam2]', 'gvhmr.enabled=false', 'event_detection.method=rule']
        module = 'src.tennis_scene.scripts.run_pipeline'
    else:
        from src.tennis_scene.generate_dataset import pseudo_annotation
        publish = pseudo_annotation.publish_annotation
        def changed_before_publication(*args, **kwargs):
            fixture['paths'][0].write_bytes(b'changed before dataset publication')
            return publish(*args, **kwargs)
        patch.setattr(pseudo_annotation, 'publish_annotation', changed_before_publication)
        overrides += ['dataset_directory=dataset', 'pipeline_overrides=["gvhmr.enabled=false","event_detection.method=rule"]']
        module = 'src.tennis_scene.scripts.generate_dataset'
    sys.argv = [module, *overrides]
    runpy.run_module(module, run_name='__main__')
"""


@pytest.mark.parametrize("entry", ["single", "dataset"])
def test_publication_failure_exits_nonzero_and_keeps_failure_record(tmp_path: Path, entry: str) -> None:
    root = tmp_path / "fixture"
    result = subprocess.run([sys.executable, "-c", _HARNESS, str(root), entry], cwd=PROJECT_ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 1, result.stdout + result.stderr
    clip = root / "dataset/videos/video_000/clips/clip_000"
    assert not (clip / "annotations/tennis_scene/annotation.json").exists()
    if entry == "single":
        assert "scene export write failure" in result.stdout + result.stderr
        receipts = list((root / "stages").glob("*/run.json"))
        assert len(receipts) == 1
        receipt = json.loads(receipts[0].read_text())
        assert receipt["status"] == "failed" and receipt["error_type"] == "OSError"
        assert "scene export write failure" in receipt["error"]
        assert json.loads((receipts[0].parent / "scene.json").read_text())["exports"] == {}
    else:
        failure_path = clip / "annotations/tennis_scene.failure.json"
        assert failure_path.exists(), result.stdout + result.stderr
        failure = json.loads(failure_path.read_text())
        assert failure["clip_id"] == "video_000/clip_000"
        assert "inputs changed during reconstruction" in failure["error"]
        # The component scene is complete, while the dataset publication failed.
        assert "scene" in json.loads((clip / "annotations/tennis_scene/scene.json").read_text())["exports"]
