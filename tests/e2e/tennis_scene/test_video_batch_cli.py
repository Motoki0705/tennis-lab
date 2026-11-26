"""Real Hydra entry and exit status with small deterministic component outputs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.utils.paths import PROJECT_ROOT
from tests.support.tennis_scene.video_dataset import add_clip

_HARNESS = """
import os
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import patch
import pytest

package = types.ModuleType('tests')
package.__path__ = [str(Path.cwd() / 'tests')]
sys.modules['tests'] = package
from src.tennis_scene.pipeline import TennisSceneOrchestrator
from src.tennis_scene.pipeline.components.court_side import PlayerCourtSideModule
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import setup_pipeline, runtime, project_fixture_player_boxes

monkeypatch = pytest.MonkeyPatch()
pipeline, _, _ = setup_pipeline(Path(os.environ['VIDEO_BATCH_TEST_ROOT']), monkeypatch)
include_ball = os.environ['VIDEO_BATCH_TEST_FAIL'] == '1'
enabled = 'true' if include_ball else 'false'
pipeline.config = runtime(Path(os.environ['VIDEO_BATCH_TEST_ROOT']), overrides=(
    'court_side.evidence=player', f'ball_detection.enabled={enabled}', f'ball_reconstruction.enabled={enabled}'))
if not include_ball:
    pipeline.components = {name: stage for name, stage in pipeline.components.items() if not name.startswith('ball_detection/')}
project_fixture_player_boxes(pipeline.components)
pipeline.publication_identity = lambda: {'fixture': 'cli'}
if os.environ['VIDEO_BATCH_TEST_FAIL'] == '1':
    def fail(*args):
        raise ReconstructionUnavailable('clip_too_long', 'fixture clip too long')
    monkeypatch.setattr(PlayerCourtSideModule, 'process', fail)
def configured_pipeline(runtime):
    assert runtime.people.runtime.dino_detector.precision == 'fp16'
    assert runtime.people.runtime.vitpose.precision == 'fp16'
    assert runtime.court_side_evidence == 'player'
    assert runtime.event_detection.method == 'rule'
    assert runtime.enabled['ball_detection'] == runtime.enabled['ball_reconstruction'] == include_ball
    return pipeline
with patch.object(TennisSceneOrchestrator, 'from_runtime_config', side_effect=configured_pipeline):
    runpy.run_module('src.tennis_scene.scripts.run_video_batch', run_name='__main__')
"""


@pytest.mark.parametrize("failed", [False, True])
def test_cli_runs_one_video_and_exits_nonzero_for_unresolved_failure(tmp_path: Path, failed: bool) -> None:
    dataset = tmp_path / "dataset"
    clip = add_clip(dataset, "video_000/clip_000")
    enabled = "true" if failed else "false"
    result = subprocess.run([sys.executable, "-c", _HARNESS,
        f"paths.data_root={tmp_path}", f"paths.output_root={tmp_path / 'outputs'}",
        "dataset_directory=dataset", "video_id=video_000", "progress_directory=progress",
        f'pipeline_overrides=["device=cuda","people_models.runtime.dino_detector.precision=fp16","people_models.runtime.vitpose.precision=fp16","court_side.evidence=player","ball_detection.enabled={enabled}","ball_reconstruction.enabled={enabled}","event_detection.method=rule"]'],
        cwd=PROJECT_ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "VIDEO_BATCH_TEST_ROOT": str(tmp_path),
                              "VIDEO_BATCH_TEST_FAIL": str(int(failed)), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                              "OPENBLAS_NUM_THREADS": "1"},
        text=True, capture_output=True, check=False, timeout=30)
    assert result.returncode == int(failed), result.stderr
    assert (clip / "annotations/tennis_scene/annotation.json").is_file() == (not failed)
    receipt = json.loads(next((tmp_path / "outputs/progress").rglob("progress.json")).read_text())
    assert receipt["status"] == ("failed" if failed else "complete")
    assert len(receipt["targets"]) == 1
    ball_nodes = {name for name in receipt["nodes"]["video_000/clip_000"] if name.startswith("ball_")}
    assert bool(ball_nodes) == failed
