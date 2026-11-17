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
from src.tennis_scene.pipeline.components.court_side import CourtSideModule
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from tests.unit.tennis_scene.pipeline.test_auto_pipeline import setup_pipeline

monkeypatch = pytest.MonkeyPatch()
pipeline, _, _ = setup_pipeline(Path(os.environ['VIDEO_BATCH_TEST_ROOT']), monkeypatch)
pipeline.publication_identity = lambda: {'fixture': 'cli'}
if os.environ['VIDEO_BATCH_TEST_FAIL'] == '1':
    def fail(*args):
        raise ReconstructionUnavailable('clip_too_long', 'fixture clip too long')
    monkeypatch.setattr(CourtSideModule, 'process', fail)
with patch.object(TennisSceneOrchestrator, 'from_runtime_config', return_value=pipeline):
    runpy.run_module('src.tennis_scene.scripts.run_video_batch', run_name='__main__')
"""


@pytest.mark.parametrize("failed", [False, True])
def test_cli_runs_one_video_and_exits_nonzero_for_unresolved_failure(tmp_path: Path, failed: bool) -> None:
    dataset = tmp_path / "dataset"
    clip = add_clip(dataset, "video_000/clip_000")
    result = subprocess.run([sys.executable, "-c", _HARNESS,
        f"paths.data_root={tmp_path}", f"paths.output_root={tmp_path / 'outputs'}",
        "dataset_directory=dataset", "video_id=video_000", "progress_directory=progress"],
        cwd=PROJECT_ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "VIDEO_BATCH_TEST_ROOT": str(tmp_path),
                              "VIDEO_BATCH_TEST_FAIL": str(int(failed)), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                              "OPENBLAS_NUM_THREADS": "1"},
        text=True, capture_output=True, check=False, timeout=30)
    assert result.returncode == int(failed), result.stderr
    assert (clip / "annotations/tennis_scene/annotation.json").is_file() == (not failed)
    receipt = json.loads(next((tmp_path / "outputs/progress").rglob("progress.json")).read_text())
    assert receipt["status"] == ("failed" if failed else "complete")
    assert len(receipt["targets"]) == 1
