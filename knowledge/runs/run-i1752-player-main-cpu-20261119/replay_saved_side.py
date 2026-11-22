"""Replay the existing player-side qualification without invoking any model."""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def replay(output: Path) -> None:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CUDA_VISIBLE_DEVICES must be empty for this CPU replay')
    import torch
    from src.submodules.models._base.inference_model import BaseInferenceModel
    from src.tennis_scene.pipeline.inference_runtime import InferenceRuntime

    def forbidden(*args, **kwargs):
        raise AssertionError('Saved-observation replay must never load models or initialize CUDA')

    baseline = REPO / 'knowledge/runs/run-i2269-player-court-side-20261119'
    expected = json.loads((baseline / 'report.json').read_text())
    common = Path(subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], text=True).strip()).parent
    output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as guards:
        for target in ('torch.cuda.init', 'torch.cuda._lazy_init', 'torch.nn.Module.__init__', 'torch.ops.load_library'):
            guards.enter_context(patch(target, side_effect=forbidden))
        guards.enter_context(patch.object(torch.cuda, 'is_available', return_value=False))
        guards.enter_context(patch.object(BaseInferenceModel, 'load', side_effect=forbidden))
        guards.enter_context(patch.object(InferenceRuntime, '__enter__', side_effect=forbidden))
        code = runpy.run_path(str(baseline / 'evaluate.py'))
        code['evaluate'](common / '.worktree-outputs', output)
    actual = json.loads((output / 'report.json').read_text())
    for key in ('expected', 'thresholds', 'scope_calibration_equal', 'source_court_digest', 'source_drift_digest', 'video_clip_outcomes'):
        assert actual[key] == expected[key], key
    for clip in expected['clips']:
        for key in ('view_half_turns', 'reason', 'source_num_frames', 'inputs_digest', 'distance_sweep', 'geometry_errors'):
            assert actual['clips'][clip][key] == expected['clips'][clip][key], (clip, key)
    assert actual['video_decision']['shared'] == expected['video_decision']['shared']
    assert not torch.cuda.is_initialized()
    summary = {
        'head': actual['git_commit'],
        'source_tree': subprocess.check_output(['git', 'rev-parse', 'HEAD:src'], text=True).strip(),
        'baseline_commit': expected['git_commit'],
        'clips': {clip: {key: row[key] for key in ('view_half_turns', 'reason', 'source_num_frames')}
                  for clip, row in actual['clips'].items()},
        'video_shared': actual['video_decision']['shared'],
        'all_standalone_distance_sweeps_exact': True, 'all_geometry_error_summaries_exact': True,
        'all_video_hypotheses_exact': True, 'admission_outcomes_exact': True,
        'model_loads': 0, 'cuda_initialized': False, 'wall_seconds': actual['wall_seconds'],
        'limits': 'The existing 3-clip qualification, including its distance sweep, is replayed. It is not an independent GT, full69 side decision, end-to-end scene replay or GPU validation.',
    }
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    replay(parser.parse_args().output.resolve())
