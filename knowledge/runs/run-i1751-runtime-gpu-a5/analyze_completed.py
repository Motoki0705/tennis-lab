"""CPU-only acceptance evidence derived from completed a5 artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from summarize import summarize


def analyze(root):
    result = summarize(root)
    comparison = json.loads((root / 'comparison-completed.json').read_text())
    result.update({key: comparison[key] for key in ('all_payloads_exact', 'all_payloads_close',
                  'scene_metadata', 'numerical_acceptance')})
    with np.load(root / 'main/scene.npz') as left, np.load(root / 'runtime/scene.npz') as right:
        delta = right['ball_3d'].astype(np.float64) - left['ball_3d'].astype(np.float64)
        uv = right['ball_uv'].astype(np.float64) - left['ball_uv'].astype(np.float64)
        result.update(ball_3d_max_abs_m=float(np.abs(delta).max()),
                      ball_3d_max_l2_m=float(np.linalg.norm(delta, axis=-1).max()),
                      ball_3d_changed_frames=np.flatnonzero(np.any(delta != 0, axis=-1)).tolist(),
                      ball_uv_max_abs=float(np.abs(uv).max()))
    result['scene_discrete_exact'] = all(item['exact'] for item in comparison['scene'].values()
                                         if np.dtype(item['dtype_main']).kind not in 'fc')
    result['component_discrete_exact'] = all(item['exact'] for node in comparison['components'].values()
                                             for item in node['arrays'].values()
                                             if np.dtype(item['dtype_main']).kind not in 'fc')
    result['unaffected_model_outputs_exact'] = all(item['arrays_exact']
        for name, item in comparison['components'].items()
        if name.split('/')[0] in ('court_detection', 'person_detection', 'person_tracking',
                                 'player_selection', 'pose_estimation', 'gvhmr', 'body_placement'))
    indexes = [json.loads((root / name / 'store/scene.json').read_text()) for name in ('main', 'runtime')]
    detections = []
    for camera in range(3):
        name = f'ball_detection/cam{camera}'
        paths = [root / variant / 'store' / index['artifacts'][name]['path']
                 for variant, index in zip(('main', 'runtime'), indexes)]
        positions = [np.load(path.parent / 'array_0001.npy', mmap_mode='r') for path in paths]
        delta = positions[1].astype(np.float64) - positions[0].astype(np.float64)
        detections.append({'camera': f'cam{camera}', 'changed_frames': np.flatnonzero(np.any(delta != 0, axis=-1)).tolist(),
                           'max_axis_px': float(np.abs(delta).max()),
                           'max_l2_px': float(np.linalg.norm(delta, axis=-1).max()),
                           'confidence_max_abs': comparison['components'][name]['arrays']['array_0002.npy']['max_abs_diff'],
                           'heatmap_max_abs': comparison['components'][name]['arrays']['array_0005.npy']['max_abs_diff']})
    result['ball_detection_differences'] = detections
    terminal = json.loads((root / 'terminal.json').read_text())
    result.update(comparison_returncode=terminal['comparison_returncode'], queue_status=terminal['status'],
                  gpu_anomaly=bool(terminal.get('anomaly')), gpu_attempts=1, gpu_retries=0)
    commits = [result['variants'][name]['commit'] for name in ('main', 'runtime')]
    changed = subprocess.check_output(['git', 'diff', '--name-only', '..'.join(commits), '--',
                                      'src/tasks/ball_detection'], text=True).splitlines()
    result['ball_implementation_changed_files'] = changed
    assert not changed
    for name, want in json.loads((root / 'frozen-files.json').read_text()).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == want
    result['seven_frozen_files_unchanged'] = True
    result['cause'] = ('Not isolated: same-main control repeat is unavailable. BF16/compiled backend numerical '
                       'variation is a hypothesis, not a demonstrated cause. Do not relax tolerances automatically.')
    result['strict_acceptance'] = 'failed; parent judgement required; PR remains Draft'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    (args.root / 'acceptance-summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('numerical_acceptance', 'runtime_loads', 'runtime_unloads',
                         'runtime_max_owned_models', 'ball_3d_max_abs_m', 'ball_3d_max_l2_m')}, indent=2))
