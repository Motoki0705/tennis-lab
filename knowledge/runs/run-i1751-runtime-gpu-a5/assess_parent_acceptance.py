"""CPU-only reassessment of completed a5 data under the parent's explicit policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def position_error(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('Coordinate shape/dtype changed')
    finite = np.isfinite(left) & np.isfinite(right)
    same = (left == right) | (np.isnan(left) & np.isnan(right))
    if not np.all(finite | same):
        raise ValueError('Non-finite coordinate pattern changed')
    delta = np.zeros(left.shape, dtype=np.float64)
    np.subtract(right, left, out=delta, where=finite, dtype=np.float64)
    return {'max_axis': float(np.abs(delta).max(initial=0)),
            'max_l2': float(np.linalg.norm(delta, axis=-1).max(initial=0))}


def assess(root, policy):
    comparison = json.loads((root / 'comparison-completed.json').read_text())
    summary = json.loads((root / 'acceptance-summary.json').read_text())
    pin = json.loads((root / 'inputs.json').read_text())
    assert pin['settings']['ball_detection']['precision'] == policy['ball_precision']
    assert comparison['tolerance'] == policy['other_outputs']
    variants = ('main', 'runtime')
    stores = [root / variant / 'store' for variant in variants]
    indexes = [json.loads((store / 'scene.json').read_text()) for store in stores]
    scene_doc = json.loads((stores[0] / indexes[0]['artifacts']['scene_assembly']['path']).read_text())
    ball_scene_arrays = {scene_doc['payload'][field]['array'] for field in ('ball_uv', 'ball_3d')}
    failures = []
    all_rows = list(comparison['scene'].values())
    for name, node in comparison['components'].items():
        all_rows.extend(node['arrays'].values())
        is_ball = name.startswith(('ball_detection/', 'ball_points/')) or name == 'ball_triangulation'
        for field, row in node['arrays'].items():
            if not is_ball and not (name == 'scene_assembly' and field in ball_scene_arrays) and not row['close']:
                failures.append(name + '/' + field)
    for name, row in comparison['scene'].items():
        if name not in ('ball_uv', 'ball_3d') and not row['close']:
            failures.append('scene/' + name)
    shape_dtype_same = all(row['shape_main'] == row['shape_runtime'] and row['dtype_main'] == row['dtype_runtime'] for row in all_rows)
    discrete_exact = all(row['exact'] for row in all_rows if np.dtype(row['dtype_main']).kind not in 'fc')
    detections = []
    for camera in pin['source']['videos']:
        name = 'ball_detection/' + camera['camera_id']
        points = []
        for store, index in zip(stores, indexes, strict=True):
            path = store / index['artifacts'][name]['path']
            doc = json.loads(path.read_text())
            points.append(np.load(path.parent / doc['payload']['uv_px']['array'], mmap_mode='r'))
        detections.append({'camera': camera['camera_id'], **position_error(*points)})
    with np.load(root / 'main/scene.npz') as left, np.load(root / 'runtime/scene.npz') as right:
        world = position_error(left['ball_3d'], right['ball_3d'])
    terminal = json.loads((root / 'terminal.json').read_text())
    checks = {
        'both_gpu_variants_complete': all(terminal['outcomes'][name]['status'] == 'complete' for name in variants),
        'same_artifact_fields': comparison['same_nodes'] and comparison['same_scene_fields'] and all(node['same_array_names'] for node in comparison['components'].values()),
        'same_shapes_and_dtypes': shape_dtype_same,
        'all_discrete_observed_masks_exact': discrete_exact,
        'ball_uv_within_0_5_px': all(item['max_l2'] <= policy['ball_detection_uv_l2_limit_px'] for item in detections),
        'ball_3d_within_1_mm': world['max_l2'] <= policy['ball_final_position_l2_limit_m'],
        'non_ball_original_float_rule': not failures,
        'json_payloads_original_rule': comparison['all_payloads_close'] and comparison['scene_metadata']['close'],
        'runtime_resources_closed': not summary['runtime_unclosed_instances'] and not summary['runtime_failed_events'],
    }
    raw_names = ('comparison.json', 'comparison-completed.json', 'acceptance-summary.json', 'terminal.json')
    return {'schema': 'c3_parent_acceptance.v1', 'policy': policy,
            'original_numerical_acceptance': comparison['numerical_acceptance'],
            'original_queue_status': terminal['status'], 'original_comparison_returncode': terminal['comparison_returncode'],
            'parent_acceptance': all(checks.values()), 'checks': checks,
            'ball_detection_error_px': detections, 'ball_final_position_error_m': world,
            'non_ball_original_rule_failures': failures,
            'raw_evidence_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in raw_names},
            'additional_gpu_runs': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--policy', type=Path, default=Path(__file__).with_name('parent-acceptance-policy.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = assess(args.root, json.loads(args.policy.read_text()))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['parent_acceptance'] else 1)
