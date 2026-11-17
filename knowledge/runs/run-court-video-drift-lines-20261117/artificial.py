"""Separate development views and exact run-77 heldout rendering inputs.

The former shared synthetic matrices were not serialized. The recover action
reconstructs ONLY those eight artificial fixtures using the unchanged run-77
code and checks their original RGB hashes / projection errors. It never opens
real video or changes the real common-H store.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path.cwd()
PREVIOUS = ROOT / 'knowledge/runs/run-court-video-shared-acceptance-20261117'
sys.path[:0] = [str(ROOT), str(PREVIOUS)]
spec = importlib.util.spec_from_file_location('run77_synthetic', PREVIOUS / 'synthetic.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
from src.tasks.court_detection.geometry.paint_observations import PAINT_NAMES
from src.tasks.court_detection.geometry.video_drift import DriftConfig, inspect_frame
from src.tasks.court_detection.geometry.video_shared.fit import SharedPaintConfig, fit_common_court
from src.tasks.court_detection.geometry.video_shared.observations import measure_frame
from build import save, sha
from evidence import assert_sources


def dev_cases():
    rng = np.random.default_rng(219178)
    cases = []
    for index in range(8):
        far, near = rng.uniform(440, 760), rng.uniform(980, 1160)
        cx = 640 + rng.uniform(-45, 45)
        top, bottom = rng.uniform(110, 195), rng.uniform(590, 665)
        corners = np.array([[cx - far / 2, top], [cx + far / 2, top + rng.uniform(-30, 30)],
                            [640 - near / 2, bottom], [640 + near / 2, bottom + rng.uniform(-30, 30)]])
        truth = cv2.getPerspectiveTransform(legacy.WORLD[:4].astype(np.float32), corners.astype(np.float32)).astype(float)
        cases.append({'id': index, 'truth': truth.tolist(), 'shared': truth.tolist(), 'supported_lines': list(PAINT_NAMES)})
    return cases


def recover(case):
    cv2.setNumThreads(1)
    truth, initial = np.array(case['truth']), np.array(case['initial'])
    groups, hashes = [], []
    for frame in range(5):
        rgb = legacy.raster(truth, frame)
        hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
        groups.extend(measure_frame(rgb, initial, clip_id=f'synthetic/clip_{frame}', frame_index=frame, config=SharedPaintConfig()).groups)
    fit = fit_common_court(initial, tuple(groups), legacy.SIZE, SharedPaintConfig())
    assert fit.matrix is not None, fit.reasons
    original = json.loads((ROOT / 'outputs/tennis_scene/evaluate/court-video-shared-acceptance-20261117/synthetic' / f'detail-{case["id"]:02d}.json').read_text())
    assert hashes == original['rgb_sha256']
    error = np.linalg.norm(legacy.project(fit.matrix) - legacy.project(truth), axis=1) * 1.5
    assert abs(error.mean() - original['mean_reference_px']) < 1e-9
    assert abs(error.max() - original['max_reference_px']) < 1e-9
    return case | {'shared': fit.matrix.tolist(), 'supported_lines': fit.diagnostic['supported_lines'],
                   'reconstructed_rgb_sha256': hashes, 'original_error_matched': True}


def evaluate(job):
    case, shift, frame, negative = job
    cv2.setNumThreads(1)
    started = time.monotonic()
    truth, shared = np.array(case['truth']), np.array(case['shared'])
    moved = truth.copy()
    moved[0] += shift / 1.5 * moved[2]
    rgb = legacy.raster(moved, frame, missing_transverse=negative)
    report = inspect_frame(rgb, shared, tuple(case['supported_lines']), DriftConfig())
    return {'case': case['id'], 'shift_reference_px': shift, 'frame_index': frame, 'negative': negative,
            'expected': 'unobservable' if negative else {0: 'ok', 5: 'warning', 11: 'failed'}[shift],
            'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'observed': report, 'seconds': time.monotonic() - started}


def main(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    args.output.mkdir(parents=True, exist_ok=True)
    if args.action == 'recover':
        protocol = json.loads((PREVIOUS / 'synthetic-protocol.json').read_text())
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            cases = list(pool.map(recover, protocol['cases']))
        save(args.output / 'recovered-synthetic-fixtures.json', {'source_protocol_sha256': sha(PREVIOUS / 'synthetic-protocol.json'), 'cases': cases})
        print('Recovered and verified eight artificial shared-H fixtures', flush=True)
        return
    if args.action == 'development':
        cases = dev_cases()
        jobs = [(case, shift, frame, False) for case in cases for shift in (0, 5, 11) for frame in (5, 6)]
        jobs += [(case, 0, 5, True) for case in cases]
    else:
        frozen = json.loads((args.output / 'frozen-profile.json').read_text())
        assert frozen['settings'] == asdict(DriftConfig())
        assert_sources(frozen)
        cases = json.loads((args.output / 'recovered-synthetic-fixtures.json').read_text())['cases']
        jobs = [(case, shift, 5, False) for case in cases for shift in (0, 5, 11)]
    save(args.output / f'{args.action}-protocol.json', {'cases': cases, 'settings': asdict(DriftConfig()), 'jobs': [list(j[1:]) for j in jobs],
        'driver_sha256': sha(Path(__file__)), 'renderer_sha256': sha(PREVIOUS / 'synthetic.py'), 'gpu_used': False})
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate, jobs))
    summary = {'cases': len(results), 'matches': sum(r['observed']['reason'] == 'drift_unobservable' if r['negative'] else
        r['observed']['status'] == r['expected'] and r['observed']['reason'] != 'drift_unobservable' for r in results),
        'unobservable': sum(r['observed']['reason'] == 'drift_unobservable' for r in results),
        'moving_safe': sum(r['shift_reference_px'] > 0 and r['observed']['status'] == 'ok' for r in results),
        'seconds': time.monotonic() - started,
        'results': [{'case': r['case'], 'shift': r['shift_reference_px'], 'frame': r['frame_index'], 'negative': r['negative'],
                     'status': r['observed']['status'], 'reason': r['observed']['reason'], 'geometry': r['observed']['geometry_reasons'],
                     'mean_px': r['observed']['mean_displacement_px'], 'std_px': r['observed']['max_visible_projection_std_px']} for r in results]}
    save(args.output / f'{args.action}-detail.json', results)
    save(args.output / f'{args.action}-summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['development', 'recover', 'heldout'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    main(parser.parse_args())
