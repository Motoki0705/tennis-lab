"""Independent heldout artificial RGB benchmark, after profile freezing.

Literal physical coordinates and a separately drawn scene define truth. The
development unit tests use a different affine-like view and are not these cases.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.tasks.court_detection.geometry.video_shared.fit import SharedPaintConfig, fit_common_court
from src.tasks.court_detection.geometry.video_shared.observations import measure_frame
from src.tasks.court_detection.geometry.video_shared.drift import inspect_drift
from build import save, sha

WORLD = np.array([[-5.485, 11.885], [5.485, 11.885], [-5.485, -11.885], [5.485, -11.885],
                  [-4.115, 11.885], [-4.115, -11.885], [4.115, 11.885], [4.115, -11.885],
                  [-4.115, 6.4], [4.115, 6.4], [-4.115, -6.4], [4.115, -6.4], [0, 6.4], [0, -6.4]])
EDGES = ((0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13))
SIZE = (1280, 720)


def project(h, xy=WORLD):
    z = np.c_[xy, np.ones(len(xy))] @ h.T
    return z[:, :2] / z[:, 2:]


def raster(h, frame, *, missing_transverse=False):
    image = np.full((720, 1280, 3), [42, 105, 62], np.uint8)
    # Render physical 5cm paint rectangles, rather than invoking a production
    # keypoint/line drawing utility or a paint extractor as the oracle.
    for index, (a, b) in enumerate(EDGES):
        if missing_transverse and index in (0, 1, 6, 7):
            continue
        p, q = WORLD[[a, b]]
        direction = q - p
        normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction) * .025
        polygon = project(h, np.array([p + normal, q + normal, q - normal, p - normal]))
        cv2.fillConvexPoly(image, np.rint(polygon).astype(np.int32), (235, 235, 230), cv2.LINE_AA)
    # Static raised net: its top band is not on the ground plane; mesh/post
    # structures persist across all frames and cannot be removed by averaging.
    ground = project(h, np.array([[-6.1, 0], [6.1, 0]]))
    top = ground - [0, 45]
    for alpha in np.linspace(0, 1, 45):
        p, q = (1 - alpha) * top[0] + alpha * top[1], (1 - alpha) * ground[0] + alpha * ground[1]
        cv2.line(image, tuple(np.rint(p).astype(int)), tuple(np.rint(q).astype(int)), (33, 44, 37), 1, cv2.LINE_AA)
    cv2.line(image, tuple(np.rint(top[0]).astype(int)), tuple(np.rint(top[1]).astype(int)), (248, 248, 248), 4, cv2.LINE_AA)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.array([[850, 80], [1180, 80], [860, 719], [620, 719]]), 1)
    image[mask.astype(bool)] = (image[mask.astype(bool)] * .55).astype(np.uint8)
    # A player and a separate occluder move across different markings.
    centre = np.rint(project(h, np.array([[-3 + frame * 1.4, -5.]]))[0]).astype(int)
    cv2.rectangle(image, tuple(centre + [-14, -65]), tuple(centre + [14, 0]), (25, 38, 30), -1)
    cv2.circle(image, tuple(centre + [0, -75]), 11, (25, 38, 30), -1)
    cv2.rectangle(image, (170 + frame * 130, 160), (220 + frame * 130, 200), (35, 45, 50), -1)
    return image


def main(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    cv2.setNumThreads(1)
    rng = np.random.default_rng(20261117)
    cases = []
    for index in range(8):
        far = rng.uniform(460, 750)
        near = rng.uniform(1020, 1160)
        cx = 640 + rng.uniform(-35, 35)
        top, bottom = rng.uniform(115, 185), rng.uniform(605, 665)
        corners = np.array([[cx - far / 2, top], [cx + far / 2, top + rng.uniform(-25, 25)],
                            [640 - near / 2, bottom], [640 + near / 2, bottom + rng.uniform(-25, 25)]])
        h = cv2.getPerspectiveTransform(WORLD[:4].astype(np.float32), corners.astype(np.float32)).astype(float)
        initial = h.copy()
        initial[0] += rng.uniform(-7, 7) * initial[2]
        initial[1] += rng.uniform(-7, 7) * initial[2]
        cases.append({'id': index, 'truth': h.tolist(), 'initial': initial.tolist()})
    config = SharedPaintConfig()
    files = {str(p): sha(p) for p in sorted(Path('src/tasks/court_detection/geometry').rglob('*.py'))}
    save(args.output / 'protocol.json', {'schema': 'video_court_synthetic_heldout_v1', 'random_seed': 20261117,
        'size_wh': SIZE, 'cases': cases, 'settings': asdict(config), 'source_hashes': files,
        'driver_sha256': sha(Path(__file__)), 'frames_per_case': 5, 'gpu_used': False,
        'motion_px_at_1920x1080': [0, 5, 11], 'reference_h_used': False,
        'thresholds_frozen_before_this_evaluation': True})
    results = []
    started = time.monotonic()
    scale = np.hypot(*SIZE) / np.hypot(1920, 1080)
    for case in cases:
        truth, initial = np.array(case['truth']), np.array(case['initial'])
        groups, hashes, measurements = [], [], []
        for frame in range(5):
            image = raster(truth, frame)
            hashes.append(hashlib.sha256(image.tobytes()).hexdigest())
            measured = measure_frame(image, initial, clip_id=f'synthetic/clip_{frame}', frame_index=frame, config=config)
            groups.extend(measured.groups)
            measurements.append(measured.diagnostic['reason'])
            if frame == 0:
                cv2.imwrite(str(args.output / f'case-{case["id"]:02d}.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        fitted = fit_common_court(initial, tuple(groups), SIZE, config)
        error = None if fitted.matrix is None else np.linalg.norm(project(fitted.matrix) - project(truth), axis=1)
        motion = []
        for shift, expected in [(0, 'ok'), (5, 'warning'), (11, 'failed')]:
            moved = truth.copy(); moved[0] += shift * scale * moved[2]
            # GT is not supplied to the drift estimator: only an accepted shared
            # fit and observations from a new independently rendered frame.
            if fitted.matrix is None:
                report = {'status': 'failed', 'reason': 'common_fit_missing'}
            else:
                measured = measure_frame(raster(moved, 5), fitted.matrix, clip_id='synthetic/moved', frame_index=5, config=config)
                report = inspect_drift(fitted.matrix, measured.groups, SIZE, config)
            motion.append({'shift_reference_px': shift, 'expected': expected, 'observed': report})
        bad_groups = []
        for frame in range(3):
            measured = measure_frame(raster(truth, frame, missing_transverse=True), initial,
                                     clip_id=f'negative/clip_{frame}', frame_index=frame, config=config)
            bad_groups.extend(measured.groups)
        rejected = fit_common_court(initial, tuple(bad_groups), SIZE, config)
        row = {'id': case['id'], 'adopted': fitted.matrix is not None, 'reasons': fitted.reasons,
               'frame_measurement_reasons': measurements, 'rgb_sha256': hashes,
               'mean_reference_px': None if error is None else float(error.mean() / scale),
               'max_reference_px': None if error is None else float(error.max() / scale),
               'negative_adopted': rejected.matrix is not None, 'negative_reasons': rejected.reasons, 'motion': motion}
        results.append(row)
        save(args.output / f'detail-{case["id"]:02d}.json', row)
        print(json.dumps({k: v for k, v in row.items() if k not in ('motion', 'rgb_sha256')}), flush=True)
    assert files == {str(p): sha(p) for p in sorted(Path('src/tasks/court_detection/geometry').rglob('*.py'))}
    summary = {'cases': len(results), 'adopted': sum(r['adopted'] for r in results),
        'within_tolerance': sum(r['adopted'] and r['mean_reference_px'] <= 10 and r['max_reference_px'] <= 25 for r in results),
        'negative_adopted': sum(r['negative_adopted'] for r in results),
        'max_mean_reference_px': max((r['mean_reference_px'] for r in results if r['adopted']), default=None),
        'max_max_reference_px': max((r['max_reference_px'] for r in results if r['adopted']), default=None),
        'motion_matches': sum(m['observed']['status'] == m['expected'] and m['observed']['reason'] != 'drift_unobservable'
                              for r in results for m in r['motion']), 'motion_cases': len(results) * 3,
        'seconds': time.monotonic() - started, 'protocol_sha256': sha(args.output / 'protocol.json')}
    save(args.output / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
