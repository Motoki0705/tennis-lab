"""Untuned transfer of the existing RGB fitter, for planning only.

Inputs: one held-out RGB frame and frozen model H. No reference H, reference
centres, or manual net masks enter fitting. Accuracy against the same-family
reference is a dependent diagnostic, not an independent acceptance result.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import time
import cv2
import numpy as np
from common import BASE, OLD, WHITE, correct, errors, source_rows, write_json

CLIPS = (5, 13, 22, 30, 39, 47, 56, 64)
spec = importlib.util.spec_from_file_location('rgb_reference_fitter', WHITE / 'fit_reference.py')
fit = importlib.util.module_from_spec(spec); spec.loader.exec_module(fit)


def run(row):
    cv2.setNumThreads(1); start = time.monotonic()
    condition = row['conditions']['postprocess_and_acceptance']
    candidate = row['current_candidates'][condition['region_index']]
    meta = json.loads((OLD / 'raw' / row['key'] / 'record.json').read_text())
    cap = cv2.VideoCapture(meta['source_video']); ok, bgr = cap.read(); cap.release()
    assert ok and bgr.shape == (1080, 1920, 3)
    assert hashlib.sha256(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()).hexdigest() == meta['rgb_sha256']
    decode_seconds = time.monotonic() - start
    entry = {'key': row['key'], 'camera': row['camera'], 'frame_index': 0,
        'model_initial_region': condition['region_index'], 'old_accepted': condition['accepted'],
        'old_correct': correct(condition['points'], row['camera']), 'old_mean_px': float(errors(condition['points'], row['camera']).mean()),
        'raw_rgb_sha256': meta['rgb_sha256'], 'decode_seconds': decode_seconds}
    if candidate['matrix'] is None:
        entry.update(status='no_model_initializer', correct=False, matrix=None)
    else:
        x, y = candidate['region'][:2]
        initial = np.array([[1., 0, x], [0, 1., y], [0, 0, 1.]]) @ np.array(candidate['matrix'])
        image = bgr.min(axis=2).astype(np.float32)
        no_exclusions = np.zeros((1080, 1920), np.uint8)
        start_fit = time.monotonic()
        coarse = [fit.fit_line(o, minimum_frames=1) for o in fit.extract([image], initial, no_exclusions, fit.CFG['radius_px'])]
        h0, supported0, diagnostic0 = fit.court_consensus(coarse, initial)
        h1 = None; supported1 = []; diagnostic1 = {'status': 'coarse_fit_failed'}
        if h0 is not None:
            fine = [fit.fit_line(o, minimum_frames=1) for o in fit.extract([image], h0, no_exclusions, fit.CFG['fine_radius_px'])]
            h1, supported1, diagnostic1 = fit.court_consensus(fine, h0)
        fit_seconds = time.monotonic() - start_fit
        points = fit.project(h1) if h1 is not None else None
        error = errors(points, row['camera'])
        entry.update(status=diagnostic1['status'], matrix=h1.tolist() if h1 is not None else None,
            correct=correct(points, row['camera']), mean_error_px=float(error.mean()) if len(error) else None,
            max_error_px=float(error.max()) if len(error) else None, coarse_diagnostic=diagnostic0,
            coarse_supported_lines=[fit.NAMES[i] for i in supported0], fine_supported_lines=[fit.NAMES[i] for i in supported1],
            fit_seconds=fit_seconds, no_fallback=True)
    write_json(BASE / 'rgb-pilot' / row['key'] / 'result.json', entry)
    return entry


def main():
    selected = [r for _, r in source_rows() if int(r['key'].split('/clip_')[1].split('/')[0]) in CLIPS]
    assert len(selected) == 32 and not set(CLIPS) & {0, 17, 34, 51, 68}
    protocol = {'clips': CLIPS, 'cases': len(selected), 'fitter_source_sha256': hashlib.sha256((WHITE / 'fit_reference.py').read_bytes()).hexdigest(),
        'RGB_only_after_model_initializer': True, 'manual_masks_used': False, 'reference_used_during_fit': False,
        'parameters': fit.CFG, 'single_frame_minimum_frames': 1, 'input_selection': 'eight spaced clip indices, disjoint from all reference-building clips',
        'not_independent_validation': 'same camera/template and fitter family as reference; shared bias remains even with disjoint frames',
        'not_a_new_acceptance_rule': 'RGB fit success is not declared accepted; no gate thresholds are tuned'}
    write_json(BASE / 'rgb-pilot/protocol.json', protocol)
    started = time.monotonic(); rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, row) for row in selected]
        for f in as_completed(futures):
            row = f.result(); rows.append(row)
            print(row['key'], row['old_correct'], row['status'], row['correct'], row.get('mean_error_px'), flush=True)
    write_json(BASE / 'rgb-pilot/results.json', sorted(rows, key=lambda r: r['key']))
    write_json(BASE / 'rgb-pilot/complete.json', {'cases': len(rows), 'seconds': time.monotonic() - started, 'workers': 4})


if __name__ == '__main__': main()
