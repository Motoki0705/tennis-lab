"""Confirm frame-derived hypotheses on all parent-approved fit frames, on CPU.

The 276 held-out frames and all reference matrices remain unread. A camera can
fail even when many individual frames have a numerical homography.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from src.tasks.court_detection.geometry.video_shared.candidates import CourtSeed
from src.tasks.court_detection.geometry.video_shared.consensus import merge_measurements
from src.tasks.court_detection.geometry.video_shared.fit import PaintGroup, SharedPaintConfig, fit_common_court
from src.tasks.court_detection.geometry.video_shared.observations import clustered_initializers, measure_frame
from probe import save, sha


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def capture_video(task):
    dataset, output, clip_id, camera, frame_rows, first_hash = task
    cv2.setNumThreads(1)
    path = dataset / 'videos/video_000/clips' / clip_id.split('/')[1] / 'media' / f'{camera}.mp4'
    digest = hash_file(path)
    capture = cv2.VideoCapture(str(path))
    ok, first = capture.read()
    if not ok or hashlib.sha256(cv2.cvtColor(first, cv2.COLOR_BGR2RGB).tobytes()).hexdigest() != first_hash:
        raise ValueError(f'Frozen model RGB does not match current media: {clip_id}/{camera}')
    records = []
    for row in frame_rows:
        capture.set(cv2.CAP_PROP_POS_FRAMES, row['frame_index'])
        ok, image = capture.read()
        if not ok or image.shape != (1080, 1920, 3):
            raise ValueError(f'Frame decode failed: {row}')
        target = output / 'frames' / camera / f'{clip_id.split("/")[1]}-{row["frame_index"]}.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), image):
            raise OSError(f'Could not write frame: {target}')
        records.append({**row, 'path': str(target.relative_to(output)), 'png_sha256': sha(target),
                        'rgb_sha256': hashlib.sha256(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()})
    capture.release()
    if hash_file(path) != digest:
        raise ValueError(f'Input media changed during capture: {path}')
    return records, {'path': str(path.relative_to(dataset)), 'sha256': digest}


def prepare(args):
    split = json.loads(args.split.read_text())
    old_manifest = json.loads(args.raw_manifest.read_text())
    rgb_hashes = {(r['clip_id'], r['camera']): r['rgb_sha256'] for r in old_manifest['records'] if r['dataset'] == 'meiji_4cam'}
    source = json.loads((args.probe / 'protocol.json').read_text())
    assert (args.probe / 'summary.json').is_file()
    search = [json.loads(p.read_text()) for p in sorted((args.probe / 'search').glob('*.json'))]
    candidates = {}
    assignments = {}
    for camera in ('cam0', 'cam1', 'cam2', 'cam3'):
        seeds = []
        for row in search:
            if row['camera'] != camera:
                continue
            for frame in row['frames']:
                if frame['reason'] == 'measured':
                    seeds.append(CourtSeed(f"{row['seed_id']}/measured/{frame['clip_id']}/{frame['frame_index']}",
                                           frame['clip_id'], frame['frame_index'], np.array(frame['chosen_homography']), False))
        kept, diagnostic = clustered_initializers(tuple(seeds), SharedPaintConfig())
        candidates[camera] = [{'id': seed.candidate_id, 'matrix': seed.matrix.tolist()} for seed in kept]
        assignments[camera] = diagnostic
    tasks = []
    clips = sorted({r['clip_id'] for r in split['fit']})
    for camera in candidates:
        for clip in clips:
            tasks.append((args.dataset, args.output, clip, camera,
                          [r for r in split['fit'] if (r['clip_id'], r['camera_id']) == (clip, camera)], rgb_hashes[clip, camera]))
    print('model-derived frame initializers', {c: len(v) for c, v in candidates.items()}, flush=True)
    records, media = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for frames, video in pool.map(capture_video, tasks):
            records.extend(frames)
            media.append(video)
    protocol = {'schema': 'shared_court_all_fit_frames_v1', 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'source_hashes': {str(p): sha(p) for p in sorted(Path('src/tasks/court_detection/geometry/video_shared').glob('*.py'))},
                'driver_sha256': sha(Path(__file__)), 'probe_protocol_sha256': sha(args.probe / 'protocol.json'),
                'settings': asdict(SharedPaintConfig()), 'candidates': candidates, 'assignments': assignments,
                'frames': records, 'media': media, 'dataset_manifest_sha256': sha(args.dataset / 'dataset.json'),
                'split_sha256': sha(args.split), 'raw_manifest_sha256': sha(args.raw_manifest),
                'reference_h_used': False, 'hand_mask_used': False, 'evaluation_frames_used': False,
                'source_model_commit': source['head']}
    save(args.output / 'protocol.json', protocol)
    print('prepared', len(records), 'fit frames; all first-frame RGB hashes matched the frozen model inputs', flush=True)


def measure(task):
    output, camera, index, seed, frame = task
    cv2.setNumThreads(1)
    rgb_path = output / frame['path']
    assert sha(rgb_path) == frame['png_sha256']
    rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
    assert hashlib.sha256(rgb.tobytes()).hexdigest() == frame['rgb_sha256']
    started = time.monotonic()
    result = measure_frame(rgb, np.array(seed['matrix']), clip_id=frame['clip_id'], frame_index=frame['frame_index'], config=SharedPaintConfig())
    return {'camera': camera, 'initializer': index, 'frame': frame, 'diagnostic': result.diagnostic,
            'groups': [{'clip_id': g.clip_id, 'frame_index': g.frame_index, 'line_index': g.line_index,
                        'points': g.points.tolist(), 'endings': g.endings} for g in result.groups],
            'seconds': time.monotonic() - started}


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only run requires CUDA_VISIBLE_DEVICES to be empty')
    started = time.monotonic()
    protocol = json.loads((args.output / 'protocol.json').read_text())
    assert protocol['driver_sha256'] == sha(Path(__file__))
    for path, digest in protocol['source_hashes'].items():
        assert sha(Path(path)) == digest
    tasks = [(args.output, camera, index, seed, frame) for camera, seeds in protocol['candidates'].items()
             for index, seed in enumerate(seeds) for frame in protocol['frames'] if frame['camera_id'] == camera]
    print('measurements', len(tasks), flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(measure, task): index for index, task in enumerate(tasks)}
        for future in as_completed(pending):
            result = future.result()
            save(args.output / 'measurements' / f'{pending[future]:05d}.json', result)
            results.append(result)
            if len(results) % 100 == 0:
                print('completed', len(results), '/', len(tasks), flush=True)
    summary = {'seconds': time.monotonic() - started, 'workers': args.workers, 'cameras': {}}
    for camera, seeds in protocol['candidates'].items():
        hypotheses = []
        individual = []
        for index, seed in enumerate(seeds):
            selected = [r for r in results if r['camera'] == camera and r['initializer'] == index]
            groups = tuple(PaintGroup(g['clip_id'], g['frame_index'], g['line_index'], np.array(g['points']), tuple(g['endings']))
                           for r in selected for g in r['groups'])
            fit = fit_common_court(np.array(seed['matrix']), groups, (1920, 1080), SharedPaintConfig())
            individual.append({'initializer': index, 'matrix': None if fit.matrix is None else fit.matrix.tolist(),
                               'fitted_matrix': None if fit.fitted_matrix is None else fit.fitted_matrix.tolist(),
                               'reasons': fit.reasons, 'diagnostic': fit.diagnostic,
                               'frame_reasons': dict(Counter(r['diagnostic']['reason'] for r in selected))})
            if fit.matrix is not None:
                hypotheses.append(groups)
        merged = None if not hypotheses else merge_measurements(np.array(seeds[0]['matrix']), tuple(hypotheses), (1920, 1080), SharedPaintConfig())
        summary['cameras'][camera] = {'individual': individual, 'accepted_individual': len(hypotheses),
                                      'matrix': None if merged is None or merged.matrix is None else merged.matrix.tolist(),
                                      'reasons': ['no_acceptable_shared_court'] if merged is None else merged.reasons,
                                      'diagnostic': None if merged is None else merged.diagnostic}
    for record in protocol['media']:
        assert hash_file(args.dataset / record['path']) == record['sha256']
    save(args.output / 'summary.json', summary)
    print({c: {'accepted_individual': v['accepted_individual'], 'reasons': v['reasons']} for c, v in summary['cameras'].items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--split', type=Path)
    parser.add_argument('--raw-manifest', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args)
