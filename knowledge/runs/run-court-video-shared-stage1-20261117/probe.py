"""CPU-only, content-pinned shared-court feasibility run. No reference H input.

Paths are CLI inputs; old run directories are read-only. Frame selection comes
from the parent-approved planning manifest. Per-candidate evidence remains in
outputs, while the small summary and protocol are promoted to knowledge.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path.cwd()))
from src.tasks.court_detection.geometry.video_shared.candidates import read_frozen_candidates
from src.tasks.court_detection.geometry.video_shared.fit import SharedPaintConfig, choose_common_solution, fit_common_court, SharedFit
from src.tasks.court_detection.geometry.video_shared.observations import clustered_initializers, measure_frame


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare(args):
    old = args.old_outputs
    inventory_spec = json.loads(Path('knowledge/runs/run-court-legacy-default-20261117/artifact-inventories.json').read_text())['comparisons']
    inventory_path = old / 'court-legacy-default-20261117/inventories/comparisons.jsonl'
    assert sha(inventory_path) == inventory_spec['sha256']
    inventory = {r['path']: r for r in map(json.loads, inventory_path.read_text().splitlines())}
    split = json.loads(args.split.read_text())
    selected = sorted({r['clip_id'] for r in split['seed']})
    candidates = []
    for camera in ('cam0', 'cam1', 'cam2', 'cam3'):
        rows = []
        for clip in selected:
            relative = f'evaluation/meiji_4cam/{clip}/{camera}/comparison.json'
            path = old / 'court-legacy-default-20261117' / relative
            assert path.stat().st_size == inventory[relative]['bytes'] and sha(path) == inventory[relative]['sha256']
            data = json.loads(path.read_text())
            for g, quality in zip(data['geometry'], data['old_candidates'], strict=True):
                rows.append({'candidate_id': f'{clip}/{camera}/region_{g["region_index"]:02d}',
                             'clip_id': clip, 'frame_index': 0, 'matrix': g['old_h'],
                             'model_accepted': quality['accepted'], 'region_xyxy': g['region'], 'source_sha256': sha(path)})
        document = {'schema': 'court_seed_manifest_v1', 'dataset_id': 'meiji_4cam', 'video_id': 'video_000',
                    'camera_id': camera, 'size_wh': [1920, 1080], 'candidates': rows,
                    'provenance': {'run': 'run-court-legacy-default-20261117', 'source_inventory_sha256': sha(inventory_path),
                                   'coordinate_convention': 'crop_pixels; translation applied by reader', 'seed_clip_ids': selected}}
        path = args.output / 'candidates' / f'{camera}.json'
        save(path, document)
        candidates.append({'camera': camera, 'path': str(path.relative_to(args.output)), 'sha256': sha(path)})
    # Probe on three time-separated clips, all from the approved fit subset.
    # No held-out evaluation or reference RGB is loaded by this script.
    probes = []
    for camera in ('cam0', 'cam1', 'cam2', 'cam3'):
        for clip in ('video_000/clip_001', 'video_000/clip_033', 'video_000/clip_067'):
            rows = [r for r in split['fit'] if r['camera_id'] == camera and r['clip_id'] == clip]
            row = min(rows, key=lambda r: r['frame_index'])
            video = args.dataset / 'videos/video_000/clips' / clip.split('/')[1] / 'media' / f'{camera}.mp4'
            capture = cv2.VideoCapture(str(video))
            capture.set(cv2.CAP_PROP_POS_FRAMES, row['frame_index'])
            ok, image = capture.read()
            capture.release()
            assert ok and image.shape == (1080, 1920, 3)
            path = args.output / 'frames' / camera / f'{clip.split("/")[1]}-{row["frame_index"]}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(path), image)
            probes.append({**row, 'path': str(path.relative_to(args.output)), 'png_sha256': sha(path),
                           'rgb_sha256': hashlib.sha256(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()})
    protocol = {'schema': 'shared_court_feasibility_v1', 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'source_hashes': {str(p): sha(p) for p in sorted(Path('src/tasks/court_detection/geometry/video_shared').glob('*.py'))},
                'driver_sha256': sha(Path(__file__)), 'settings': asdict(SharedPaintConfig()),
                'dataset_manifest_sha256': sha(args.dataset / 'dataset.json'), 'split_sha256': sha(args.split),
                'candidates': candidates, 'probe_frames': probes, 'seed_clip_ids': selected,
                'reference_h_used': False, 'hand_mask_used': False, 'gpu_used': False,
                'scope': 'stage1 initializer search on three fit frames/camera, not the full acceptance evaluation'}
    save(args.output / 'protocol.json', protocol)
    print('prepared', len(probes), 'RGB frames', flush=True)


def process(payload):
    cv2.setNumThreads(1)
    output, camera, seed, frames = payload
    started = time.monotonic()
    config = SharedPaintConfig()
    measurements = []
    groups = []
    for record in frames:
        path = output / record['path']
        assert sha(path) == record['png_sha256']
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == record['rgb_sha256']
        observed = measure_frame(rgb, seed.matrix, clip_id=record['clip_id'], frame_index=record['frame_index'], config=config)
        measurements.append(observed.diagnostic)
        groups.extend(observed.groups)
    fit = fit_common_court(seed.matrix, tuple(groups), (1920, 1080), config)
    result = {'camera': camera, 'seed_id': seed.candidate_id, 'seed_homography': seed.matrix.tolist(),
              'seed_model_accepted': seed.model_accepted, 'frames': measurements,
              'matrix': None if fit.matrix is None else fit.matrix.tolist(),
              'fitted_matrix': None if fit.fitted_matrix is None else fit.fitted_matrix.tolist(),
              'reasons': fit.reasons, 'diagnostic': fit.diagnostic, 'seconds': time.monotonic() - started}
    return result


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only run requires CUDA_VISIBLE_DEVICES to be empty')
    protocol = json.loads((args.output / 'protocol.json').read_text())
    assert protocol['driver_sha256'] == sha(Path(__file__))
    for path, digest in protocol['source_hashes'].items():
        assert sha(Path(path)) == digest
    started = time.monotonic()
    tasks = []
    selections = {}
    for item in protocol['candidates']:
        camera = item['camera']
        seeds = read_frozen_candidates(args.output / item['path'], sha256=item['sha256'], dataset_id='meiji_4cam',
                                      video_id='video_000', camera_id=camera, size_wh=(1920, 1080),
                                      clip_ids=tuple(protocol['seed_clip_ids']))
        initializers, diagnostic = clustered_initializers(seeds, SharedPaintConfig())
        save(args.output / 'initializers' / f'{camera}.json', diagnostic)
        selections[camera] = diagnostic
        frames = [r for r in protocol['probe_frames'] if r['camera_id'] == camera]
        tasks.extend((args.output, camera, seed, frames) for seed in initializers)
    print('initializers', {camera: value['initializers'] for camera, value in selections.items()}, flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(process, task): index for index, task in enumerate(tasks)}
        for future in as_completed(pending):
            result = future.result()
            result['index'] = pending[future]
            save(args.output / 'search' / f'{pending[future]:04d}.json', result)
            results.append(result)
            if len(results) % 10 == 0:
                print('completed', len(results), '/', len(tasks), 'accepted', sum(r['matrix'] is not None for r in results), flush=True)
    summary = {'seconds': time.monotonic() - started, 'workers': args.workers, 'cameras': {}}
    for camera in selections:
        selected = sorted((r for r in results if r['camera'] == camera), key=lambda r: r['seed_id'])
        fits = tuple(SharedFit(None if r['matrix'] is None else np.array(r['matrix']),
                               None if r['fitted_matrix'] is None else np.array(r['fitted_matrix']),
                               tuple(r['reasons']), r['diagnostic']) for r in selected)
        choice = choose_common_solution(fits, SharedPaintConfig())
        summary['cameras'][camera] = {'initializers': len(selected), 'accepted_candidates': sum(r['matrix'] is not None for r in selected),
                                      'reasons': choice.reasons, 'diagnostic': choice.diagnostic,
                                      'matrix': None if choice.matrix is None else choice.matrix.tolist()}
    save(args.output / 'summary.json', summary)
    print(json.dumps({camera: {k: v for k, v in value.items() if k != 'diagnostic'} for camera, value in summary['cameras'].items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--old-outputs', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--split', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args)
