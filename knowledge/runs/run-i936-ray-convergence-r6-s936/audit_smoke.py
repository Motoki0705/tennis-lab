"""Replay every fixed smoke frame in bounded CPU batches, without regeneration."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import yaml

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.refiner_3d.synthetic.generator import write_json
from src.tasks.ball_refiner.refiner_3d.triangulation import frame_observations
from src.utils.geometry.probabilistic_triangulation import GaussianPrior3D, LaplaceConfig
from src.utils.geometry.probabilistic_triangulation.convergence import convergence_config, triangulate_converged
from src.utils.geometry.triangulation import PinholeCamera


@lru_cache(maxsize=2)
def source_rally(source, rally_id):
    torch.set_num_threads(1)
    manifest = json.loads((Path(source) / 'manifest.json').read_text())
    record = next(r for r in manifest['rallies'] if r['rally_id'] == rally_id)
    path = Path(source) / (rally_id + '.npz')
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['npz_sha256']:
        raise ValueError('Source rally SHA mismatch')
    with np.load(path, allow_pickle=False) as z:
        a = {key: z[key] for key in z.files}
    cameras = tuple(PinholeCamera(str(i), a['camera_estimated_K'][i], a['camera_estimated_R'][i], a['camera_estimated_t'][i]) for i in range(3))
    distribution = BallGMM2D(*(torch.from_numpy(a[key]) for key in ('gmm2d_means_uv', 'gmm2d_scale_tril_uv', 'gmm2d_mixture_logits', 'gmm2d_presence_logits')))
    return a, cameras, distribution


def chunk(job):
    source, output, settings, rally_id, start, stop = job
    a, cameras, distribution = source_rally(source, rally_id)
    prior = GaussianPrior3D(np.array(settings['prior_mean_m']), np.diag(settings['prior_covariance_diagonal_m2']))
    config = convergence_config(settings['boundary_convergence'])
    records, checks = [], []
    for frame in range(start, stop):
        began = time.monotonic()
        observations = frame_observations(distribution, torch.from_numpy(a['source_size_wh'].astype(float)), frame=frame)
        try:
            checked = triangulate_converged(observations, cameras, prior=prior, laplace=LaplaceConfig(64, settings['max_nfev']), config=config)
        except Exception as exc:
            raise RuntimeError(f'{rally_id}/frame{frame}: {exc}') from exc
        records.append({'rally_id': rally_id, 'frame': frame, 'converged': checked.converged, 'rounds': checked.rounds,
                        'nll_at_synthetic_truth': -float(checked.posterior.distribution.log_prob(a['positions_3d_m'][frame])),
                        'seconds': time.monotonic() - began, 'history': checked.history})
        checks.append(checked)
    name = f'{rally_id}-{start:05d}-{stop:05d}'
    np.savez_compressed(Path(output) / (name + '.npz'), frames=np.arange(start, stop),
        method_codes=np.array([[__import__('src.utils.geometry.probabilistic_triangulation.solver',fromlist=['COMPONENT_METHODS']).COMPONENT_METHODS.index(m) for m in r.posterior.component_methods] for r in checks], dtype=np.uint8),
        log_evidence=np.stack([r.posterior.component_log_evidence for r in checks]),
        means=np.stack([r.posterior.distribution.means for r in checks]),
        covariance=np.stack([r.posterior.distribution.covariance for r in checks]),
        weights=np.stack([r.posterior.distribution.weights for r in checks]),
        component_changes=np.stack([r.component_changes for r in checks]),
        component_converged=np.stack([r.component_converged for r in checks]))
    write_json(Path(output) / (name + '.json'), records)
    return records


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'output', 'plan'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not all(p.is_absolute() for p in (args.source, args.output, args.plan)):
        parser.error('Use absolute paths')
    manifest = json.loads((args.source / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or len(manifest['rallies']) != 12:
        raise ValueError('Require complete fixed 12-rally smoke')
    settings = yaml.safe_load(args.plan.read_text())['degradation']
    selected_start, selected_stop = args.batch * args.batch_size, (args.batch + 1) * args.batch_size
    if args.workers not in range(1, 5) or args.batch < 0 or args.batch_size < 1 or selected_start >= sum(r['frames'] for r in manifest['rallies']):
        raise ValueError('Invalid batch')
    args.output.mkdir(parents=True, exist_ok=True)
    progress = args.output / f'progress-{args.batch}.json'
    if progress.exists():
        raise FileExistsError(progress)
    offset, jobs = 0, []
    for record in manifest['rallies']:
        begin, end = max(0, selected_start - offset), min(record['frames'], selected_stop - offset)
        for start in range(begin, end, 32):
            jobs.append((str(args.source), str(args.output), settings, record['rally_id'], start, min(start + 32, end)))
        offset += record['frames']
    began = time.monotonic()
    records, failures = [], []
    state = {'status': 'running', 'batch': args.batch, 'batch_size': args.batch_size, 'workers': args.workers, 'source_manifest_sha256': hashlib.sha256((args.source / 'manifest.json').read_bytes()).hexdigest(), 'plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(), 'settings': settings['boundary_convergence']}
    write_json(progress, state)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(chunk, job): job[3:] for job in jobs}
        for future in as_completed(futures):
            try:
                records.extend(future.result())
            except Exception as exc:
                failures.append({'chunk': futures[future], 'error': repr(exc)})
            state.update(completed_frames=len(records), nonconverged_frames=sum(not r['converged'] for r in records), failures=failures, elapsed_seconds=time.monotonic() - began)
            write_json(progress, state)
    state.update(status='failed' if failures else 'complete', records=sorted(records, key=lambda r: (r['rally_id'], r['frame'])))
    write_json(progress, state)
    print(json.dumps({k: v for k, v in state.items() if k != 'records'}), flush=True)
    if failures:
        raise RuntimeError('All chunk failures recorded')


if __name__ == '__main__':
    main()
