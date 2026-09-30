"""Fixed sample replay. Persist each frame, including failures, without reselection."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.refiner_3d.triangulation import frame_observations
from src.utils.geometry.probabilistic_triangulation import GaussianPrior3D, LaplaceConfig
from src.utils.geometry.probabilistic_triangulation.convergence import convergence_config, triangulate_converged
from src.utils.geometry.triangulation import PinholeCamera


@lru_cache(maxsize=10)
def load_rally(path: str, sha: str):
    torch.set_num_threads(1)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
        raise ValueError('Source hash mismatch')
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files if not k.startswith(('gmm3d', 'integration'))}
    cameras = tuple(PinholeCamera(str(i), a['camera_estimated_K'][i], a['camera_estimated_R'][i], a['camera_estimated_t'][i]) for i in range(3))
    gmm = BallGMM2D(*(torch.from_numpy(a[k]) for k in ('gmm2d_means_uv', 'gmm2d_scale_tril_uv', 'gmm2d_mixture_logits', 'gmm2d_presence_logits')))
    return a, cameras, gmm


@lru_cache(maxsize=10)
def recalibrated_rally(path, sha, settings_json, seed):
    from src.tasks.ball_refiner.refiner_3d.synthetic.calibration import load_calibration
    from src.tasks.ball_refiner.refiner_3d.synthetic.observations import calibrated_distribution
    a, cameras, _ = load_rally(path, sha)
    settings = json.loads(settings_json)
    bank = load_calibration(Path(settings['calibration']['bank']), settings['calibration']['bank_sha256'])
    projection = np.stack([c.project(a['positions_3d_m'])[0] for c in cameras])
    distribution, _, _ = calibrated_distribution(projection, a['source_size_wh'].astype(float), settings, np.random.default_rng(seed),
        calibration=bank, occlusion=a['occlusion_mask'], out_of_frame=a['out_of_frame_mask'])
    return a, cameras, distribution


def one(job):
    row, source, settings, output = job
    a, cameras, gmm = load_rally(source['path'], source['sha256'])
    if settings.get('audit_recalibration', False):
        a, cameras, gmm = recalibrated_rally(source['path'], source['sha256'], json.dumps(settings, sort_keys=True), source['seed'])
    observations = frame_observations(gmm, torch.from_numpy(a['source_size_wh'].astype(float)), frame=row['frame'])
    began = time.perf_counter()
    cpu_began = time.process_time()
    result = dict(row)
    name = f"{row['rally_id']}-{row['frame']:05d}"
    try:
        checked = triangulate_converged(observations, cameras, prior=GaussianPrior3D(np.array(settings['prior_mean_m']), np.diag(settings['prior_covariance_diagonal_m2'])), laplace=LaplaceConfig(settings['max_components'], settings['max_nfev']), config=convergence_config(settings['boundary_convergence']))
        result.update(seconds=time.perf_counter()-began, cpu_seconds=time.process_time()-cpu_began,
                      converged=checked.converged, rounds=checked.rounds, history=checked.history,
                      components=len(checked.component_converged), methods=checked.posterior.component_methods,
                      integration_diagnostics=getattr(checked.posterior, 'component_integration_diagnostics', ()),
                      nonconverged_mass=float(checked.posterior.distribution.weights[~checked.component_converged].sum()))
        if len(checked.component_converged) != 125:
            raise ValueError('K4 audit requires all 125 components')
        p = checked.posterior
        np.savez_compressed(Path(output)/(name+'.npz'), means=p.distribution.means, covariance=p.distribution.covariance,
                            weights=p.distribution.weights, log_evidence=p.component_log_evidence, camera_subsets=p.camera_subsets,
                            component_changes=checked.component_changes, component_converged=checked.component_converged)
    except Exception as exc:
        result.update(seconds=time.perf_counter()-began, cpu_seconds=time.process_time()-cpu_began, converged=False, error=repr(exc))
    (Path(output)/(name+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settings', type=Path)
    parser.add_argument('--calibration-report', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers not in range(1, 5) or not args.output.is_absolute():
        raise ValueError('Use 1..4 workers and absolute output')
    sample = json.loads(args.sample.read_text())
    settings = sample['settings'] if args.settings is None else json.loads(args.settings.read_text())
    if args.calibration_report is not None:
        from src.tasks.ball_refiner.refiner_3d.synthetic.calibration import with_calibration_report
        if not args.calibration_report.is_absolute():
            raise ValueError('Use an absolute calibration report')
        settings = with_calibration_report(settings, args.calibration_report)
        settings['audit_recalibration'] = True
    args.output.mkdir(parents=True, exist_ok=False)
    import src.utils.geometry.probabilistic_triangulation as module
    code = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(module.__file__).parent.glob('*.py'))}
    state = {'sample_sha256': hashlib.sha256(args.sample.read_bytes()).hexdigest(), 'settings': settings,
             'code': code, 'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'workers': args.workers}
    (args.output/'provenance.json').write_text(json.dumps(state, indent=2)+'\n')
    jobs = [(row, sample['sources'][row['rally_id']], settings, str(args.output)) for row in sample['frames']]
    began = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 20 == 0:
                print(json.dumps({'done': len(rows), 'converged': sum(r['converged'] for r in rows), 'failures': sum('error' in r for r in rows), 'wall_seconds': time.perf_counter()-began}), flush=True)
    state.update(wall_seconds=time.perf_counter()-began, records=sorted(rows, key=lambda r: (r['rally_id'], r['frame'])))
    (args.output/'results.json').write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps({k: v for k, v in state.items() if k not in ('records', 'code', 'settings')}), flush=True)


if __name__ == '__main__':
    main()
