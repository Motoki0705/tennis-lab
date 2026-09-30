"""Run one fixed-budget method on the pre-registered sample, preserving all outcomes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import resource
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from src.tasks.ball_refiner.refiner_3d.comparison import triangulate_stratified_samples
from src.tasks.ball_refiner.refiner_3d.scoring import score_mixture
from src.tasks.ball_refiner.refiner_3d.triangulation import frame_observations
from src.utils.geometry.probabilistic_triangulation import GaussianPrior3D, LaplaceConfig, triangulate_gmm
from src.utils.geometry.probabilistic_triangulation.convergence import convergence_config, triangulate_converged
from src.utils.geometry.probabilistic_triangulation.ray import RayConfig
from src.utils.geometry.probabilistic_triangulation.solver import HybridConfig, _triangulate, triangulate_hybrid
from src.utils.geometry.probabilistic_triangulation.volume import VoxelConfig

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'knowledge/runs/run-i936-k4-audit-r7-s93607'
# Reuse the historical hash-checking BallGMM2D input loader. No GT in fitting.
spec = importlib.util.spec_from_file_location('fixed_audit', OLD / 'audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

BUDGETS = {
    'A': {'max_nfev': 100, 'diagnose_nonregular': True},
    'H': {'max_nfev': 100, 'initial_cells': 8, 'levels': 4, 'refine_cells': 64, 'prior_sigmas': 5.},
    'ray': {'settings': 'sample.json:settings.boundary_convergence'},
    'adaptive_ray': {'settings': 'adaptive-settings.json:boundary_convergence'},
    'C': {'samples_per_product': 8, 'seed': 93608, 'max_nfev': 100},
    'Q20': {'max_nfev': 100, 'quadrature_order': 20},
}


def one(job):
    number, row, source, settings, method, output = job
    torch.set_num_threads(1)
    arrays, cameras, gmm = audit.load_rally(source['path'], source['sha256'])
    adapter_began = time.perf_counter()
    obs = frame_observations(gmm, torch.from_numpy(arrays['source_size_wh'].astype(float)), frame=row['frame'])
    adapter_seconds = time.perf_counter() - adapter_began
    prior = GaussianPrior3D(np.asarray(settings['prior_mean_m']), np.diag(settings['prior_covariance_diagonal_m2']))
    began, cpu_began = time.perf_counter(), time.process_time()
    record = dict(row, method=method, adapter_seconds=adapter_seconds)
    name = f"{row['rally_id']}-{row['frame']:05d}"
    try:
        checked = None
        if method == 'A':
            posterior = triangulate_gmm(obs, cameras, prior=prior, config=LaplaceConfig(125, **BUDGETS[method]))
        elif method == 'H':
            b = BUDGETS[method]
            posterior = triangulate_hybrid(obs, cameras, prior=prior, config=HybridConfig(LaplaceConfig(125, b['max_nfev']), VoxelConfig(b['initial_cells'], b['levels'], b['refine_cells'], b['prior_sigmas'])))
        elif method in ('ray', 'adaptive_ray'):
            cfg = settings['boundary_convergence'] if method == 'ray' else json.loads((OLD / 'adaptive-settings.json').read_text())['boundary_convergence']
            checked = triangulate_converged(obs, cameras, prior=prior, laplace=LaplaceConfig(125, 100), config=convergence_config(cfg))
            posterior = checked.posterior
        elif method == 'C':
            posterior = triangulate_stratified_samples(obs, cameras, prior=prior, **BUDGETS[method])
        elif method == 'Q20':
            posterior = _triangulate(obs, cameras, prior=prior, config=LaplaceConfig(125, 100), volume=RayConfig(20))
        else:
            raise ValueError(method)
        record.update(seconds=time.perf_counter() - began, cpu_seconds=time.process_time() - cpu_began)
        p = posterior.distribution
        if len(p.weights) != 125:
            raise ValueError('Every candidate must retain 125 component products')
        np.linalg.cholesky(p.covariance.astype(np.float32))
        error = 0.
        for mask in np.unique(posterior.camera_subsets, axis=0):
            mass = p.weights[(posterior.camera_subsets == mask).all(1)].sum()
            error = max(error, abs(float(mass) - float(np.prod(np.where(mask, obs.presence, 1 - obs.presence)))))
        record.update(components=len(p.weights), subset_mass_error=error, prior_only_probability=posterior.prior_only_probability,
                      mean_presence=obs.presence.tolist(), amodal_present=(~arrays['out_of_frame_mask'][:, row['frame']]).tolist(),
                      component_methods=posterior.component_methods,
                      optimizer=posterior.component_optimization_diagnostics,
                      integration=posterior.component_integration_diagnostics,
                      converged=None if checked is None else checked.converged,
                      history=() if checked is None else checked.history)
        if method in ('A', 'C'):
            record['optimizer_converged'] = all(d['optimizer_converged'] for d in posterior.component_optimization_diagnostics)
        saved = dict(means=p.means, covariance=p.covariance, weights=p.weights, camera_subsets=posterior.camera_subsets,
                     log_evidence=posterior.component_log_evidence)
        if checked is not None:
            saved.update(component_converged=checked.component_converged, component_changes=checked.component_changes)
        np.savez_compressed(Path(output) / (name + '.npz'), **saved)
        scoring_began = time.perf_counter()
        record.update(score_mixture(p, arrays['positions_3d_m'][row['frame']].astype(float), samples=8192, seed=936080000 + number))
        record['scoring_seconds'] = time.perf_counter() - scoring_began
        record['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception as exc:
        record.setdefault('seconds', time.perf_counter() - began)
        record.setdefault('cpu_seconds', time.process_time() - cpu_began)
        record['error'] = repr(exc)
    (Path(output) / (name + '.json')).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=BUDGETS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute():
        raise ValueError('Absolute output required')
    sample_path = OLD / 'sample.json'
    sample_hash = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    if sample_hash != 'df7753a4177ba5171971681064619cb06aaab3dac481409d7cf14781477dee7d':
        raise ValueError('Pre-registered sample changed')
    sample = json.loads(sample_path.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [*sorted((ROOT / 'src/utils/geometry/probabilistic_triangulation').glob('*.py')),
             ROOT / 'src/tasks/ball_refiner/refiner_3d/comparison.py', ROOT / 'src/tasks/ball_refiner/refiner_3d/scoring.py', Path(__file__)]
    state = dict(method=args.method, budget=BUDGETS[args.method], workers=4, native_threads=1,
                 sample_sha256=sample_hash, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                 source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                 scoring={'samples': 8192, 'seed': '936080000 + pre-registered frame index', 'HDR': 'log q(GT) >= quantile_(1-a)(log q(X)), X~q', 'cost': 'solver-only wall seconds per frame with 4 concurrent workers; scoring and I/O excluded'})
    (args.output / 'provenance.json').write_text(json.dumps(state, indent=2) + '\n')
    began = time.perf_counter()
    rows = []
    jobs = [(i, r, sample['sources'][r['rally_id']], sample['settings'], args.method, str(args.output)) for i, r in enumerate(sample['frames'])]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 20 == 0:
                print(json.dumps(dict(method=args.method, done=len(rows), failures=sum('error' in r for r in rows), wall_seconds=time.perf_counter() - began)), flush=True)
    state.update(wall_seconds=time.perf_counter() - began, records=sorted(rows, key=lambda r: (r['rally_id'], r['frame'])))
    (args.output / 'results.json').write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in state.items() if k not in ('records', 'source_sha256')}), flush=True)


if __name__ == '__main__':
    main()
