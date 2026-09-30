"""Audit stored H dev data and measure synthetic 2D coverage without test scoring."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.ball_refiner.evaluation.hdr import highest_density_regions
from src.tasks.ball_refiner.refiner_2d.distribution import BallGMM2D
from src.tasks.ball_refiner.refiner_3d.diffusion.data import rally_window
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset
from src.tasks.ball_refiner.refiner_3d.synthetic.generator import write_json


def verify_dev_dataset(dataset: Path, output: Path, *, samples: int = 2048) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    torch.set_num_threads(1)
    source = SyntheticDataset(dataset)
    plan = source.manifest['plan']
    if source.manifest['counts'] != {'train': 64, 'val': 16, 'test': 16} or plan['degradation']['boundary_convergence']['method'] != 'fixed_hybrid':
        raise ValueError('Require the complete 96-rally H development set')
    for name, digest in source.manifest['input_hashes'].items():
        if sha256(Path(name)) != digest:
            raise ValueError(f'Generation input changed: {name}')
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {'status': 'running', 'dataset': str(dataset), 'manifest_sha256': sha256(dataset / 'manifest.json'),
        'rallies': [], 'samples': samples, 'levels': [.5, .9, .95],
        'coverage_scope': 'all in-image camera-frames in train/val; observed vs artificial gap; test structural validation only',
        'hdr_method': 'shared #935 highest_density_regions, full R2 Gaussian mixture, float64 Monte Carlo; conditional on existence',
        'presence_mass_max_error': 0., 'weight_sum_max_error': 0.}
    aggregate: dict[str, list[np.ndarray[Any, Any]]] = defaultdict(list)
    try:
        for record in source.records:
            arrays = source.load(record)
            identity = record['rally_id']
            if json.loads((dataset / (identity + '.json')).read_text()) != record:
                raise ValueError('Rally JSON/manifest mismatch')
            split_index = ('train', 'val', 'test').index(record['split'])
            index = int(identity.rsplit('-', 1)[1])
            expected_seed = int(np.random.SeedSequence([plan['seed'], split_index, index]).generate_state(1)[0])
            geometry = plan['geometry']['sources'][split_index]
            if record['seed'] != expected_seed or record['geometry_clip'] != geometry['clip_id'] or record['geometry_sha256'] != geometry['sha256']:
                raise ValueError('Seed/geometry/split identity changed')
            for key in ('gmm3d_means_m', 'gmm3d_covariance_m2', 'gmm3d_weights', 'gmm2d_means_uv', 'gmm2d_scale_tril_uv'):
                if arrays[key].dtype != np.float32:
                    raise ValueError('Expected float32 storage: ' + key)
            np.linalg.cholesky(arrays['gmm3d_covariance_m2'])
            tril = arrays['gmm2d_scale_tril_uv']
            np.linalg.cholesky(tril @ tril.swapaxes(-1, -2))
            subsets, weights = arrays['gmm3d_camera_subsets'], arrays['gmm3d_weights']
            presence = torch.sigmoid(torch.from_numpy(arrays['gmm2d_presence_logits'])).numpy().T.astype(np.float64)
            mass_error = max(float(np.max(np.abs(weights[:, (subsets[0] == mask).all(-1)].astype(np.float64).sum(-1)
                                                     - np.prod(np.where(mask, presence, 1 - presence), axis=-1))))
                             for mask in np.unique(subsets[0], axis=0))
            report['presence_mass_max_error'] = max(report['presence_mass_max_error'], mass_error)
            report['weight_sum_max_error'] = max(report['weight_sum_max_error'], float(np.max(np.abs(weights.astype(np.float64).sum(-1) - 1))))
            row: dict[str, Any] = {'rally_id': identity, 'seed': record['seed'], 'frames': record['frames'],
                'npz_sha256': record['npz_sha256'], 'npz_bytes': record['npz_bytes'], 'components': weights.shape[1],
                'float32_spd': True, 'unassessed_frames': int((~arrays['integration_convergence_assessed']).sum()),
                'converged_frames': int(arrays['integration_converged'].sum()),
                'zero_weight_components': int((weights == 0).sum()),
                'all_camera_gap_frames': int(arrays['occlusion_mask'].all(0).sum()),
                'out_of_frame_camera_frames': int(arrays['out_of_frame_mask'].sum()),
                'presence_mass_max_error': mass_error, 'hdr': {}}
            # Test data is loaded solely for storage/contract verification above.
            if record['split'] != 'test':
                batch = rally_window(arrays, record, start=0, frames=record['frames'], allow_nonconverged=True)
                if batch.condition.means_m.shape != (1, record['frames'], 125, 3):
                    raise ValueError('Reader-to-training round trip lost components')
                truth = arrays['positions_3d_m'].astype(np.float64)
                camera_xyz = np.einsum('vij,tj->vti', arrays['camera_true_R'], truth) + arrays['camera_true_t'][:, None]
                homogeneous = np.einsum('vij,vtj->vti', arrays['camera_true_K'], camera_xyz)
                truth_uv = homogeneous[..., :2] / homogeneous[..., 2, None] / (arrays['source_size_wh'] - 1)[:, None]
                coverage = np.zeros((3, record['frames'], 3), dtype=bool)
                thresholds = np.full((3, record['frames'], 3), np.nan)
                in_image = ~arrays['out_of_frame_mask']
                for camera in range(3):
                    selected = in_image[camera]
                    if not selected.any():
                        continue
                    distribution = BallGMM2D(*(torch.from_numpy(arrays[key][camera:camera + 1, selected]) for key in (
                        'gmm2d_means_uv', 'gmm2d_scale_tril_uv', 'gmm2d_mixture_logits', 'gmm2d_presence_logits')))
                    hdr = highest_density_regions(distribution, torch.from_numpy(truth_uv[camera:camera + 1, selected]),
                                                  levels=(.5, .9, .95), samples=samples, seed=record['seed'] + camera, chunk_size=32)
                    coverage[camera, selected] = hdr.covered[0].numpy()
                    thresholds[camera, selected] = hdr.log_threshold[0].numpy()
                np.savez_compressed(output / (identity + '-hdr.npz'), covered=coverage, log_threshold_uv=thresholds,
                                    in_image=in_image, occlusion=arrays['occlusion_mask'])
                for condition, selected in (('observed', in_image & ~arrays['occlusion_mask']), ('gap', in_image & arrays['occlusion_mask'])):
                    values = coverage[selected]
                    row['hdr'][condition] = {'count': len(values), 'covered': values.sum(0).tolist()}
                    aggregate[record['split'] + '/' + condition].append(values)
                    aggregate['train_val/' + condition].append(values)
            report['rallies'].append(row)
            write_json(output / 'report.json', report)
            print(json.dumps({'verified': identity, 'frames': record['frames'], 'hdr': row['hdr']}), flush=True)
        report['coverage'] = {}
        for name, chunks in aggregate.items():
            values = np.concatenate(chunks)
            report['coverage'][name] = {'count': len(values), 'covered': values.sum(0).tolist(),
                                       'hdr50_90_95': values.mean(0).tolist() if len(values) else None}
        report.update(status='complete', frames=sum(r['frames'] for r in report['rallies']),
                      components=sum(r['frames'] * r['components'] for r in report['rallies']),
                      npz_bytes=sum(r['npz_bytes'] for r in report['rallies']),
                      dataset_bytes=sum(p.stat().st_size for p in dataset.iterdir() if p.is_file()),
                      elapsed_seconds=time.perf_counter() - started)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter() - started)
        write_json(output / 'report.json', report)
        raise
    write_json(output / 'report.json', report)
    return report
