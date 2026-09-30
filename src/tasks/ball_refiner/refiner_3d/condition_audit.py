"""Audit all dev identities, then score train/val full 3D conditions only."""
from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.utils.geometry.probabilistic_triangulation.distributions import (
    GaussianMixture3D,
)
from src.utils.paths import PROJECT_ROOT

from .baselines import condition_points, evaluate_baselines
from .condition_metrics import LEVELS, mixture_hdr_metrics
from .diffusion.dev_evaluation import RallyInput
from .diffusion.metrics import Array
from .synthetic.calibration import with_calibration_report
from .synthetic.configuration import sha256
from .synthetic.dataset import SyntheticDataset
from .synthetic.generator import write_json

# These must match across banks, before interpreting a paired comparison.
IDENTITY_ARRAYS = ('native_positions_3d_m', 'positions_3d_m', 'timestamps_seconds',
    'event_labels', 'event_region_mask', 'free_flight_mask', 'occlusion_mask', 'out_of_frame_mask',
    'source_size_wh', 'camera_base_K', 'camera_base_R', 'camera_base_t',
    'camera_true_K', 'camera_true_R', 'camera_true_t',
    'camera_estimated_K', 'camera_estimated_R', 'camera_estimated_t')
IDENTITY_METADATA = ('rally_id', 'split', 'seed', 'frames', 'native_frames', 'native_hz',
    'fps_numerator', 'fps_denominator', 'physics', 'physics_proposals', 'events',
    'geometry_clip', 'geometry_sha256', 'gap_intervals', 'end_reason')


def audit_manifest(source: SyntheticDataset) -> dict[str, Any]:
    """File hashes/JSON only for test: no test NPZ array reads or GT scoring."""
    manifest = source.manifest
    if manifest['counts'] != {'train': 64, 'val': 16, 'test': 16} or manifest['failures']:
        raise ValueError('Require a failure-free 96-rally dev set')
    expected_ids = {f'{split}-{i:05d}' for split, count in manifest['counts'].items() for i in range(count)}
    if {r['rally_id'] for r in source.records} != expected_ids or list(source.directory.glob('failed-*.json')):
        raise ValueError('Missing/unexpected rally IDs or failure artifacts')
    plan = manifest['plan']
    if plan['degradation']['boundary_convergence']['method'] != 'fixed_hybrid':
        raise ValueError('Require fixed H without changes to generator method')
    for name, digest in manifest['input_hashes'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('Generation input changed: ' + name)
    plan_paths = [Path(name) for name in manifest['input_hashes'] if Path(name).name == 'dataset_plan.yaml']
    if len(plan_paths) != 1:
        raise ValueError('Need an unambiguous source plan')
    expected = yaml.safe_load(plan_paths[0].read_text())
    calibration = plan['degradation']['calibration']
    expected['degradation'] = with_calibration_report(expected['degradation'], PROJECT_ROOT / calibration['report'])
    if expected != plan or plan['counts']['dev_rallies'] != manifest['counts']:
        raise ValueError('Expanded manifest disagrees with source plan/report')
    records = []
    for record in sorted(source.records, key=lambda r: r['rally_id']):
        identity = record['rally_id']
        stored = source.directory / (identity + '.npz')
        if sha256(stored) != record['npz_sha256'] or stored.stat().st_size != record['npz_bytes']:
            raise ValueError('Stored rally hash/size mismatch: ' + identity)
        if json.loads((source.directory / (identity + '.json')).read_text()) != record:
            raise ValueError('Rally metadata disagrees with manifest: ' + identity)
        split_index = ('train', 'val', 'test').index(record['split'])
        index = int(identity.rsplit('-', 1)[1])
        seed = int(np.random.SeedSequence([plan['seed'], split_index, index]).generate_state(1)[0])
        geometry = plan['geometry']['sources'][split_index]
        if (record['calibration_bank_sha256'] != calibration['bank_sha256']
                or record['seed'] != seed or record['geometry_clip'] != geometry['clip_id']
                or record['geometry_sha256'] != geometry['sha256'] or record['components_per_frame'] != 125):
            raise ValueError('Rally seed/geometry/component identity mismatch')
        records.append({key: record[key] for key in (*IDENTITY_METADATA, 'npz_sha256', 'npz_bytes', 'components_per_frame')})
    if len(records) != 96 or sum(r['frames'] for r in records) != manifest['total_frames']:
        raise ValueError('Aggregate manifest count/frame mismatch')
    return {'manifest_sha256': sha256(source.directory / 'manifest.json'), 'counts': manifest['counts'], 'plan': plan,
            'rallies': records, 'failures': manifest['failures'], 'source_plan_matches': True,
            'input_hashes': manifest['input_hashes'], 'calibration': calibration,
            'frames': manifest['total_frames'], 'components': manifest['total_frames'] * 125,
            'npz_bytes': sum(r['npz_bytes'] for r in records),
            'dataset_bytes': sum(p.stat().st_size for p in source.directory.iterdir() if p.is_file()),
            'test_scope': 'metadata and file SHA only; test arrays never opened',
            'generation_seconds': manifest['elapsed_seconds'], 'sum_rally_seconds': manifest['sum_rally_seconds'],
            'pilot_projection': manifest['pilot_projection']}


def compare_condition_reports(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Reject unpaired settings/trajectories/masks before making a comparison."""
    if any(r['status'] != 'complete' or r['audit_only'] for r in (control, candidate)):
        raise ValueError('Need two complete scored audits')
    if control['samples_each_threshold_and_volume'] != candidate['samples_each_threshold_and_volume']:
        raise ValueError('HDR sampling budgets differ')
    plans = []
    for report in (control, candidate):
        plan = deepcopy(report['audit']['plan'])
        plan['degradation'].pop('status')
        for key in ('bank', 'bank_sha256', 'report', 'report_sha256'):
            plan['degradation']['calibration'].pop(key)
        plans.append(plan)
    if plans[0] != plans[1]:
        raise ValueError('Changes beyond residual bank in generation plans')
    for before, after in zip(control['audit']['rallies'], candidate['audit']['rallies'], strict=True):
        if any(before[key] != after[key] for key in IDENTITY_METADATA):
            raise ValueError('Paired rally metadata differ: ' + before['rally_id'])
    for before, after in zip(control['array_audits'], candidate['array_audits'], strict=True):
        if (before['rally_id'] != after['rally_id'] or before['identity_array_hashes'] != after['identity_array_hashes']):
            raise ValueError('Paired trajectories/cameras/masks differ: ' + before['rally_id'])
    return {'status': 'complete', 'paired_metadata_rallies': len(control['audit']['rallies']),
            'paired_quality_rallies': len(control['array_audits']),
            'control': control['condition_metrics'], 'candidate': candidate['condition_metrics'],
            'baselines': {name: report['baselines'] for name, report in (('control', control), ('candidate', candidate))},
            'test_arrays_read': 0, 'only_plan_change': '2D residual bank/report/status'}


def _summary(chunks: list[dict[str, Array]], camera_count: int | None) -> dict[str, Any]:
    merged = {key: np.concatenate([r[key] for r in chunks]) for key in chunks[0]}
    mask = np.ones(len(merged['nll_nat']), bool) if camera_count is None else merged['visible_cameras'] == camera_count
    count = int(mask.sum())
    if not count:
        return {'frames': 0}
    return {'frames': count, 'nll_nat': float(merged['nll_nat'][mask].mean()),
            'mixture_mean_rmse_m': float(np.sqrt(merged['mean_error2_m2'][mask].mean())),
            'hdr_coverage': merged['covered'][mask].mean(0).tolist(),
            'hdr_volume_mean_m3': merged['volume_m3'][mask].mean(0).tolist(),
            'hdr_volume_median_m3': np.median(merged['volume_m3'][mask], axis=0).tolist(),
            'hdr_volume_mean_frame_mc_se_m3': merged['volume_mc_se_m3'][mask].mean(0).tolist()}


def run_condition_audit(dataset: Path, output: Path, *, samples: int, audit_only: bool = False) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if samples < 100:
        raise ValueError('Need >=100 HDR samples')
    torch.set_num_threads(1)
    started = time.perf_counter()
    source = SyntheticDataset(dataset)
    report: dict[str, Any] = {'status': 'running', 'dataset': str(dataset), 'audit': audit_manifest(source),
        'samples_each_threshold_and_volume': samples, 'levels': LEVELS,
        'metric_scope': 'all train/val frames, no test quality scoring; baseline all 16 val',
        'hdr_volume': 'independent MC volume draw in R3; reported MC SE conditional on estimated threshold',
        'visibility': 'number of cameras with neither occlusion nor out_of_frame; not presence',
        'array_audits': [], 'condition_metrics': {}, 'audit_only': audit_only}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', report)
    minimum_available = 2**63 - 1
    aggregate: dict[str, list[dict[str, Array]]] = defaultdict(list)
    validation = []
    try:
        for record in sorted(source.records, key=lambda r: r['rally_id']):
            if record['split'] == 'test':
                continue
            available = next(int(s.split()[1]) * 1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
            minimum_available = min(minimum_available, available)
            if available < 8 * 1024**3:
                raise MemoryError('Available host RAM fell below 8 GiB')
            arrays = source.load(record)
            weights = arrays['gmm3d_weights']
            if weights.shape != (record['frames'], 125):
                raise ValueError('Lost mixture components')
            for key in ('gmm3d_means_m', 'gmm3d_covariance_m2', 'gmm3d_weights', 'gmm2d_means_uv', 'gmm2d_scale_tril_uv'):
                if arrays[key].dtype != np.float32:
                    raise ValueError('Changed storage dtype: ' + key)
            identity_hashes = {key: hashlib.sha256(str((arrays[key].dtype, arrays[key].shape)).encode() + arrays[key].tobytes()).hexdigest() for key in IDENTITY_ARRAYS}
            report['array_audits'].append({'rally_id': record['rally_id'], 'frames': record['frames'],
                'identity_array_hashes': identity_hashes, 'float32_spd': True, 'full_components': 125,
                'zero_weight_components': int((weights == 0).sum()),
                'unassessed_frames': int((~arrays['integration_convergence_assessed']).sum()),
                'weight_sum_max_error': float(np.abs(weights.astype(np.float64).sum(-1) - 1).max())})
            if not audit_only:
                points = condition_points(arrays)['mixture_mean']
                scores = []
                for frame in range(record['frames']):
                    mixture = GaussianMixture3D(arrays['gmm3d_means_m'][frame].astype(np.float64),
                        arrays['gmm3d_covariance_m2'][frame].astype(np.float64), weights[frame].astype(np.float64))
                    scores.append(mixture_hdr_metrics(mixture, arrays['positions_3d_m'][frame].astype(np.float64),
                        samples=samples, seed=int(np.random.SeedSequence([record['seed'], frame, 93612]).generate_state(1)[0])))
                values = {key: np.stack([s[key] for s in scores]) for key in scores[0]}
                values['visible_cameras'] = (~(arrays['occlusion_mask'] | arrays['out_of_frame_mask'])).sum(0)
                values['mean_error2_m2'] = np.square(points - arrays['positions_3d_m']).sum(-1)
                np.savez_compressed(output / (record['rally_id'] + '.npz'), **values)
                for key in (record['split'], 'train_val'):
                    aggregate[key].append(values)
                if record['split'] == 'val':
                    validation.append(RallyInput(record, arrays))
            write_json(output / 'manifest.json', report)
            print(json.dumps({'condition_audit_completed': record['rally_id']}), flush=True)
        for split, chunks in aggregate.items():
            report['condition_metrics'][split] = {'overall': _summary(chunks, None),
                'by_visible_cameras': {str(i): _summary(chunks, i) for i in range(4)}}
        if not audit_only:
            report['baselines'] = evaluate_baselines(validation, predictions=output / 'baseline_predictions')
        if sha256(dataset / 'manifest.json') != report['audit']['manifest_sha256']:
            raise ValueError('Dataset manifest changed during audit')
        for record in source.records:
            if sha256(dataset / (record['rally_id'] + '.npz')) != record['npz_sha256']:
                raise ValueError('Dataset file changed during audit')
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'manifest.json', report)
        raise
    report['resources'] = {'seconds': time.perf_counter() - started, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                           'minimum_available_host_bytes': minimum_available, 'gpu_jobs': 0}
    report['artifacts'] = {str(p.relative_to(output)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                           for p in sorted(output.rglob('*.npz'))}
    write_json(output / 'manifest.json', report)
    return report
