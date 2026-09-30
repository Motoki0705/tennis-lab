"""Paired full-rally/T128 inference with fixed sample/frame noise, validation only."""
from __future__ import annotations

import json
import resource
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset
from src.tasks.ball_refiner.refiner_3d.synthetic.generator import write_json
from src.utils.schema.court_normalization import (
    normalize_court_position,
)

from .context_inference import predict_context
from .data import rally_window
from .losses import LossConfig, trajectory_loss
from .metrics import TrajectoryMetrics
from .model import ModelConfig, TrajectoryDenoiser


def run_context_probe(dataset: Path, training_output: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    started = time.perf_counter()
    minimum_available = 2**63 - 1

    def budget() -> None:
        nonlocal minimum_available
        available = next(int(s.split()[1]) * 1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        minimum_available = min(available, minimum_available)
        if available < 8 * 1024**3:
            raise MemoryError('Available host RAM fell below 8 GiB')
        if time.perf_counter() - started > 1050:
            raise TimeoutError('CPU context comparison exceeded 1050 seconds')

    budget()
    paths = {'dataset_manifest': dataset / 'manifest.json', 'training_manifest': training_output / 'manifest.json'}
    hashes = {key: sha256(path) for key, path in paths.items()}
    training = json.loads(paths['training_manifest'].read_text())
    config = training['config']
    if training['status'] != 'complete' or training['source_manifest_sha256'] != hashes['dataset_manifest']:
        raise ValueError('Need the exact dataset of the completed training run')
    if config['updates'] != 20000 or config['frames'] != 128 or config['stride'] != 128:
        raise ValueError('This diagnostic requires the fixed 20k T128/stride128 run')
    source = SyntheticDataset(dataset)
    if source.manifest['counts'] != config['expected_counts']:
        raise ValueError('Dataset split counts changed')
    records = [r for r in source.records if r['split'] == 'val']
    records.sort(key=lambda r: r['rally_id'])
    identities = [{'rally_id': r['rally_id'], 'npz_sha256': r['npz_sha256']} for r in records]
    if identities != [r for r in training['read_rallies'] if r['rally_id'].startswith('val-')]:
        raise ValueError('Validation inputs changed')
    objectives: tuple[Literal['flow', 'regression'], ...] = ('flow', 'regression')
    models: dict[Literal['flow', 'regression'], TrajectoryDenoiser] = {}
    for arm in objectives:
        paths[arm + '_checkpoint'] = training_output / arm / 'dev-only.pt'
        hashes[arm + '_checkpoint'] = sha256(paths[arm + '_checkpoint'])
        saved_arm = training['arms'][arm]
        if saved_arm['status'] != 'complete' or saved_arm['checkpoint_sha256'] != hashes[arm + '_checkpoint']:
            raise ValueError('Incomplete/changed checkpoint')
        saved = torch.load(paths[arm + '_checkpoint'], map_location='cpu', weights_only=True)
        if (not saved['diagnostic_only'] or saved['objective'] != arm or saved['updates'] != 20000
                or saved['config'] != {**config, 'evaluate_updates': tuple(config['evaluate_updates'])}):
            raise ValueError('Checkpoint identity/config mismatch')
        model = TrajectoryDenoiser(ModelConfig(**config['model'])).eval().requires_grad_(False)
        model.load_state_dict(saved['state_dict'], strict=True)
        models[arm] = model
    report: dict[str, Any] = {
        'status': 'running', 'dataset': str(dataset), 'training_output': str(training_output),
        'input_hashes': hashes, 'device': 'cpu', 'native_threads': 1, 'updates': 20000,
        'samples': config['samples'], 'steps': config['steps'], 'read_rallies': [], 'results': {},
        'selection': 'all 16 validation rallies; no train/test NPZ reads or checkpoint/seed selection',
        'noise': 'one full-rally CPU draw per sample with rally seed+2, sliced identically; probe time/noise seed+1',
        'stitching': 'T128/stride128, absolute times, right padding; earlier window owns overlap; derivatives include seams',
        'decision_rule': {'arm': 'flow', 't128_rmse_at_most_m': 8.012, 'paired_rmse_improvement_at_least_m': 1.5},
        'old_gpu_comparison': 'CPU RNG/numerics differ; causal contrast is paired whole vs T128 on CPU',
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', report)
    collected = {(arm, mode, kind): TrajectoryMetrics() for arm in models for mode in ('whole', 't128') for kind in ('mean', 'samples', 'truth')}
    loss_sums: dict[tuple[str, str], dict[str, float]] = {(arm, mode): {} for arm in models for mode in ('whole', 't128')}
    total_frames = 0
    try:
        for record in records:
            budget()
            arrays = source.load(record)
            batch = rally_window(arrays, record, start=0, frames=record['frames'], allow_nonconverged=True)
            clean = normalize_court_position(batch.target_positions_m)
            generator = torch.Generator().manual_seed(record['seed'] + 2)
            noise = torch.stack([torch.randn(clean.shape, generator=generator) for _ in range(config['samples'])])
            probe_generator = torch.Generator().manual_seed(record['seed'] + 1)
            flow_time = torch.rand(1, generator=probe_generator)
            flow_state = (1 - flow_time[:, None, None]) * torch.randn(clean.shape, generator=probe_generator) + flow_time[:, None, None] * clean
            saved_arrays = {'initial_noise': noise.numpy(), 'probe_flow_time': flow_time.numpy(), 'probe_flow_state': flow_state.numpy()}
            owners = {}
            for objective in objectives:
                for mode, width in (('whole', None), ('t128', config['frames'])):
                    trajectories, probe, ownership = predict_context(
                        models[objective], batch, objective=objective, initial_noise=noise,
                        probe_state=flow_state if objective == 'flow' else torch.zeros_like(clean),
                        probe_time=flow_time if objective == 'flow' else torch.zeros(1),
                        steps=config['steps'], frames=width, check_budget=budget,
                    )
                    loss, terms = trajectory_loss(probe, batch, LossConfig(**config['loss']))
                    for key, value in {'loss': loss, **terms}.items():
                        sums = loss_sums[objective, mode]
                        sums[key] = sums.get(key, 0.) + float(value) * record['frames']
                    sample = trajectories.numpy()
                    for kind, value in (('mean', sample.mean(0)[None]), ('samples', sample), ('truth', arrays['positions_3d_m'][None])):
                        collected[objective, mode, kind].add(value, arrays)
                    saved_arrays[objective + '_' + mode + '_samples_m'] = sample
                    owners[objective + '/' + mode] = ownership
            np.savez_compressed(output / (record['rally_id'] + '.npz'), **saved_arrays)
            report['read_rallies'].append({**identities[len(report['read_rallies'])], 'frames': record['frames'], 'windows': owners})
            total_frames += record['frames']
            write_json(output / 'manifest.json', report)
            print(json.dumps({'context_probe_completed': record['rally_id']}), flush=True)
        for objective in models:
            report['results'][objective] = {}
            for mode in ('whole', 't128'):
                report['results'][objective][mode] = {
                    'loss': {key: value / total_frames for key, value in loss_sums[objective, mode].items()},
                    **{kind: collected[objective, mode, kind].summarize() for kind in ('mean', 'samples', 'truth')},
                }
        flow = report['results']['flow']
        whole = flow['whole']['mean']['metrics']['rmse_m_overall']['value']
        short = flow['t128']['mean']['metrics']['rmse_m_overall']['value']
        report['decision'] = {'whole_rmse_m': whole, 't128_rmse_m': short, 'paired_improvement_m': whole - short,
                              'context_explains_degradation': short <= 8.012 and whole - short >= 1.5}
        if hashes != {key: sha256(path) for key, path in paths.items()}:
            raise ValueError('Inputs changed during context probe')
        for record in records:
            if sha256(dataset / (record['rally_id'] + '.npz')) != record['npz_sha256']:
                raise ValueError('Validation rally changed during context probe')
        report.update(status='complete', frames=total_frames)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'manifest.json', report)
        raise
    report['resources'] = {'seconds': time.perf_counter() - started, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                           'minimum_available_host_bytes': minimum_available, 'gpu_jobs': 0}
    report['artifacts'] = {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(output.glob('*.npz'))}
    write_json(output / 'manifest.json', report)
    return report
