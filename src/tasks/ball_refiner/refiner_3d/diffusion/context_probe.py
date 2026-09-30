"""Paired full-rally/T128 inference with fixed sample/frame noise, validation only."""
from __future__ import annotations

import json
import resource
import time
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset
from src.tasks.ball_refiner.refiner_3d.synthetic.generator import write_json
from src.utils.schema.court_normalization import (
    denormalize_court_position,
    normalize_court_position,
)

from .data import rally_window, window_starts
from .flow import sample_trajectories
from .losses import LossConfig, TrainingBatch, trajectory_loss
from .metrics import TrajectoryMetrics
from .model import DenoiserOutput, MixtureCondition, ModelConfig, TrajectoryDenoiser


def _window(value: Tensor, start: int, frames: int, *, axis: int) -> Tensor:
    shape = list(value.shape)
    shape[axis] = frames
    result = value.new_zeros(shape)
    length = min(frames, value.shape[axis] - start)
    result.narrow(axis, 0, length).copy_(value.narrow(axis, start, length))
    return result


def predict_context(
    model: TrajectoryDenoiser, batch: TrainingBatch, *, objective: Literal['flow', 'regression'],
    initial_noise: Tensor, probe_state: Tensor, probe_time: Tensor, steps: int,
    frames: int | None, check_budget: Callable[[], None],
) -> tuple[Tensor, DenoiserOutput, list[dict[str, int]]]:
    """Keep earliest-window ownership, absolute timestamps, and every real frame.

    The unpadded input is one whole rally. Padding matches training windows.
    Metrics/loss are computed on the assembled original timeline, including seams.
    """
    condition = batch.condition
    if condition.padding_mask.shape[0] != 1 or bool(condition.padding_mask.any()):
        raise ValueError('Context probe requires one complete unpadded rally')
    length = condition.padding_mask.shape[1]
    width = length if frames is None else frames
    starts = [0] if frames is None else window_starts(length, width, width)
    samples = initial_noise.shape[0] if objective == 'flow' else 1
    prediction = condition.means_m.new_empty((samples, length, 3))
    probe_positions = torch.empty_like(batch.target_positions_m)
    probe_events = condition.means_m.new_empty((1, length, 2))
    covered = 0
    ownership = []
    with torch.no_grad():
        for start in starts:
            check_budget()
            stop = min(length, start + width)
            if start > covered:
                raise ValueError('Context windows left uncovered frames')
            values = {f.name: _window(getattr(condition, f.name), start, width, axis=1)
                      for f in fields(MixtureCondition)}
            values['padding_mask'][:, stop - start:] = True
            selected = MixtureCondition(**values)
            probe = model(_window(probe_state, start, width, axis=1), probe_time, selected)
            if objective == 'flow':
                trajectories = sample_trajectories(
                    model, selected, samples=samples, steps=steps,
                    generator=torch.Generator().manual_seed(0),
                    initial_noise=_window(initial_noise, start, width, axis=2),
                ).positions_m[:, 0]
            else:
                trajectories = denormalize_court_position(probe.positions_norm)
            offset = covered - start
            prediction[:, covered:stop] = trajectories[:, offset:stop - start]
            probe_positions[:, covered:stop] = probe.positions_norm[:, offset:stop - start]
            probe_events[:, covered:stop] = probe.event_logits[:, offset:stop - start]
            ownership.append({'window_start': start, 'real_stop': stop, 'owned_start': covered,
                              'owned_stop': stop, 'padded_frames': width - (stop - start)})
            covered = stop
    if covered != length or not bool(torch.isfinite(prediction).all()):
        raise ValueError('Incomplete/nonfinite context predictions')
    return prediction, DenoiserOutput(probe_positions, probe_events), ownership


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
