"""One preregistered CPU comparison; fixed 512-train/20k weights and 16 val."""
from __future__ import annotations

import json
import resource
import runpy
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from src.tasks.ball_refiner.refiner_3d.baseline_comparison import comparison_markdown
from src.tasks.ball_refiner.refiner_3d.diffusion.context_inference import (
    predict_context,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.data import rally_window
from src.tasks.ball_refiner.refiner_3d.diffusion.metrics import Array, TrajectoryMetrics
from src.tasks.ball_refiner.refiner_3d.diffusion.model import (
    ModelConfig,
    TrajectoryDenoiser,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.overlap_inference import (
    predict_overlap,
)
from src.tasks.ball_refiner.refiner_3d.diffusion.roughness import (
    magnitude_summary,
    stencil_masks,
    window_owners,
)
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset
from src.tasks.ball_refiner.refiner_3d.synthetic.generator import write_json


def decide(methods: dict, roughness: dict) -> dict:
    rule = runpy.run_path(str(Path(__file__).resolve().parent.parent / 'run-i936-physics10-t128-r14-s936/collect.py'))['rule']
    results = {}
    formal = {}
    for arm, kind in (('flow', 'mean'), ('flow', 'samples'), ('regression', 'mean')):
        candidate, reference = f'{arm}_overlap_{kind}', f'{arm}_control_{kind}'
        original = rule(methods[candidate], methods[reference])
        axes = {}
        for name, row in original['axes'].items():
            if name.startswith(('acceleration_', 'jerk_')):
                continue
            axes[name] = {**row, 'maximum_ratio': 1.05,
                          'pass': row['candidate'] <= 1.05 * row['reference'] + row['tolerance']}
        for derivative in ('acceleration', 'jerk'):
            checks = [('seam_free', 'squared_sum', .5), ('free', 'squared_sum', 1.),
                      ('free', 'p95', 1.), ('inside_free', 'p95', 1.)]
            for support, statistic, maximum in checks:
                a = roughness[f'{candidate}/{derivative}/{support}']
                b = roughness[f'{reference}/{derivative}/{support}']
                assert a['count'] == b['count'] and a['count'] > 0
                axes[f'{derivative}_{support}_{statistic}'] = {
                    'candidate': a[statistic], 'reference': b[statistic], 'maximum_ratio': maximum,
                    'ratio': a[statistic] / b[statistic],
                    'pass': a[statistic] <= maximum * b[statistic] + 1e-6 + 1e-6 * abs(b[statistic])}
        behind = {key: methods[name]['metrics']['behind_all'] for key, name in (('candidate', candidate), ('reference', reference))}
        assert behind['candidate']['count'] == behind['reference']['count']
        behind['pass'] = behind['candidate']['invalid_count'] <= behind['reference']['invalid_count']
        results[candidate] = {'axes': axes, 'behind': behind, 'pass': all(row['pass'] for row in axes.values()) and behind['pass']}
        formal[candidate] = {name: rule(methods[candidate], methods[name])
                             for name in ('mixture_mean', 'mixture_mean_rts', 'regression_overlap_mean') if name != candidate}
    return {'diagnostic': results, 'common_improvement': all(r['pass'] for r in results.values()), 'formal': formal}


def main() -> None:
    torch.set_num_threads(1)
    bundle = Path(__file__).resolve().parent
    plan = json.loads((bundle / 'plan.json').read_text())
    for name, digest in plan['pinned_files'].items():
        assert sha256(Path(name)) == digest, name
    output = bundle / 'results'
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    minimum_available = 2**63 - 1

    def budget() -> None:
        nonlocal minimum_available
        available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        minimum_available = min(available, minimum_available)
        if available < 6 * 1024**3:
            raise MemoryError('MemAvailable below 6 GiB')
        if time.perf_counter() - started > 1080:
            raise TimeoutError('CPU probe exceeded 18 minutes; no retry')

    training_output = Path(plan['training_output'])
    training = json.loads((training_output / 'manifest.json').read_text())
    config = training['config']
    assert training['status'] == 'complete' and config['updates'] == 20000 and config['frames'] == config['stride'] == 128
    dataset = SyntheticDataset(Path(training['dataset']))
    assert sha256(dataset.directory / 'manifest.json') == training['source_manifest_sha256']
    expected = {r['rally_id']: r['npz_sha256'] for r in training['read_rallies'] if r['rally_id'].startswith('val-')}
    assert len(expected) == 16
    records = sorted((r for r in dataset.records if r['rally_id'] in expected), key=lambda r: r['rally_id'])
    assert {r['rally_id']: r['npz_sha256'] for r in records} == expected
    objectives: tuple[Literal['flow', 'regression'], ...] = ('flow', 'regression')
    models = {}
    for arm in objectives:
        path = training_output / arm / 'dev-only.pt'
        assert sha256(path) == training['arms'][arm]['checkpoint_sha256']
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        assert checkpoint['diagnostic_only'] and checkpoint['objective'] == arm and checkpoint['updates'] == 20000
        assert checkpoint['config'] == {**config, 'evaluate_updates': tuple(config['evaluate_updates'])}
        model = TrajectoryDenoiser(ModelConfig(**config['model'])).eval().requires_grad_(False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        models[arm] = model
    accumulators = {f'{arm}_{mode}_{kind}': TrajectoryMetrics() for arm in objectives for mode in ('control', 'overlap') for kind in ('mean', 'samples')}
    derivatives: dict[str, list[Array]] = defaultdict(list)
    report: dict[str, Any] = {'status': 'running', 'plan_sha256': sha256(bundle / 'plan.json'), 'read_rallies': [],
                              'test_arrays_read': 0, 'extra_val_arrays_read': 0, 'device': 'cpu', 'native_threads': 1,
                              'primary_update': 20000, 'prediction_windows': {}}
    write_json(output / 'manifest.json', report)
    fields = ('positions_3d_m', 'timestamps_seconds', 'occlusion_mask', 'out_of_frame_mask', 'event_region_mask',
              'free_flight_mask', 'camera_true_K', 'camera_true_R', 'camera_true_t')
    try:
        for record in records:
            budget()
            arrays = dataset.load(record)
            batch = rally_window(arrays, record, start=0, frames=record['frames'], allow_nonconverged=True)
            generator = torch.Generator().manual_seed(record['seed'] + 2)
            noise = torch.stack([torch.randn((1, record['frames'], 3), generator=generator) for _ in range(config['samples'])])
            saved = {'initial_noise': noise.numpy(), **{k: arrays[k] for k in fields}}
            windows = {}
            dt = float(np.diff(arrays['timestamps_seconds'])[0])
            for arm in objectives:
                control, _, owners = predict_context(models[arm], batch, objective=arm, initial_noise=noise,
                    probe_state=torch.zeros_like(batch.target_positions_m), probe_time=torch.zeros(1),
                    steps=config['steps'], frames=128, check_budget=budget)
                assert owners == training['arms'][arm]['validation'][-1]['windows'][record['rally_id']]
                overlap, spans = predict_overlap(models[arm], batch, objective=arm, initial_noise=noise,
                    steps=config['steps'], frames=128, stride=64, check_budget=budget)
                windows[arm] = {'control': owners, 'overlap': spans}
                ownership = window_owners(owners, record['frames'])
                for mode, prediction in (('control', control), ('overlap', overlap)):
                    sample = prediction.numpy()
                    saved[f'{arm}_{mode}_samples_m'] = sample
                    for kind, value in (('mean', sample.mean(0)[None]), ('samples', sample)):
                        name = f'{arm}_{mode}_{kind}'
                        accumulators[name].add(value, arrays)
                        for order, label in ((2, 'acceleration'), (3, 'jerk')):
                            vectors = np.diff(value.astype(np.float64), n=order, axis=1) / dt**order
                            for support, mask in stencil_masks(arrays['free_flight_mask'], ownership, order).items():
                                derivatives[f'{name}/{label}/{support}'].append(vectors[:, mask].reshape(-1, 3))
            np.savez_compressed(output / (record['rally_id'] + '.npz'), **saved)
            report['read_rallies'].append({'rally_id': record['rally_id'], 'npz_sha256': record['npz_sha256'], 'frames': record['frames']})
            report['prediction_windows'][record['rally_id']] = windows
            write_json(output / 'manifest.json', report)
            print(json.dumps({'completed_rally': record['rally_id'], 'seconds': time.perf_counter() - started}), flush=True)
        audited = json.loads(Path(plan['audited_a_comparison']).read_text())['methods']
        methods = {k: audited[k] for k in ('mixture_mean', 'mixture_mean_rts', 'truth')}
        methods.update({name: values.summarize() for name, values in accumulators.items()})
        roughness = {key: magnitude_summary(np.concatenate(values)) for key, values in derivatives.items()}
        for values in methods.values():
            assert values['metrics']['rmse_m_overall']['count'] in (6383, 4 * 6383)
        result = decide(methods, roughness)
        for name, digest in plan['pinned_files'].items():
            assert sha256(Path(name)) == digest, name
        for record in records:
            assert sha256(dataset.directory / (record['rally_id'] + '.npz')) == record['npz_sha256']
        write_json(output / 'comparison.json', {'methods': methods})
        (output / 'comparison.md').write_text(comparison_markdown(methods))
        write_json(output / 'roughness.json', roughness)
        write_json(output / 'rules.json', result)
        report.update(status='complete', frames=sum(r['frames'] for r in records), all_pinned_hashes_unchanged=True,
                      resources={'elapsed_seconds': time.perf_counter() - started, 'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                 'minimum_available_host_bytes': minimum_available, 'gpu_jobs': 0},
                      artifacts={p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(output.iterdir()) if p.is_file() and p.name != 'manifest.json'})
        write_json(output / 'manifest.json', report)
        print(json.dumps({'resources': report['resources'], 'diagnostic': {k:v['pass'] for k,v in result['diagnostic'].items()}}, indent=2))
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'manifest.json', report)
        raise


if __name__ == '__main__':
    main()
