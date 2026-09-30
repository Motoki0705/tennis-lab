"""Reproduce CPU diagnostics from the committed run-13 predictions; no fitting."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.tasks.ball_refiner.refiner_3d.diffusion.metrics import Array
from src.tasks.ball_refiner.refiner_3d.diffusion.roughness import (
    local_error_components,
    magnitude_summary,
    stencil_masks,
    window_owners,
)
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256


def main() -> None:
    bundle = Path(__file__).resolve().parent
    source = bundle.parent / 'run-i936-anchored-t128-flow-regression-r13-s936/collected'
    manifest = json.loads((source / 'manifest.json').read_text())
    collection = json.loads((source / 'collection.json').read_text())
    windows = manifest['arms']['flow']['validation'][-1]['windows']
    assert windows == manifest['arms']['regression']['validation'][-1]['windows']
    derivative: dict[str, list[Array]] = defaultdict(list)
    positions: dict[str, list[Array]] = defaultdict(list)
    local: dict[str, list[Array]] = defaultdict(list)
    sample_spread: dict[str, list[Array]] = defaultdict(list)
    sources = {}
    for rally in sorted(windows):
        arrays = {}
        for arm in ('flow', 'regression'):
            path = source / arm / 'predictions' / (rally + '.npz')
            sources[str(path.relative_to(source))] = sha256(path)
            with np.load(path, allow_pickle=False) as stored:
                arrays[arm] = {key: stored[key] for key in stored.files}
        baseline = source / 'baseline_predictions' / (rally + '.npz')
        sources[str(baseline.relative_to(source))] = sha256(baseline)
        with np.load(baseline, allow_pickle=False) as stored:
            methods = {key: stored[key][None] for key in ('truth', 'mixture_mean', 'mixture_mean_rts')}
        methods.update(flow_mean=arrays['flow']['mean_m'][None], flow_samples=arrays['flow']['samples_m'],
                       regression=arrays['regression']['mean_m'][None])
        truth = arrays['flow']['positions_3d_m'].astype(np.float64)
        free = arrays['flow']['free_flight_mask']
        visible = (~(arrays['flow']['occlusion_mask'] | arrays['flow']['out_of_frame_mask'])).sum(0)
        owners = window_owners(windows[rally], len(free))
        dt = float(np.diff(arrays['flow']['timestamps_seconds'])[0])
        for name, prediction in methods.items():
            prediction = prediction.astype(np.float64)
            error = prediction - truth
            bias = np.broadcast_to(error.mean(1, keepdims=True), error.shape)
            for key, value in (('error', error), ('rally_bias', bias), ('demeaned_error', error - bias)):
                positions[name + '/' + key].append(value.reshape(-1, 3))
            if name != 'flow_samples':
                for key, value in local_error_components(error[0], free, owners, dt).items():
                    local[name + '/' + key].append(value)
            for order, label in ((2, 'acceleration'), (3, 'jerk')):
                values = np.diff(prediction, n=order, axis=1) / dt**order
                masks = stencil_masks(free, owners, order)
                for support, mask in masks.items():
                    derivative[f'{name}/{label}/{support}'].append(values[:, mask].reshape(-1, 3))
                    for count in range(4):
                        stratum = visible[order // 2:len(free) - order + order // 2] == count
                        derivative[f'{name}/{label}/{support}/camera{count}'].append(values[:, mask & stratum].reshape(-1, 3))
                if name == 'flow_samples':
                    residual = values - values.mean(0, keepdims=True)
                    for support, mask in masks.items():
                        sample_spread[label + '/' + support].append(residual[:, mask].reshape(-1, 3))
        sample_error = methods['flow_samples'].astype(np.float64)
        sample_spread['position'].append((sample_error - sample_error.mean(0, keepdims=True)).reshape(-1, 3))
    def summarize(groups: dict[str, list[Array]]) -> dict[str, Any]:
        return {key: magnitude_summary(np.concatenate(parts)) for key, parts in groups.items()}

    report = {'source_collection_sha256': sha256(source / 'collection.json'), 'prediction_sha256': sources,
              'rallies': sorted(windows), 'primary_update': 20000, 'test_arrays_read': 0,
              'derivative': summarize(derivative), 'position': summarize(positions),
              'local_five_frame_error_decomposition': summarize(local), 'flow_sample_spread': summarize(sample_spread),
              'definitions': {'seam': 'full derivative stencil intersects two source-window owners',
                              'free': 'every derivative support frame is free-flight',
                              'local_error': 'five-frame centered box trend; only within same window, all support frames free; derivatives use seven original frames',
                              'bias': 'per-rally constant mean XYZ error, repeated per frame for frame-weighted energy',
                              'sample_spread': 'each sample minus sample mean; derivative squared energy is orthogonal to mean energy'}}
    losses = {}
    training_output = Path(collection['output'])
    for arm in ('flow', 'regression'):
        path = training_output / arm / 'updates.jsonl'
        assert sha256(path) == collection['source_sha256'][str(path.relative_to(training_output))]['sha256']
        blocks = defaultdict(list)
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                blocks[(row['update'] - 1) // 2000].append(row)
        terms = manifest['config']['loss']
        results = []
        for _block, rows in sorted(blocks.items()):
            raw = {key: float(np.mean([r[key] for r in rows])) for key in terms}
            weighted = {key: terms[key] * raw[key] for key in terms}
            results.append({'updates': [rows[0]['update'], rows[-1]['update']], 'count': len(rows),
                            'raw': raw, 'weighted': weighted,
                            'physics_fraction': weighted['physics'] / sum(weighted.values()),
                            'gradient_norm_all_terms_mean': float(np.mean([r['gradient_norm'] for r in rows]))})
        val = manifest['arms'][arm]['validation'][-1]['loss']
        losses[arm] = {'train_blocks': results, 'validation_probe_raw': val,
                       'validation_probe_weighted': {key: terms[key] * val[key] for key in terms}}
    report['losses'] = losses
    (bundle / 'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name in ('truth', 'mixture_mean', 'mixture_mean_rts', 'flow_mean', 'flow_samples', 'regression'):
        print(name)
        for label in ('acceleration', 'jerk'):
            d = report['derivative']
            print(label, {s: {k: d[f'{name}/{label}/{s}'][k] for k in ('count', 'p95', 'squared_sum')} for s in ('free', 'seam_free', 'inside_free')})
        if name != 'flow_samples':
            print('position/local', {key: value['rms'] for group in ('position', 'local_five_frame_error_decomposition') for key, value in report[group].items() if key.startswith(name + '/')})
    print('sample_spread', report['flow_sample_spread'])
    print('last2k loss', {arm: value['train_blocks'][-1] for arm, value in losses.items()})


if __name__ == '__main__':
    main()
