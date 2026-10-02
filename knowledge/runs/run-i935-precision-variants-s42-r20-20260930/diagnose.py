"""Reproduce r20 CPU diagnosis from immutable r18/r4 paired outputs and caches."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.tasks.ball_refiner.evaluation.detector_selection import validation_clips
from src.tasks.ball_refiner.evaluation.precision import detector_groups, precision_rows
from src.utils.checksum import dual_sha256

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BUNDLE = Path(__file__).resolve().parent
ROOT = Path('/home/kamimura/projects/tennis-lab')
PAIRED = ROOT / 'outputs/ball_refiner/evaluate/paired/i935-mixed-e9-vs-ft-e13-r18-20260930'
RUNS = {'new': ROOT / 'outputs/ball_refiner/train/detector_only/i935-mixed-e9-s42-r18-20260930',
        'old': ROOT / 'outputs/ball_refiner/train/detector_only/i935-ft-e13-s42-r4-20260928'}


def main() -> None:
    torch.set_num_threads(2)
    manifest = json.loads((PAIRED / 'manifest.json').read_text())
    assert json.loads((PAIRED / 'run_state.json').read_text())['status'] == 'complete'
    entries = {(x['clip_id'], x['method'], x['condition']): x for x in manifest['artifacts']}
    store = BallFrameStore(Path(manifest['recipe']['store']))
    caches = {name: EvidenceCache(ROOT / 'data/ball_refiner' / directory, store) for name, directory in (
        ('new', 'detector-mixed-e9-trainval-r17-20260930'), ('old', 'detector-ft-e13-trainval-r3-20260928'))}
    hashes = {str(PAIRED / 'manifest.json'): dual_sha256(PAIRED / 'manifest.json')}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grids = set()
    recalls: dict[str, list[np.ndarray]] = defaultdict(list)

    def load(clip_id: str, method: str) -> dict[str, np.ndarray]:
        entry = entries[(clip_id, method, 'observed')]
        path = PAIRED / entry['path']
        digest = dual_sha256(path)
        assert digest == entry['sha256']
        hashes[str(path)] = digest
        with np.load(path, allow_pickle=False) as saved:
            return dict(saved)

    for record in validation_clips(store):
        new_detector = load(record.clip_id, 'new_detector')
        new_target = new_detector['target_uv']
        observed = new_detector['target_reason'] == 0
        groups = [record.source]
        if record.camera_id:
            groups.append(f'{record.source}/{record.camera_id}')
        if record.source == 'meiji':
            half = 'selection' if record.clip_id in manifest['recipe']['partition']['selection'] else 'calibration'
            groups.extend([f'meiji/{half}', f'meiji/{half}/{record.camera_id}'])
        for name in RUNS:
            arrays = load(record.clip_id, f'{name}_refiner')
            detector = load(record.clip_id, f'{name}_detector')
            evidence = caches[name].load(record.clip_id)
            grids.add(evidence.heatmap_size_hw)
            for other in (arrays, detector):
                for key in ('target_uv', 'target_reason', 'frame_index', 'pts'):
                    np.testing.assert_equal(other[key], new_detector[key])
            np.testing.assert_allclose(detector['argmax_uv'], evidence.argmax_uv, atol=1e-5, rtol=1e-5)
            rows = precision_rows(arrays['means'][observed], arrays['scale_tril'][observed], arrays['mixture_logits'][observed],
                                  new_target[observed], new_detector['argmax_uv'][observed],
                                  evidence.candidates.coords[0].numpy()[observed], evidence.candidates.valid[0].numpy()[observed],
                                  (record.source_width, record.source_height))
            np.testing.assert_allclose(rows['error_px'], arrays['error_px'][observed], atol=1e-4)
            factor = np.array([record.source_width - 1, record.source_height - 1])
            own_error = np.linalg.norm((detector['argmax_uv'][observed] - new_target[observed]) * factor, axis=-1)
            rows['own_detector_error_px'] = own_error
            for basis, bucket_errors in [('e9', rows['detector_error_px']), ('own', own_error)]:
                for label, mask in detector_groups(bucket_errors).items():
                    for group in groups:
                        grouped[f'{name}/{basis}/{group}/{label}'].append({k: v[mask] for k, v in rows.items()})
            if name == 'new':
                distances = np.linalg.norm((evidence.candidates.coords[0].numpy()[observed] - new_target[observed, None]) * factor, axis=-1)
                distances[~evidence.candidates.valid[0].numpy()[observed]] = np.inf
                recalls[record.source].append(np.stack([(distances[:, :k].min(-1) <= 20) for k in (1, 3, 4, 8)], axis=-1))
    result = {}
    for key, parts in grouped.items():
        values = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
        row: dict[str, Any] = {'frames': len(values['error_px'])}
        for metric, vector in values.items():
            vector = vector[np.isfinite(vector)]
            row[metric] = {'n': len(vector), 'mean': float(vector.mean()) if len(vector) else None,
                           'quantiles': np.quantile(vector, [.05, .5, .9, .95]).tolist() if len(vector) else None}
        row['component_ellipse_coverage'] = {str(p): float((values['z_radius'] <= np.sqrt(-2 * np.log(1 - p))).mean())
                                            if len(values['z_radius']) else None for p in (.5, .9, .95)}
        near = values['nearest_candidate_px']
        row['nearest_candidate_fractions'] = {str(px): float((near <= px).mean()) if len(near) else None for px in (1, 2, 8)}
        result[key] = row
    curves = {name: [json.loads(x) for x in (run / 'learning_curve.jsonl').read_text().splitlines()] for name, run in RUNS.items()}
    epoch_bias = []
    for name, run in RUNS.items():
        for saved_dir in sorted(run.glob('validation-epoch-*')):
            deltas = []
            errors = []
            for record in validation_clips(store):
                path = saved_dir / f'clip-{record.index:05d}-observed.npz'
                if not path.is_file():
                    continue
                hashes[str(path)] = dual_sha256(path)
                with np.load(path, allow_pickle=False) as saved:
                    evidence = caches[name].load(record.clip_id)
                    factor = np.array([record.source_width - 1, record.source_height - 1])
                    mask = saved['position_valid'] & (np.linalg.norm((saved['target_uv'] - evidence.argmax_uv) * factor, axis=-1) <= 8)
                    mean = saved['means'][np.arange(record.frame_count), saved['mixture_logits'].argmax(-1)]
                    deltas.append(((mean - evidence.argmax_uv) * factor)[mask])
                    errors.append(np.linalg.norm((mean - saved['target_uv']) * factor, axis=-1)[mask])
            delta, error = np.concatenate(deltas), np.concatenate(errors)
            epoch_bias.append({'model': name, 'epoch': int(saved_dir.name[-3:]), 'frames': len(delta),
                               'median_dx_px': float(np.median(delta[:, 0])), 'median_dy_px': float(np.median(delta[:, 1])),
                               'median_error_px': float(np.median(error))})
    for cache in caches.values():
        hashes[str(cache.directory / 'manifest.json')] = dual_sha256(cache.directory / 'manifest.json')
    for run in RUNS.values():
        hashes[str(run / 'learning_curve.jsonl')] = dual_sha256(run / 'learning_curve.jsonl')
    payload = {'groups': result, 'curves': curves, 'saved_selection_epoch_bias': epoch_bias, 'input_sha256': hashes, 'heatmap_size_hw': sorted(grids),
               'e9_candidate_recall20px_at_1_3_4_8': {k: np.concatenate(v).mean(0).tolist() for k, v in recalls.items()},
               'scope': 'CPU only, observed validation; e9 and own-detector buckets. z is top-component radial Mahalanobis, not mixture HDR.'}
    (BUNDLE / 'diagnosis.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    lines = ['# run20 CPU精度診断', '', 'bucketはepoch9 top-1誤差≤8 / (8,20] / >20 source px。新旧とも同じframe。', '',
             '| source | bucket | n | 新refiner誤差 p50/p90/p95 | 旧refiner誤差 p50/p90/p95 | 新refiner→e9距離 p50/p90/p95 |',
             '|---|---|---:|---|---|---|']

    def quant(row: dict[str, Any], name: str) -> str:
        values = row[name]['quantiles']
        return 'N/A' if values is None else ' / '.join(f'{x:.2f}' for x in values[1:])

    for source in recalls:
        for bucket in ('within8', '8to20', 'wrong', 'all'):
            new, old = (result[f'{name}/e9/{source}/{bucket}'] for name in RUNS)
            lines.append(f"| {source} | {bucket} | {new['frames']} | {quant(new, 'error_px')} | {quant(old, 'error_px')} | {quant(new, 'detector_distance_px')} |")
    lines.extend(['', 'sigmaは最大weight成分のsource px共分散の長軸1σ。z=√(誤差ᵀΣ⁻¹誤差)、2次元正規の期待p50/p90/p95は1.177/2.146/2.448。',
                  '成分ellipse coverageはGMM HDR coverageではない。', '',
                  '| source | model | sigma p50/p90/p95 px | z p50/p90/p95 | 成分50/90/95% coverage | 最近候補距離 p50/p90/p95 | ≤2px比率 |',
                  '|---|---|---|---|---|---|---:|'])
    for source in recalls:
        for name in RUNS:
            row = result[f'{name}/e9/{source}/all']
            coverage = ' / '.join(f'{v:.3f}' for v in row['component_ellipse_coverage'].values())
            lines.append(f"| {source} | {name} | {quant(row, 'sigma_major_px')} | {quant(row, 'z_radius')} | {coverage} | {quant(row, 'nearest_candidate_px')} | {row['nearest_candidate_fractions']['2']:.3f} |")
    lines.extend(['', '| model | step | train joint NLL uv | val observed位置NLL uv | val gap位置NLL uv | 選択NLL uv |', '|---|---:|---:|---:|---:|---:|'])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    for ax, (name, curve) in zip(axes, curves.items(), strict=True):
        for row in curve:
            lines.append(f"| {name} | {row['step']} | {row['train_joint_nll']:.4f} | {row['observed']['position_nll_uv']:.4f} | {row['evidence_gap']['position_nll_uv']:.4f} | {row['selection_nll_uv']:.4f} |")
        for metric, curve_values in [('train joint', [r['train_joint_nll'] for r in curve]),
                               ('val observed position', [r['observed']['position_nll_uv'] for r in curve]),
                               ('val gap position', [r['evidence_gap']['position_nll_uv'] for r in curve])]:
            ax.plot([r['step'] for r in curve], curve_values, label=metric)
        ax.set(title=name, xlabel='step', ylabel='NLL (UV); lower is better')
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
    figure.savefig(BUNDLE / 'learning-curves.png', dpi=140)
    plt.close(figure)
    lines.extend(['', 'trainはdropout/人工gapありのepoch平均joint NLL、valはepoch末evalの条件付き位置NLL。train位置項単独は元ログにないので正確な汎化gapには読み替えない。', '', '![curves](learning-curves.png)', ''])
    lines.extend(['保存済みbest更新epochのMeiji選択側。各モデル自身のdetector誤差≤8pxに限定。epochで偏りの向きが変わる。', '',
                  '| model | epoch (0-based) | n | x偏り中央値 px | y偏り中央値 px | 誤差中央値 px |', '|---|---:|---:|---:|---:|---:|'])
    for row in epoch_bias:
        lines.append(f"| {row['model']} | {row['epoch']} | {row['frames']} | {row['median_dx_px']:.2f} | {row['median_dy_px']:.2f} | {row['median_error_px']:.2f} |")
    (BUNDLE / 'diagnosis-tables.md').write_text('\n'.join(lines))
    print(json.dumps({'groups': len(result), 'verified_input_files': len(hashes), 'recall20': payload['e9_candidate_recall20px_at_1_3_4_8']}))


if __name__ == '__main__':
    main()
