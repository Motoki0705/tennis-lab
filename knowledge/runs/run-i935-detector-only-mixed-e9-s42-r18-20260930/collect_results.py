"""CPU-only collection of the immutable r18 outputs; fail before reporting on mismatch."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.gaps import fixed_gap_mask
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.tasks.ball_refiner.evaluation.detector_selection import validation_clips
from src.tasks.ball_refiner.evaluation.paired_metrics import Array, strata, summarize
from src.utils.checksum import dual_sha256

BUNDLE = Path(__file__).resolve().parent
ROOT = Path('/home/kamimura/projects/tennis-lab')
JOB = '1790730250749474097_274619_i935-detector-only-mixed-e9-s42-r18-20260930'
METHODS = ('new_refiner', 'old_refiner', 'new_detector', 'old_detector')
KEYS = ('error_px', 'nll_uv', 'nll_px', 'presence_nll', 'coverage', 'area_px2')
LEVELS = (0.5, 0.9, 0.95)


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def write(name: str, value: Any) -> None:
    (BUNDLE / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main() -> None:
    torch.set_num_threads(2)
    plan = read(BUNDLE / 'plan.json')
    config = OmegaConf.load(plan['training_config'])
    training = ROOT / 'outputs' / config.run.output_dir
    paired = Path(plan['comparison_output'])
    diagnostic = ROOT / 'outputs' / OmegaConf.load(plan['calibration_config']).run.output_dir
    report = Path(plan['report'])
    queue = ROOT / '.training_queue'
    state = (queue / 'state' / f'{JOB}.state').read_text()
    assert 'state=done\n' in state and (queue / 'done' / f'{JOB}.job').is_file()
    worker_lines = [x for x in (queue / 'worker.log').read_text(errors='replace').splitlines() if JOB in x]
    assert any(' done: ' in x for x in worker_lines)
    resource = read(report / 'resource_usage.json')
    assert resource['status'] == 'complete' and resource['device_monitor']['failure'] is None
    assert read(report / 'plan.json') == plan
    input_hashes = dict(plan['input_sha256'])
    for location in (paired, diagnostic):
        input_hashes.update(read(location / 'manifest.json')['input_sha256'])
    for name, digest in input_hashes.items():
        assert dual_sha256(Path(name)) == digest, name

    checkpoints = []
    curve = [json.loads(x) for x in (training / 'learning_curve.jsonl').read_text().splitlines()]
    assert [x['epoch'] for x in curve] == list(range(12))
    assert [x['step'] for x in curve] == list(range(250, 3001, 250))
    assert len(list(training.glob('epoch-*.pt'))) == 12
    data_hash = dual_sha256(training / 'data_manifest.json')
    for row in curve:
        p = training / f"epoch-{row['epoch']:03d}.pt"
        checkpoint = torch.load(p, map_location='cpu', weights_only=True)
        assert checkpoint['epoch'] == row['epoch'] and checkpoint['step'] == row['step']
        assert checkpoint['data_manifest_sha256'] == data_hash
        assert checkpoint['selection_nll_uv'] == row['selection_nll_uv']
        assert abs(row['selection_nll_uv'] - (row['observed']['position_nll_uv'] + row['evidence_gap']['position_nll_uv']) / 2) < 1e-12
        assert all(torch.isfinite(x).all() for x in checkpoint['state_dict'].values())
        checkpoints.append({'path': str(p), 'sha256': dual_sha256(p), 'bytes': p.stat().st_size,
                            'epoch': row['epoch'], 'step': row['step'], 'selection_nll_uv': row['selection_nll_uv']})
    best = read(training / 'best.json')
    assert best['epoch'] == min(curve, key=lambda x: x['selection_nll_uv'])['epoch']
    assert best['checkpoint_sha256'] == checkpoints[best['epoch']]['sha256']
    assert read(training / 'run_state.json') == {'best_epoch': best['epoch'], 'selection_nll_uv': best['selection_nll_uv'], 'status': 'complete', 'steps': 3000}

    manifest = read(paired / 'manifest.json')
    assert read(paired / 'run_state.json') == {'status': 'complete', 'clips': 70, 'files': 560}
    assert read(paired / 'progress.json')['artifacts'] == manifest['artifacts']
    store = BallFrameStore(Path(manifest['recipe']['store']))
    records = {x.clip_id: x for x in validation_clips(store)}
    assert len(records) == 70 and sum(r.frame_count for r in records.values()) == 40144
    assert len(list(paired.glob('*/*.npz'))) == len(manifest['artifacts']) == 560
    targets = {k: project_store_targets(store, r) for k, r in records.items()}
    grouped: dict[str, list[dict[str, Array]]] = defaultdict(list)
    seen = set()
    artifacts = []
    denominators = {}
    for entry in manifest['artifacts']:
        identity = (entry['clip_id'], entry['method'], entry['condition'])
        assert identity not in seen
        seen.add(identity)
        r, target = records[entry['clip_id']], targets[entry['clip_id']]
        p = paired / entry['path']
        assert dual_sha256(p) == entry['sha256'], str(p)
        artifacts.append({**entry, 'path': str(p), 'bytes': p.stat().st_size})
        with np.load(p, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        assert entry['frames'] == r.frame_count and entry['source'] == r.source and entry['camera'] == r.camera_id
        for name, value in [('frame_index', target.frame_index), ('pts', target.pts), ('target_reason', target.reason),
                            ('target_uv', target.uv), ('presence', target.presence), ('presence_valid', target.presence_valid)]:
            np.testing.assert_equal(a[name], value)
        target_hash = hashlib.sha256(target.frame_index.tobytes() + target.pts.tobytes() + target.reason.tobytes() + target.uv.tobytes()).hexdigest()
        assert manifest['paired_frame_identity'][r.clip_id] == {'frames': r.frame_count, 'frame_pts_sha256': target_hash}
        gap = fixed_gap_mask(r.frame_count, clip_id=r.clip_id, block_length=config.data.window_length,
                             lengths=tuple(config.training.gap_lengths), seed=config.data.partition_seed)
        if entry['condition'] == 'observed':
            gap[:] = False
        np.testing.assert_equal(a['gap_mask'], gap)
        for k in KEYS:
            assert len(a[k]) == r.frame_count
            assert not np.isinf(a[k]).any()
        located = np.isin(target.reason, [0, 5, 6])
        for key in ('nll_uv', 'nll_px', 'coverage', 'area_px2'):
            assert np.isfinite(a[key][located]).all() and np.isnan(a[key][~located]).all()
        if entry['method'].endswith('refiner'):
            assert np.isfinite(a['presence_nll'][target.presence_valid]).all()
            assert np.isnan(a['presence_nll'][~target.presence_valid]).all()
            for k in ('means', 'scale_tril', 'mixture_logits', 'presence_logits'):
                assert np.isfinite(a[k]).all()
            assert np.all(a['scale_tril'][:, :, (0, 1), (0, 1)] > 0)
        else:
            assert np.isnan(a['presence_nll']).all()
            if entry['condition'] == 'evidence_gap':
                assert np.isnan(a['error_px']).all()
                np.testing.assert_equal(a['coverage'][located], 1)
                np.testing.assert_equal(a['area_px2'][located], (r.source_width - 1) * (r.source_height - 1))
        groups = [r.source]
        if r.camera_id is not None:
            groups.append(f'{r.source}/{r.camera_id}')
        if r.source == 'meiji':
            half = 'selection' if r.clip_id in manifest['recipe']['partition']['selection'] else 'calibration'
            groups.extend([f'meiji/{half}', f'meiji/{half}/{r.camera_id}'])
        for label, selected in strata(target, gap, entry['condition']).items():
            for group in groups:
                grouped[f"{entry['method']}/{group}/{entry['condition']}/{label}"].append({k: a[k][selected] for k in KEYS})
        denominators[r.clip_id] = {'frames': r.frame_count, 'source': r.source, 'camera': r.camera_id, **target.counts()}
    assert seen == {(c, m, k) for c in records for m in METHODS for k in ('observed', 'evidence_gap')}
    metrics = {k: summarize(v, LEVELS) for k, v in grouped.items()}
    saved = read(paired / 'metrics.json')
    for k, v in saved.items():
        assert metrics[k] == v, k
    for k, v in grouped.items():
        errors = np.concatenate([x['error_px'] for x in v])
        errors = errors[np.isfinite(errors)]
        metrics[k]['p90_error_px'] = float(np.quantile(errors, .9)) if len(errors) else None

    dm = read(diagnostic / 'manifest.json')
    assert read(diagnostic / 'run_state.json') == {'status': 'complete', 'camera_clips': 18, 'partition': 'calibration', 'prediction_files': 36}
    assert dm['checkpoint'] == best
    assert dm['clip_ids'] == manifest['recipe']['partition']['calibration']
    assert len(dm['artifacts']) == len(list((diagnostic / 'predictions').glob('*.npz'))) == 36
    diag_seen = set()
    for entry in dm['artifacts']:
        p = diagnostic / entry['path']
        assert dual_sha256(p) == entry['sha256']
        artifacts.append({**entry, 'path': str(p), 'bytes': p.stat().st_size})
        target = targets[entry['clip_id']]
        r = records[entry['clip_id']]
        with np.load(p, allow_pickle=False) as z:
            for k in ('frame_index', 'pts'):
                np.testing.assert_equal(z[k], getattr(target, k))
            scored = target.position_valid & (z['gap_mask'] if entry['condition'] == 'evidence_gap' else True)
            assert entry['scored_frames'] == int(scored.sum())
            np.testing.assert_equal(z['scored_frame_index'], target.frame_index[scored])
            # Separate MC seeds mean HDR realizations differ; compare the actual GMM and NLL.
            other = paired / 'new_refiner' / f"clip-{r.index:05d}-{entry['condition']}.npz"
            with np.load(other, allow_pickle=False) as pair:
                for k in ('means', 'scale_tril', 'mixture_logits', 'presence_logits'):
                    np.testing.assert_allclose(z[k], pair[k], rtol=1e-5, atol=1e-6)
                np.testing.assert_allclose(z['nll_uv'], pair['nll_uv'][scored], rtol=1e-5, atol=1e-5)
        diag_seen.add((entry['clip_id'], entry['condition']))
    assert diag_seen == {(c, k) for c in dm['clip_ids'] for k in ('observed', 'evidence_gap')}

    # Register historical logs/repro verbatim; do not rewrite the captured command.
    for directory, source in [('queue_repro', queue / 'repro' / JOB), ('training', training), ('diagnostic', diagnostic), ('paired', paired)]:
        destination = BUNDLE / directory
        destination.mkdir(exist_ok=True)
        for p in source.iterdir():
            if p.is_file() and p.suffix in ('.json', '.jsonl', '.yaml', '.txt', '.sh', '.patch'):
                shutil.copy2(p, destination / p.name)
    shutil.copy2(queue / 'logs' / f'{JOB}.log', BUNDLE / 'queue.log')
    shutil.copy2(queue / 'done' / f'{JOB}.job', BUNDLE / 'queue.job')
    (BUNDLE / 'queue.state').write_text(state)
    (BUNDLE / 'worker_excerpt.log').write_text('\n'.join(worker_lines) + '\n')
    shutil.copy2(report / 'resource_usage.json', BUNDLE / 'resource_usage.json')
    write('artifact_hashes.json', artifacts + checkpoints)
    write('paired_recomputed.json', metrics)
    write('collection.json', {
        'status': 'verified', 'queue_job': JOB, 'queue_state': 'done',
        'exit_code': 0, 'exit_evidence': 'worker done branch only reached on rc=0; no separate success exit record',
        'training': {'checkpoints': len(checkpoints), 'steps': 3000, 'best': best},
        'diagnostic': {'files': len(diag_seen), 'clips': 18, 'status': 'complete', 'gmm_and_nll_match_paired': True},
        'paired': {'files': len(seen), 'clips': len(records), 'frames_per_method_condition': 40144,
                   'saved_metric_groups_reproduced_exactly': len(saved), 'camera_half_groups_added': True},
        'verified_input_files': len(input_hashes), 'denominators': denominators,
        'output_bytes': {str(p): sum(x.stat().st_size for x in p.rglob('*') if x.is_file()) for p in (training, diagnostic, paired, report)},
        'limitations': ['Original media/annotation hashes inherited from store, not rehashed.',
                        'Detector evidence_gap NPZ rows outside gap are placeholders and must not be scored or overlaid.',
                        'HDR Monte Carlo samples not re-run during collection; hashes, masks, GMM and reductions verified.',
                        'Absent/unknown remain N/A in this report; raw legacy absent presence metrics are retained.'],
    })
    print(json.dumps({'status': 'verified', 'files': len(seen), 'best': best}))


if __name__ == '__main__':
    main()
