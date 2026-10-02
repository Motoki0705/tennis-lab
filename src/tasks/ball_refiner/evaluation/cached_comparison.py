"""Evaluate one declared refiner variant from cache against immutable paired rows."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.tasks.ball_refiner.data.gaps import fixed_gap_mask, validation_partition
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.tasks.ball_refiner.data.windows import LoadedClip
from src.tasks.ball_refiner.evaluation.detector_selection import validation_clips
from src.tasks.ball_refiner.evaluation.paired_metrics import (
    Array,
    gmm_rows,
    strata,
    summarize,
)
from src.tasks.ball_refiner.evaluation.pilot_comparison import (
    ComparisonSettings,
    load_pilot,
)
from src.tasks.ball_refiner.training.evaluation import predict_clip
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256

METRICS = ('error_px', 'nll_uv', 'nll_px', 'presence_nll', 'coverage', 'area_px2')


def paired_reference(path: Path, digest: str, target: dict[str, Array]) -> dict[str, Array]:
    """Never reuse references from another timeline/teacher/gap or changed file."""
    if dual_sha256(path) != digest:
        raise ValueError(f'Reference hash changed: {path}')
    with np.load(path, allow_pickle=False) as saved:
        arrays = dict(saved)
    for key, value in target.items():
        if not np.array_equal(value, arrays[key], equal_nan=True):
            raise ValueError(f'Reference target/frame identity differs: {key}')
    return arrays


def run_cached_comparison(
    training_run: Path, reference: Path, output: Path, *, settings: ComparisonSettings, device: torch.device,
) -> Path:
    """No detector model, JPEGs or context; settings and paired targets stay fixed."""
    if output.exists():
        raise FileExistsError(output)
    reference_hash = dual_sha256(reference / 'manifest.json')
    manifest = json.loads((reference / 'manifest.json').read_text())
    if json.loads((reference / 'run_state.json').read_text())['status'] != 'complete':
        raise ValueError('References require complete paired validation')
    if json.loads(json.dumps(asdict(settings))) != manifest['settings']:
        raise ValueError('HDR settings differ from saved references')
    config, pair, hashes = load_pilot(training_run, device)
    store = BallFrameStore(config.store)
    cache = EvidenceCache(config.evidence, store)
    records = validation_clips(store)
    partition = validation_partition(tuple(r for r in records if r.source == 'meiji'), config.partition_seed)
    if partition != manifest['recipe']['partition'] or str(config.store) != manifest['recipe']['store']:
        raise ValueError('Reference store or partition differs')
    recipe = manifest['recipe']
    if (config.seed != recipe['seed'] or config.window_length != recipe['window'] or config.stride != recipe['stride']
            or list(config.training.gap_lengths) != recipe['training']['gap_lengths']
            or hashes[str(config.evidence / 'manifest.json')] != manifest['input_sha256'].get(str(config.evidence / 'manifest.json'))):
        raise ValueError('Reference seed/window/gaps/evidence differs')
    hashes.update({str(config.store / name): digest for name, digest in cache.manifest['store']['sha256'].items()})
    entries = {(x['clip_id'], x['method'], x['condition']): x for x in manifest['artifacts']}
    rows: dict[str, list[dict[str, Array]]] = defaultdict(list)
    artifacts: list[dict[str, Any]] = []
    output.mkdir(parents=True)
    write_json_atomic(output / 'run_state.json', {'status': 'running'})
    hashes[str(reference / 'manifest.json')] = reference_hash
    for record in records:
        clip = LoadedClip(record, cache.load(record.clip_id), project_store_targets(store, record))
        target = clip.targets
        groups = [record.source]
        if record.camera_id is not None:
            groups.append(f'{record.source}/{record.camera_id}')
        if record.source == 'meiji':
            half = 'selection' if record.clip_id in partition['selection'] else 'calibration'
            groups.extend([f'meiji/{half}', f'meiji/{half}/{record.camera_id}'])
        gap = fixed_gap_mask(record.frame_count, clip_id=record.clip_id, block_length=config.window_length,
                             lengths=config.training.gap_lengths, seed=config.partition_seed)
        for condition in ('observed', 'evidence_gap'):
            mask = gap if condition == 'evidence_gap' else np.zeros_like(gap)
            fixed: dict[str, Array] = {'frame_index': target.frame_index, 'pts': target.pts, 'target_uv': target.uv,
                                      'target_reason': target.reason, 'gap_mask': mask, 'presence': target.presence,
                                      'presence_valid': target.presence_valid}
            methods = {}
            for method in ('new_refiner', 'new_detector'):
                entry = entries[(record.clip_id, method, condition)]
                path = reference / entry['path']
                methods[method] = paired_reference(path, entry['sha256'], fixed)
                hashes[str(path)] = entry['sha256']
            prediction = predict_clip(pair, clip, config, device=device, gap=mask)
            values = gmm_rows(prediction, target, (record.source_width, record.source_height),
                              **{key: value for key, value in asdict(settings).items() if key != 'uniform_weight'},
                              clip_id=record.clip_id, condition=condition, device=device)
            methods['variant'] = values
            for method, metrics in methods.items():
                for label, chosen in strata(target, mask, condition).items():
                    for group in groups:
                        rows[f'{method}/{group}/{condition}/{label}'].append({k: metrics[k][chosen] for k in METRICS})
            path = output / f'clip-{record.index:05d}-{condition}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, **values, **fixed,
                                    **{f.name: getattr(prediction, f.name)[0].numpy() for f in fields(prediction)})
            artifacts.append({'clip_id': record.clip_id, 'source': record.source, 'camera': record.camera_id,
                              'condition': condition, 'frames': record.frame_count, 'path': path.name, 'sha256': dual_sha256(path)})
            write_json_atomic(output / 'progress.json', {'artifacts': artifacts})
        print(json.dumps({'cached_comparison_clip': record.clip_id}), flush=True)
    metrics = {}
    for key, parts in sorted(rows.items()):
        value = summarize(parts, settings.levels)
        errors = np.concatenate([p['error_px'] for p in parts])
        errors = errors[np.isfinite(errors)]
        value['p90_error_px'] = float(np.quantile(errors, .9)) if len(errors) else None
        metrics[key] = value
    if any(dual_sha256(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('Cached evaluation inputs changed during execution')
    write_json_atomic(output / 'metrics.json', metrics)
    write_json_atomic(output / 'manifest.json', {'schema': 'ball_refiner_cached_comparison.v1', 'settings': asdict(settings),
                                               'input_sha256': hashes, 'training_run': str(training_run), 'reference': str(reference),
                                               'artifacts': artifacts, 'partition': partition,
                                               'scope': 'Same val frames and gap masks; no detector replay, test or context; uncalibrated GMM HDR on R2'})
    write_json_atomic(output / 'run_state.json', {'status': 'complete', 'clips': len(records), 'files': len(artifacts)})
    return output
