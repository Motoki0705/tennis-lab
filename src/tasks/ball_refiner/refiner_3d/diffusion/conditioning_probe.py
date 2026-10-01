"""CPU-only frozen-token linear read-out; never trains the trajectory model."""
from __future__ import annotations

import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset
from src.tasks.ball_refiner.refiner_3d.synthetic.generator import write_json
from src.utils.schema.court_normalization import (
    court_coordinate_normalization_metadata,
    denormalize_court_position,
    normalize_court_position,
)

from .cohort import reference_validation
from .data import rally_window
from .model import ModelConfig, TrajectoryDenoiser, condition_features

RCOND = 1e-12
READOUT_TOLERANCE_M = 0.1  # Diagnostic criterion, not a model acceptance gate.


def _design(tokens: Tensor) -> Tensor:
    if tokens.ndim != 2 or min(tokens.shape) < 1 or not tokens.is_floating_point():
        raise ValueError('Need a nonempty floating frame-by-feature matrix')
    if tokens.device.type != 'cpu' or not bool(torch.isfinite(tokens).all()):
        raise ValueError('Read-out requires finite CPU tokens')
    return torch.cat((tokens.double(), torch.ones((len(tokens), 1), dtype=torch.float64)), dim=1)


@dataclass(frozen=True)
class LinearReadout:
    coefficients: Tensor
    rank: int
    singular_values: Tensor

    def predict(self, tokens: Tensor) -> Tensor:
        return _design(tokens) @ self.coefficients


def fit_readout(train_tokens: Tensor, train_targets_norm: Tensor) -> LinearReadout:
    """Unregularized minimum-norm OLS, including an intercept, in float64."""
    design = _design(train_tokens)
    if train_targets_norm.shape != (len(design), 3) or not train_targets_norm.is_floating_point():
        raise ValueError('Need XYZ targets for each train frame')
    if train_targets_norm.device.type != 'cpu' or not bool(torch.isfinite(train_targets_norm).all()):
        raise ValueError('Read-out requires finite CPU targets')
    fitted = torch.linalg.lstsq(design, train_targets_norm.double(), rcond=RCOND, driver='gelsd')
    if not bool(torch.isfinite(fitted.solution).all()):
        raise FloatingPointError('Nonfinite read-out coefficients')
    # Rank deficiency is reported, not repaired with a second solver or ridge.
    return LinearReadout(fitted.solution, int(fitted.rank), fitted.singular_values)


def error_summary(prediction_m: Tensor, target_m: Tensor) -> dict[str, Any]:
    if prediction_m.shape != target_m.shape or prediction_m.ndim != 2 or prediction_m.shape[1] != 3:
        raise ValueError('Expected matching frame-by-XYZ predictions and targets')
    if not bool(torch.isfinite(prediction_m).all() & torch.isfinite(target_m).all()):
        raise FloatingPointError('Nonfinite read-out prediction or target')
    if not len(target_m):
        return {'frames': 0, 'rmse_m': None, 'p50_m': None, 'p95_m': None, 'maximum_m': None}
    errors = torch.linalg.vector_norm(prediction_m - target_m, dim=-1)
    return {'frames': len(errors), 'rmse_m': float(errors.square().mean().sqrt()),
            'p50_m': float(torch.quantile(errors, .5)), 'p95_m': float(torch.quantile(errors, .95)),
            'maximum_m': float(errors.max())}


def run_conditioning_probe(
    dataset: Path, training_output: Path, output: Path, *, objective: str = 'flow',
) -> dict[str, Any]:
    if objective not in ('flow', 'regression'):
        raise ValueError('Read-out objective must be flow or regression')
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    started = time.perf_counter()
    training_path, dataset_path = training_output / 'manifest.json', dataset / 'manifest.json'
    hashes = {'training_manifest': sha256(training_path), 'dataset_manifest': sha256(dataset_path)}
    training = json.loads(training_path.read_text())
    if training['status'] != 'complete' or hashes['dataset_manifest'] != training['source_manifest_sha256']:
        raise ValueError('Require the completed training run and its exact dataset')
    arm = training['arms'][objective]
    checkpoint = training_output / objective / 'dev-only.pt'
    hashes['checkpoint'] = sha256(checkpoint)
    if arm['status'] != 'complete' or hashes['checkpoint'] != arm['checkpoint_sha256']:
        raise ValueError(f'Incomplete or changed {objective} checkpoint')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not saved['diagnostic_only'] or saved['objective'] != objective or saved['updates'] != arm['updates']:
        raise ValueError('Unexpected diagnostic checkpoint identity')
    if saved['config'] != {**training['config'], 'evaluate_updates': tuple(training['config']['evaluate_updates'])}:
        raise ValueError('Checkpoint/config mismatch')
    model = TrajectoryDenoiser(ModelConfig(**training['config']['model'])).eval().requires_grad_(False)
    model.load_state_dict(saved['state_dict'], strict=True)
    if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError('Nonfinite encoder state')
    source = SyntheticDataset(dataset)
    if source.manifest['counts'] != training['config']['expected_counts']:
        raise ValueError('Unexpected split sizes')
    val_ids = reference_validation(source.records, training)
    records = sorted((r for r in source.records if r['split'] == 'train' or r['rally_id'] in val_ids),
                     key=lambda r: r['rally_id'])
    identities = [{'rally_id': r['rally_id'], 'npz_sha256': r['npz_sha256']} for r in records]
    if identities != training['read_rallies']:
        raise ValueError('Probe must use exactly the historical train/val inputs')
    manifest: dict[str, Any] = {
        'status': 'running', 'diagnostic_only': True, 'device': 'cpu',
        'dataset': str(dataset), 'training_output': str(training_output), 'input_hashes': hashes,
        'encoder': {'objective': objective, 'updates': saved['updates'], 'frozen': True,
                    'location': 'weighted nonlinear component tokens before state addition and temporal attention'},
        'solver': {'driver': 'gelsd', 'rcond': RCOND, 'ridge': 0., 'intercept': True, 'dtype': 'float64'},
        'readout_tolerance_m': READOUT_TOLERANCE_M, 'court_normalization': court_coordinate_normalization_metadata(),
        'selection': 'all train frames fit, exact historical val cohort evaluated; no extra val/test NPZ reads or GT fit target',
        'unused_val_rallies': sorted(r['rally_id'] for r in source.records
                                    if r['split'] == 'val' and r['rally_id'] not in val_ids),
        'read_rallies': [], 'results': {},
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', manifest)
    tokens: dict[str, list[Tensor]] = {'train': [], 'val': []}
    targets: dict[str, list[Tensor]] = {'train': [], 'val': []}
    visibility: dict[str, list[Tensor]] = {'train': [], 'val': []}
    max_round_trip = 0.
    max_raw_feature_error = 0.
    minimum_available = 2**63 - 1
    try:
        with torch.no_grad():
            for record in records:
                available = next(int(s.split()[1]) * 1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
                minimum_available = min(minimum_available, available)
                if available < 6 * 1024**3:
                    raise MemoryError('Available host RAM fell below 6 GiB')
                arrays = source.load(record)
                condition = rally_window(arrays, record, start=0, frames=record['frames'], allow_nonconverged=True).condition
                # High-precision target uses every stored weight, without renormalizing.
                mean_m = (condition.means_m.double() * condition.weights.double()[..., None]).sum(-2)[0]
                normalized = normalize_court_position(mean_m)
                round_trip = denormalize_court_position(normalized)
                raw_readout = denormalize_court_position((condition_features(condition)[..., :3].double() * condition.weights.double()[..., None]).sum(-2)[0])
                max_round_trip = max(max_round_trip, float((round_trip - mean_m).abs().max()))
                max_raw_feature_error = max(max_raw_feature_error, float((raw_readout - mean_m).abs().max()))
                encoded = model.encode_condition(condition)[0]
                if not bool(torch.isfinite(encoded).all()):
                    raise FloatingPointError('Nonfinite pooled condition tokens')
                split = record['split']
                tokens[split].append(encoded)
                targets[split].append(mean_m)
                visibility[split].append(torch.from_numpy((~(arrays['occlusion_mask'] | arrays['out_of_frame_mask'])).sum(0)))
                manifest['read_rallies'].append({**identities[len(manifest['read_rallies'])], 'split': split,
                                               'frames': record['frames'], 'components': condition.weights.shape[-1]})
                write_json(output / 'manifest.json', manifest)
        if max_round_trip > 1e-10 or max_raw_feature_error > 1e-5:
            raise ValueError('Court normalization/raw feature read-out exceeds numeric tolerance')
        fit = fit_readout(torch.cat(tokens['train']), normalize_court_position(torch.cat(targets['train'])))
        np.savez_compressed(output / 'head.npz', coefficients=fit.coefficients.numpy(),
                            singular_values=fit.singular_values.numpy(), rank=fit.rank)
        manifest['solver'].update(rank=fit.rank, columns=fit.coefficients.shape[0])
        for split in ('train', 'val'):
            target = torch.cat(targets[split])
            predicted = denormalize_court_position(fit.predict(torch.cat(tokens[split])))
            visible = torch.cat(visibility[split])
            manifest['results'][split] = {
                'overall': error_summary(predicted, target),
                'by_visible_cameras': {str(i): error_summary(predicted[visible == i], target[visible == i]) for i in range(4)},
            }
            np.savez_compressed(output / (split + '.npz'), prediction_m=predicted.numpy(),
                                target_m=target.numpy(), visible_cameras=visible.numpy())
        if hashes != {'training_manifest': sha256(training_path), 'dataset_manifest': sha256(dataset_path), 'checkpoint': sha256(checkpoint)}:
            raise ValueError('Input identity changed during the probe')
        for record in records:
            if sha256(dataset / (record['rally_id'] + '.npz')) != record['npz_sha256']:
                raise ValueError('Rally changed during the probe')
        manifest.update(status='complete', meets_diagnostic_tolerance=manifest['results']['val']['overall']['rmse_m'] <= READOUT_TOLERANCE_M,
                        maximum_round_trip_abs_m=max_round_trip, maximum_raw_feature_readout_abs_m=max_raw_feature_error)
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'manifest.json', manifest)
        raise
    manifest['resources'] = {'elapsed_seconds': time.perf_counter() - started,
                             'peak_process_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                             'minimum_available_host_bytes': minimum_available, 'gpu_jobs': 0}
    manifest['artifacts'] = {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size} for p in sorted(output.glob('*.npz'))}
    write_json(output / 'manifest.json', manifest)
    return manifest
