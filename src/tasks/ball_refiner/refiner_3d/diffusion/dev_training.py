"""Bounded train/val bring-up of x0 flow and the same-backbone regression."""
from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import torch

from src.tasks.ball_refiner.refiner_3d.baselines import evaluate_baselines
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset
from src.tasks.ball_refiner.refiner_3d.synthetic.generator import write_json

from .curves import plot_dev_curves
from .data import collate_windows, rally_window, window_starts
from .dev_config import DevConfig, load_config
from .dev_evaluation import RallyInput, evaluate_dev
from .flow import training_objective
from .losses import TrainingBatch
from .model import TrajectoryDenoiser


class RunBudget:
    def __init__(self, config: DevConfig, device: str) -> None:
        self.config, self.device = config, device
        self.started = time.perf_counter()
        self.peak_device_used_bytes = 0
        self.minimum_available_bytes = 2**63 - 1
        if device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable')
            capacity = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(config.allocator_limit_gib * 1024**3 / capacity, 0)
            torch.cuda.reset_peak_memory_stats()
        elif device != 'cpu':
            raise ValueError('Device must be explicitly cpu or cuda')
        self.check()

    def check(self) -> None:
        if time.perf_counter() - self.started > self.config.maximum_seconds:
            raise TimeoutError('Dev job exceeded its shared wall-time budget')
        available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
        self.minimum_available_bytes = min(self.minimum_available_bytes, available)
        if available < 6 * 1024**3:
            raise MemoryError('Available host RAM fell below 6 GiB')
        if self.device == 'cuda':
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            self.peak_device_used_bytes = max(self.peak_device_used_bytes, total - free)
            if self.peak_device_used_bytes > self.config.maximum_device_bytes:
                raise MemoryError('Dev job exceeded 10 GB device-memory budget')

    def report(self) -> dict[str, Any]:
        return {'elapsed_seconds': time.perf_counter() - self.started,
                'peak_allocated_bytes': torch.cuda.max_memory_allocated() if self.device == 'cuda' else None,
                'peak_reserved_bytes': torch.cuda.max_memory_reserved() if self.device == 'cuda' else None,
                'peak_device_used_bytes': self.peak_device_used_bytes if self.device == 'cuda' else None,
                'minimum_available_host_bytes': self.minimum_available_bytes,
                'driver_measurement': 'total-minus-free sampled each update and val rally; includes other usage; not a continuous driver peak'}


def _train_one(
    config: DevConfig, windows: list[TrainingBatch], validation: list[RallyInput], *,
    device: str, objective: Literal['flow', 'regression'], output: Path, budget: RunBudget,
) -> dict[str, Any]:
    # Same initialization and shuffle sequence for both arms; flow RNG is separate.
    torch.manual_seed(config.seed)
    model = TrajectoryDenoiser(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    flow_rng = torch.Generator(device=device).manual_seed(config.seed + 1)
    order_rng = torch.Generator().manual_seed(config.seed + 2)
    order: list[int] = []
    initial_state_hash_path = output / 'initial-state.pt'
    output.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), initial_state_hash_path)
    arm: dict[str, Any] = {'objective': objective, 'status': 'running', 'parameters': sum(p.numel() for p in model.parameters()),
                          'initial_state_sha256': sha256(initial_state_hash_path), 'validation': []}
    write_json(output / 'manifest.json', arm)

    def evaluate(update: int) -> None:
        metrics = evaluate_dev(model, validation, config, device=device, objective=objective,
                               check_budget=budget.check,
                               predictions=output / 'predictions' / f'update-{update:05d}')
        arm['validation'].append({'update': update, **metrics})
        write_json(output / 'manifest.json', arm)
        print(json.dumps({'objective': objective, 'update': update, 'val': metrics['metrics']['mean'], 'budget': budget.report()}), flush=True)

    evaluate(0)
    model.train()
    train_seconds, trained_frames = 0., 0
    with (output / 'updates.jsonl').open('w') as log:
        for update in range(1, config.updates + 1):
            budget.check()
            started = time.perf_counter()
            while len(order) < config.batch_size:
                order.extend(torch.randperm(len(windows), generator=order_rng).tolist())
            indices, order = order[:config.batch_size], order[config.batch_size:]
            batch = collate_windows([windows[i] for i in indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = training_objective(model, batch, config.loss, flow_rng, objective=objective)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            budget.check()
            seconds = time.perf_counter() - started
            frames = int((~batch.condition.padding_mask).sum())
            train_seconds += seconds
            trained_frames += frames
            row = {'update': update, 'window_indices': indices, 'real_frames': frames, 'seconds': seconds,
                   'loss': loss.item(), 'gradient_norm': norm.item(), **{key: value.item() for key, value in terms.items()},
                   **budget.report()}
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
            if update in config.evaluate_updates:
                evaluate(update)
    checkpoint = output / 'dev-only.pt'
    torch.save({'diagnostic_only': True, 'objective': objective, 'config': asdict(config),
                'state_dict': model.state_dict(), 'updates': config.updates}, checkpoint)
    arm.update(status='complete', updates=config.updates, training_seconds=train_seconds,
               training_frames=trained_frames, training_frames_per_second=trained_frames / train_seconds,
               updates_per_second=config.updates / train_seconds, checkpoint_sha256=sha256(checkpoint),
               checkpoint_bytes=checkpoint.stat().st_size, resources=budget.report())
    write_json(output / 'manifest.json', arm)
    plot_dev_curves(output)
    return arm


def run_dev_training(dataset: Path, config_path: Path, output: Path, *, device: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    config = load_config(config_path)
    torch.set_num_threads(1)
    source = SyntheticDataset(dataset)
    if source.manifest['counts'] != config.expected_counts or source.manifest['plan']['degradation']['boundary_convergence']['method'] != 'fixed_hybrid':
        raise ValueError('Dev run requires the declared counts and fixed-budget H')
    budget = RunBudget(config, device)
    manifest: dict[str, Any] = {'status': 'running', 'diagnostic_only': True, 'device': device,
        'config': asdict(config), 'config_sha256': sha256(config_path), 'dataset': str(dataset),
        'source_manifest_sha256': sha256(dataset / 'manifest.json'), 'arms': {},
        'selection': 'fixed updates and seed; no checkpoint selection, tuning or test-rally reads',
        'precision': 'fp32 eager', 'windows': [], 'read_rallies': [],
        'input_diagnostics': {'frames': 0, 'unassessed_frames': 0, 'nonconverged_frames': 0}}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', manifest)
    try:
        windows: list[TrainingBatch] = []
        validation = []
        for record in sorted(source.records, key=lambda row: row['rally_id']):
            if record['split'] == 'test':
                continue
            if record['split'] not in ('train', 'val'):
                raise ValueError('Unknown split')
            budget.check()
            arrays = source.load(record)
            manifest['read_rallies'].append({'rally_id': record['rally_id'], 'npz_sha256': record['npz_sha256']})
            assessed = arrays['integration_convergence_assessed']
            diagnostics = manifest['input_diagnostics']
            diagnostics['frames'] += record['frames']
            diagnostics['unassessed_frames'] += int((~assessed).sum())
            diagnostics['nonconverged_frames'] += int((assessed & ~arrays['integration_converged']).sum())
            if record['split'] == 'train':
                for start in window_starts(record['frames'], config.frames, config.stride):
                    windows.append(rally_window(arrays, record, start=start, frames=config.frames, allow_nonconverged=True))
                    manifest['windows'].append({'rally_id': record['rally_id'], 'start': start, 'frames': min(config.frames, record['frames'] - start)})
            else:
                validation.append(RallyInput(record, arrays))
        write_json(output / 'manifest.json', manifest)
        manifest['baselines'] = evaluate_baselines(validation, predictions=output / 'baseline_predictions')
        write_json(output / 'baselines.json', manifest['baselines'])
        for objective in ('flow', 'regression'):
            manifest['arms'][objective] = _train_one(config, windows, validation, device=device,
                                                     objective=objective, output=output / objective, budget=budget)
            write_json(output / 'manifest.json', manifest)
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}', resources=budget.report())
        write_json(output / 'manifest.json', manifest)
        raise
    manifest['resources'] = budget.report()
    manifest['output_bytes'] = sum(path.stat().st_size for path in output.rglob('*') if path.is_file())
    write_json(output / 'manifest.json', manifest)
    return manifest
