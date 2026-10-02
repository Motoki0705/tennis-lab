"""One bounded queue job: r4 recipe, r5 diagnostic, paired validation comparison."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import fields
from pathlib import Path
from types import FrameType
from typing import Any

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from src.tasks.ball_detection.data.store import BallFrameStore
from src.tasks.ball_refiner.data.evidence_cache import EvidenceCache
from src.tasks.ball_refiner.data.targets import project_store_targets
from src.tasks.ball_refiner.data.windows import LoadedClip
from src.tasks.ball_refiner.evaluation.detector_selection import validation_clips
from src.tasks.ball_refiner.evaluation.pilot_comparison import (
    ComparisonSettings,
    run_comparison,
)
from src.tasks.ball_refiner.evaluation.runner import run_evaluation
from src.tasks.ball_refiner.refiner_2d import build_ball_refiner_2d
from src.tasks.ball_refiner.training.configuration import PilotConfig
from src.tasks.ball_refiner.training.evaluation import predict_clip
from src.tasks.ball_refiner.training.runner import prepare_data, run_training
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256


def differences(a: dict[str, Any], b: dict[str, Any], prefix: str = '') -> dict[str, Any]:
    result = {}
    for key in sorted(set(a) | set(b)):
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
            result.update(differences(a[key], b[key], name))
        elif a.get(key) != b.get(key):
            result[name] = {'old': a.get(key), 'new': b.get(key)}
    return result


def preflight(plan: dict[str, Any]) -> dict[str, Any]:
    for name, digest in plan['input_sha256'].items():
        if dual_sha256(Path(name)) != digest:
            raise ValueError(f'Fixed input changed: {name}')
    new_config = OmegaConf.load(plan['training_config'])
    config = PilotConfig.from_config(new_config)
    old_path = Path(plan['old_run'])
    old_config = PilotConfig.from_config(OmegaConf.load(old_path / 'config.yaml'))
    diff = differences(old_config.resolved, config.resolved)
    if set(diff) != {'data.evidence', 'run.output_dir'}:
        raise ValueError(f'Unexpected change from r4 recipe: {diff}')
    for path in (config.output, Path(plan['comparison_output']), Path(plan['report'])):
        if path.exists():
            raise FileExistsError(path)
    calibration = OmegaConf.load(plan['calibration_config'])
    calibration_output = Path(calibration.paths.output_root) / calibration.run.output_dir
    if calibration_output.exists():
        raise FileExistsError(calibration_output)
    if shutil.disk_usage(config.output.parents[3]).free < plan['disk_budget_bytes']:
        raise RuntimeError('Insufficient disk headroom')
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if available < 8 * 1024**3:
        raise RuntimeError('Require 8 GiB host headroom at launch')
    dataset, selection, manifest = prepare_data(config)
    old_manifest = json.loads((old_path / 'data_manifest.json').read_text())
    manifest_diff = differences(old_manifest, manifest)
    if set(manifest_diff) - {'evidence_manifest_sha256', 'detector.checkpoint', 'detector.sha256', 'detector.device'}:
        raise ValueError(f'Training windows/sampling/gaps/partition changed: {manifest_diff}')
    store = BallFrameStore(config.store)
    cache = EvidenceCache(config.evidence, store)
    references: dict[str, dict[str, int]] = {}
    for clip in validation_clips(store):
        cache.load(clip.clip_id)
        references[clip.clip_id] = project_store_targets(store, clip).counts()
    # Check the current shared inference refactor against the actual r4 best artifact.
    best = json.loads((old_path / 'best.json').read_text())
    checkpoint = torch.load(old_path / best['checkpoint'], map_location='cpu', weights_only=True)
    pair = build_ball_refiner_2d(old_config.model)
    pair.model.load_state_dict(checkpoint['state_dict'], strict=True)
    old_cache = EvidenceCache(old_config.evidence, store)
    record = min((store.clip_by_id(name) for name in old_manifest['validation']['selection']), key=lambda r: r.frame_count)
    clip = LoadedClip(record, old_cache.load(record.clip_id), project_store_targets(store, record))
    prediction = predict_clip(pair, clip, old_config, device=torch.device('cpu'), gap=np.zeros(record.frame_count, np.bool_))
    saved_path = old_path / best['predictions'] / f'clip-{record.index:05d}-observed.npz'
    errors = {}
    with np.load(saved_path, allow_pickle=False) as saved:
        for field in fields(prediction):
            actual = getattr(prediction, field.name)[0].numpy()
            np.testing.assert_allclose(actual, saved[field.name], rtol=3e-4, atol=3e-4 if 'logits' in field.name else 3e-5)
            errors[field.name] = float(np.max(np.abs(actual - saved[field.name])))
    return {'status': 'passed', 'config_diff': diff, 'data_manifest_diff': manifest_diff,
            'source_windows': manifest['source_windows'], 'training_windows': len(dataset),
            'train_frames': manifest['train_frames'], 'selection_clips': len(selection),
            'validation_partition': manifest['validation'], 'validation_target_counts': references,
            'old_checkpoint_cpu_replay': {'clip_id': record.clip_id, 'frames': record.frame_count,
                                           'saved_sha256': dual_sha256(saved_path), 'max_abs_differences': errors},
            'ram_available_bytes': available}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--preflight-output', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    checked = preflight(plan)
    if args.preflight_output is not None:
        write_json_atomic(args.preflight_output, checked)
        print(json.dumps({'preflight': 'passed', 'output': str(args.preflight_output)}))
        return
    report = Path(plan['report'])
    report.mkdir(parents=True, exist_ok=False)
    write_json_atomic(report / 'preflight.json', checked)
    shutil.copyfile(args.plan, report / 'plan.json')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(plan['allocator_limit_bytes'] / total, device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    status, phases = 'failed', {}
    stop = threading.Event()
    monitor: dict[str, Any] = {'peak_device_used_bytes': 0, 'samples': 0, 'failure': None}

    def terminate(signum: int, frame: FrameType | None) -> None:
        raise RuntimeError(f'Queue timeout/resource stop signal {signum}: {monitor["failure"]}')

    signal.signal(signal.SIGTERM, terminate)

    def watchdog() -> None:
        while not stop.wait(1):
            try:
                result = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                        check=True, capture_output=True, text=True, timeout=5)
                used = int(result.stdout.strip()) * 1024**2
                monitor['peak_device_used_bytes'] = max(monitor['peak_device_used_bytes'], used)
                monitor['samples'] += 1
                available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                                 if line.startswith('MemAvailable:'))
                if used > plan['gpu_device_used_limit_bytes'] or available < 6_000_000_000:
                    raise RuntimeError(f'Resource budget exceeded: device VRAM {used}, host available {available}')
            except Exception as error:
                monitor['failure'] = repr(error)
                os.kill(os.getpid(), signal.SIGTERM)
                return

    thread = threading.Thread(target=watchdog, daemon=True)
    thread.start()
    try:
        step = time.monotonic()
        new_run = run_training(OmegaConf.load(plan['training_config']))
        phases['training_seconds'] = time.monotonic() - step
        gc.collect()
        torch.cuda.empty_cache()
        step = time.monotonic()
        run_evaluation(OmegaConf.load(plan['calibration_config']))
        phases['calibration_diagnostic_seconds'] = time.monotonic() - step
        gc.collect()
        torch.cuda.empty_cache()
        step = time.monotonic()
        settings = {**plan['comparison_settings'], 'levels': tuple(plan['comparison_settings']['levels'])}
        run_comparison(Path(plan['old_run']), new_run, Path(plan['comparison_output']),
                       settings=ComparisonSettings(**settings), device=device)
        phases['comparison_seconds'] = time.monotonic() - step
        if monitor['failure'] is not None:
            raise RuntimeError(monitor['failure'])
        status = 'complete'
    finally:
        stop.set()
        thread.join(timeout=6)
        write_json_atomic(report / 'resource_usage.json', {
            'status': status, 'seconds': time.monotonic() - started, 'phases': phases, 'device_monitor': monitor,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'allocator_limit_bytes': plan['allocator_limit_bytes'], 'queue_job': os.environ.get('TENNIS_RUN_ID'),
        })


if __name__ == '__main__':
    main()
