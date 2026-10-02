"""A single bounded queue allocation: predeclared cached training/evaluation arms."""

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

import numpy as np
import torch
from omegaconf import OmegaConf
from render_comparison import render

from src.tasks.ball_refiner.evaluation.cached_comparison import run_cached_comparison
from src.tasks.ball_refiner.evaluation.pilot_comparison import ComparisonSettings
from src.tasks.ball_refiner.refiner_2d import build_ball_refiner_2d
from src.tasks.ball_refiner.training.configuration import PilotConfig
from src.tasks.ball_refiner.training.evaluation import predict_clip
from src.tasks.ball_refiner.training.runner import prepare_data, run_training
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256
from src.utils.resource_guard import HostRAMGuard, available_ram_bytes, process_tree_rss_bytes


def differences(a: dict[str, Any], b: dict[str, Any], prefix: str = '') -> dict[str, Any]:
    result = {}
    for key in sorted(set(a) | set(b)):
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(a.get(key), dict) and isinstance(b.get(key), dict):
            result.update(differences(a[key], b[key], name))
        elif a.get(key) != b.get(key):
            result[name] = {'baseline': a.get(key), 'variant': b.get(key)}
    return result


def ram_available() -> int:
    return available_ram_bytes()


def output_size(path: Path) -> int:
    """Count surviving files without following directory links or hiding IO errors."""
    total = 0
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        total += output_size(Path(entry.path))
                    elif entry.is_file():
                        total += entry.stat().st_size
                except FileNotFoundError:
                    # A compiler temporary file disappeared after discovery.
                    continue
    except FileNotFoundError:
        # The directory itself may disappear before opening or while iterating.
        # Handle each subtree separately so surviving siblings are still counted.
        pass
    return total


def preflight(plan: dict[str, Any]) -> dict[str, Any]:
    for name, digest in plan['input_sha256'].items():
        if dual_sha256(Path(name)) != digest:
            raise ValueError(f'Frozen input changed: {name}')
    if ram_available() < 8 * 1024**3 or shutil.disk_usage(plan['output_root']).free < plan['disk_budget_bytes']:
        raise RuntimeError('Insufficient launch headroom (8 GiB RAM and disk budget required)')
    if Path(plan['report']).exists():
        raise FileExistsError(plan['report'])
    baseline = Path(plan['baseline'])
    base = PilotConfig.from_config(OmegaConf.load(baseline / 'config.yaml'))
    expected_manifest = json.loads((baseline / 'data_manifest.json').read_text())
    configs = []
    for variant in plan['variants']:
        cfg = PilotConfig.from_config(OmegaConf.load(variant['training_config']))
        diff = differences(base.resolved, cfg.resolved)
        if diff != variant['declared_diff']:
            raise ValueError(f'Undeclared variant factor: {variant["name"]}: {diff}')
        if cfg.output.exists() or Path(variant['evaluation_output']).exists():
            raise FileExistsError(variant['name'])
        dataset, selection, manifest = prepare_data(cfg)
        if manifest != expected_manifest:
            raise ValueError('Sampling/targets/cache/selection differs from r18')
        configs.append({'name': variant['name'], 'diff': diff, 'training_windows': len(dataset),
                        'source_windows': manifest['source_windows'], 'selection_clips': len(selection)})
        del dataset, selection
        gc.collect()
    # Actual legacy checkpoint replay proves unchanged absolute-model semantics.
    _, selection, _ = prepare_data(base)
    best = json.loads((baseline / 'best.json').read_text())
    checkpoint = torch.load(baseline / best['checkpoint'], map_location='cpu', weights_only=True)
    pair = build_ball_refiner_2d(base.model)
    pair.model.load_state_dict(checkpoint['state_dict'], strict=True)
    clip = min(selection, key=lambda c: c.record.frame_count)
    prediction = predict_clip(pair, clip, base, device=torch.device('cpu'), gap=np.zeros(clip.record.frame_count, np.bool_))
    path = baseline / best['predictions'] / f'clip-{clip.record.index:05d}-observed.npz'
    errors = {}
    with np.load(path, allow_pickle=False) as saved:
        for field in fields(prediction):
            actual = getattr(prediction, field.name)[0].numpy()
            np.testing.assert_allclose(actual, saved[field.name], atol=3e-4 if 'logits' in field.name else 3e-5, rtol=3e-4)
            errors[field.name] = float(np.max(np.abs(actual - saved[field.name])))
    return {'status': 'passed', 'variants': configs, 'input_files': len(plan['input_sha256']),
            'baseline_cpu_replay': {'clip': clip.record.clip_id, 'frames': clip.record.frame_count, 'max_abs_error': errors},
            'ram_available_bytes': ram_available()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--preflight-output', type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    torch.set_num_threads(2)
    started = time.monotonic()
    checked = preflight(plan)
    if args.preflight_output:
        write_json_atomic(args.preflight_output, checked)
        print(json.dumps(checked), flush=True)
        return
    report = Path(plan['report'])
    report.mkdir(parents=True)
    write_json_atomic(report / 'preflight.json', checked)
    shutil.copyfile(args.plan, report / 'plan.json')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(plan['allocator_limit_bytes'] / torch.cuda.get_device_properties(device).total_memory, device)
    torch.cuda.reset_peak_memory_stats(device)
    status = 'running'
    phases: list[dict[str, Any]] = []
    stop = threading.Event()
    monitor: dict[str, Any] = {'peak_device_used_bytes': 0, 'samples': 0, 'failure': None}
    ram_guard = HostRAMGuard()
    ram_guard.sample(ram_available(), time.monotonic() - started, rss_bytes=process_tree_rss_bytes())

    def terminate(signum: int, frame: FrameType | None) -> None:
        raise RuntimeError(f'Queue timeout/resource signal {signum}: {monitor["failure"]}')

    signal.signal(signal.SIGTERM, terminate)
    output_paths = [report, Path(plan['compiler_cache'])]
    for variant in plan['variants']:
        output_paths.extend([PilotConfig.from_config(OmegaConf.load(variant['training_config'])).output, Path(variant['evaluation_output'])])

    def watchdog() -> None:
        while not stop.wait(1):
            try:
                result = subprocess.run(['nvidia-smi', '-i', '0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                        check=True, capture_output=True, text=True, timeout=5)
                used = int(result.stdout.strip()) * 1024**2
                monitor['peak_device_used_bytes'] = max(monitor['peak_device_used_bytes'], used)
                monitor['samples'] += 1
                ram_failure = ram_guard.sample(ram_available(), time.monotonic() - started, rss_bytes=process_tree_rss_bytes())
                if used > plan['gpu_device_used_limit_bytes'] or ram_failure:
                    raise RuntimeError(f'Resource limit exceeded: device {used}, RAM: {ram_failure}')
                if monitor['samples'] % 10 == 0:
                    disk_bytes = sum(output_size(p) for p in output_paths)
                    if disk_bytes > plan['disk_budget_bytes']:
                        raise RuntimeError(f'Output disk limit exceeded: {disk_bytes}')
            except Exception as error:
                monitor['failure'] = repr(error)
                os.kill(os.getpid(), signal.SIGTERM)
                return

    def publish() -> None:
        write_json_atomic(report / 'resource_usage.json', {
            'status': status, 'seconds': time.monotonic() - started, 'phases': phases, 'device_monitor': monitor,
            'host_ram': ram_guard.report(),
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device), 'queue_job': os.environ.get('TENNIS_RUN_ID'),
            'output_bytes': {str(p): output_size(p) for p in output_paths},
        })

    thread = threading.Thread(target=watchdog, daemon=True)
    thread.start()
    try:
        settings = ComparisonSettings(**{**plan['comparison_settings'], 'levels': tuple(plan['comparison_settings']['levels'])})
        for variant in plan['variants']:
            torch.compiler.reset()
            begin = time.monotonic()
            training = run_training(OmegaConf.load(variant['training_config']))
            phases.append({'variant': variant['name'], 'phase': 'training', 'seconds': time.monotonic() - begin})
            publish()
            gc.collect()
            torch.cuda.empty_cache()
            begin = time.monotonic()
            run_cached_comparison(training, Path(plan['reference']), Path(variant['evaluation_output']), settings=settings, device=device)
            phases.append({'variant': variant['name'], 'phase': 'evaluation', 'seconds': time.monotonic() - begin})
            publish()
            gc.collect()
            torch.cuda.empty_cache()
        render(plan, report)
        if monitor['failure']:
            raise RuntimeError(monitor['failure'])
        for name, digest in plan['input_sha256'].items():
            if dual_sha256(Path(name)) != digest:
                raise ValueError(f'Frozen input changed during job: {name}')
        status = 'complete'
    except BaseException:
        status = 'failed'
        raise
    finally:
        stop.set()
        thread.join(timeout=6)
        publish()


if __name__ == '__main__':
    main()
