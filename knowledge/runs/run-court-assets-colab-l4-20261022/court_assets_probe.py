"""Measure the pinned public NHT renderer on existing exported cameras."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

import torch
from nht_pipeline import render as nht

CAMERAS = {
    'B00': ['frame_000000', 'frame_000163', 'frame_000326', 'frame_000490'],
    'B01': ['frame_000000', 'frame_000088', 'frame_000176', 'frame_000266'],
    'B02': ['frame_000000', 'frame_000052', 'frame_000104', 'frame_000156'],
    'B03': ['frame_000000', 'frame_000050', 'frame_000101', 'frame_000161'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--working', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.working.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.cuda.init()
    torch.cuda.synchronize()
    report = {
        'schema': 'court_render_probe_v1',
        'batch_size': 1,
        'without_source_images': True,
        'camera_ids': CAMERAS,
        'context_init_seconds': time.perf_counter() - started,
        'environment': {
            'python': platform.python_version(),
            'gpu': torch.cuda.get_device_name(),
            'cuda': torch.version.cuda,
            'packages': {name: importlib.metadata.version(name) for name in ('torch', 'gsplat', 'numpy', 'tinycudann')},
            'nvidia_smi': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip(),
        },
        'renders': [],
    }
    original_init = nht.ResidentSceneRenderer.__init__
    original_batch = nht.ResidentSceneRenderer.render_batch
    original_load = torch.load
    for repeat in range(args.repeats):
        for scene_id, ids in CAMERAS.items():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            item = {'scene_id': scene_id, 'repeat': repeat, 'load_seconds': [], 'checkpoint_load_seconds': [], 'batch_seconds': []}

            def measured_load(*a, **kw):
                start = time.perf_counter()
                value = original_load(*a, **kw)
                torch.cuda.synchronize()
                item['checkpoint_load_seconds'].append(time.perf_counter() - start)
                return value

            def measured_init(self, *a, **kw):
                start = time.perf_counter()
                original_init(self, *a, **kw)
                torch.cuda.synchronize()
                item['load_seconds'].append(time.perf_counter() - start)

            def measured_batch(self, *a, **kw):
                torch.cuda.synchronize()
                start = time.perf_counter()
                value = original_batch(self, *a, **kw)
                torch.cuda.synchronize()
                item['batch_seconds'].append(time.perf_counter() - start)
                return value

            target = args.working / f'repeat-{repeat}' / scene_id
            device_samples = []
            monitor = subprocess.Popen(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-lms', '100'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            def collect_device_memory():
                for line in monitor.stdout:
                    device_samples.append(float(line.strip()) * 1024 ** 2)
            thread = threading.Thread(target=collect_device_memory, daemon=True)
            thread.start()
            started = time.perf_counter()
            with patch.object(nht.ResidentSceneRenderer, '__init__', measured_init), patch.object(nht.ResidentSceneRenderer, 'render_batch', measured_batch), patch.object(torch, 'load', measured_load):
                manifest = nht.render_scene(args.scenes / scene_id / 'reconstruction/export/scene.json', target, camera_ids=ids, batch_size=1, without_source_images=True)
            item['public_seconds'] = time.perf_counter() - started
            monitor.terminate()
            monitor.wait(timeout=10)
            thread.join(timeout=10)
            if not device_samples:
                raise RuntimeError('nvidia-smi memory sampler returned no samples')
            item['sampled_device_peak_used_bytes'] = max(device_samples)
            item['device_memory_samples'] = len(device_samples)
            item['device_memory_sample_interval_ms'] = 100
            item['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            item['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            item['frame_count'] = len(manifest['renders'])
            item['steady_frames_per_second'] = 3 / sum(item['batch_seconds'][1:])
            item['output_bytes'] = sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
            published = args.output / f'repeat-{repeat}' / scene_id
            published.parent.mkdir(parents=True, exist_ok=True)
            target.replace(published)
            report['renders'].append(item)
            metrics = args.working / 'metrics.json'
            metrics.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            metrics.replace(args.output / 'metrics.json')
            print(json.dumps(item), flush=True)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
