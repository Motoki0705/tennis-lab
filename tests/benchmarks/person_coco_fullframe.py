"""One bounded COCO-only full-frame dev run; invoke infer through the GPU queue."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import torch
from player_detection_far_diagnosis import (  # type: ignore[import-not-found]  # sibling CLI module
    infer_camera,
    preflight,
)

from src.submodules.models import DinoPersonDetector
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256


def plan_run(repo: Path, comparison: Path, report: Path) -> dict[str, Any]:
    checked = preflight(repo, comparison, report)
    if checked['baseline_size'] != [800, 1333]:
        raise ValueError('This grant permits only the original 800/1333 input')
    return {key: checked[key] for key in ('comparison', 'comparison_sha256', 'reservation',
        'reservation_sha256', 'pipeline_config_sha256', 'repository', 'baseline_size', 'inputs')} | {
        'schema': 'coco_fullframe_dev_v1', 'status': 'planned', 'floor': .01,
        'weights': {'coco': checked['weights']['coco']}, 'scope': 'full_frame_no_roi',
        'runtime_scope': 'synchronized predictor; excludes video decode, model load and archive IO',
        'torch_allocator_limit_bytes': 6 * 1024**3, 'wall_limit_seconds': 5400,
        'archives': {}, 'report': str(report)}


def infer(plan: dict[str, Any], report: Path) -> None:
    if (report / 'progress.json').exists():
        raise FileExistsError('Never overwrite an earlier or partially completed run')
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    total = torch.cuda.get_device_properties(0).total_memory
    # Reserve <=6 GiB for torch, leaving >2 GB for CUDA/non-torch allocations
    # within the 8 GB grant. This is not a claimed whole-process measurement.
    torch.cuda.set_per_process_memory_fraction(plan['torch_allocator_limit_bytes'] / total)
    torch.cuda.reset_peak_memory_stats()
    weight = plan['weights']['coco']
    if dual_sha256(Path(weight['path'])) != weight['sha256']:
        raise ValueError('COCO weights changed after preflight')
    detector = DinoPersonDetector(Path(weight['path']), Path(plan['repository']), device='cuda',
        confidence=.01, short_side=800, max_long_side=1333)
    started = time.perf_counter()
    plan['status'] = 'running'
    write_json_atomic(report / 'progress.json', plan)
    try:
        detector.load()
        saved = plan['archives']['coco_fullframe_0.01'] = {}
        for record in plan['inputs']:
            key = f"{record['clip']}/{record['camera']}"
            plan['current'] = key
            write_json_atomic(report / 'progress.json', plan)
            # Direct predictor output: no calibration/ROI is passed to it.
            archive = infer_camera(detector, record, tiled=False)
            saved[key] = archive.save(report / 'raw/coco_fullframe_0.01' / f'{key}.npz')
            plan.update(elapsed_seconds=time.perf_counter() - started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            write_json_atomic(report / 'progress.json', plan)
            print(json.dumps({'key': key, **saved[key], 'elapsed_seconds': plan['elapsed_seconds']}), flush=True)
        plan['status'] = 'ok'
        write_json_atomic(report / 'inference.json', plan)
    except Exception as error:
        plan.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json_atomic(report / 'progress.json', plan)
        detector.unload()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('preflight', 'infer'), required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    plan = plan_run(args.repo.resolve(), args.comparison.resolve(), args.report.resolve())
    target = args.report / 'plan.json'
    if args.phase == 'preflight':
        args.report.mkdir(parents=True, exist_ok=True)
        with target.open('x') as handle:
            json.dump(plan, handle, indent=2)
        print(json.dumps({'camera_clips': len(plan['inputs']),
            'frames': sum(r['video']['num_frames'] for r in plan['inputs'])}))
    else:
        if json.loads(target.read_text()) != plan:
            raise ValueError('Preflight changed; do not run different inputs under the same grant')
        infer(plan, args.report)


if __name__ == '__main__':
    main()
