"""One bounded GPU queue job: native KPR parts for the same 40,531 dev rows.

Requires recorded CPU parity before planning. Does not run detection, pose,
tracking, selection or unseen evaluation. Native parts/visibility stay explicit.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
from person_tracking_matrix import (  # type: ignore[import-not-found]
    CLIP,
    checked,
    record_file,
)

from src.tasks.person_tracking.archive import load_features
from src.tasks.player_association.appearance.kpr import WEIGHT_SHA256, KprEncoder
from src.tasks.player_association.appearance.sampling import crop
from src.tasks.player_detection.evaluation.person_sources import DEV_CLIPS
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256
from src.utils.video import OpenCVVideoFrameReader

CODE = Path(__file__).resolve().parents[2]


def plan(args: argparse.Namespace) -> None:
    if (args.report / 'plan.json').exists():
        raise FileExistsError(args.report / 'plan.json')
    parity = json.loads(args.parity.read_text())
    if parity['status'] != 'ok' or not parity['state_keys_equal'] or max(parity['max_abs_diff'].values()) > 1e-6 \
            or not parity['prompts_bitwise_equal'] or 'all_other_detection_poses_negative' not in parity['trials']:
        raise ValueError('KPR features require successful native output and prompt parity')
    for path, sha in parity['port_files'].items():
        checked({'path': path, 'sha256': sha})
    features = json.loads((args.features / 'features.json').read_text())
    source_plan = json.loads((args.features / 'plan.json').read_text())
    expected = {f'{clip}/{cam}' for clip in DEV_CLIPS for cam in ('cam0', 'cam1', 'cam2')}
    if features['status'] != 'ok' or set(features['records'][CLIP]) != expected \
            or sum(v['detections'] for v in features['records'][CLIP].values()) != 40531 \
            or features['plan_sha256'] != dual_sha256(args.features / 'plan.json'):
        raise ValueError('Require the exact completed all-person dev feature run')
    for entry in features['records'][CLIP].values():
        checked(entry)
    weight = args.repo / 'ckpt/player_association/kpr/kpr_market_SOLIDER_93.25_96.59_41453430.pth.tar'
    checked({'path': str(weight), 'sha256': WEIGHT_SHA256})
    write_json_atomic(args.report / 'plan.json', {'schema': 'kpr_native_dev_plan_v1', 'weights': record_file(weight),
        'parent': record_file(args.features / 'features.json'), 'parent_plan': record_file(args.features / 'plan.json'),
        'parity': record_file(args.parity), 'port_files': parity['port_files'],
        'entry_code': record_file(Path(__file__)), 'inputs': features['records'][CLIP],
        'reservation': source_plan['reservation'], 'rows': 40531, 'frames': 10491, 'batch_size': 2,
        'negative_prompts': 'all other detections in the same frame, outside-crop joints invisible',
        'output_contract': 'person_kpr_native_features_v1: normalized (N,6,512) parts, (N,6) visibility; no flattened proxy',
        'timeout_seconds': 5400, 'allocator_bytes': 7 * 1024**3, 'disk_limit_bytes': 1_500_000_000})


def extract(args: argparse.Namespace) -> None:
    path = args.report / 'plan.json'
    run = json.loads(path.read_text())
    if (args.report / 'features.progress.json').exists():
        raise FileExistsError('KPR runs are immutable, including failures')
    for key in ('weights', 'parent', 'parent_plan', 'parity', 'entry_code', 'reservation'):
        checked(run[key])
    for name, sha in run['port_files'].items():
        checked({'path': name, 'sha256': sha})
    if set(json.loads(checked(run['reservation']).read_text())['clips']) & set(DEV_CLIPS):
        raise ValueError('Reserved clips overlap dev input')
    if psutil.virtual_memory().available < 6 * 1024**3:
        raise RuntimeError('Require >=6 GiB available host RAM')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(run['allocator_bytes'] / torch.cuda.get_device_properties(0).total_memory)
    encoder = KprEncoder(checked(run['weights']), device='cuda')
    start = time.perf_counter()
    result: dict[str, Any] = {'schema': 'kpr_native_dev_run_v1', 'status': 'running', 'plan_sha256': dual_sha256(path), 'records': {}}
    write_json_atomic(args.report / 'features.progress.json', result)
    try:
        for key, record in run['inputs'].items():
            frames, provenance = load_features(checked(record))
            source = provenance['source']
            checked(source)
            n = record['detections']
            embeddings: np.ndarray = np.zeros((n, 6, 512), np.float32)
            visibility: np.ndarray = np.zeros((n, 6), bool)
            row_count = frame_count = 0
            for packet in OpenCVVideoFrameReader(Path(source['path'])):
                frame = frames[packet.index]
                count = len(frame.rows)
                clipped = np.rint(frame.boxes).clip([0, 0, 0, 0], [source['width'], source['height']] * 2)
                size = clipped[:, 2:] - clipped[:, :2]
                if (size <= 0).any():
                    raise ValueError('KPR requires nonempty input crops; no replacement embedding')
                for begin in range(0, count, run['batch_size']):
                    indices = np.arange(begin, min(count, begin + run['batch_size']))
                    pixels = torch.from_numpy(np.stack([crop(packet.frame, frame.boxes[i], encoder.input_size) for i in indices]))
                    positive = frame.poses[indices].astype(np.float64).copy()
                    positive[..., :2] = (positive[..., :2] - clipped[indices, None, :2]) / size[indices, None]
                    negative = np.stack([np.delete(frame.poses, i, axis=0) for i in indices]).astype(np.float64)
                    negative[..., :2] = (negative[..., :2] - clipped[indices, None, None, :2]) / size[indices, None, None]
                    embedded, visible, _ = encoder.extract(pixels, positive, negative)
                    if not torch.allclose(torch.linalg.vector_norm(embedded, dim=-1)[visible], torch.ones(int(visible.sum())), atol=1e-4):
                        raise ValueError('Visible KPR parts have nonunit embeddings')
                    embeddings[frame.rows[indices]] = embedded.numpy()
                    visibility[frame.rows[indices]] = visible.numpy()
                row_count += count
                frame_count += 1
            if (row_count, frame_count) != (n, record['frames']):
                raise ValueError('KPR did not cover every input frame/row')
            arrays = {k: np.concatenate([getattr(f, k) for f in frames]) for k in ('rows', 'boxes', 'scores', 'poses')}
            if not np.array_equal(arrays['rows'], np.arange(n)):
                raise ValueError('KPR parent row axis is not the original complete detection order')
            arrays.update(offsets=np.cumsum([0, *[len(f.rows) for f in frames]], dtype=np.int64),
                part_embeddings=embeddings, part_visibility=visibility,
                metadata=np.asarray(json.dumps({'schema': 'person_kpr_native_features_v1', 'parent': record,
                    'plan_sha256': result['plan_sha256'], 'weight_sha256': WEIGHT_SHA256, 'source': source})))
            destination = args.report / f'{key}.kpr.npz'
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as out:
                np.savez_compressed(out, **arrays)
            with np.load(destination, allow_pickle=False) as saved:
                if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
                    raise ValueError('KPR archive read-back changed a field')
            result['records'][key] = {**record_file(destination), 'frames': frame_count, 'detections': row_count,
                                     'visible_parts': int(visibility.sum()), 'read_back_equal': True}
            result.update(elapsed_seconds=time.perf_counter() - start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if sum(p.stat().st_size for p in args.report.rglob('*') if p.is_file()) > run['disk_limit_bytes']:
                raise RuntimeError('KPR output exceeded 1.5 GB budget')
            write_json_atomic(args.report / 'features.progress.json', result)
            print(f'{key}: {row_count} rows, {result["elapsed_seconds"]:.1f}s', flush=True)
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json_atomic(args.report / 'features.progress.json', result)
        raise
    result['status'] = 'ok'
    write_json_atomic(args.report / 'features.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('plan', 'extract'), required=True)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--parity', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.phase == 'plan' and (args.features is None or args.parity is None):
        parser.error('plan requires --features and --parity')
    {'plan': plan, 'extract': extract}[args.phase](args)
