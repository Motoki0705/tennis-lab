"""Immutable COCO-.30 all-person ViTPose/CLIP/SOLIDER features; no tracking/evaluation.

Plan is CPU-only. Extract is one GPU queue job, bounded externally by 5400 s.
KPR is explicitly absent until its inference port is ready. Never reads GT labels.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
from hydra import compose, initialize_config_dir

from src.submodules.models import ViTPosePose2D
from src.tasks.person_tracking.archive import load_features, save_features
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.features import (
    FeatureConfig,
    FeatureExtractor,
    UnpromptedEncoder,
    encode_appearance,
)
from src.tasks.player_association.appearance.encoders import (
    build_encoder,
    encoder_weights,
)
from src.tasks.player_detection.evaluation.far_archive import DetectionArchive
from src.tasks.player_detection.evaluation.person_sources import DEV_CLIPS
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.utils.checksum import dual_sha256
from src.utils.video import OpenCVVideoFrameReader

CODE_ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'coco_fullframe_0.30'
ENCODERS = {'clipreid_vitb16_market1501': 1280, 'solider_swin_base_msmt17': 1024}


def runtime(repo: Path) -> PipelineRuntimeConfig:
    with initialize_config_dir(version_base='1.3', config_dir=str(CODE_ROOT / 'src/tennis_scene/configs')):
        config = compose(config_name='pipeline', overrides=[f'paths.project_root={CODE_ROOT}',
            f'paths.data_root={repo / "data"}', f'paths.checkpoint_root={repo / "ckpt"}',
            f'paths.external_asset_root={repo / "third_party"}', 'device=cuda', 'people_models.runtime.vitpose.batch_size=4'])
    return PipelineRuntimeConfig.from_config(config, bind_inputs=False)


def checked_file(record: dict[str, Any]) -> Path:
    path = Path(record['path'])
    if dual_sha256(path) != record['sha256']:
        raise ValueError(f'Input content changed: {path}')
    return path


def source_inputs(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = json.loads(path.read_text())
    if source['schema'] != 'fullframe_person_sources_v1':
        raise ValueError('Require run-6 full-frame sources')
    expected = {f'{clip}/{camera}' for clip in DEV_CLIPS for camera in ('cam0', 'cam1', 'cam2')}
    if set(source['archives'][SOURCE]) != expected or len(source['inputs']) != 12 \
            or {f"{r['clip']}/{r['camera']}" for r in source['inputs']} != expected:
        raise ValueError('Require all and only four dev clips x three cameras')
    inference = checked_file({'path': source['coco_inference'], 'sha256': source['coco_inference_sha256']})
    inferred = json.loads(inference.read_text())
    if inferred['status'] != 'ok' or inferred['scope'] != 'full_frame_no_roi' or inferred['baseline_size'] != [800, 1333]:
        raise ValueError('Wrong COCO source scope/resize')
    reservation = checked_file({'path': source['reservation'], 'sha256': source['reservation_sha256']})
    if set(json.loads(reservation.read_text())['clips']) & set(DEV_CLIPS):
        raise ValueError('Development set overlaps reserved unseen clips')
    records = []
    for record in source['inputs']:
        key = f"{record['clip']}/{record['camera']}"
        saved = source['archives'][SOURCE][key]
        archive = DetectionArchive.load(saved)
        video = record['video']
        checked_file(video)
        if len(archive.scores) != saved['detections'] or (archive.scores < np.float32(.3)).any() \
                or len(archive.milliseconds) != video['num_frames'] or video['camera_id'] != record['camera']:
            raise ValueError(f'COCO .30 source rows/timeline mismatch: {key}')
        records.append({'key': key, 'video': video, 'detection': saved})
    return source, records


def plan(args: argparse.Namespace) -> None:
    target = args.report / 'plan.json'
    if target.exists():
        raise FileExistsError(target)
    source, records = source_inputs(args.sources)
    cfg = runtime(args.repo)
    detector = source['weights']['coco']['coco']
    if checked_file(detector).resolve() != cfg.people.detector_checkpoint.resolve() \
            or cfg.people.runtime.dino_detector.confidence != .3:
        raise ValueError('Source differs from default COCO .30 checkpoint')
    models: dict[str, dict[str, Any]] = {name: {'path': str(encoder_weights(name, checkpoint_root=args.repo / 'ckpt', external_root=args.repo / 'third_party')),
                     'dimension': dimension} for name, dimension in ENCODERS.items()}
    models['pose'] = {'path': str(cfg.people.vitpose_checkpoint), 'runtime': json_value(cfg.people.runtime.vitpose), 'precision': 'float32'}
    for model in models.values():
        model['sha256'] = dual_sha256(Path(model['path']))
    manifest = {'schema': 'all_person_dev_features_plan_v1', 'source_name': SOURCE,
        'sources': {'path': str(args.sources), 'sha256': dual_sha256(args.sources)}, 'inputs': records,
        'reservation': {'path': source['reservation'], 'sha256': source['reservation_sha256']},
        'detector': detector, 'models': models, 'repo': str(args.repo),
        'feature_config': asdict(FeatureConfig(bbox_enlarge=cfg.people.runtime.tracking.bbox_enlarge,
                                             min_appearance_height_px=1., appearance_batch_size=8)),
        'frames': sum(r['video']['num_frames'] for r in records),
        'detections': sum(r['detection']['detections'] for r in records),
        'scope': 'all COCO .30 persons, full frame, no ROI/player selection, four dev clips only',
        'excluded_encoders': {'kpr': 'inference port not ready; no replacement'},
        'budget': {'timeout_seconds': 5400, 'allocator_bytes': 7 * 1024**3, 'peak_vram_gb_estimate': 9., 'disk_bytes_limit': 5_000_000_000}}
    write_json_atomic(target, manifest)
    print(json.dumps({'plan': str(target), 'frames': manifest['frames'], 'detections': manifest['detections'], 'models': models}), flush=True)


def camera_features(video: dict[str, Any], detections: DetectionArchive, extractor: FeatureExtractor | UnpromptedEncoder,
                    config: FeatureConfig, saved: list[DetectionFeatures] | None) -> list[DetectionFeatures]:
    checked_file(video)
    frames = []
    for packet in OpenCVVideoFrameReader(Path(video['path']), max_frames=video['num_frames']):
        start, end = detections.offsets[packet.index:packet.index + 2]
        rows: np.ndarray = np.arange(start, end, dtype=np.int64)
        boxes, scores = detections.boxes[start:end], detections.scores[start:end]
        if isinstance(extractor, FeatureExtractor):
            if saved is not None:
                raise ValueError('Pose extraction cannot receive saved features')
            frame = extractor.extract(packet.index, packet.frame, rows, boxes, scores)
        else:
            if saved is None or len(saved) != video['num_frames']:
                raise ValueError('Additional encoder requires the full saved pose timeline')
            old = saved[packet.index]
            if not np.array_equal(old.rows, rows) or not np.array_equal(old.boxes, boxes) or not np.array_equal(old.scores, scores):
                raise ValueError('Saved pose changed the input detection rows')
            frame = encode_appearance(packet.index, packet.frame, rows, boxes, scores, old.poses, extractor, config)
        frames.append(frame)
    if len(frames) != video['num_frames'] or sum(len(f.rows) for f in frames) != len(detections.scores):
        raise ValueError('Feature extraction did not preserve every input frame/person')
    return frames


def extract(args: argparse.Namespace) -> None:
    manifest = json.loads((args.report / 'plan.json').read_text())
    if manifest['schema'] != 'all_person_dev_features_plan_v1' or manifest['source_name'] != SOURCE \
            or set(manifest['models']) != {*ENCODERS, 'pose'}:
        raise ValueError('Unsupported or incomplete feature plan')
    if (args.report / 'features.progress.json').exists() or (args.report / 'features.json').exists():
        raise FileExistsError('Feature runs, including failures, are immutable')
    if args.repo.resolve() != Path(manifest['repo']).resolve():
        raise ValueError('Feature asset root changed')
    checked_file(manifest['sources'])
    _, records = source_inputs(Path(manifest['sources']['path']))
    if records != manifest['inputs']:
        raise ValueError('Feature inputs changed after planning')
    checked_file(manifest['reservation'])
    for record in manifest['models'].values():
        checked_file(record)
    if psutil.virtual_memory().available < 6 * 1024**3:
        raise RuntimeError('Feature job requires at least 6 GiB available host RAM')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(manifest['budget']['allocator_bytes'] / total, 0)
    cfg = runtime(args.repo)
    pose_runtime = manifest['models']['pose']['runtime']
    if pose_runtime != json_value(cfg.people.runtime.vitpose):
        raise ValueError('Pose runtime changed after planning')
    config = FeatureConfig(**manifest['feature_config'])
    progress: dict[str, Any] = {'schema': 'all_person_dev_features_run_v1', 'plan_sha256': dual_sha256(args.report / 'plan.json'),
        'status': 'running', 'records': {}, 'models': manifest['models'], 'scope': manifest['scope']}
    start_time = time.perf_counter()
    write_json_atomic(args.report / 'features.progress.json', progress)
    try:
        for name, dimension in ENCODERS.items():
            encoder = UnpromptedEncoder(build_encoder(name, checkpoint_root=args.repo / 'ckpt',
                external_root=args.repo / 'third_party', device='cuda'), dimension)
            pose = None
            extractor: FeatureExtractor | UnpromptedEncoder = encoder
            if name == 'clipreid_vitb16_market1501':
                pose = ViTPosePose2D(cfg.people.vitpose_checkpoint, device='cuda', flip_test=cfg.people.runtime.vitpose.flip_test,
                    batch_size=4, head_config=cfg.people.runtime.vitpose.head, precision='float32')
                extractor = FeatureExtractor(pose, encoder, config)
            try:
                for record in records:
                    key = record['key']
                    progress['current_input'] = {'encoder': name, 'key': key}
                    write_json_atomic(args.report / 'features.progress.json', progress)
                    detections = DetectionArchive.load(record['detection'])
                    saved = None
                    if pose is None:
                        prior = progress['records']['clipreid_vitb16_market1501'][key]
                        saved, _ = load_features(checked_file(prior))
                    frames = camera_features(record['video'], detections, extractor, config, saved)
                    path = args.report / name / f'{key}.features.npz'
                    provenance = {'source': record['video'], 'detection': record['detection'],
                        'models': {'pose': manifest['models']['pose'], 'appearance': manifest['models'][name]},
                        'encoder': name, 'config': asdict(config), 'plan_sha256': progress['plan_sha256'], 'scope': manifest['scope']}
                    save_features(path, frames, provenance)
                    restored, restored_provenance = load_features(path)
                    if restored_provenance != provenance or any(not np.array_equal(getattr(a, field), getattr(b, field))
                            for a, b in zip(frames, restored, strict=True)
                            for field in ('rows', 'boxes', 'scores', 'poses', 'embeddings', 'appearance_valid')):
                        raise ValueError('Feature archive roundtrip changed values/provenance')
                    progress['records'].setdefault(name, {})[key] = {'path': str(path), 'sha256': dual_sha256(path),
                        'frames': len(frames), 'detections': len(detections.scores), 'bytes': path.stat().st_size,
                        'appearance_valid': sum(int(f.appearance_valid.sum()) for f in frames)}
                    del frames, restored, saved
                    progress['elapsed_seconds'] = time.perf_counter() - start_time
                    progress['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                    progress['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                    if sum(p.stat().st_size for p in args.report.rglob('*') if p.is_file()) > manifest['budget']['disk_bytes_limit']:
                        raise RuntimeError('Feature output exceeded its disk budget')
                    write_json_atomic(args.report / 'features.progress.json', progress)
                    print(f'{name}: {key} complete ({progress["elapsed_seconds"]:.1f}s)', flush=True)
            finally:
                if pose is not None:
                    pose.unload()
                del extractor, encoder, pose
                gc.collect()
                torch.cuda.empty_cache()
    except Exception as error:
        progress.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json_atomic(args.report / 'features.progress.json', progress)
        raise
    progress['status'] = 'ok'
    write_json_atomic(args.report / 'features.json', progress)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('plan', 'extract'), required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    args.repo, args.report = args.repo.resolve(), args.report.resolve()
    if args.phase == 'plan':
        if args.sources is None:
            parser.error('plan requires --sources')
        args.sources = args.sources.resolve()
        plan(args)
    else:
        extract(args)


if __name__ == '__main__':
    main()
