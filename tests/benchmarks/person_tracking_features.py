"""Shared per-detection features and CPU tracking on a saved #937 detection store.

features: GPU, training queue only. track: CPU, same immutable features for all
methods. A --max-frames prefix is a smoke test, not a full-clip evaluation.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.submodules.models import ViTPosePose2D
from src.tasks.person_tracking.archive import load_features, save_features
from src.tasks.person_tracking.botsort_pose import BotSortPoseConfig
from src.tasks.person_tracking.features import (
    FeatureConfig,
    FeatureExtractor,
    UnpromptedEncoder,
)
from src.tasks.person_tracking.methods import build_tracker
from src.tasks.player_association.appearance.encoders import (
    build_encoder,
    encoder_weights,
)
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256
from src.utils.video import OpenCVVideoFrameReader
from tests.benchmarks.player_detection_clips import compose_detector_runtime


def extract(args: argparse.Namespace) -> None:
    report = args.report.resolve()
    if (report / 'features.json').exists():
        raise FileExistsError(report / 'features.json')
    report.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.store / 'scene.json').read_text())['source']
    store = ClipStore(args.store, source)
    runtime, _ = compose_detector_runtime(args.repo.resolve(), report, args.device, [], 'features')
    people = runtime.people
    # Fixed for this first adapter. Other encoders must declare their own dimensions/prompts.
    name = 'clipreid_vitb16_market1501'
    weights = encoder_weights(name, checkpoint_root=args.repo / 'ckpt', external_root=args.repo / 'third_party')
    model_identity = {'appearance': {'encoder': name, 'path': str(weights), 'sha256': dual_sha256(weights)},
                      'pose': {'path': str(people.vitpose_checkpoint), 'sha256': dual_sha256(people.vitpose_checkpoint),
                               'runtime': json_value(people.runtime.vitpose), 'precision': 'float32'}}
    feature_config = FeatureConfig(bbox_enlarge=people.runtime.tracking.bbox_enlarge)
    manifest: dict[str, Any] = {'schema': 'tracking_feature_run_v1', 'source': source, 'cameras': {},
                                'models': model_identity, 'feature_config': asdict(feature_config),
                                'scope': 'prefix_smoke' if args.max_frames is not None else 'full_clip',
                                'detector_scope': 'saved_pipeline_court_roi', 'status': 'running'}
    write_json_atomic(report / 'features.progress.json', manifest)
    pose = ViTPosePose2D(people.vitpose_checkpoint, device=args.device, flip_test=people.runtime.vitpose.flip_test,
                        batch_size=people.runtime.vitpose.batch_size, head_config=people.runtime.vitpose.head, precision='float32')
    encoder = UnpromptedEncoder(build_encoder(name, checkpoint_root=args.repo / 'ckpt',
                                              external_root=args.repo / 'third_party', device=args.device), dimension=1280)
    extractor = FeatureExtractor(pose, encoder, feature_config)
    try:
        for video in source['videos']:
            if dual_sha256(Path(video['path'])) != video['sha256']:
                raise ValueError('Source video content changed since detector run')
            camera = video['camera_id']
            reference = store.active(f'person_detection/{camera}')
            if reference is None:
                raise ValueError(f'Missing detector artifact for {camera}')
            detections = store.load(reference, ArtifactCodec(PersonDetectionOutput))
            if detections.camera_id != camera or len(detections.frame_offsets) != video['num_frames'] + 1:
                raise ValueError('Detector camera/timeline mismatch')
            count = video['num_frames'] if args.max_frames is None else min(args.max_frames, video['num_frames'])
            frames = []
            for packet in OpenCVVideoFrameReader(Path(video['path']), max_frames=count):
                start, end = detections.frame_offsets[packet.index:packet.index + 2]
                frames.append(extractor.extract(packet.index, packet.frame, np.arange(start, end, dtype=np.int64),
                                                detections.boxes_xyxy[start:end], detections.confidence[start:end]))
            if len(frames) != count:
                raise ValueError('Incomplete source decoding')
            path = report / f'{camera}.features.npz'
            save_features(path, frames, {'source': video, 'detection': json_value(reference), 'models': model_identity,
                                        'config': asdict(feature_config), 'scope': manifest['scope']})
            restored, provenance = load_features(path)
            if len(restored) != count or provenance['detection'] != json_value(reference):
                raise ValueError('Feature roundtrip lost its provenance')
            manifest['cameras'][camera] = {'path': str(path), 'sha256': dual_sha256(path), 'frames': count,
                'detections': sum(len(f.rows) for f in frames),
                'appearance_valid': sum(int(f.appearance_valid.sum()) for f in frames)}
            write_json_atomic(report / 'features.progress.json', manifest)
    finally:
        pose.unload()
    manifest['status'] = 'ok'
    write_json_atomic(report / 'features.json', manifest)


def track(args: argparse.Namespace) -> None:
    report = args.report.resolve()
    target = report / f'tracking.{args.method}.json'
    if target.exists():
        raise FileExistsError(target)
    source = json.loads((report / 'features.json').read_text())
    if source['status'] != 'ok':
        raise ValueError('Incomplete features')
    config = BotSortPoseConfig()
    result: dict[str, Any] = {'method': args.method, 'config': asdict(config), 'cameras': {},
                               'features_sha256': dual_sha256(report / 'features.json'), 'scope': source['scope']}
    for camera, record in source['cameras'].items():
        path = Path(record['path'])
        if dual_sha256(path) != record['sha256']:
            raise ValueError(f'Feature content changed for {camera}')
        frames, provenance = load_features(path)
        method = build_tracker(args.method, fps=float(provenance['source']['fps']), config=config)
        assignments = [method.update(frame) for frame in frames]
        result['cameras'][camera] = {'frames': len(frames),
            'track_ids': sorted({int(i) for a in assignments for i in a.track_ids}),
            'observed_detections': sum(len(a.detection_rows) for a in assignments),
            'assignments': [json_value(a) for a in assignments]}
    write_json_atomic(target, result)
    print(json.dumps({c: {k: v for k, v in r.items() if k != 'assignments'} for c, r in result['cameras'].items()}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('features', 'track'), required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--store', type=Path, help='features: saved detector ClipStore')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--method', default='botsort_pose')
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    if args.phase == 'features':
        if args.store is None:
            parser.error('features requires --store')
        extract(args)
    else:
        track(args)


if __name__ == '__main__':
    main()
