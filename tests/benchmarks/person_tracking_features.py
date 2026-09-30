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
from hydra import compose, initialize_config_dir

from src.submodules.models import (
    BotSortAssociator,
    PersonDetectionResult,
    ViTPosePose2D,
)
from src.tasks.person_tracking.archive import load_features, save_features
from src.tasks.person_tracking.botsort_pose import BotSortPoseConfig
from src.tasks.person_tracking.contracts import DetectionFeatures, TrackCapacityExceeded
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
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256
from src.utils.video import OpenCVVideoFrameReader


def extract(args: argparse.Namespace) -> None:
    report = args.report.resolve()
    if any(report.glob('*.features.npz')) or (report / 'features.progress.json').exists() or (report / 'features.json').exists():
        raise FileExistsError(f'Existing feature run (including failed output) is immutable: {report}')
    report.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.store / 'scene.json').read_text())['source']
    store = ClipStore(args.store, source)
    code_root = Path(__file__).resolve().parents[2]
    repo = args.repo.resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(code_root / "src/tennis_scene/configs")):
        config = compose(config_name="pipeline", overrides=[
            f"paths.project_root={code_root}", f"paths.data_root={repo / 'data'}",
            f"paths.checkpoint_root={repo / 'ckpt'}", f"paths.external_asset_root={repo / 'third_party'}",
            f"paths.output_root={report}", f"paths.artifact_root={report}",
            f"paths.cache_root={report / 'cache'}", f"device={args.device}"])
    runtime = PipelineRuntimeConfig.from_config(config, bind_inputs=False)
    people = runtime.people
    # Fixed for this first adapter. Other encoders must declare their own dimensions/prompts.
    name = 'clipreid_vitb16_market1501'
    weights = encoder_weights(name, checkpoint_root=args.repo / 'ckpt', external_root=args.repo / 'third_party')
    model_identity = {'appearance': {'encoder': name, 'path': str(weights), 'sha256': dual_sha256(weights)},
                      'pose': {'path': str(people.vitpose_checkpoint), 'sha256': dual_sha256(people.vitpose_checkpoint),
                               'runtime': json_value(people.runtime.vitpose), 'precision': 'float32'}}
    feature_config = FeatureConfig(bbox_enlarge=people.runtime.tracking.bbox_enlarge)
    manifest: dict[str, Any] = {'schema': 'tracking_feature_run_v2', 'source': source, 'cameras': {},
                                'models': model_identity, 'feature_config': asdict(feature_config),
                                'scope': 'prefix_smoke' if args.max_frames is not None else 'full_clip',
                                'pose_score_semantics': 'unbounded_raw_vitpose_heatmap_peak',
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
                manifest['current_input'] = {'camera': camera, 'frame': packet.index,
                    'rows': [int(start), int(end)], 'boxes': detections.boxes_xyxy[start:end].tolist()}
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
                'appearance_valid': sum(int(f.appearance_valid.sum()) for f in frames),
                'pose_score_outside_0_1': sum(int(((f.poses[..., 2] < 0) | (f.poses[..., 2] > 1)).sum()) for f in frames)}
            write_json_atomic(report / 'features.progress.json', manifest)
    except Exception as error:
        manifest.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json_atomic(report / 'features.progress.json', manifest)
        raise
    finally:
        pose.unload()
    manifest['status'] = 'ok'
    write_json_atomic(report / 'features.json', manifest)


def track_baseline(frames: list[DetectionFeatures], provenance: dict[str, Any], max_tracks: int | None) -> dict[str, Any]:
    """Actual old Ultralytics path, including image-based sparse optical flow.

    Replay exactly the same #937 detections as the derivative. This smoke
    records the wrapper's observed boxes/IDs; they are Kalman-updated boxes,
    not falsely claimed as a detection-row mapping or final tracking metrics.
    """
    video = provenance['source']
    if dual_sha256(Path(video['path'])) != video['sha256']:
        raise ValueError('Baseline video content changed')
    tracker = BotSortAssociator()
    observed = []
    seen: set[int] = set()
    for packet in OpenCVVideoFrameReader(Path(video['path']), max_frames=len(frames)):
        feature = frames[packet.index]
        tracks = tracker.update(PersonDetectionResult(feature.boxes, feature.scores), packet.frame)
        seen.update(track['id'] for track in tracks)
        if max_tracks is not None and len(seen) > max_tracks:
            raise TrackCapacityExceeded(f'Ultralytics baseline exceeds cumulative camera cap {max_tracks}')
        observed.append({'frame': packet.index, 'tracks': json_value(tracks)})
    if len(observed) != len(frames):
        raise ValueError('Baseline video timeline differs from shared features')
    return {'frames': len(frames), 'track_ids': sorted(seen),
            'observed_detections': sum(len(item['tracks']) for item in observed),
            'assignment_schema': 'ultralytics_observed_boxes_not_detection_rows', 'observations': observed}


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
    if args.method == 'ultralytics_botsort':
        result['config'] = {'backend': 'src.submodules.models.BotSortAssociator',
                            'with_reid': False, 'gmc_method': 'sparseOptFlow', 'max_tracks': config.max_tracks}
        result['method_label'] = 'Ultralytics BoT-SORT baseline (same #937 detections)'
    else:
        result['method_label'] = 'BoT-SORT-style derivative with appearance and pose'
    for camera, record in source['cameras'].items():
        path = Path(record['path'])
        if dual_sha256(path) != record['sha256']:
            raise ValueError(f'Feature content changed for {camera}')
        frames, provenance = load_features(path)
        if args.method == 'ultralytics_botsort':
            result['cameras'][camera] = track_baseline(frames, provenance, config.max_tracks)
            continue
        method = build_tracker(args.method, fps=float(provenance['source']['fps']), config=config)
        assignments = [method.update(frame) for frame in frames]
        result['cameras'][camera] = {'frames': len(frames),
            'track_ids': sorted({int(i) for a in assignments for i in a.track_ids}),
            'observed_detections': sum(len(a.detection_rows) for a in assignments),
            'assignments': [json_value(a) for a in assignments]}
    write_json_atomic(target, result)
    print(json.dumps({c: {k: v for k, v in r.items() if k not in {'assignments', 'observations'}} for c, r in result['cameras'].items()}))


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
