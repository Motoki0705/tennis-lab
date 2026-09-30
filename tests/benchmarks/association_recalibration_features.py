"""Plan and extract the precommitted unlabeled #964 calibration inputs.

No label reader, fitting or association scoring is imported by this entrypoint.
Only metadata selects clips. All GPU work belongs to a single bounded queue job.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
from pipeline_preflight import compose_config  # type: ignore[import-not-found]

from src.submodules.models import ViTPosePose2D
from src.tasks.person_tracking.archive import load_features, save_features
from src.tasks.person_tracking.contracts import DetectionFeatures
from src.tasks.person_tracking.features import (
    FeatureConfig,
    FeatureExtractor,
    UnpromptedEncoder,
)
from src.tasks.player_association.appearance.encoders import build_encoder
from src.tennis_scene.configuration import PipelineRuntimeConfig
from src.tennis_scene.generate_dataset.manifest import ClipManifest
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.person_detection import (
    PersonDetectionInput,
    PersonDetectionModule,
)
from src.tennis_scene.pipeline.contracts import SourceVideo
from src.tennis_scene.pipeline.definition import file_identity
from src.utils.checksum import dual_sha256
from src.utils.video import OpenCVVideoFrameReader

CODE = Path(__file__).resolve().parents[2]
PROTOCOL = CODE / 'knowledge/runs/run-i964-recalibration-r12-20260930/protocol.md'
DEV = frozenset(('video_000/clip_000', 'video_000/clip_007', 'video_001/clip_001', 'video_002/clip_013'))
CAMERAS = ('cam0', 'cam1', 'cam2')


def clip_path(dataset: Path, clip_id: str) -> Path:
    video, clip = clip_id.split('/')
    return dataset / 'videos' / video / 'clips' / clip


def checked(record: dict[str, Any]) -> Path:
    path = Path(record['path'])
    if dual_sha256(path) != record['sha256']:
        raise ValueError(f'Input content changed: {path}')
    return path


def select_clips(metadata: dict[str, dict[str, Any]], reserved: set[str], labelled: set[str]) -> list[str]:
    """Metadata-only rank; no successful/failed result influences selection."""
    result = []
    videos = sorted({clip.split('/')[0] for clip in metadata})
    if videos != ['video_000', 'video_001', 'video_002']:
        raise ValueError('Calibration requires all three source recordings')
    for video in videos:
        eligible = [c for c in metadata if c.split('/')[0] == video and c not in DEV | reserved | labelled]
        ordered = sorted(eligible, key=lambda c: (metadata[c]['num_frames'] / metadata[c]['fps'], c))
        if len(ordered) < 2:
            raise ValueError(f'{video}: fewer than two unlabeled calibration clips')
        result.extend(ordered[:2])
    return result


def assert_disjoint(selected: dict[str, dict[str, Any]], excluded: dict[str, dict[str, Any]]) -> None:
    """Check source-frame intervals without opening excluded media or labels."""
    for a_id, a in selected.items():
        for b_id, b in excluded.items():
            for ca in a['cameras']:
                for cb in b['cameras']:
                    if ca['source_path'] == cb['source_path'] and max(ca['source_frame_start'], cb['source_frame_start']) \
                            <= min(ca['source_frame_end'], cb['source_frame_end']):
                        raise ValueError(f'Overlapping source intervals: {a_id} / {b_id}')


def restrict_appearance(frames: list[DetectionFeatures], before: FeatureConfig, after: FeatureConfig,
                        size: tuple[int, int]) -> tuple[list[DetectionFeatures], int]:
    """Explicitly project a denser cache to the production eligibility mask.

    Crops/weights must already match. Batching is an execution setting, not a new
    crop. A less restrictive request cannot be satisfied by a restricted cache.
    """
    if before.bbox_enlarge != after.bbox_enlarge or before.min_appearance_height_px > after.min_appearance_height_px:
        raise ValueError('Cannot project incompatible or less complete cached features')
    width, height = size
    scale = np.hypot(width, height) / np.hypot(1920, 1080)
    changed, result = 0, []
    for frame in frames:
        clipped = np.rint(frame.boxes).astype(np.float32)
        clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
        clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
        eligible = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] - clipped[:, 1] >= after.min_appearance_height_px * scale)
        valid = frame.appearance_valid & eligible
        changed += int(np.count_nonzero(valid != frame.appearance_valid))
        embeddings = frame.embeddings.copy()
        embeddings[~valid] = 0
        result.append(replace(frame, embeddings=embeddings, appearance_valid=valid))
    return result, changed


def runtime(repo: Path, report: Path, first_clip: Path, device: str) -> PipelineRuntimeConfig:
    return PipelineRuntimeConfig.from_config(compose_config(repo, report, first_clip, device), bind_inputs=False)


def plan(repo: Path, report: Path) -> None:
    target = report / 'plan.json'
    if target.exists():
        raise FileExistsError(target)
    dataset = repo / 'data/tennis_multivew/processed/meiji_3cam/dataset'
    previous = repo / 'outputs/player_association/evaluate/meiji_clips/i933-observe-v1-20260927'
    observed = previous / 'observe.json'
    reservation = dataset / 'annotations/player_association/unseen_protocol.json'
    reserved = set(json.loads(reservation.read_text())['clips'])
    # Only the keys of the historical report enter selection.
    metadata = {c: json.loads((clip_path(dataset, c) / 'clip.json').read_text())
                for c in json.loads(observed.read_text())['clips']}
    labelled = {c for c in metadata if any((clip_path(dataset, c) / 'annotations/player_association' / name).exists()
                                          for name in ('labels.json', 'review.yaml'))}
    chosen = select_clips(metadata, reserved, labelled)
    excluded = {c: json.loads((clip_path(dataset, c) / 'clip.json').read_text()) for c in DEV | reserved}
    assert_disjoint({c: metadata[c] for c in chosen}, excluded)
    cfg = runtime(repo, report, clip_path(dataset, chosen[0]), 'cpu')
    feature_config = replace(cfg.tracking.features, appearance_batch_size=8)
    audit = [{'clip': c, 'duration_s': m['num_frames'] / m['fps'],
              'manifest': file_identity(clip_path(dataset, c) / 'clip.json'),
              'selected': c in chosen,
              'reason': 'dev' if c in DEV else 'reserved' if c in reserved else 'labelled' if c in labelled
                        else 'two_shortest_in_recording' if c in chosen else 'longer_in_recording'}
             for c, m in sorted(metadata.items())]
    inputs: list[dict[str, Any]] = []
    for c in chosen:
        manifest = ClipManifest.load(clip_path(dataset, c))
        if tuple(manifest.camera_ids) != CAMERAS:
            raise ValueError(f'{c}: incomplete camera set')
        m = metadata[c]
        for camera in CAMERAS:
            inputs.append({'key': f'{c}/{camera}', 'clip': c, 'camera': camera,
                'video': {**file_identity(manifest.media_path(camera)), 'camera_id': camera,
                          'num_frames': m['num_frames'], 'fps': m['fps'], 'width': m['width'], 'height': m['height']},
                'action': 'extract_missing'})
    # Completed feature manifests owned by #964 only; do not inspect other lanes.
    inventory = []
    for path in sorted((repo / 'outputs/person_tracking').glob('**/features.json')):
        data = json.loads(path.read_text())
        records = data.get('records', {}).get(cfg.tracking.encoder, {})
        inventory.append({**file_identity(path), 'matching_unlabeled_keys': sorted(set(records) & {r['key'] for r in inputs}),
                          'status': data['status']})
        if set(records) & {r['key'] for r in inputs}:
            raise ValueError('Matching calibration cache discovered: audit identity before scheduling extraction')
    model_paths = {'detector': cfg.people.detector_checkpoint, 'pose': cfg.people.vitpose_checkpoint,
                   'clip': cfg.tracking_encoder_weights, 'aflink': cfg.aflink_checkpoint}
    manifest = {
        'schema': 'i964_recalibration_features_plan_v1', 'repo': str(repo), 'dataset': str(dataset),
        'protocol': file_identity(PROTOCOL), 'historical_observe': file_identity(observed),
        'reservation': file_identity(reservation), 'selected_clips': chosen, 'candidate_audit': audit,
        'excluded_manifest_hashes': {c: file_identity(clip_path(dataset, c) / 'clip.json') for c in sorted(excluded)},
        'source_intervals_disjoint': True, 'inputs': inputs, 'inventory': inventory,
        'models': {k: file_identity(p) for k, p in model_paths.items()},
        'detector_runtime': json_value(cfg.people.runtime.dino_detector),
        'scope': 'full_frame', 'merge_duplicates': cfg.merge_duplicate_person_boxes,
        'pose_runtime': json_value(cfg.people.runtime.vitpose), 'pose_precision': 'float32',
        'feature_config': asdict(feature_config), 'production_tracking': cfg.tracking.identity(),
        'frames': sum(r['video']['num_frames'] for r in inputs),
        'code': {str(p.relative_to(CODE)): dual_sha256(p) for folder in
                 ('src/tasks/person_tracking', 'src/tasks/player_association', 'src/submodules', 'src/tennis_scene')
                 for p in sorted((CODE / folder).rglob('*.py'))},
        'benchmark': file_identity(Path(__file__)),
        'entrypoints': [file_identity(CODE / 'tests/benchmarks' / name) for name in
                        ('pipeline_preflight.py', 'association_feature_guard.py', 'association_recalibration_features.sh',
                         'build_dino_extension.sh')],
        'budget': {'wall_seconds': 7200, 'allocator_bytes': 7 * 1024**3, 'vram_stop_bytes': 9_500_000_000,
                   'disk_limit_bytes': 5_000_000_000, 'expected_seconds': [3600, 6900], 'expected_disk_bytes': 2_000_000_000},
    }
    write_json_atomic(target, manifest)
    print(json.dumps({'selected': chosen, 'camera_clips': len(inputs), 'frames': manifest['frames']}), flush=True)


def validate_plan(manifest: dict[str, Any], repo: Path) -> None:
    if manifest['schema'] != 'i964_recalibration_features_plan_v1' or manifest['repo'] != str(repo):
        raise ValueError('Unexpected plan or asset root')
    for key in ('protocol', 'historical_observe', 'reservation', 'benchmark'):
        checked(manifest[key])
    for record in (*manifest['models'].values(), *manifest['excluded_manifest_hashes'].values(), *manifest['entrypoints']):
        checked(record)
    for record in manifest['candidate_audit']:
        checked(record['manifest'])
    for path, sha in manifest['code'].items():
        checked({'path': str(CODE / path), 'sha256': sha})
    reserved = set(json.loads(checked(manifest['reservation']).read_text())['clips'])
    selected = set(manifest['selected_clips'])
    if len(selected) != 6 or selected & (DEV | reserved):
        raise ValueError('Calibration split changed or overlaps dev/reserved')
    expected = {f'{c}/{camera}' for c in selected for camera in CAMERAS}
    if len(manifest['inputs']) != len(expected) or {r['key'] for r in manifest['inputs']} != expected:
        raise ValueError('Calibration camera set is incomplete')
    for c in selected:
        annotation = clip_path(Path(manifest['dataset']), c) / 'annotations/player_association'
        if any((annotation / name).exists() for name in ('labels.json', 'review.yaml')):
            raise ValueError('A selected calibration clip became labelled')
    for record in manifest['inputs']:
        checked(record['video'])


def extract(repo: Path, report: Path) -> None:
    target = report / 'features.progress.json'
    if target.exists() or (report / 'features.json').exists():
        raise FileExistsError('Feature runs, including failures, are immutable')
    manifest = json.loads((report / 'plan.json').read_text())
    validate_plan(manifest, repo)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(manifest['budget']['allocator_bytes'] / total, 0)
    cfg = runtime(repo, report, clip_path(Path(manifest['dataset']), manifest['selected_clips'][0]), 'cuda')
    actual_models = {'detector': cfg.people.detector_checkpoint, 'pose': cfg.people.vitpose_checkpoint,
                     'clip': cfg.tracking_encoder_weights, 'aflink': cfg.aflink_checkpoint}
    if {k: file_identity(p) for k, p in actual_models.items()} != manifest['models']:
        raise ValueError('Runtime asset paths/content changed after planning')
    if cfg.tracking.identity() != manifest['production_tracking'] \
            or json_value(cfg.people.runtime.vitpose) != manifest['pose_runtime'] \
            or json_value(cfg.people.runtime.dino_detector) != manifest['detector_runtime'] \
            or cfg.merge_duplicate_person_boxes != manifest['merge_duplicates']:
        raise ValueError('Runtime changed after planning')
    config = FeatureConfig(**manifest['feature_config'])
    progress: dict[str, Any] = {'schema': 'i964_recalibration_features_v1', 'status': 'running',
        'plan_sha256': dual_sha256(report / 'plan.json'), 'records': {}, 'detections': {}}
    start_time = time.monotonic()
    write_json_atomic(target, progress)
    try:
        # Separate model stages keep DINO out of memory while ViTPose/CLIP run.
        for record in manifest['inputs']:
            if psutil.virtual_memory().available < 6 * 1024**3:
                raise RuntimeError('Require at least 6 GiB available host RAM')
            key, meta = record['key'], record['video']
            progress['current'] = {'key': key, 'stage': 'detector'}
            write_json_atomic(target, progress)
            video = SourceVideo(**{k: Path(v) if k == 'path' else v for k, v in meta.items()})
            detections = PersonDetectionModule(cfg.people, merge_duplicates=False).process(PersonDetectionInput(video))
            path = report / 'detections' / f'{key}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                np.savez_compressed(handle, offsets=detections.frame_offsets, boxes=detections.boxes_xyxy,
                                    scores=detections.confidence, source_rows=detections.source_rows)
            progress['detections'][key] = file_identity(path)
            progress['current']['stage'] = 'pose_clip'
            write_json_atomic(target, progress)
            pose = ViTPosePose2D(cfg.people.vitpose_checkpoint, device='cuda',
                flip_test=cfg.people.runtime.vitpose.flip_test, batch_size=4,
                head_config=cfg.people.runtime.vitpose.head, precision='float32')
            encoder = None
            extractor = None
            try:
                encoder = build_encoder(cfg.tracking.encoder, checkpoint_root=repo / 'ckpt',
                                        external_root=repo / 'third_party', device='cuda')
                extractor = FeatureExtractor(pose, UnpromptedEncoder(encoder, 1280), config)
                frames = []
                assert detections.source_rows is not None
                for packet in OpenCVVideoFrameReader(video.path, max_frames=video.num_frames):
                    a, b = detections.frame_offsets[packet.index:packet.index + 2]
                    frames.append(extractor.extract(packet.index, packet.frame, detections.source_rows[a:b],
                                                    detections.boxes_xyxy[a:b], detections.confidence[a:b]))
                if len(frames) != video.num_frames:
                    raise ValueError('Incomplete feature timeline')
                path = report / cfg.tracking.encoder / f'{key}.features.npz'
                provenance = {'source': meta, 'detection': progress['detections'][key], 'models': manifest['models'],
                              'config': asdict(config), 'encoder': cfg.tracking.encoder,
                              'plan_sha256': progress['plan_sha256']}
                save_features(path, frames, provenance)
                restored, restored_provenance = load_features(path)
                if restored_provenance != provenance or any(not np.array_equal(getattr(a, field), getattr(b, field))
                        for a, b in zip(frames, restored, strict=True)
                        for field in ('rows', 'boxes', 'scores', 'poses', 'embeddings', 'appearance_valid')):
                    raise ValueError('Feature roundtrip changed values or provenance')
                progress['records'][key] = {**file_identity(path), 'frames': len(frames),
                    'detections': len(detections.confidence), 'bytes': path.stat().st_size}
                del frames, restored
            finally:
                pose.unload()
                del pose, encoder, extractor
                gc.collect()
                torch.cuda.empty_cache()
            progress.update(elapsed_seconds=time.monotonic() - start_time,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if sum(p.stat().st_size for p in report.rglob('*') if p.is_file()) > manifest['budget']['disk_limit_bytes']:
                raise RuntimeError('Feature output exceeded its disk budget')
            write_json_atomic(target, progress)
            print(f'{key}: complete ({progress["elapsed_seconds"]:.1f}s)', flush=True)
    except Exception as error:
        progress.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json_atomic(target, progress)
        raise
    progress['status'] = 'ok'
    write_json_atomic(report / 'features.json', progress)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('plan', 'extract'), required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    (plan if args.phase == 'plan' else extract)(args.repo.resolve(), args.report.resolve())


if __name__ == '__main__':
    main()
