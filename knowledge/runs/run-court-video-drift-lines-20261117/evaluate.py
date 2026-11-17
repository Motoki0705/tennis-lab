"""Frozen real-video diagnostic evaluation; no common fit or reference input."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.tasks.court_detection.geometry.video_drift import DriftConfig
from src.tennis_scene.pipeline.clip_inputs import prepare_clip_source, resolve_dataset_clip, select_dataset_clips
from src.tennis_scene.pipeline.video_court.store import VideoCourtStore
from src.tennis_scene.pipeline.video_court.drift_build import build_video_drift
from evidence import assert_sources


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    args.dataset = args.dataset.resolve()
    started = time.monotonic()
    pinned = json.loads((args.output / 'immutable-inputs.json').read_text())
    frozen = json.loads((args.output / 'frozen-profile.json').read_text())
    assert frozen['settings'] == asdict(DriftConfig())
    assert_sources(frozen)
    assert all(sha(Path(p)) == digest for p, digest in pinned['files'].items())
    assert sha(args.output / 'frame-protocol.json') == frozen['frame_protocol_sha256']
    selection = json.loads((args.output / 'frame-protocol.json').read_text())
    base = VideoCourtStore(Path(pinned['base_store'])).load(pinned['base_bundle_digest'])
    base.verify()
    selected = {(r['camera_id'], r['clip_id'], r['frame_index']): r for r in selection['frames']}
    plan = {}
    sources = []
    snapshots = []
    for record in select_dataset_clips(args.dataset):
        if record.video_id != base.manifest['video_id']:
            continue
        bound = resolve_dataset_clip(args.dataset, record)
        prepared = prepare_clip_source(bound.video_paths, bound.camera_ids, clip_id=record.clip_id)
        sources.append(prepared.source)
        snapshots.extend((bound.snapshot, prepared.snapshot))
        metadata = json.loads((bound.clip_dir / 'clip.json').read_text())
        for camera in metadata['cameras']:
            indices = [r['frame_index'] for r in selection['frames'] if r['camera_id'] == camera['camera_id'] and r['clip_id'] == record.clip_id]
            for index in indices:
                assert selected[camera['camera_id'],record.clip_id,index]['source_frame'] == camera['source_frame_start'] + index
            plan.setdefault(camera['camera_id'], {})[record.clip_id] = indices
    provenance = {'frame_protocol_sha256': sha(args.output / 'frame-protocol.json'),
                  'frozen_profile_sha256': sha(args.output / 'frozen-profile.json'), 'numeric_sources': frozen['numeric_sources'],
                  'driver_sha256': sha(Path(__file__)), 'gpu_used': False, 'reference_h_used': False,
                  'reference_frames_used': False, 'real_evaluation_started_at': time.time()}
    save(args.output / 'evaluation-protocol.json', provenance | {'frame_count': len(selected), 'base_bundle_digest': base.digest})
    artifact = build_video_drift(base, tuple(sources), frame_plan=plan, directory=args.output / 'drift-store',
                                 provenance=provenance, config=DriftConfig(), workers=args.workers)
    for snapshot in snapshots:
        snapshot.assert_unchanged()
    assert_sources(frozen)
    assert all(sha(Path(p)) == digest for p, digest in pinned['files'].items())
    summary = {'base_bundle_digest': base.digest, 'drift_bundle_digest': artifact.drift.digest,
               'base_camera_digests': base.manifest['cameras'], 'drift_camera_digests': artifact.drift_manifest['cameras'],
               'base_artifact_bytes_unchanged': True, 'numeric_sources_unchanged': True,
               'seconds': time.monotonic() - started, 'cameras': {}}
    for camera, doc in artifact.drift_cameras.items():
        clips = list(doc['clips'].values())
        frames = [f for c in clips for f in c['frames']]
        summary['cameras'][camera] = {'clips': len(clips), 'frames': len(frames),
            'clip_status_counts': dict(Counter(c['status'] for c in clips)),
            'clip_reason_counts': dict(Counter(str(c['reason']) for c in clips)),
            'frame_status_counts': dict(Counter(f['status'] for f in frames)),
            'frame_reason_counts': dict(Counter(str(f['reason']) for f in frames)),
            'geometry_reason_counts': dict(Counter(g for f in frames for g in f['geometry_reasons'])),
            'usable_clips': sum(c['status'] != 'failed' for c in clips)}
    summary['complete_four_camera_clips'] = sum(all(artifact.drift_cameras[c]['clips'][clip]['status'] != 'failed' for c in artifact.drift_cameras)
        for clip in next(iter(artifact.drift_cameras.values()))['clips'])
    save(args.output / 'evaluation-summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=Path('data/tennis_multiview/processed/meiji_4cam/dataset'))
    parser.add_argument('--workers', type=int, default=4)
    main(parser.parse_args())
