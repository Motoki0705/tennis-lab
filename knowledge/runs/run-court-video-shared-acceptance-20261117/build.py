"""Frozen CPU production build; the reference is never an estimator input."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time

import cv2
from hydra import compose, initialize_config_dir

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from src.tennis_scene.pipeline.orchestrator import TennisSceneOrchestrator


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def freeze():
    return {str(p): sha(p) for p in sorted(Path('src').rglob('*.py'))} | {
        'src/tennis_scene/configs/pipeline.yaml': sha(Path('src/tennis_scene/configs/pipeline.yaml'))}


def prepare(args):
    split = json.loads(args.split.read_text())
    fit = json.loads((args.stage1 / 'all-fit/protocol.json').read_text())
    raw = json.loads(args.raw_manifest.read_text())
    hashes = {(r['clip_id'], r['camera']): r['rgb_sha256'] for r in raw['records'] if r['dataset'] == 'meiji_4cam'}
    media = {r['path']: r['sha256'] for r in fit['media']}
    clips = sorted({r['clip_id'] for r in split['seed']})
    refs = {}
    for camera in ('cam0', 'cam1', 'cam2', 'cam3'):
        candidates = json.loads((args.stage1 / f'candidates/{camera}.json').read_text())
        assert candidates['provenance']['seed_clip_ids'] == clips
        candidates['provenance']['source_frames'] = {
            f'{clip}/0': {'media_sha256': media[f'videos/video_000/clips/{clip.split("/")[1]}/media/{camera}.mp4'],
                           'rgb_sha256': hashes[clip, camera]} for clip in clips}
        path = args.output / f'inputs/{camera}.json'
        save(path, candidates)
        refs[camera] = {'path': str(path.resolve().relative_to((ROOT / 'outputs').resolve())), 'sha256': sha(path)}
    request = {'schema': 'video_court_build_v1', 'profile': 'temporal_rgb_candidate_consensus_v1',
               'dataset_directory': 'tennis_multiview/processed/meiji_4cam/dataset', 'video_id': 'video_000',
               'candidate_manifests': refs, 'seed_clip_ids': clips, 'fit_clip_ids': None,
               'bootstrap_clip_ids': ['video_000/clip_001', 'video_000/clip_033', 'video_000/clip_067'],
               'fit_numerators': [1, 2, 3], 'diagnostic_numerators': [4], 'denominator': 5, 'workers': args.workers}
    save(args.output / 'inputs/request.json', request)
    save(args.output / 'inputs/provenance.json', {'split_sha256': sha(args.split), 'fit_protocol_sha256': sha(args.stage1 / 'all-fit/protocol.json'),
                                               'frozen_raw_manifest_sha256': sha(args.raw_manifest), 'seed_clip_count': len(clips),
                                               'reference_frames_excluded_from_all_subsets': True})


def build(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    cv2.setNumThreads(1)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    request = args.output / 'inputs/request.json'
    output_relative = args.output.resolve().relative_to((ROOT / 'outputs').resolve())
    overrides = ['device=cpu', 'gvhmr.enabled=false', f'paths.artifact_root={(ROOT / "outputs").resolve()}',
                 'court_kp.mode=video_shared', 'court_kp.video_shared.action=build',
                 f'court_kp.video_shared.directory={output_relative}/store', 'court_kp.video_shared.dataset_id=meiji_4cam',
                 'court_kp.video_shared.video_id=video_000', f'court_kp.video_shared.request={output_relative}/inputs/request.json',
                 f'court_kp.video_shared.request_sha256={sha(request)}']
    with initialize_config_dir(version_base='1.3', config_dir=str(ROOT / 'src/tennis_scene/configs')):
        cfg = compose(config_name='pipeline', overrides=overrides)
    runner = TennisSceneOrchestrator.from_config(cfg)
    protocol = {'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), 'sources': freeze(),
                'driver_sha256': sha(Path(__file__)), 'code_identity': runner.code_identity, 'overrides': overrides,
                'request_sha256': sha(request), 'request': json.loads(request.read_text()),
                'reference_h_used': False, 'manual_points_used': False, 'manual_net_mask_used': False,
                'reference_frames_used': False, 'gpu_used': False, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}
    save(args.output / 'build-protocol.json', protocol)
    started = time.monotonic()
    artifact = runner.prepare_video_court()
    assert artifact is not None
    assert freeze() == protocol['sources'], 'Production sources changed during build'
    artifact.verify()
    save(args.output / 'build-summary.json', {'artifact_digest': artifact.digest, 'manifest': artifact.manifest,
         'seconds': time.monotonic() - started, 'cameras': {camera: {
             'status': value['status'], 'reasons': value['diagnostics']['reasons'],
             'clip_status_counts': value['diagnostics']['clip_status_counts'],
             'clip_reason_counts': dict(Counter(f['reason'] for clip in value['clips'].values() for f in clip['frames'])),
             'selected_initializer': value['diagnostics']['selected_initializer'],
             'fit_frames': len(value['diagnostics']['fit_frames'])} for camera, value in artifact.cameras.items()}})
    print((args.output / 'build-summary.json').read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'build'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage1', type=Path)
    parser.add_argument('--split', type=Path)
    parser.add_argument('--raw-manifest', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else build)(args)
