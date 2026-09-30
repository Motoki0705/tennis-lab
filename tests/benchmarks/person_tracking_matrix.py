"""CPU matrix governed by the run-8 precommitted protocol, with durable stages."""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil  # type: ignore[import-untyped]
import scipy
import torch
from person_selection_cpu import (  # type: ignore[import-not-found]
    calibration,
    dense_tracks,
)

from src.submodules.models import (
    BotSortAssociator,
    TrackRequest,
    select_and_complete_tracks,
)
from src.tasks.person_tracking.appearance_cache import cached_appearance
from src.tasks.person_tracking.archive import load_features
from src.tasks.person_tracking.botsort_pose import BotSortPoseConfig
from src.tasks.person_tracking.court_linking import (
    LinkingConfig,
    exclusive_region,
    select_linked_candidates,
)
from src.tasks.person_tracking.deep_ocsort_pose import DeepOCSortPoseConfig
from src.tasks.person_tracking.evaluation import (
    score_units,
    stratified_scores,
    tracking_units,
)
from src.tasks.person_tracking.feature_tracks import sampled_appearance, scatter_tracks
from src.tasks.person_tracking.linked_timeline import linked_timeline
from src.tasks.person_tracking.methods import build_tracker
from src.tasks.player_association.appearance.encoders import build_encoder
from src.tasks.player_association.appearance.sampling import TrackAppearance
from src.tasks.player_association.association.associate import (
    AssociationUndecided,
    CameraTracks,
    associate,
)
from src.tasks.player_association.association.config import load_association_config
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.evaluation.metrics import CameraPrediction, evaluate
from src.tasks.player_association.geometry.footpoints import ground_footpoints
from src.tasks.player_detection.evaluation.far_archive import DetectionArchive
from src.tasks.player_detection.evaluation.person_sources import DEV_CLIPS, write_csv
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.tracking_identity import (
    TrackletLinkPolicy,
    link_tracklets,
    torso_appearance_lab,
)
from src.tennis_scene.pipeline.errors import ReconstructionUnavailable
from src.utils.checksum import dual_sha256
from src.utils.video import OpenCVVideoFrameReader

CODE = Path(__file__).resolve().parents[2]
CLIP = 'clipreid_vitb16_market1501'
SOLIDER = 'solider_swin_base_msmt17'
METHODS = ('botsort_pose', 'deep_ocsort_pose')
VARIANTS = ('old_old', 'new_old', *[f'{m}__{e}' for m in METHODS for e in (CLIP, SOLIDER)])
PROTOCOL = CODE / 'knowledge/runs/run-i964-tracker-matrix-r8-20260930/protocol.md'


def checked(record: dict[str, Any]) -> Path:
    path = Path(record['path'])
    if dual_sha256(path) != record['sha256']:
        raise ValueError(f'Changed evidence: {path}')
    return path


def record_file(path: Path) -> dict[str, Any]:
    return {'path': str(path), 'sha256': dual_sha256(path), 'bytes': path.stat().st_size}


def inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = json.loads((args.features / 'plan.json').read_text())
    features = json.loads((args.features / 'features.json').read_text())
    if features['status'] != 'ok' or features['plan_sha256'] != dual_sha256(args.features / 'plan.json'):
        raise ValueError('Feature run incomplete or changed')
    source = json.loads(checked(plan['sources']).read_text())
    if {r['clip'] for r in source['inputs']} != set(DEV_CLIPS):
        raise ValueError('Matrix is only the fixed four dev clips')
    for r in source['inputs']:
        checked({'path': r['label_path'], 'sha256': r['label_sha256']})
    checked(plan['reservation'])
    if set(json.loads(Path(plan['reservation']['path']).read_text())['clips']) & set(DEV_CLIPS):
        raise ValueError('Dev set overlaps reserved clips')
    return plan, features, source


def old_replay(record: dict[str, Any], detection: dict[str, Any], camera: Any) -> tuple[CameraTracks, dict[str, Any]]:
    archive = DetectionArchive.load(detection)
    checked(record['video'])
    tracker = BotSortAssociator()
    history = []
    for packet in OpenCVVideoFrameReader(Path(record['video']['path'])):
        tracked = tracker.update(archive.at(packet.index), packet.frame)
        history.append([{**item, 'appearance_lab': torso_appearance_lab(packet.frame, item['bbx_xyxy'])} for item in tracked])
    if len(history) != record['video']['num_frames']:
        raise ValueError('Legacy replay video ended early')
    linked = link_tracklets(history, TrackletLinkPolicy())
    result = select_and_complete_tracks(linked.history, TrackRequest(Path(record['video']['path']), None, False), len(history))
    ids = result.track_ids
    boxes = np.stack([result.tracks[i].numpy() for i in ids]) if ids else np.zeros((0, len(history), 4), np.float32)
    observed = np.stack([result.observed_mask(i).numpy() for i in ids]) if ids else np.zeros((0, len(history)), bool)
    return CameraTracks(camera, (1920, 1080), np.asarray(ids, np.int64), boxes, observed), {'lab_links': json_value(linked.links)}


def setup(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    sides_path = args.repo / 'outputs/court_side/evaluate/meiji_clips/i932-detector-v1-20260927/decisions_v2.json'
    cfg = load_association_config(CODE / 'src/tasks/player_association/configs/association.yaml', players_per_side=1)
    identity = {'protocol': record_file(PROTOCOL), 'features': record_file(args.features / 'features.json'),
        'plan': record_file(args.features / 'plan.json'), 'side': record_file(sides_path),
        'code': {str(p.relative_to(CODE)): dual_sha256(p) for p in sorted((CODE / 'src/tasks/person_tracking').rglob('*.py'))},
        'benchmark': record_file(Path(__file__)), 'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip(),
        'association': json_value(cfg), 'selection': asdict(LinkingConfig()), 'lab': asdict(TrackletLinkPolicy()),
        'botsort_pose': asdict(BotSortPoseConfig()), 'deep_ocsort_pose': asdict(DeepOCSortPoseConfig()),
        'versions': {'numpy': np.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__},
        'models': plan['models'], 'seed': 0, 'device': 'cpu'}
    path = args.report / 'identity.json'
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError('Matrix resume identity changed; keep old run and use new directory')
    else:
        write_json_atomic(path, identity)
    return identity


def track(args: argparse.Namespace) -> None:
    plan, features, source = inputs(args)
    identity = setup(args, plan)
    sides = json.loads(checked(identity['side']).read_text())
    for record in source['inputs']:
        key = f"{record['clip']}/{record['camera']}"
        side = next(s for s in sides['clips'] if s['clip_id'] == record['clip'])
        turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
        camera = calibration(record, turns[record['camera']])
        loaded = {e: load_features(checked(features['records'][e][key]))[0] for e in (CLIP, SOLIDER)}
        for variant in VARIANTS:
            target = args.report / 'tracks' / variant / f'{key}.json'
            if target.exists():
                value = json.loads(target.read_text())
                if value['identity_sha256'] != dual_sha256(args.report / 'identity.json'):
                    raise ValueError('Cached track identity changed')
                checked(value['arrays'])
                continue
            start = time.perf_counter()
            result: dict[str, Any] = {'status': 'ok', 'identity_sha256': dual_sha256(args.report / 'identity.json')}
            origins = None
            try:
                if variant == 'old_old':
                    tracks, provenance = dense_tracks(source, {}, 'old_pipeline', record, camera, args.repo)
                    result['legacy'] = provenance
                elif variant == 'new_old':
                    tracks, provenance = old_replay(record, source['archives']['coco_fullframe_0.30'][key], camera)
                    result['legacy'] = provenance
                else:
                    method, encoder = variant.split('__')
                    frames = loaded[encoder]
                    tracker = build_tracker(method, fps=record['video']['fps'])
                    assignments = [tracker.update(f) for f in frames]
                    repeated = build_tracker(method, fps=record['video']['fps'])
                    for feature, first in zip(frames, assignments, strict=True):
                        again = repeated.update(feature)
                        if not np.array_equal(first.track_ids, again.track_ids) or not np.array_equal(first.detection_rows, again.detection_rows):
                            raise ValueError('Tracker repeat changed IDs/rows')
                    tracks, origins = scatter_tracks(camera, frames, assignments, (1920, 1080))
                    result['deterministic_repeat_equal'] = True
            except ReconstructionUnavailable as error:
                # Explicit baseline failure, not a substitute tracker or partial result.
                result.update(status='failed', reason=str(error))
                tracks = CameraTracks(camera, (1920, 1080), np.empty(0, np.int64),
                    np.zeros((0, record['video']['num_frames'], 4), np.float32), np.zeros((0, record['video']['num_frames']), bool))
            path = target.with_suffix('.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as out:
                np.savez_compressed(out, track_ids=tracks.track_ids, boxes=tracks.boxes_xyxy, observed=tracks.observed,
                    origins=origins if origins is not None else np.full(tracks.observed.shape, -1, np.int64))
            result.update(arrays=record_file(path), elapsed_seconds=time.perf_counter() - start,
                          track_count=len(tracks.track_ids), source_detection_rows=origins is not None)
            write_json_atomic(target, result)
            print(f'{variant} {key}: {result["status"]}, {len(tracks.track_ids)} IDs', flush=True)


def assess(args: argparse.Namespace) -> None:
    plan, feature_manifest, source = inputs(args)
    identity = setup(args, plan)
    sides = json.loads(checked(identity['side']).read_text())
    config = load_association_config(CODE / 'src/tasks/player_association/configs/association.yaml', players_per_side=1)
    clip_encoder = None
    for variant in VARIANTS:
        for clip in DEV_CLIPS:
            target = args.report / 'evaluation' / variant / clip / 'result.json'
            if target.exists():
                continue
            records = sorted([r for r in source['inputs'] if r['clip'] == clip], key=lambda r: r['camera'])
            side = next(s for s in sides['clips'] if s['clip_id'] == clip)
            turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
            labels = ClipLabels.load(Path(records[0]['label_path']))
            units, originals, masks = [], [], []
            groups: dict[str, list[CameraTracks]] = {CLIP: [], SOLIDER: []}
            result: dict[str, Any] = {'variant': variant, 'clip': clip, 'cameras': {}, 'association': {}}
            for record in records:
                cam, key = record['camera'], f"{clip}/{record['camera']}"
                camera = calibration(record, turns[cam])
                saved = json.loads((args.report / 'tracks' / variant / f'{key}.json').read_text())
                with np.load(checked(saved['arrays']), allow_pickle=False) as a:
                    origins = a['origins']
                    raw = CameraTracks(camera, (1920, 1080), a['track_ids'], a['boxes'], a['observed'])
                appearances: dict[str, list[TrackAppearance] | tuple[TrackAppearance, ...]] = {}
                if saved['source_detection_rows']:
                    for encoder in (CLIP, SOLIDER):
                        frames, _ = load_features(checked(feature_manifest['records'][encoder][key]))
                        appearances[encoder] = sampled_appearance(raw, origins, frames)
                else:
                    xy, valid = ground_footpoints(raw.boxes_xyxy, raw.observed, camera, 1080, config.footpoints)
                    core = valid & exclusive_region(xy, LinkingConfig(), core=True)
                    if clip_encoder is None:
                        clip_encoder = build_encoder(CLIP, checkpoint_root=args.repo / 'ckpt', external_root=args.repo / 'third_party', device='cpu')
                    appearances[CLIP], sampling = cached_appearance(raw, core, record['video'], clip_encoder,
                        plan['models'][CLIP]['sha256'], args.report / 'baseline_crop_cache')
                    saved['sampling'] = sampling
                tracks = replace(raw, appearance=appearances[CLIP])
                mask, selection = select_linked_candidates(tracks, record['video']['fps'], LinkingConfig(), config.footpoints)
                group, group_origins = linked_timeline(tracks, selection)
                groups[CLIP].append(group)
                if SOLIDER in appearances:
                    groups[SOLIDER].append(linked_timeline(replace(raw, appearance=appearances[SOLIDER]), selection)[0])
                local, unknown = tracking_units(tracks, mask, labels)
                units.extend(local)
                originals.append(tracks)
                masks.append(group_origins)
                path = target.parent / f'{cam}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as out:
                    np.savez_compressed(out, track_ids=raw.track_ids, boxes=raw.boxes_xyxy, observed=raw.observed,
                        selected=mask, group_origins=group_origins, group_boxes=group.boxes_xyxy, group_observed=group.observed)
                result['cameras'][cam] = {'tracking': saved, 'selection': selection, 'arrays': record_file(path),
                    'unlabelled_selected_boxes': unknown, 'metrics': score_units(local)}
            for encoder, candidate in groups.items():
                if not candidate:
                    continue
                try:
                    prediction = associate(candidate, records[0]['video']['fps'], config)
                except AssociationUndecided as error:
                    result['association'][encoder] = {'status': 'undecided', 'reason': error.reason, 'diagnostics': error.diagnostics}
                else:
                    predictions = {}
                    identity_arrays = {}
                    for tracks, origin, ids in zip(originals, masks, prediction.player_ids, strict=True):
                        frame_ids = np.full(tracks.observed.shape, -1, np.int64)
                        for row in range(len(ids)):
                            at = np.flatnonzero((ids[row] >= 0) & (origin[row] >= 0))
                            frame_ids[origin[row, at], at] = ids[row, at]
                        predictions[tracks.camera.camera_id] = CameraPrediction(tracks.track_ids, tracks.boxes_xyxy, tracks.observed, frame_ids)
                        identity_arrays[tracks.camera.camera_id] = frame_ids
                    identity_path = target.parent / f'{encoder}.identities.npz'
                    with identity_path.open('xb') as out:
                        np.savez_compressed(out, **identity_arrays)
                    result['association'][encoder] = {'status': 'ok', 'metrics': evaluate(labels, predictions),
                                                       'diagnostics': prediction.diagnostics, 'arrays': record_file(identity_path)}
            path = target.parent / 'units.jsonl.gz'
            with gzip.open(path, 'wt') as out:
                for unit in units:
                    out.write(json.dumps(unit) + '\n')
            result['units'] = record_file(path)
            result['metrics'] = score_units(units)
            write_json_atomic(target, result)
            print(f'evaluated {variant}/{clip}: IDF1 {result["metrics"]["idf1"]:.6f}', flush=True)


def summarize(args: argparse.Namespace) -> None:
    tables: list[dict[str, Any]] = []
    records = []
    for variant in VARIANTS:
        units: list[dict[str, Any]] = []
        for clip in DEV_CLIPS:
            path = args.report / 'evaluation' / variant / clip / 'result.json'
            result = json.loads(path.read_text())
            with gzip.open(checked(result['units']), 'rt') as src:
                units.extend(json.loads(line) for line in src)
            records.append({'variant': variant, 'clip': clip, 'result': record_file(path),
                'metrics': result['metrics'], 'association': result['association'],
                'failed_cameras': [cam for cam, r in result['cameras'].items() if r['tracking']['status'] != 'ok']})
        tables.extend({'variant': variant, **row} for row in stratified_scores(units))
    write_csv(args.report / 'comparison.csv', tables)
    write_json_atomic(args.report / 'comparison.json', {'table': tables, 'records': records,
                                                      'identity': record_file(args.report / 'identity.json')})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--phase', choices=('track', 'evaluate', 'report'), required=True)
    args = parser.parse_args()
    if psutil.virtual_memory().available < 6 * 1024**3:
        raise RuntimeError('Require >=6 GiB available RAM')
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    {'track': track, 'evaluate': assess, 'report': summarize}[args.phase](args)


if __name__ == '__main__':
    main()
