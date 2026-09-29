"""Saved-dev-only #964 person source / dwell diagnostic, entirely on CPU.

Phases are durable: sources -> tracks -> select -> video. Detector inference
is never imported or launched. The derivative lacks all-person pose/CLIP
features (only an old 120-frame smoke exists), so it is explicitly untested.
"""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
from numpy.typing import NDArray

from src.tasks.person_tracking.court_candidates import DwellConfig, select_candidates
from src.tasks.person_tracking.court_consistency import court_consistency
from src.tasks.person_tracking.selection_metrics import aggregate_units, selection_units
from src.tasks.person_tracking.selection_report import write_review_report
from src.tasks.player_association.appearance.encoders import (
    build_encoder,
    encoder_weights,
)
from src.tasks.player_association.appearance.sampling import (
    CropSamplingConfig,
    embed_tracks,
)
from src.tasks.player_association.association.associate import (
    AssociationUndecided,
    CameraTracks,
    associate,
)
from src.tasks.player_association.association.config import load_association_config
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_detection.evaluation.far_archive import DetectionArchive
from src.tasks.player_detection.evaluation.person_sources import (
    DEV_CLIPS,
    summarize_sources,
    write_csv,
)
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.court_calibration import (
    CourtCalibrationOutput,
)
from src.tennis_scene.pipeline.components.person_tracking import PersonTrackingOutput
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256
from src.utils.video import OpenCVVideoFrameReader

SOURCES = ('ft_base_0.01', 'coco_0.30', 'union_0.30')
TRACKER_CONFIG = dict(track_high_thresh=0., track_low_thresh=0., new_track_thresh=0., track_buffer=30,
    match_thresh=.8, fuse_score=False, gmc_method='sparseOptFlow', proximity_thresh=.5, appearance_thresh=.8,
    with_reid=False, model='auto')
CODE_ROOT = Path(__file__).resolve().parents[2]


def load_sources(report: Path) -> dict[str, Any]:
    source: dict[str, Any] = json.loads((report / 'sources.json').read_text())
    if set(r['clip'] for r in source['inputs']) != set(DEV_CLIPS):
        raise ValueError('Require only the fixed dev set')
    for record in source['inputs']:
        if dual_sha256(Path(record['label_path'])) != record['label_sha256']:
            raise ValueError('Labels changed')
    return source


def replay(report: Path) -> None:
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.bot_sort import BOTSORT
    source = load_sources(report)
    target = report / 'tracks.json'
    if target.exists():
        raise FileExistsError(target)
    result: dict[str, Any] = {'sources_sha256': dual_sha256(report / 'sources.json'),
        'ultralytics_version': ultralytics.__version__, 'config': TRACKER_CONFIG, 'records': {},
        'method': 'Ultralytics BoT-SORT; score gates disabled, fuse_score=False; original detector scores retained',
        'score_policy': 'source threshold only; no further #937 gate or score fusion; no person capacity',
        'boxes': 'raw observed detection row for scoring/footpoint, Kalman box also archived',
        'derivative': 'not ready for these inputs: full all-person pose/appearance not saved; no substitute features'}
    for record in source['inputs']:
        key = f"{record['clip']}/{record['camera']}"
        video = record['video']
        if dual_sha256(Path(video['path'])) != video['sha256']:
            raise ValueError('Video changed')
        archives = {name: DetectionArchive.load(source['archives'][name][key]) for name in SOURCES}
        trackers = {name: BOTSORT(SimpleNamespace(**TRACKER_CONFIG)) for name in SOURCES}
        # Like the old wrapper, track_buffer is 30 frames at source ~60Hz.
        ids: dict[str, NDArray[np.int64]] = {name: np.full(len(a.scores), -1, np.int64) for name, a in archives.items()}
        kalman = {name: np.zeros_like(a.boxes) for name, a in archives.items()}
        elapsed = dict.fromkeys(SOURCES, 0.)
        count = 0
        for packet in OpenCVVideoFrameReader(Path(video['path'])):
            for name, archive in archives.items():
                det = archive.at(packet.index)
                data = np.column_stack((det.boxes_xyxy, det.scores, np.zeros(len(det.scores), np.float32)))
                start = time.perf_counter()
                output = trackers[name].update(Boxes(data, orig_shape=packet.frame.shape[:2]), packet.frame)
                elapsed[name] += time.perf_counter() - start
                offset = int(archive.offsets[packet.index])
                for row in output:
                    if len(row) != 8 or not float(row[7]).is_integer() or not 0 <= row[7] < len(det.scores):
                        raise ValueError('Ultralytics did not return an exact source detection index')
                    index = offset + int(row[7])
                    if ids[name][index] != -1:
                        raise ValueError('Two tracks assigned to one detection')
                    ids[name][index], kalman[name][index] = int(row[4]), row[:4]
            count += 1
        if count != video['num_frames']:
            raise ValueError('Video ended before complete timeline')
        for name in SOURCES:
            path = report / 'tracks' / name / f'{key}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as handle:
                np.savez_compressed(handle, track_ids=ids[name], kalman_boxes=kalman[name])
            result['records'].setdefault(name, {})[key] = {'path': str(path), 'sha256': dual_sha256(path),
                'source_sha256': source['archives'][name][key]['sha256'], 'frames': count,
                'track_count': len(np.unique(ids[name][ids[name] >= 0])), 'ms_per_frame': elapsed[name] * 1000 / count}
        write_json_atomic(report / 'tracks.progress.json', result)
        print(f'tracked {key}: ' + str({n: result['records'][n][key]['track_count'] for n in SOURCES}), flush=True)
    write_json_atomic(target, result)


def calibration(record: dict[str, Any], half_turn: bool) -> Any:
    root = Path(record['baseline_store'])
    store = ClipStore(root, json.loads((root / 'scene.json').read_text())['source'])
    ref = store.active('court_calibration')
    if ref is None or json_value(ref) != record['calibration_reference']:
        raise ValueError('Calibration changed')
    value = store.load(ref, ArtifactCodec(CourtCalibrationOutput))
    cameras = {v.camera.camera_id: v.camera for v in value.calibration.views}
    return cameras[record['camera']].half_turned(half_turn)


def dense_tracks(source: dict[str, Any], tracking: dict[str, Any], name: str,
                 record: dict[str, Any], camera: Any, repo: Path) -> tuple[CameraTracks, dict[str, Any]]:
    key = f"{record['clip']}/{record['camera']}"
    if name == 'old_pipeline':
        root = repo / 'outputs/player_association/evaluate/meiji_clips/i933-observe-v1-20260927/stores' / record['clip']
        store = ClipStore(root, json.loads((root / 'scene.json').read_text())['source'])
        ref = store.active(f"person_tracking/{record['camera']}")
        if ref is None:
            raise ValueError('Missing old detector + old tracker baseline')
        old = store.load(ref, ArtifactCodec(PersonTrackingOutput))
        return CameraTracks(camera, (1920, 1080), old.track_ids, old.boxes_xyxy, old.observed), {'store': str(root), 'reference': json_value(ref)}
    archive = DetectionArchive.load(source['archives'][name][key])
    saved = tracking['records'][name][key]
    if dual_sha256(Path(saved['path'])) != saved['sha256'] or saved['source_sha256'] != source['archives'][name][key]['sha256']:
        raise ValueError('Tracking/source hash mismatch')
    with np.load(saved['path'], allow_pickle=False) as data:
        assignments = data['track_ids']
    tids = np.unique(assignments[assignments >= 0])
    boxes: NDArray[np.float32] = np.zeros((len(tids), len(archive.milliseconds), 4), np.float32)
    observed = np.zeros(boxes.shape[:2], bool)
    for frame in range(len(archive.milliseconds)):
        begin, end = archive.offsets[frame:frame + 2]
        for index in range(begin, end):
            if assignments[index] >= 0:
                row = int(np.searchsorted(tids, assignments[index]))
                if observed[row, frame]:
                    raise ValueError('A track has two observations in one frame')
                boxes[row, frame], observed[row, frame] = archive.boxes[index], True
    return CameraTracks(camera, (1920, 1080), tids, boxes, observed), saved


def select(repo: Path, report: Path) -> None:
    source = load_sources(report)
    tracking = json.loads((report / 'tracks.json').read_text())
    if tracking['sources_sha256'] != dual_sha256(report / 'sources.json'):
        raise ValueError('Source manifest changed since replay')
    if (report / 'selection.json').exists():
        raise FileExistsError(report / 'selection.json')
    config_path = CODE_ROOT / 'src/tasks/player_association/configs/association.yaml'
    config = load_association_config(config_path, players_per_side=1)
    dwell = DwellConfig(config.region, config.footpoints, config.min_presence_fraction)
    sides_path = repo / 'outputs/court_side/evaluate/meiji_clips/i932-detector-v1-20260927/decisions_v2.json'
    sides = json.loads(sides_path.read_text())
    name = 'clipreid_vitb16_market1501'
    weights = encoder_weights(name, checkpoint_root=repo / 'ckpt', external_root=repo / 'third_party')
    encoder = build_encoder(name, checkpoint_root=repo / 'ckpt', external_root=repo / 'third_party', device='cpu')
    result: dict[str, Any] = {'sources_sha256': dual_sha256(report / 'sources.json'), 'tracks_sha256': dual_sha256(report / 'tracks.json'),
        'association_config': json_value(config), 'association_config_sha256': dual_sha256(config_path),
        'dwell_config': json_value(dwell), 'side_decisions': str(sides_path), 'side_sha256': dual_sha256(sides_path),
        'encoder': {'name': name, 'weights': str(weights), 'sha256': dual_sha256(weights), 'device': 'cpu'},
        'sampling': asdict(CropSamplingConfig()), 'records': {}, 'table': [], 'per_clip': [],
        'identity_metric': 'known player identity kept if >=50% labelled units selected; nonplayer identity rejected only if zero units selected',
        'stages': 'dwell candidate is primary; existing CLIP+court association is secondary. An undecided second check has no final metrics, never a fallback.'}
    all_units: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with gzip.open(report / 'selection_units.jsonl.gz', 'wt') as handle:
        for variant in (*SOURCES, 'old_pipeline'):
            for clip in DEV_CLIPS:
                records = sorted((r for r in source['inputs'] if r['clip'] == clip), key=lambda r: r['camera'])
                side = [s for s in sides['clips'] if s['clip_id'] == clip]
                if len(side) != 1 or not side[0]['annotation']['decided']:
                    raise ValueError('Existing court side is undecided')
                turns = dict(zip(side[0]['camera_ids'], side[0]['annotation']['view_half_turns'], strict=True))
                selected_cameras, original_cameras, chosen_rows = [], [], []
                clip_result: dict[str, Any] = {'cameras': {}, 'association': {}}
                labels = ClipLabels.load(Path(records[0]['label_path']))
                candidate_units = []
                for record in records:
                    cam = record['camera']
                    tracks, provenance = dense_tracks(source, tracking, variant, record, calibration(record, turns[cam]), repo)
                    chosen, diagnostics = select_candidates(tracks, dwell)
                    selected = np.zeros(tracks.observed.shape, bool)
                    selected[chosen] = True
                    units = selection_units(tracks, selected, labels)
                    candidate_units += units
                    appearance, samples = embed_tracks(Path(record['video']['path']), tracks.boxes_xyxy[chosen], tracks.observed[chosen],
                        tracks.image_size, encoder, CropSamplingConfig())
                    candidate = CameraTracks(tracks.camera, tracks.image_size, tracks.track_ids[chosen], tracks.boxes_xyxy[chosen],
                                             tracks.observed[chosen], appearance)
                    selected_cameras.append(candidate)
                    original_cameras.append(tracks)
                    chosen_rows.append(chosen)
                    path = report / 'selection' / variant / clip / f'{cam}.npz'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('xb') as dst:
                        np.savez_compressed(dst, track_ids=tracks.track_ids, boxes=tracks.boxes_xyxy,
                                            observed=tracks.observed, chosen=chosen)
                    feature_path = path.with_suffix('.appearance.npz')
                    arrays = {f'{i}/{field}': getattr(a, field) for i, a in enumerate(appearance) for field in ('frames', 'embeddings')}
                    with feature_path.open('xb') as dst:
                        np.savez_compressed(dst, **arrays)
                    clip_result['cameras'][cam] = {'path': str(path), 'sha256': dual_sha256(path), 'tracking': provenance,
                        'appearance_path': str(feature_path), 'appearance_sha256': dual_sha256(feature_path),
                        'samples': json_value(samples), 'candidates': len(chosen), 'all_tracks': len(tracks.track_ids), 'dwell': diagnostics}
                stages = {'dwell': candidate_units}
                try:
                    associated = associate(selected_cameras, records[0]['video']['fps'], config)
                except AssociationUndecided as error:
                    clip_result['association'] = {'status': 'undecided', 'reason': error.reason, 'diagnostics': error.diagnostics}
                else:
                    clip_result['association'] = {'status': 'ok', 'diagnostics': associated.diagnostics}
                    final_units = []
                    for cam_index, tracks in enumerate(original_cameras):
                        chosen = chosen_rows[cam_index]
                        ids = np.full(tracks.observed.shape, -1, np.int64)
                        ids[chosen] = associated.player_ids[cam_index]
                        final_units += selection_units(tracks, ids >= 0, labels)
                        target = report / 'selection' / variant / clip / f'{tracks.camera.camera_id}.identities.npz'
                        with target.open('xb') as dst:
                            np.savez_compressed(dst, player_ids=ids)
                        clip_result['cameras'][tracks.camera.camera_id]['identities'] = {'path': str(target), 'sha256': dual_sha256(target)}
                    stages['associated'] = final_units
                for stage, units in stages.items():
                    all_units.setdefault((variant, stage), []).extend(units)
                    result['per_clip'].append({'source': variant, 'stage': stage, 'clip': clip, **aggregate_units(units)})
                    for unit in units:
                        handle.write(json.dumps({'source': variant, 'stage': stage, **unit}) + '\n')
                result['records'].setdefault(variant, {})[clip] = clip_result
                write_json_atomic(report / 'selection.progress.json', result)
                print(f'selected {variant}/{clip}: {[len(c.track_ids) for c in selected_cameras]}, association={clip_result["association"]["status"]}', flush=True)
    result['table'] = [{'source': name, 'stage': stage, 'clips': len({u['clip'] for u in units}), **aggregate_units(units)}
                       for (name, stage), units in all_units.items()]
    write_json_atomic(report / 'selection.json', result)
    write_csv(report / 'selection.csv', result['table'])
    write_csv(report / 'selection_per_clip.csv', result['per_clip'])


def draw_box(image: np.ndarray, raw: np.ndarray, color: tuple[int, int, int], text: str = '') -> None:
    x1, y1, x2, y2 = np.round(raw / 3).astype(int)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2 if text else 1)
    if text:
        cv2.putText(image, text, (x1, max(13, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)


def consistency_report(report: Path) -> None:
    source = load_sources(report)
    selection = json.loads((report / 'selection.json').read_text())
    sides_path = Path(selection['side_decisions'])
    if dual_sha256(sides_path) != selection['side_sha256']:
        raise ValueError('Court side input changed')
    sides = json.loads(sides_path.read_text())
    config = load_association_config(CODE_ROOT / 'src/tasks/player_association/configs/association.yaml', players_per_side=1)
    table: list[dict[str, Any]] = []
    for variant, clips in selection['records'].items():
        for clip, verdict in clips.items():
            if verdict['association']['status'] != 'ok':
                continue  # Explicitly undecided: no inferred identity to verify.
            side = next(s for s in sides['clips'] if s['clip_id'] == clip)
            turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
            cameras, identities = [], []
            for record in sorted((r for r in source['inputs'] if r['clip'] == clip), key=lambda r: r['camera']):
                saved = verdict['cameras'][record['camera']]
                for entry in (saved, saved['identities']):
                    if dual_sha256(Path(entry['path'])) != entry['sha256']:
                        raise ValueError('Saved selection changed')
                with np.load(saved['path']) as data, np.load(saved['identities']['path']) as ids:
                    cameras.append(CameraTracks(calibration(record, turns[record['camera']]), (1920, 1080), data['track_ids'], data['boxes'], data['observed']))
                    identities.append(ids['player_ids'])
            table.extend({'source': variant, 'clip': clip, **row} for row in court_consistency(cameras, identities, config.footpoints))
    write_json_atomic(report / 'court_consistency.json', {'selection_sha256': dual_sha256(report / 'selection.json'),
        'scope': 'Post-association footpoint distances on z=0; successful clips only; no new threshold; not full body 3D', 'table': table})
    write_csv(report / 'court_consistency.csv', table)


def render(report: Path) -> None:
    source, selection = load_sources(report), json.loads((report / 'selection.json').read_text())
    variant, clip = 'union_0.30', 'video_000/clip_000'
    verdict = selection['records'][variant][clip]
    if verdict['association']['status'] != 'ok':
        raise ValueError('Do not render guessed cross-camera identities after an undecided association')
    records = sorted((r for r in source['inputs'] if r['clip'] == clip), key=lambda r: r['camera'])
    # 12 real-time seconds, 20fps sampling, 3 synchronized 640x360 cameras.
    fps, seconds = 20., 12.
    source_frames = np.round(np.arange(int(seconds * fps)) * records[0]['video']['fps'] / fps).astype(int)
    captures, data, all_people = [], [], []
    for r in records:
        value = verdict['cameras'][r['camera']]
        for entry in (value, value['identities']):
            if dual_sha256(Path(entry['path'])) != entry['sha256']:
                raise ValueError('Video selection input changed')
        with np.load(value['path']) as x, np.load(value['identities']['path']) as y:
            data.append({**dict(x), **dict(y)})
        all_people.append(DetectionArchive.load(source['archives'][variant][f"{clip}/{r['camera']}"]))
        captures.append(cv2.VideoCapture(r['video']['path']))
    temp, target = report / 'selection_3cam.raw.mp4', report / 'selection_3cam.mp4'
    if target.exists() or temp.exists():
        raise FileExistsError(target)
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter.fourcc(*'mp4v'), fps, (1920, 410))
    if not writer.isOpened():
        raise RuntimeError('Video writer failed')
    colors = ((80, 210, 80), (220, 170, 40))
    try:
        for frame in source_frames:
            panels = []
            for camera, capture in enumerate(captures):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
                ok, image = capture.read()
                if not ok:
                    raise ValueError('Video decode failed')
                image = cv2.resize(image, (640, 360))
                for raw in all_people[camera].at(frame).boxes_xyxy:
                    draw_box(image, raw, (160, 160, 160))
                for row in np.flatnonzero(data[camera]['observed'][:, frame] & (data[camera]['player_ids'][:, frame] >= 0)):
                    identity = int(data[camera]['player_ids'][row, frame])
                    draw_box(image, data[camera]['boxes'][row, frame], colors[identity % len(colors)], f'P{identity}')
                cv2.putText(image, f"{records[camera]['camera']}  frame {frame}", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                panels.append(image)
            canvas: NDArray[np.uint8] = np.zeros((410, 1920, 3), np.uint8)
            canvas[:360] = np.concatenate(panels, axis=1)
            cv2.putText(canvas, 'DEV video_000/clip_000 | FT>=0.3 + saved ROI COCO | grey: all available persons | colour: court dwell + CLIP association',
                        (10, 381), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, 'Predicted P0/P1 (no reviewed labels used for selection or colours); COCO outside ROI was not saved.',
                        (10, 403), cv2.FONT_HERSHEY_SIMPLEX, .58, (210, 210, 210), 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
        for cap in captures:
            cap.release()
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(temp), '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], check=True)
    read = cv2.VideoCapture(str(target))
    frames, shape, got = 0, None, True
    while got:
        got, frame = read.read()
        if got:
            frames += 1
            shape = frame.shape
    read.release()
    if frames != len(source_frames) or shape != (410, 1920, 3):
        raise ValueError('Encoded review video failed full readback')
    write_json_atomic(report / 'video.json', {'path': str(target), 'sha256': dual_sha256(target), 'frames': frames,
        'fps': fps, 'shape': shape, 'source': variant, 'clip': clip, 'source_frames': source_frames.tolist(),
        'selection_sha256': dual_sha256(report / 'selection.json'), 'readback': 'all frames verified'})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--phase', choices=('sources', 'tracks', 'select', 'video', 'report'), required=True)
    parser.add_argument('--progress', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    if args.phase == 'sources':
        if args.progress is None:
            parser.error('--progress is required for sources')
        summarize_sources(args.progress, args.report)
    elif args.phase == 'tracks':
        replay(args.report)
    elif args.phase == 'select':
        select(args.repo, args.report)
    elif args.phase == 'video':
        render(args.report)
    else:
        consistency_report(args.report)
        write_review_report(args.report)


if __name__ == '__main__':
    main()
