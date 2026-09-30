"""CPU-only run-9 addendum; immutable old results and explicit native KPR distances."""
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
import torch
from person_selection_cpu import calibration  # type: ignore[import-not-found]
from person_tracking_matrix import (  # type: ignore[import-not-found]
    CLIP,
    CODE,
    SOLIDER,
    VARIANTS,
    checked,
    inputs,
    record_file,
)
from person_tracking_matrix_video import (  # type: ignore[import-not-found]
    recommendation,
)

from src.tasks.person_tracking.court_linking import (
    LinkingConfig,
    select_linked_candidates,
)
from src.tasks.person_tracking.deep_ocsort_pose import DeepOCSortPose
from src.tasks.person_tracking.evaluation import (
    score_units,
    stratified_scores,
    tracking_units,
)
from src.tasks.person_tracking.feature_tracks import sampled_appearance, scatter_tracks
from src.tasks.person_tracking.linked_timeline import linked_timeline
from src.tasks.person_tracking.part_archive import load_part_features
from src.tasks.person_tracking.strongsort import (
    InvalidPrediction,
    StrongSort,
    StrongSortConfig,
)
from src.tasks.person_tracking.strongsort_offline import AFLink, gaussian_interpolation
from src.tasks.player_association.association.associate import (
    AssociationUndecided,
    CameraTracks,
    associate,
)
from src.tasks.player_association.association.config import load_association_config
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.evaluation.metrics import CameraPrediction, evaluate
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

KPR = 'kpr_market_solider_parts'
DEEP = f'deep_ocsort_pose__{CLIP}'
LAB = f'deep_ocsort_pose_lab__{CLIP}'
STRONG = f'strongsort_pp__{CLIP}'
NATIVE = f'best_kpr__{KPR}'
PROTOCOL = CODE / 'knowledge/runs/run-i964-tracker-linking-r9-20260930/protocol-addendum.md'


def setup(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, features, source = inputs(args)
    native = json.loads((args.kpr / 'features.json').read_text())
    if native['status'] != 'ok' or native['plan_sha256'] != dual_sha256(args.kpr / 'plan.json'):
        raise ValueError('Native features incomplete or changed')
    previous = json.loads((args.previous / 'identity.json').read_text())
    identity = {'protocol': record_file(PROTOCOL), 'plan': record_file(args.features / 'plan.json'),
                'previous': record_file(args.previous / 'comparison.json'), 'features': record_file(args.features / 'features.json'),
                'native': record_file(args.kpr / 'features.json'), 'aflink': record_file(args.aflink),
                'side': previous['side'], 'selection': asdict(LinkingConfig()), 'strongsort': asdict(StrongSortConfig()),
                'lab': asdict(TrackletLinkPolicy()), 'association': previous['association'],
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip(),
                'code': {str(p.relative_to(CODE)): dual_sha256(p) for folder in ('src/tasks/person_tracking', 'src/tasks/player_association')
                         for p in sorted((CODE / folder).rglob('*.py'))}, 'benchmark': record_file(Path(__file__))}
    dest = args.report / 'identity.json'
    if dest.exists():
        if json.loads(dest.read_text()) != identity:
            raise ValueError('Run identity changed; preserve old output and choose a new directory')
    else:
        write_json_atomic(dest, identity)
    return features, native, source


def read_tracks(record: dict[str, Any], camera: Any) -> tuple[CameraTracks, np.ndarray]:
    with np.load(checked(record['arrays']), allow_pickle=False) as saved:
        return CameraTracks(camera, (1920, 1080), saved['track_ids'], saved['boxes'], saved['observed']), saved['origins']


def save_tracks(path: Path, tracks: CameraTracks, origins: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as target:
        np.savez_compressed(target, track_ids=tracks.track_ids, boxes=tracks.boxes_xyxy,
                            observed=tracks.observed, origins=origins)
    return dict(record_file(path))


def merge_tracks(raw: CameraTracks, origins: np.ndarray, roots: dict[int, int]) -> tuple[CameraTracks, np.ndarray]:
    groups = sorted(set(roots.values()))
    boxes = np.zeros((len(groups), *raw.boxes_xyxy.shape[1:]), np.float32)
    rows = np.full(boxes.shape[:2], -1, np.int64)
    for source, root in roots.items():
        destination = groups.index(root)
        at = raw.observed[source]
        if (rows[destination, at] >= 0).any():
            raise ValueError('AFLink created overlapping observations')
        boxes[destination, at], rows[destination, at] = raw.boxes_xyxy[source, at], origins[source, at]
    return CameraTracks(raw.camera, raw.image_size, raw.track_ids[groups], boxes, rows >= 0), rows


def lab_relink(raw: CameraTracks, origins: np.ndarray, video: dict[str, Any]) -> tuple[CameraTracks, np.ndarray, Any]:
    history = []
    for packet in OpenCVVideoFrameReader(checked(video)):
        history.append([{'id': int(raw.track_ids[row]), 'bbx_xyxy': raw.boxes_xyxy[row, packet.index],
                         'appearance_lab': torso_appearance_lab(packet.frame, raw.boxes_xyxy[row, packet.index]),
                         'detection_row': int(origins[row, packet.index])}
                        for row in np.flatnonzero(raw.observed[:, packet.index])])
    if len(history) != raw.observed.shape[1]:
        raise ValueError('Lab video timeline ended early')
    linked = link_tracklets(history, TrackletLinkPolicy())
    ids = sorted(linked.source_ids)
    boxes: np.ndarray = np.zeros((len(ids), len(history), 4), np.float32)
    rows = np.full(boxes.shape[:2], -1, np.int64)
    for frame, observations in enumerate(linked.history):
        for observation in observations:
            row = ids.index(observation['id'])
            boxes[row, frame], rows[row, frame] = observation['bbx_xyxy'], observation['detection_row']
    return CameraTracks(raw.camera, raw.image_size, np.asarray(ids, np.int64), boxes, rows >= 0), rows, json_value(linked.links)


def track_variant(args: argparse.Namespace, variant: str, method: str, encoder: str) -> None:
    features, native, source = setup(args)
    identity = json.loads((args.report / 'identity.json').read_text())
    sides = json.loads(checked(identity['side']).read_text())
    af = AFLink(args.aflink) if method == STRONG else None
    from src.tasks.person_tracking.archive import load_features
    for record in source['inputs']:
        key = f"{record['clip']}/{record['camera']}"
        target = args.report / 'tracks' / variant / f'{key}.json'
        if target.exists():
            checked(json.loads(target.read_text())['arrays'])
            continue
        side = next(s for s in sides['clips'] if s['clip_id'] == record['clip'])
        turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
        camera = calibration(record, turns[record['camera']])
        frames = load_part_features(checked(native['records'][key]))[0] if encoder == KPR else load_features(checked(features['records'][CLIP][key]))[0]
        start = time.perf_counter()
        result: dict[str, Any] = {'status': 'ok', 'method': method, 'encoder': encoder, 'source_detection_rows': True}
        try:
            if method == LAB and encoder == CLIP:
                previous = json.loads((args.previous / 'tracks' / DEEP / f'{key}.json').read_text())
                raw, origins = read_tracks(previous, camera)
                result['online_source'] = previous['arrays']
            else:
                assignments = []
                tracker = StrongSort() if method == STRONG else DeepOCSortPose(record['video']['fps'])
                repeat = StrongSort() if method == STRONG else DeepOCSortPose(record['video']['fps'])
                for frame in frames:
                    first, second = tracker.update(frame), repeat.update(frame)
                    if not np.array_equal(first.track_ids, second.track_ids) or not np.array_equal(first.detection_rows, second.detection_rows):
                        raise ValueError('Tracker repeat changed source rows or IDs')
                    assignments.append(first)
                raw, origins = scatter_tracks(camera, frames, assignments, (1920, 1080))
                result['deterministic_repeat_equal'] = True
            result['online'] = save_tracks(target.with_suffix('.online.npz'), raw, origins)
            if method == LAB:
                raw, origins, result['links'] = lab_relink(raw, origins, record['video'])
            elif method == STRONG:
                assert af is not None
                roots, result['link_candidates'] = af.links(raw.boxes_xyxy, raw.observed)
                result['links'] = {str(k): v for k, v in roots.items() if k != v}
                raw, origins = merge_tracks(raw, origins, roots)
                smooth = gaussian_interpolation(raw.boxes_xyxy, raw.observed)
                path = target.with_suffix('.gsi.npz')
                with path.open('xb') as out:
                    np.savez_compressed(out, boxes=smooth.boxes, observed=smooth.observed,
                                        interpolated=smooth.interpolated, origins=origins, track_ids=raw.track_ids)
                result.update(gsi=record_file(path), interpolated_boxes=int(smooth.interpolated.sum()))
        except (ReconstructionUnavailable, InvalidPrediction) as error:
            result.update(status='failed', reason=str(error))
            raw = CameraTracks(camera, (1920, 1080), np.empty(0, np.int64),
                               np.zeros((0, len(frames), 4), np.float32), np.zeros((0, len(frames)), bool))
            origins = np.full(raw.observed.shape, -1, np.int64)
        result.update(arrays=save_tracks(target.with_suffix('.npz'), raw, origins), elapsed_seconds=time.perf_counter() - start,
                      identity_sha256=dual_sha256(args.report / 'identity.json'), track_count=len(raw.track_ids))
        write_json_atomic(target, result)
        print(f'{variant} {key}: {result["status"]} {len(raw.track_ids)} tracks', flush=True)


def save_units(path: Path, units: list[dict[str, Any]]) -> dict[str, Any]:
    with gzip.open(path, 'xt') as stream:
        for unit in units:
            stream.write(json.dumps(unit) + '\n')
    return dict(record_file(path))


def assess_variant(args: argparse.Namespace, variant: str, *, old: bool = False) -> None:
    from src.tasks.person_tracking.archive import load_features
    features, native, source = setup(args)
    identity = json.loads((args.report / 'identity.json').read_text())
    sides = json.loads(checked(identity['side']).read_text())
    config = load_association_config(CODE / 'src/tasks/player_association/configs/association.yaml', players_per_side=1)
    for clip in DEV_CLIPS:
        target = args.report / 'evaluation' / variant / clip / 'result.json'
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        records = sorted([r for r in source['inputs'] if r['clip'] == clip], key=lambda r: r['camera'])
        labels = ClipLabels.load(Path(records[0]['label_path']))
        side = next(s for s in sides['clips'] if s['clip_id'] == clip)
        turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
        previous = json.loads((args.previous / 'evaluation' / variant / clip / 'result.json').read_text()) if old else None
        result: dict[str, Any] = {'variant': variant, 'clip': clip, 'cameras': {}, 'association': {} if previous is None else previous['association'].copy()}
        units, group_units, originals, mappings = [], [], [], []
        all_groups: dict[str, list[CameraTracks]] = {e: [] for e in (CLIP, SOLIDER, KPR)}
        for record in records:
            cam, key = record['camera'], f"{clip}/{record['camera']}"
            camera = calibration(record, turns[cam])
            parent = args.previous if old else args.report
            saved = json.loads((parent / 'tracks' / variant / f'{key}.json').read_text())
            raw, origins = read_tracks(saved, camera)
            if previous is not None:
                entry = previous['cameras'][cam].copy()
                with np.load(checked(entry['arrays']), allow_pickle=False) as arrays:
                    mask, selection = arrays['selected'], entry['selection']
                    group = CameraTracks(camera, (1920, 1080), np.arange(len(arrays['group_boxes']), dtype=np.int64),
                                         arrays['group_boxes'], arrays['group_observed'])
                    group_origins = arrays['group_origins']
            else:
                frames = load_features(checked(features['records'][CLIP][key]))[0]
                raw = replace(raw, appearance=tuple(sampled_appearance(raw, origins, frames)))
                mask, selection = select_linked_candidates(raw, record['video']['fps'], LinkingConfig(), config.footpoints)
                group, group_origins = linked_timeline(raw, selection)
                path = target.parent / f'{cam}.npz'
                with path.open('xb') as out:
                    np.savez_compressed(out, track_ids=raw.track_ids, boxes=raw.boxes_xyxy, observed=raw.observed,
                                        selected=mask, group_origins=group_origins, group_boxes=group.boxes_xyxy, group_observed=group.observed)
                entry = {'tracking': saved, 'selection': selection, 'arrays': record_file(path)}
            local, unknown = tracking_units(raw, mask, labels)
            gu, group_unknown = tracking_units(group, group.observed, labels)
            if previous is not None and score_units(local) != previous['cameras'][cam]['metrics']:
                raise ValueError('Historical raw primary changed')
            entry.update(metrics=score_units(local), unlabelled_selected_boxes=unknown,
                         group_metrics=score_units(gu), group_unlabelled_selected_boxes=group_unknown)
            result['cameras'][cam] = entry
            units.extend(local)
            group_units.extend(gu)
            originals.append(raw)
            mappings.append(group_origins)
            if saved['source_detection_rows']:
                for encoder in ((KPR,) if old else (CLIP, SOLIDER, KPR)):
                    frames = load_part_features(checked(native['records'][key]))[0] if encoder == KPR else load_features(checked(features['records'][encoder][key]))[0]
                    appearance = tuple(sampled_appearance(raw, origins, frames))
                    linked, mapping = linked_timeline(replace(raw, appearance=appearance), selection)
                    if not np.array_equal(mapping, group_origins):
                        raise ValueError('Appearance substitution changed the fixed group timeline')
                    all_groups[encoder].append(linked)
        for encoder, candidates in all_groups.items():
            if not candidates:
                continue
            if len(candidates) != 3:
                raise ValueError('Association requires all three cameras')
            try:
                prediction = associate(candidates, records[0]['video']['fps'], config)
            except AssociationUndecided as error:
                result['association'][encoder] = {'status': 'undecided', 'reason': error.reason, 'diagnostics': error.diagnostics}
            else:
                predictions, arrays = {}, {}
                for raw, mapping, ids in zip(originals, mappings, prediction.player_ids, strict=True):
                    frame_ids = np.full(raw.observed.shape, -1, np.int64)
                    for row in range(len(ids)):
                        at = np.flatnonzero((ids[row] >= 0) & (mapping[row] >= 0))
                        frame_ids[mapping[row, at], at] = ids[row, at]
                    cam = raw.camera.camera_id
                    predictions[cam] = CameraPrediction(raw.track_ids, raw.boxes_xyxy, raw.observed, frame_ids)
                    arrays[cam] = frame_ids
                path = target.parent / f'{encoder}.identities.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
                result['association'][encoder] = {'status': 'ok', 'metrics': evaluate(labels, predictions),
                                                   'diagnostics': prediction.diagnostics, 'arrays': record_file(path)}
        result.update(units=save_units(target.parent / 'units.jsonl.gz', units), metrics=score_units(units),
                      group_units=save_units(target.parent / 'group_units.jsonl.gz', group_units), group_metrics=score_units(group_units))
        write_json_atomic(target, result)
        print(f'{variant}/{clip}: raw={result["metrics"]["idf1"]:.6f} group={result["group_metrics"]["idf1"]:.6f}', flush=True)


def summarize(args: argparse.Namespace, variants: tuple[str, ...]) -> list[dict[str, Any]]:
    tables, records = [], []
    for variant in variants:
        units: list[dict[str, Any]] = []
        groups: list[dict[str, Any]] = []
        for clip in DEV_CLIPS:
            path = args.report / 'evaluation' / variant / clip / 'result.json'
            result = json.loads(path.read_text())
            for key, destination in (('units', units), ('group_units', groups)):
                with gzip.open(checked(result[key]), 'rt') as stream:
                    destination.extend(json.loads(line) for line in stream)
            records.append({'variant': variant, 'clip': clip, 'result': record_file(path),
                            'association': result['association'], 'metrics': result['metrics'], 'group_metrics': result['group_metrics'],
                            'failed_cameras': [c for c, r in result['cameras'].items() if r['tracking']['status'] != 'ok']})
        for raw, group in zip(stratified_scores(units), stratified_scores(groups), strict=True):
            tables.append({'variant': variant, **raw, **{f'group_{k}': v for k, v in group.items() if k not in ('camera', 'near_far')}})
    write_csv(args.report / 'comparison.csv', tables)
    write_json_atomic(args.report / 'comparison.json', {'table': tables, 'records': records, 'identity': record_file(args.report / 'identity.json')})
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'features', 'kpr', 'aflink', 'previous', 'report'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--phase', choices=('base', 'kpr', 'report'), required=True)
    args = parser.parse_args()
    if psutil.virtual_memory().available < 6 * 1024**3:
        raise RuntimeError('Require >=6 GiB available RAM')
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    np.random.seed(0)
    torch.manual_seed(0)
    if args.phase == 'base':
        setup(args)
        for variant in VARIANTS:
            assess_variant(args, variant, old=True)
        for variant in (LAB, STRONG):
            track_variant(args, variant, variant, CLIP)
            assess_variant(args, variant)
        table = summarize(args, (*VARIANTS, LAB, STRONG))
        chosen = recommendation([r for r in table if r['variant'] in (DEEP, LAB, STRONG)])
        write_json_atomic(args.report / 'kpr_method.json', {'chosen': chosen, 'comparison': record_file(args.report / 'comparison.json'),
                                                          'rule': 'precommitted raw IDF1, switches, fragments, name'})
    elif args.phase == 'kpr':
        choice = json.loads((args.report / 'kpr_method.json').read_text())
        track_variant(args, NATIVE, choice['chosen'], KPR)
        assess_variant(args, NATIVE)
        summarize(args, (*VARIANTS, LAB, STRONG, NATIVE))
    else:
        summarize(args, (*VARIANTS, LAB, STRONG, NATIVE))


if __name__ == '__main__':
    main()
