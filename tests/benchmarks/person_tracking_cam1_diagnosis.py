"""Post-hoc cam1-far diagnosis under the run-9 protocol, never a tuning loop."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from person_selection_cpu import calibration  # type: ignore[import-not-found]
from person_tracking_linking import (  # type: ignore[import-not-found]
    CLIP,
    DEEP,
    checked,
    record_file,
)
from person_tracking_matrix_video import (  # type: ignore[import-not-found]
    load_result,
    states,
)

from src.tasks.person_tracking.archive import load_features
from src.tasks.person_tracking.deep_ocsort_pose import (
    DeepOCSortPose,
    DeepOCSortPoseConfig,
)
from src.tasks.person_tracking.evaluation import score_units, tracking_units
from src.tasks.person_tracking.feature_tracks import scatter_tracks
from src.tasks.person_tracking.pose_distance import local_pose, pose_distance
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_detection.evaluation.person_sources import DEV_CLIPS
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.geometry.bbox import pairwise_iou


def diagnose(args: argparse.Namespace) -> None:
    identity = json.loads((args.matrix / 'identity.json').read_text())
    plan = json.loads(checked(identity['plan']).read_text())
    source = json.loads(checked(plan['sources']).read_text())
    manifest = json.loads(checked(identity['features']).read_text())
    sides = json.loads(checked(identity['side']).read_text())
    result: dict[str, Any] = {'source_identity': record_file(args.matrix / 'identity.json'), 'scope': 'post-hoc diagnosis only; not candidate selection', 'clips': {}}
    candidates = []
    for clip in DEV_CLIPS:
        record = next(r for r in source['inputs'] if r['clip'] == clip and r['camera'] == 'cam1')
        baseline, units = load_result(args.matrix, DEEP, clip)
        state = states(baseline, units)
        far = [u for u in units if u['camera'] == 'cam1' and u['role'] == 'player' and u['near_far'] == 'far']
        errors = np.zeros(record['video']['num_frames'], np.int64)
        for unit in far:
            errors[unit['frame']] += state['cam1', unit['frame'], unit['person']] != 'correct'
        width = min(len(errors), round(5 * record['video']['fps']))
        for start, count in enumerate(np.convolve(errors, np.ones(width, np.int64), mode='valid')):
            candidates.append({'clip': clip, 'start': start, 'end': start + width, 'error_units': int(count)})
        result['clips'][clip] = {'far_units': len(far), 'states': dict(Counter(state['cam1', u['frame'], u['person']] for u in far)),
                                 'events': {key: [e for e in baseline['metrics'][key] if e['camera'] == 'cam1' and e['near_far'] == 'far']
                                            for key in ('switch_events', 'fragment_events')}}
    windows: list[dict[str, Any]] = []
    for window in sorted(candidates, key=lambda w: (-w['error_units'], w['clip'], w['start'])):
        if not any(window['clip'] == w['clip'] and max(window['start'], w['start']) < min(window['end'], w['end']) for w in windows):
            windows.append(window)
        if len(windows) == 3:
            break
    result['windows'] = windows
    for clip in DEV_CLIPS:
        record = next(r for r in source['inputs'] if r['clip'] == clip and r['camera'] == 'cam1')
        side = next(s for s in sides['clips'] if s['clip_id'] == clip)
        turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
        camera = calibration(record, turns['cam1'])
        frames = load_features(checked(manifest['records'][CLIP][f'{clip}/cam1']))[0]
        labels = ClipLabels.load(Path(record['label_path']))
        baseline, original_units = load_result(args.matrix, DEEP, clip)
        state = states(baseline, original_units)
        trace: list[dict[str, Any]] = []
        tracker = DeepOCSortPose(record['video']['fps'], trace=trace.append)
        assignments, internal = [], []
        for feature in frames:
            assignments.append(tracker.update(feature))
            internal.append({t.detection_row: {'id': t.identity, 'hit_streak': int(t.state.hit_streak),
                                               'time_since_update': int(t.state.time_since_update)}
                             for t in tracker.tracks if t.state.time_since_update == 0})
        raw, origins = scatter_tracks(camera, frames, assignments, (1920, 1080))
        with np.load(checked(baseline['cameras']['cam1']['tracking']['arrays']), allow_pickle=False) as saved:
            if not all(np.array_equal(a, saved[k]) for a, k in ((raw.track_ids, 'track_ids'), (raw.boxes_xyxy, 'boxes'),
                                                               (raw.observed, 'observed'), (origins, 'origins'))):
                raise ValueError('Traced rerun differs from run 8')
        directory = args.report / 'diagnosis' / clip
        directory.mkdir(parents=True, exist_ok=True)
        with gzip.open(directory / 'trace.jsonl.gz', 'xt') as output:
            for entry in trace:
                output.write(json.dumps(entry) + '\n')
        by_frame: dict[int, list[dict[str, Any]]] = {}
        for entry in trace:
            by_frame.setdefault(entry['frame'], []).append(entry)
        details: list[dict[str, Any]] = []
        previous: dict[str, Any] = {}
        for unit in [u for u in original_units if u['camera'] == 'cam1' and u['role'] == 'player' and u['near_far'] == 'far']:
            f = unit['frame']
            feature = frames[f]
            reference = labels.cameras['cam1']
            at = reference.at(f)
            person = next(i for i, p in enumerate(labels.people) if p.person_id == unit['person'])
            boxes = reference.boxes_xyxy[at][reference.person_index[at] == person]
            overlaps = pairwise_iou(feature.boxes, boxes).max(1)
            row = int(overlaps.argmax()) if len(overlaps) else None
            item: dict[str, Any] = {**unit, 'state': state['cam1', f, unit['person']], 'detection_iou': 0. if row is None else float(overlaps[row])}
            if row is not None and overlaps[row] >= .5:
                source_row = int(feature.rows[row])
                box = feature.boxes[row]
                pose = local_pose(box, feature.poses[row])
                prior = previous.get(unit['person'])
                item.update(detection_row=source_row, box=box.tolist(), height_px=float(box[3] - box[1]), score=float(feature.scores[row]),
                            pose_joints_ge03=int((feature.poses[row, :, 2] >= .3).sum()), internal=internal[f].get(source_row),
                            previous_reference_cosine=None if prior is None else float(feature.embeddings[row] @ prior['embedding']),
                            previous_reference_pose_distance=None if prior is None else pose_distance(pose, prior['pose']))
                previous[unit['person']] = {'embedding': feature.embeddings[row], 'pose': pose}
                emitted = {int(r): int(t) for r, t in zip(assignments[f].detection_rows, assignments[f].track_ids, strict=True)}
                item['emitted_id'] = emitted.get(source_row)
                comparisons = []
                for entry in by_frame.get(f, []):
                    if source_row not in entry['rows']:
                        continue
                    index = entry['rows'].index(source_row)
                    for j, tid in enumerate(entry['track_ids']):
                        competing = [{'row': int(other), 'iou': entry['iou'][k][j], 'similarity': entry['similarity'][k][j]}
                                     for k, other in enumerate(entry['rows']) if k != index]
                        comparisons.append({'stage': 'first' if entry['first'] else 'ocr', 'track_id': tid,
                                            **{k: entry[k][index][j] for k in ('iou', 'pose_cost', 'appearance_similarity', 'appearance_cost', 'similarity')},
                                            'proposed': [index, j] in [list(p) for p in entry['proposed']],
                                            'accepted': (source_row, tid) in [tuple(p) for p in entry['accepted']],
                                            'competing_detections': competing})
                item['matching'] = comparisons
            details.append(item)
        with gzip.open(directory / 'far_units.jsonl.gz', 'xt') as output:
            for entry in details:
                output.write(json.dumps(entry) + '\n')
        summary = result['clips'][clip]
        summary.update(trace=record_file(directory / 'trace.jsonl.gz'), details=record_file(directory / 'far_units.jsonl.gz'), rerun_equal=True)
        bins = {}
        for category in ('correct', 'wrong_identity', 'miss'):
            subset = [d for d in details if d['state'] == category]
            valid = [d for d in subset if 'detection_row' in d]
            bins[category] = {'units': len(subset), 'detection_available_iou05': len(valid),
                              'height_median': float(np.median([d['height_px'] for d in valid])) if valid else None,
                              'pose_joints_median': float(np.median([d['pose_joints_ge03'] for d in valid])) if valid else None,
                              'not_emitted': sum(d['emitted_id'] is None for d in valid),
                              'emitted_not_selected_or_different_match': sum(d['emitted_id'] is not None and d['track_id'] is None for d in valid)}
        summary['by_state'] = bins
        summary['counterfactuals'] = {}
        for name, cfg in (('pose_zero', replace(DeepOCSortPoseConfig(), pose_weight=0.)),
                          ('appearance_zero', replace(DeepOCSortPoseConfig(), appearance_weight=0., adaptive_weight=0.))):
            alternative = DeepOCSortPose(record['video']['fps'], cfg)
            alt, _ = scatter_tracks(camera, frames, [alternative.update(f) for f in frames], (1920, 1080))
            # All raw observations: these are diagnostics, not selected candidate metrics.
            units, _ = tracking_units(alt, alt.observed, labels)
            far = [u for u in units if u['near_far'] == 'far' and u['role'] == 'player']
            summary['counterfactuals'][name] = {'config': cfg.__dict__, 'far_raw_unselected': score_units(far)}
        raw_units, _ = tracking_units(raw, raw.observed, labels)
        summary['factual_far_raw_unselected'] = score_units([u for u in raw_units if u['near_far'] == 'far' and u['role'] == 'player'])
        write_json_atomic(args.report / 'diagnosis.progress.json', result)
        print(f'diagnosed {clip}: {summary["states"]}', flush=True)
    write_json_atomic(args.report / 'diagnosis.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    diagnose(parser.parse_args())
