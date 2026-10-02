"""Reserve long unseen association clips using metadata/history only; never decode.

One longest eligible clip per source video, >=600 frames. Prior design,
evaluation and pseudo-label calibration reports are all exclusion evidence,
including failed clips. A reservation with evaluation attempts cannot change.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256


def select_clips(manifests: list[dict[str, Any]], excluded: set[str]) -> tuple[list[str], list[dict[str, Any]]]:
    selected = []
    audit = []
    for video in sorted({m['clip_id'].split('/')[0] for m in manifests}):
        candidates = sorted((m for m in manifests if m['clip_id'].startswith(video + '/')),
                            key=lambda m: (-m['num_frames'], m['clip_id']))
        eligible = [m for m in candidates if m['num_frames'] >= 600 and m['clip_id'] not in excluded]
        if not eligible:
            raise ValueError(f'No >=600-frame unseen candidate for {video}')
        chosen = eligible[0]['clip_id']
        selected.append(chosen)
        for candidate in candidates:
            clip = candidate['clip_id']
            reason = ('prior_person_design_or_calibration' if clip in excluded else
                      'below_600_frames' if candidate['num_frames'] < 600 else
                      'longest_eligible_in_this_video' if clip == chosen else
                      'shorter_than_selected_clip_from_same_video')
            audit.append({'clip_id': clip, 'num_frames': candidate['num_frames'],
                          'selected': clip == chosen, 'reason': reason})
    return selected, audit


def reserve(repo: Path, report: Path) -> None:
    dataset = repo / 'data/tennis_multivew/processed/meiji_3cam/dataset'
    protocol_path = dataset / 'annotations/player_association/unseen_protocol.json'
    previous = json.loads(protocol_path.read_text())
    if previous['evaluation_attempts'] != 0 or previous['tuning_complete'] or previous['status'] != 'reserved_not_labelled':
        raise ValueError('Only an unlabelled, untuned reservation with zero evaluations may change')
    report.mkdir(parents=True, exist_ok=False)
    previous_hash = dual_sha256(protocol_path)
    (report / 'previous_protocol.json').write_bytes(protocol_path.read_bytes())
    excluded = set(previous['excluded_previously_observed_clips'])
    evidence = []
    for path in sorted((repo / 'outputs/player_association/evaluate').glob('*/*/*.json')):
        if path.name not in {'observe.json', 'calibration.json', 'evaluate.json', 'appearance.json'}:
            continue
        clips = sorted({c.replace('/clips/', '/') for c in re.findall(r'video_\d{3}/(?:clips/)?clip_\d{3}', path.read_text())})
        excluded.update(clips)
        evidence.append({'path': str(path), 'sha256': dual_sha256(path), 'mentioned_clips': clips})
    if not evidence:
        raise ValueError('No person design/calibration history found')
    paths = sorted(dataset.glob('videos/*/clips/*/clip.json'))
    manifests = [json.loads(p.read_text()) for p in paths]
    selected, audit = select_clips(manifests, excluded)
    records = {}
    for clip in selected:
        path = next(p for p, m in zip(paths, manifests, strict=True) if m['clip_id'] == clip)
        manifest = json.loads(path.read_text())
        if (path.parent / 'annotations/player_association/labels.json').exists():
            raise ValueError(f'Candidate already labelled: {clip}')
        if manifest['camera_ids'] != ['cam0', 'cam1', 'cam2']:
            raise ValueError(f'Expected three cameras: {clip}')
        videos = {}
        for camera, relative in zip(manifest['camera_ids'], manifest['video_paths'], strict=True):
            video = (path.parent / relative).resolve()
            videos[camera] = {'path': str(video), 'sha256': dual_sha256(video)}
        records[clip] = {'num_frames': manifest['num_frames'], 'fps': manifest['fps'],
                         'duration_s': manifest['num_frames'] / manifest['fps'], 'videos': videos,
                         'manifest_sha256': dual_sha256(path), 'reason': 'longest_eligible_in_this_video'}
    protocol = {**previous, 'schema': 'association_unseen_reservation_v2', 'clips': records,
        'previous_protocol_sha256': previous_hash, 'previous_protocol_snapshot': str(report / 'previous_protocol.json'),
        'excluded_previously_observed_clips': sorted(excluded), 'history_evidence': evidence,
        'candidate_audit': audit, 'reservation_method': 'metadata_only_longest_per_video_min_600_no_visual_inspection',
        'superseded_short_clips': sorted(set(previous['clips']) - set(selected)),
        'scope_limit': 'Unseen for person tracking/association design and calibration; same players/recordings. '
                       'Ball/court automatic processing is not person-identity tuning. No labels or model evaluation here.'}
    write_json_atomic(protocol_path, protocol)
    receipt = {'protocol': str(protocol_path), 'sha256': dual_sha256(protocol_path),
               'previous_sha256': previous_hash, 'selected': records, 'candidate_audit': audit,
               'evaluation_attempts': 0, 'history_evidence': evidence}
    write_json_atomic(report / 'receipt.json', receipt)
    print(json.dumps({k: {f: v[f] for f in ('num_frames', 'duration_s', 'reason')} for k, v in records.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    reserve(args.repo.resolve(), args.report.resolve())
