"""CPU comparison of saved person sources; never infer a missing camera-clip.

FT archives precede ROI filtering; the historical COCO store follows it.
Outside-ROI COCO counts are therefore unknown, not zero. Unit agreement uses
the reviewed COCO-derived boxes and is not independent detection recall.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.submodules.models import PersonDetectionResult
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_detection.evaluation.far_archive import DetectionArchive
from src.tasks.player_detection.evaluation.far_player import (
    merge_extra,
    threshold,
    unit_rows,
)
from src.tennis_scene.pipeline.artifacts import json_value, write_json_atomic
from src.tennis_scene.pipeline.components.person_detection import PersonDetectionOutput
from src.tennis_scene.pipeline.storage.clip_store import ClipStore
from src.tennis_scene.pipeline.storage.codec import ArtifactCodec
from src.utils.checksum import dual_sha256

DEV_CLIPS = ('video_000/clip_000', 'video_000/clip_007', 'video_001/clip_001', 'video_002/clip_013')
MISSING_1080 = 'video_002/clip_013/cam2'
THRESHOLDS = (.01, .02, .05, .1, .3)


def roi_mask(boxes: NDArray[np.floating], roi: list[list[float]]) -> NDArray[np.bool_]:
    polygon = np.asarray(roi, np.float32)
    return np.asarray([cv2.pointPolygonTest(polygon, (float((b[0] + b[2]) / 2), float(b[3])), False) >= 0
                       for b in boxes], bool)


def pack(frames: list[PersonDetectionResult], milliseconds: NDArray[np.float64]) -> DetectionArchive:
    return DetectionArchive(np.r_[0, np.cumsum([len(f.scores) for f in frames])].astype(np.int64),
                            np.concatenate([f.boxes_xyxy for f in frames]),
                            np.concatenate([f.scores for f in frames]), milliseconds)


def source_variants(raw: dict[str, DetectionArchive], coco: DetectionArchive) -> dict[str, DetectionArchive]:
    result = {}
    for name, archive in raw.items():
        for score in THRESHOLDS:
            result[f'{name}_{score:.2f}'] = pack([threshold(archive.at(f), score)
                for f in range(len(archive.milliseconds))], archive.milliseconds)
    result['coco_0.30'] = coco
    ft = result['ft_base_0.30']
    # No small-box restriction: all available COCO people. FT wins overlaps;
    # scores from different detectors are never compared to choose a box.
    result['union_0.30'] = pack([merge_extra(ft.at(f), coco.at(f), dedup_iou=.5)
        for f in range(len(ft.milliseconds))], ft.milliseconds + coco.milliseconds)
    return result


def source_counts(archive: DetectionArchive, labels: ClipLabels, camera: str,
                  roi: list[list[float]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = {side: Counter() for side in ('all', 'near', 'far', 'unknown')}
    for frame in range(labels.num_frames):
        prediction = archive.at(frame)
        rows = unit_rows(prediction, labels.cameras[camera], labels.roles, frame)
        players = [row for row in rows if row['role'] == 'player']
        midpoint = sum(row['old_box'][3] for row in players) / 2 if len(players) == 2 and players[0]['old_box'][3] != players[1]['old_box'][3] else None
        inside = roi_mask(prediction.boxes_xyxy, roi)
        sides = np.full(len(inside), 'unknown', dtype='<U7') if midpoint is None else np.where(prediction.boxes_xyxy[:, 3] < midpoint, 'far', 'near')
        for side, counter in counts.items():
            keep = np.ones(len(inside), bool) if side == 'all' else sides == side
            counter.update(frames=1, persons=int(keep.sum()), inside=int((inside & keep).sum()), outside=int((~inside & keep).sum()))
        for row in rows:
            for side in ('all', row['near_far']):
                role = row['role']
                counts[side].update({f'{role}_units': 1, f'{role}_hit_05': int(row['matched']), f'{role}_hit_03': int(row['matched_iou03'])})
    return counts


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w') as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_sources(progress: Path, report: Path) -> dict[str, Any]:
    if (report / 'sources.json').exists() or (report / 'sources').exists():
        raise FileExistsError(f'Use a fresh report directory: {report}')
    plan = json.loads(progress.read_text())
    expected = {f'{clip}/{camera}' for clip in DEV_CLIPS for camera in ('cam0', 'cam1', 'cam2')}
    if {f"{r['clip']}/{r['camera']}" for r in plan['inputs']} != expected:
        raise ValueError('Only the fixed four dev clips may be used')
    reservation = Path(plan['reservation'])
    if dual_sha256(reservation) != plan['reservation_sha256'] or set(json.loads(reservation.read_text())['clips']) & set(DEV_CLIPS):
        raise ValueError('Unseen reservation changed or overlaps dev')
    if set(plan['archives']) != {'ft_base', 'ft_1080'} or set(plan['archives']['ft_base']) != expected \
            or set(plan['archives']['ft_1080']) != expected - {MISSING_1080}:
        raise ValueError('Expected the explicitly cancelled 12/12 and 11/12 archive set')
    comparison = Path(plan['comparison'])
    if dual_sha256(comparison) != plan['comparison_sha256']:
        raise ValueError('Historical comparison changed')
    historical = json.loads(comparison.read_text())
    report.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {'schema': 'person_sources_cpu_v1', 'progress': str(progress),
        'progress_sha256': dual_sha256(progress), 'comparison_sha256': dual_sha256(comparison),
        'reservation_sha256': dual_sha256(reservation), 'inputs': plan['inputs'], 'archives': {}, 'reuse': [],
        'missing': {'ft_1080': [MISSING_1080]}, 'thresholds': THRESHOLDS, 'weights': plan['weights'],
        'scope': 'FT: full saved frame; COCO: saved court ROI only; union: full FT plus ROI COCO',
        'interpretation': 'Reviewed COCO-derived old-box agreement, favouring COCO; not detection recall. Unlabelled persons are not FP.',
        'runtime_scope': 'FT: saved GPU forward only; COCO: historical component wall time incl load/decode/ROI/save; union sum is mixed-scope estimate, not measured union runtime.'}
    grouped: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    times: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0., 0., 0.])
    for record in plan['inputs']:
        camera, clip = record['camera'], record['clip']
        key = f'{clip}/{camera}'
        label = Path(record['label_path'])
        if dual_sha256(label) != record['label_sha256'] or dual_sha256(Path(record['video']['path'])) != record['video']['sha256']:
            raise ValueError(f'Source content changed: {key}')
        labels = ClipLabels.load(label)
        raw = {}
        for name, records in plan['archives'].items():
            if key in records:
                raw[name] = DetectionArchive.load(records[key])
                manifest['reuse'].append({'variant': name, 'key': key, **records[key], 'verified': True})
        old = historical['variants']['coco_person']['clips'][clip]
        root = Path(old['store'])
        store = ClipStore(root, json.loads((root / 'scene.json').read_text())['source'])
        ref = store.active(f'person_detection/{camera}')
        if ref is None or json_value(ref) != old['cameras'][camera]['artifact']:
            raise ValueError('COCO reference changed')
        coco = store.load(ref, ArtifactCodec(PersonDetectionOutput))
        coco_archive = DetectionArchive(coco.frame_offsets.astype(np.int64), coco.boxes_xyxy, coco.confidence,
            np.full(labels.num_frames, old['seconds'][f'person_detection/{camera}'] * 1000 / labels.num_frames, np.float64))
        manifest['reuse'].append({'variant': 'coco_0.30', 'key': key, 'store': str(root), 'reference': json_value(ref), 'verified': True})
        for name, archive in source_variants(raw, coco_archive).items():
            if len(archive.milliseconds) != labels.num_frames:
                raise ValueError(f'Incomplete timeline: {key}/{name}')
            saved = archive.save(report / 'sources' / name / f'{key}.npz')
            manifest['archives'].setdefault(name, {})[key] = saved
            counts = source_counts(archive, labels, camera, record['roi'])
            for coverage in ('available', 'common11'):
                if coverage == 'common11' and key == MISSING_1080:
                    continue
                for cam in (camera, 'all'):
                    for side, counter in counts.items():
                        grouped[(name, coverage, cam, side)].update(counter)
                    timing = times[(name, coverage, cam)]
                    timing[0] += float(archive.milliseconds.sum())
                    timing[1] += len(archive.milliseconds)
                    timing[2] += 1
        print(f'sources: {key}', flush=True)
    table = []
    for (name, coverage, camera, side), count in sorted(grouped.items()):
        ms, frames, clips = times[(name, coverage, camera)]
        row: dict[str, Any] = {'source': name, 'coverage': coverage, 'camera': camera, 'near_far': side,
            'camera_clips': int(clips), **count, 'persons_per_frame': count['persons'] / frames,
            'inside_per_frame': count['inside'] / frames,
            'outside_per_frame': None if name == 'coco_0.30' else count['outside'] / frames,
            'outside_scope': 'unavailable' if name == 'coco_0.30' else 'FT only; COCO unavailable' if name == 'union_0.30' else 'all saved FT',
            'ms_per_frame': ms / frames, 'runtime_kind': 'component_wall' if name == 'coco_0.30' else 'mixed_scope_sum' if name == 'union_0.30' else 'forward'}
        for role in ('player', 'non_player'):
            for iou in ('03', '05'):
                row[f'{role}_agreement_{iou}'] = count[f'{role}_hit_{iou}'] / count[f'{role}_units'] if count[f'{role}_units'] else None
        if name == 'coco_0.30':
            row['outside'] = None
        table.append(row)
    manifest['table'] = table
    write_json_atomic(report / 'sources.json', manifest)
    write_csv(report / 'sources.csv', table)
    return manifest
