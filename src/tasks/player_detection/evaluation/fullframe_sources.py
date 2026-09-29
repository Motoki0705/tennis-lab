"""Fixed-dev, pre-ROI source comparison; no detector invocation or GT selection."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_detection.evaluation.far_archive import DetectionArchive
from src.tasks.player_detection.evaluation.far_player import merge_extra, threshold
from src.tasks.player_detection.evaluation.person_sources import (
    DEV_CLIPS,
    pack,
    source_counts,
    write_csv,
)
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256

FT_THRESHOLDS = (.01, .02, .05)
COCO_THRESHOLDS = (.05, .10, .30)
SOURCES = (*(f'ft_base_{v:.2f}' for v in FT_THRESHOLDS),
           *(f'coco_fullframe_{v:.2f}' for v in COCO_THRESHOLDS), 'union_fullframe_0.30')


def fullframe_variants(ft: DetectionArchive, coco: DetectionArchive) -> dict[str, DetectionArchive]:
    if len(ft.milliseconds) != len(coco.milliseconds):
        raise ValueError('FT/COCO timelines differ')
    result = {}
    for name, archive, thresholds in (('ft_base', ft, FT_THRESHOLDS), ('coco_fullframe', coco, COCO_THRESHOLDS)):
        for score in thresholds:
            result[f'{name}_{score:.2f}'] = pack([threshold(archive.at(f), score)
                for f in range(len(archive.milliseconds))], archive.milliseconds)
    result['union_fullframe_0.30'] = pack([merge_extra(threshold(ft.at(f), .3), threshold(coco.at(f), .3), dedup_iou=.5)
        for f in range(len(ft.milliseconds))], ft.milliseconds + coco.milliseconds)
    return result


def summarize_fullframe(ft_progress: Path, coco_inference: Path, report: Path) -> None:
    if (report / 'sources.json').exists() or (report / 'sources').exists():
        raise FileExistsError(report)
    ft, coco = (json.loads(p.read_text()) for p in (ft_progress, coco_inference))
    expected = {f'{clip}/{cam}' for clip in DEV_CLIPS for cam in ('cam0', 'cam1', 'cam2')}
    if ft['baseline_size'] != [800, 1333] or ft['floor'] != .01:
        raise ValueError('FT baseline resize/floor differ')
    if coco['status'] != 'ok' or coco['scope'] != 'full_frame_no_roi' or coco['floor'] != .01 or coco['baseline_size'] != [800, 1333]:
        raise ValueError('COCO full-frame inference is incomplete or has different settings')
    for plan, name in ((ft, 'ft_base'), (coco, 'coco_fullframe_0.01')):
        if set(plan['archives'][name]) != expected or {f"{r['clip']}/{r['camera']}" for r in plan['inputs']} != expected:
            raise ValueError('Require all and only 12 fixed-dev camera-clips')
        reservation = Path(plan['reservation'])
        if dual_sha256(reservation) != plan['reservation_sha256'] or set(json.loads(reservation.read_text())['clips']) & set(DEV_CLIPS):
            raise ValueError('Unseen reservation changed/overlaps')
    if ft['comparison_sha256'] != coco['comparison_sha256'] or ft['reservation_sha256'] != coco['reservation_sha256']:
        raise ValueError('Source plans have different development inputs')
    report.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {'schema': 'fullframe_person_sources_v1', 'inputs': ft['inputs'], 'archives': {}, 'reuse': [],
        'ft_progress': str(ft_progress), 'ft_progress_sha256': dual_sha256(ft_progress),
        'coco_inference': str(coco_inference), 'coco_inference_sha256': dual_sha256(coco_inference),
        'reservation': ft['reservation'], 'reservation_sha256': ft['reservation_sha256'],
        'scope': 'all sources full frame before ROI; same 800/1333 resize; fixed 4 dev clips',
        'interpretation': 'Reference boxes come from COCO and favour COCO. Agreement is not detection recall. Unlabelled predictions are not FP.',
        'union': 'FT >=.30 first, add COCO >=.30 at IoU<.5; no cross-model score comparison, no ROI/height gate',
        'weights': {'ft': ft['weights'], 'coco': coco['weights']}, 'table': []}
    grouped: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for record in ft['inputs']:
        key = f"{record['clip']}/{record['camera']}"
        other = next(r for r in coco['inputs'] if f"{r['clip']}/{r['camera']}" == key)
        for field in ('video', 'label_sha256', 'calibration_reference'):
            if record[field] != other[field]:
                raise ValueError(f'Input mismatch: {key}/{field}')
        for path, sha in ((record['label_path'], record['label_sha256']), (record['video']['path'], record['video']['sha256'])):
            if dual_sha256(Path(path)) != sha:
                raise ValueError(f'Input changed: {path}')
        labels = ClipLabels.load(Path(record['label_path']))
        raw = []
        for plan, name in ((ft, 'ft_base'), (coco, 'coco_fullframe_0.01')):
            entry = plan['archives'][name][key]
            raw.append(DetectionArchive.load(entry))
            manifest['reuse'].append({'source': name, 'key': key, **entry, 'verified': True})
        for name, archive in fullframe_variants(*raw).items():
            if len(archive.milliseconds) != labels.num_frames:
                raise ValueError('Incomplete label timeline')
            manifest['archives'].setdefault(name, {})[key] = archive.save(report / 'sources' / name / f'{key}.npz')
            counts = source_counts(archive, labels, record['camera'], record['roi'])
            for camera in (record['camera'], 'all'):
                for side, count in counts.items():
                    grouped[name, camera, side].update(count)
        print(f'full-frame sources {key}', flush=True)
    for (name, camera, side), counts in sorted(grouped.items()):
        manifest['table'].append({'source': name, 'camera': camera, 'near_far': side, **counts,
            'persons_per_frame': counts['persons'] / counts['frames']})
    write_json_atomic(report / 'sources.json', manifest)
    write_csv(report / 'sources.csv', manifest['table'])
