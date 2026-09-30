"""Explicit whole-camera resumption; partial files never count as complete."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from association_recalibration_features import (  # type: ignore[import-not-found]
    CODE,
    checked,
    validate_plan,
)

from src.tasks.person_tracking.archive import load_features
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.definition import file_identity

# Only orchestration and the new, explicitly granted budget may change.
_CONTROL_FIELDS = frozenset(('benchmark', 'entrypoints', 'budget', 'resume'))


def verify_completed(plan: dict[str, Any], progress: dict[str, Any],
                     plan_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check all declared complete cameras, including exact source detection rows."""
    if progress['schema'] != 'i964_recalibration_features_v1' or progress['plan_sha256'] != plan_sha:
        raise ValueError('Completed cameras belong to a different feature plan')
    inputs = {r['key']: r for r in plan['inputs']}
    records, detections = progress['records'], progress['detections']
    if not set(records) <= set(detections) <= set(inputs):
        raise ValueError('Unknown camera or complete features without detections')
    for key, record in records.items():
        path = checked(record)
        frames, provenance = load_features(path)
        detection = detections[key]
        origin_sha = plan_sha
        if key in plan.get('resume', {}).get('reuse_keys', []):
            origin_sha = plan['resume']['source_plan']['sha256']
        expected = {'source': inputs[key]['video'], 'detection': detection, 'models': plan['models'],
                    'config': plan['feature_config'], 'encoder': plan['production_tracking']['encoder'],
                    'plan_sha256': origin_sha}
        if provenance != expected:
            raise ValueError(f'{key}: feature provenance differs')
        if len(frames) != inputs[key]['video']['num_frames'] or record['frames'] != len(frames) \
                or record['detections'] != sum(len(f.rows) for f in frames) or record['bytes'] != path.stat().st_size:
            raise ValueError(f'{key}: incomplete camera timeline/row count')
        with np.load(checked(detection), allow_pickle=False) as raw:
            if set(raw.files) != {'offsets', 'source_rows', 'boxes', 'scores'} or not np.array_equal(
                    raw['offsets'], np.cumsum([0, *[len(f.rows) for f in frames]], dtype=np.int64)):
                raise ValueError(f'{key}: detection timeline differs')
            for field, name in (('rows', 'source_rows'), ('boxes', 'boxes'), ('scores', 'scores')):
                if not np.array_equal(np.concatenate([getattr(f, field) for f in frames]), raw[name]):
                    raise ValueError(f'{key}: detection {field} differs')
    return copy.deepcopy(records), {key: copy.deepcopy(detections[key]) for key in records}


def resume_records(plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate the pinned source and the explicit reuse list on every launch."""
    if 'resume' not in plan:
        return {}, {}
    resume = plan['resume']
    source = json.loads(checked(resume['source_plan']).read_text())
    progress = json.loads(checked(resume['source_progress']).read_text())
    if {k: v for k, v in source.items() if k not in _CONTROL_FIELDS} != \
            {k: v for k, v in plan.items() if k not in _CONTROL_FIELDS}:
        raise ValueError('Resume changed feature identity, split or production code')
    records, detections = verify_completed(source, progress, resume['source_plan']['sha256'])
    if sorted(records) != resume['reuse_keys']:
        raise ValueError('Explicit reuse list differs from verified complete cameras')
    return records, detections


def plan_resume(repo: Path, report: Path, source: Path) -> None:
    """Keep the old plan's scientific identity; update only the audited producer."""
    if report.exists():
        raise FileExistsError('Resume requires a new output directory')
    source_plan, source_progress = source / 'plan.json', source / 'features.progress.json'
    old = json.loads(source_plan.read_text())
    if 'resume' in old:
        raise ValueError('This resume plan expects the original run-12 producer')
    progress = json.loads(source_progress.read_text())
    records, _ = verify_completed(old, progress, file_identity(source_plan)['sha256'])
    plan = copy.deepcopy(old)
    plan['benchmark'] = file_identity(CODE / 'tests/benchmarks/association_recalibration_features.py')
    plan['entrypoints'] = [file_identity(Path(r['path'])) for r in old['entrypoints']]
    plan['entrypoints'].append(file_identity(Path(__file__)))
    plan['budget'].update(wall_seconds=16200, expected_seconds=[4800, 12600],
                          expected_disk_bytes=500_000_000)
    plan['resume'] = {
        'source_plan': file_identity(source_plan), 'source_progress': file_identity(source_progress),
        'reuse_keys': sorted(records),
        'producer_changes': [{'before': r, 'after': file_identity(Path(r['path']))}
                             for r in [old['benchmark'], *old['entrypoints']]
                             if file_identity(Path(r['path'])) != r],
    }
    validate_plan(plan, repo)
    resume_records(plan)
    write_json_atomic(report / 'plan.json', plan)
    write_json_atomic(report / 'reuse.json', {
        'source_plan': plan['resume']['source_plan'], 'source_progress': plan['resume']['source_progress'],
        'verified_complete': [{'key': key, 'feature': records[key], 'detection': progress['detections'][key]}
                              for key in sorted(records)],
        'recompute': [r['key'] for r in plan['inputs'] if r['key'] not in records],
        'unlisted_partial_files_reused': False,
    })
    print(f'Resume verified: reuse {len(records)}, recompute {len(plan["inputs"]) - len(records)} cameras')
