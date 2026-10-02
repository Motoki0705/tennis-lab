"""Apply the agreed nine-row projection and retrack two dev cameras on CPU.

No label reader or association scorer is imported. The other ten track archives
are checked and referenced unchanged. All old artifacts remain immutable.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
from association_recalibration_features import (  # type: ignore[import-not-found]
    CODE,
    checked,
    restrict_appearance,
)

from src.tasks.person_tracking.archive import load_features
from src.tasks.person_tracking.features import FeatureConfig
from src.tasks.person_tracking.sequence import TrackingConfig, track_sequence
from src.tasks.person_tracking.strongsort_offline import AFLink
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.definition import file_identity
from src.utils.checksum import dual_sha256

_CHANGED = {'video_000/clip_000/cam1': 1, 'video_001/clip_001/cam0': 8}
_FIELDS = ('rows', 'boxes', 'scores', 'poses', 'embeddings', 'appearance_valid')


def retrack(reuse_path: Path, report: Path) -> None:
    if report.exists():
        raise FileExistsError('Dev projection/retracking output is immutable')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reuse = json.loads(reuse_path.read_text())
    plan = json.loads(checked(reuse['plan']).read_text())
    old_identity = json.loads(checked(reuse['old_tracking']).read_text())
    if old_identity['profile'] != TrackingConfig().identity() or old_identity['profile'] != plan['production_tracking']:
        raise ValueError('Production tracking profile differs from the frozen run')
    for path, sha in old_identity['code'].items():
        checked({'path': str(CODE / path), 'sha256': sha})
    aflink = AFLink(checked(plan['models']['aflink']))
    changed = {k: len(r['changed_appearance_rows']) for k, r in reuse['dev'].items() if r['changed_appearance_rows']}
    if changed != _CHANGED or len(reuse['dev']) != 12:
        raise ValueError('Dev projection camera/row set differs from the approved audit')
    identity = {'reuse': file_identity(reuse_path), 'profile': TrackingConfig().identity(),
                'aflink': plan['models']['aflink'], 'benchmark': file_identity(Path(__file__))}
    write_json_atomic(report / 'identity.json', identity)
    results: dict[str, Any] = {}
    for key, record in reuse['dev'].items():
        if psutil.virtual_memory().available < 6 * 1024**3:
            raise RuntimeError('Require at least 6 GiB available host RAM')
        original, provenance = load_features(checked(record['source']))
        frames, _ = load_features(checked(record['reuse_features']))
        projected, count = restrict_appearance(original, FeatureConfig(**provenance['config']),
                                               FeatureConfig(**plan['feature_config']), (1920, 1080))
        if count != len(record['changed_appearance_rows']) or any(
                not np.array_equal(getattr(a, field), getattr(b, field))
                for a, b in zip(projected, frames, strict=True) for field in _FIELDS):
            raise ValueError(f'{key}: saved projection differs from the production mask')
        old = json.loads(checked(record['previous_track']).read_text())
        for field in ('arrays', 'gsi'):
            checked(old[field])
        if old['status'] != 'ok' or old['identity_sha256'] != reuse['old_tracking']['sha256']:
            raise ValueError('Saved dev track identity/status differs')
        if not count:
            results[key] = {'action': 'reuse_exact_input_track', 'track': record['previous_track'],
                            'features': record['reuse_features']}
            continue
        start = time.monotonic()
        sequence = track_sequence(frames, fps=provenance['source']['fps'], config=TrackingConfig(), aflink=aflink)
        smooth = sequence.reconstruction
        if smooth is None:
            raise ValueError('StrongSORT++ reconstruction is missing')
        origins = sequence.evidence.detection_rows
        if not np.array_equal(origins >= 0, sequence.observed) or (smooth.interpolated & sequence.observed).any() \
                or (origins[smooth.interpolated] != -1).any():
            raise ValueError('Synthetic GSI was promoted to a real observation')
        for f, features in enumerate(frames):
            for row in np.flatnonzero(sequence.observed[:, f]):
                positions = np.flatnonzero(features.rows == origins[row, f])
                if len(positions) != 1 or not np.array_equal(sequence.boxes[row, f], features.boxes[positions[0]]):
                    raise ValueError('Retracked box lost its exact source row')
        path = report / 'tracks' / f'{key}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {'track_ids': sequence.track_ids, 'boxes': sequence.boxes, 'observed': sequence.observed,
                  'origins': origins}
        with path.open('xb') as handle:
            np.savez_compressed(handle, track_ids=sequence.track_ids, boxes=sequence.boxes,
                                observed=sequence.observed, origins=origins)
        with np.load(path, allow_pickle=False) as saved:
            for field, value in arrays.items():
                np.testing.assert_array_equal(saved[field], value)
        gsi = path.with_suffix('.gsi.npz')
        with gsi.open('xb') as handle:
            np.savez_compressed(handle, boxes=smooth.boxes, observed=smooth.observed,
                                interpolated=smooth.interpolated, origins=origins, track_ids=sequence.track_ids)
        descriptor = path.with_suffix('.json')
        write_json_atomic(descriptor, {'status': 'ok', 'arrays': file_identity(path), 'gsi': file_identity(gsi),
            'features': record['reuse_features'], 'previous_track': record['previous_track'],
            'changed_appearance_rows': record['changed_appearance_rows'], 'source_track_ids': sequence.source_track_ids,
            'link_candidates': sequence.link_candidates, 'frames': len(frames), 'track_count': len(sequence.track_ids),
            'identity_sha256': dual_sha256(report / 'identity.json'), 'elapsed_seconds': time.monotonic() - start,
            'real_observations': int(sequence.observed.sum()), 'interpolated_boxes': int(smooth.interpolated.sum())})
        results[key] = {'action': 'cpu_retracked', 'track': file_identity(descriptor), 'features': record['reuse_features']}
        write_json_atomic(report / 'tracks.progress.json', results)
        print(f'{key}: retracked, projected rows={count}, seconds={time.monotonic() - start:.2f}', flush=True)
    write_json_atomic(report / 'tracks.json', {'status': 'ok', 'records': results, 'identity': file_identity(report / 'identity.json'),
        'retracked_cameras': sorted(_CHANGED), 'changed_rows': 9, 'dev_labels_opened': False, 'dev_scoring_executed': False})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    retrack(args.reuse.resolve(), args.report.resolve())


if __name__ == '__main__':
    main()
