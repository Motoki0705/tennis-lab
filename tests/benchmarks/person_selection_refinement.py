"""CPU-only refinement on the fixed four dev clips, reusing run-4 tracks/CLIP."""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from person_selection_cpu import (  # type: ignore[import-not-found]  # sibling CLI
    calibration,
)

from src.tasks.person_tracking.court_linking import (
    LinkingConfig,
    select_linked_candidates,
)
from src.tasks.person_tracking.selection_diagnosis import diagnose_tracks
from src.tasks.person_tracking.selection_metrics import aggregate_units, selection_units
from src.tasks.player_association.appearance.sampling import TrackAppearance
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.geometry.footpoints import FootpointConfig
from src.tasks.player_detection.evaluation.person_sources import DEV_CLIPS, write_csv
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256

SOURCES = ('ft_base_0.01', 'union_0.30', 'old_pipeline')


def load_saved(record: dict[str, Any], camera: Any) -> tuple[CameraTracks, np.ndarray]:
    for prefix in ('', 'appearance_'):
        if dual_sha256(Path(record[prefix + 'path'])) != record[prefix + 'sha256']:
            raise ValueError('Run-4 selection/appearance archive changed')
    with np.load(record['path'], allow_pickle=False) as a:
        ids, boxes, seen, chosen = a['track_ids'], a['boxes'], a['observed'], a['chosen']
    appearance = [TrackAppearance(np.empty(0, np.int64), np.empty((0, 0), np.float32)) for _ in ids]
    with np.load(record['appearance_path'], allow_pickle=False) as a:
        for index, row in enumerate(chosen):
            appearance[row] = TrackAppearance(a[f'{index}/frames'], a[f'{index}/embeddings'])
    previous = np.zeros_like(seen)
    previous[chosen] = True
    return CameraTracks(camera, (1920, 1080), ids, boxes, seen, tuple(appearance)), previous


def evaluate(previous: Path, report: Path) -> None:
    if (report / 'selection.json').exists():
        raise FileExistsError(report)
    report.mkdir(parents=True, exist_ok=True)
    source = json.loads((previous / 'sources.json').read_text())
    old = json.loads((previous / 'selection.json').read_text())
    if old['sources_sha256'] != dual_sha256(previous / 'sources.json'):
        raise ValueError('Run-4 sources changed')
    reservation = json.loads(Path(source['progress']).read_text())['reservation']
    if dual_sha256(Path(reservation)) != source['reservation_sha256'] or set(json.loads(Path(reservation).read_text())['clips']) & set(DEV_CLIPS):
        raise ValueError('Unseen reservation changed/overlaps')
    expected = {f'{clip}/{cam}' for clip in DEV_CLIPS for cam in ('cam0', 'cam1', 'cam2')}
    if {f"{r['clip']}/{r['camera']}" for r in source['inputs']} != expected:
        raise ValueError('Expected exactly the 12 dev camera-clips')
    if dual_sha256(Path(old['side_decisions'])) != old['side_sha256']:
        raise ValueError('Side decisions changed')
    sides = json.loads(Path(old['side_decisions']).read_text())
    config = LinkingConfig()
    result: dict[str, Any] = {'previous': str(previous), 'previous_sha256': dual_sha256(previous / 'selection.json'),
        'sources_sha256': dual_sha256(previous / 'sources.json'), 'config': asdict(config),
        'encoder': old['encoder'], 'appearance_policy': 'reuse existing CLIP crops where available; missing explicitly marked, no new GPU/encoder run',
        'scope': source['scope'], 'inputs': source['inputs'], 'records': {}, 'table': [],
        'association': 'not rerun; existing person_identities v3, CLIP default, ambiguity/handoff rules unchanged'}
    units: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    diagnosis: dict[str, Any] = {'records': {}, 'table': []}
    with gzip.open(report / 'selection_units.jsonl.gz', 'wt') as handle:
        for record in source['inputs']:
            clip, cam = record['clip'], record['camera']
            if dual_sha256(Path(record['label_path'])) != record['label_sha256']:
                raise ValueError('Dev labels changed')
            labels = ClipLabels.load(Path(record['label_path']))
            side = next(s for s in sides['clips'] if s['clip_id'] == clip)
            if not side['annotation']['decided']:
                raise ValueError('Existing court side is undecided')
            turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
            camera = calibration(record, turns[cam])
            for name in SOURCES:
                saved = old['records'][name][clip]['cameras'][cam]
                tracks, old_mask = load_saved(saved, camera)
                diagnostic = diagnose_tracks(tracks, old_mask, labels)
                diagnosis['records'].setdefault(name, {})[f'{clip}/{cam}'] = diagnostic
                for row in diagnostic['tracks']:
                    diagnosis['table'].append({'source': name, 'clip': clip, 'camera': cam, **row})
                selected, linked = select_linked_candidates(tracks, record['video']['fps'], config, FootpointConfig())
                region, unlinked = select_linked_candidates(tracks, record['video']['fps'], config, FootpointConfig(), link_fragments=False)
                for stage, mask in (('old_rule', old_mask), ('region_only', region), ('linked_dwell', selected)):
                    current = selection_units(tracks, mask, labels)
                    units[name, stage].extend(current)
                    for unit in current:
                        handle.write(json.dumps({'source': name, 'stage': stage, **unit}) + '\n')
                path = report / 'selection' / name / clip / f'{cam}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as dst:
                    np.savez_compressed(dst, selected=selected, region_only=region)
                result['records'].setdefault(name, {})[f'{clip}/{cam}'] = {'path': str(path), 'sha256': dual_sha256(path),
                    'previous_path': saved['path'], 'previous_sha256': saved['sha256'],
                    'appearance_path': saved['appearance_path'], 'appearance_sha256': saved['appearance_sha256'],
                    'linked': linked, 'region_only': unlinked}
                print(f'{name} {clip}/{cam}: {linked["all_tracks"]} tracks -> {linked["selected_groups"]} candidates, {len(linked["links"])} links', flush=True)
            write_json_atomic(report / 'selection.progress.json', result)
    camera_table, identity_table, clip_table = [], [], []
    for (name, stage), rows in units.items():
        result['table'].append({'source': name, 'stage': stage, **aggregate_units(rows)})
        for clip in DEV_CLIPS:
            clip_table.append({'source': name, 'stage': stage, 'clip': clip, **aggregate_units([r for r in rows if r['clip'] == clip])})
        for camera in ('cam0', 'cam1', 'cam2'):
            for side in ('all', 'near', 'far', 'unknown'):
                subset = [r for r in rows if r['camera'] == camera and (side == 'all' or r['near_far'] == side)]
                camera_table.append({'source': name, 'stage': stage, 'camera': camera, 'near_far': side, **aggregate_units(subset)})
        for clip, person in sorted({(r['clip'], r['person']) for r in rows}):
            identity_table.append({'source': name, 'stage': stage, 'clip': clip, 'person': person,
                **aggregate_units([r for r in rows if (r['clip'], r['person']) == (clip, person)])})
    write_json_atomic(report / 'selection.json', result)
    write_json_atomic(report / 'diagnosis.json', diagnosis)
    for filename, rows in (('selection', result['table']), ('camera', camera_table), ('identity', identity_table),
                           ('clip', clip_table), ('diagnosis', diagnosis['table'])):
        write_csv(report / f'{filename}.csv', rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.previous, args.report)


if __name__ == '__main__':
    main()
