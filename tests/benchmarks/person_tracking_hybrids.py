"""Run-10 fixed hybrids, exact historical score replay, and immutable CPU outputs."""
from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
from person_selection_cpu import calibration  # type: ignore[import-not-found]
from person_tracking_linking import (  # type: ignore[import-not-found]
    CLIP,
    CODE,
    DEEP,
    KPR,
    LAB,
    NATIVE,
    STRONG,
    VARIANTS,
    assess_variant,
    checked,
    inputs,
    merge_tracks,
    read_tracks,
    record_file,
    save_tracks,
    summarize,
)

from src.tasks.person_tracking.archive import load_features
from src.tasks.person_tracking.court_linking import LinkingConfig
from src.tasks.person_tracking.deep_ocsort_pose import (
    DeepOCSortPose,
    DeepOCSortPoseConfig,
)
from src.tasks.person_tracking.evaluation import score_units, tracking_units
from src.tasks.person_tracking.feature_tracks import scatter_tracks
from src.tasks.person_tracking.part_archive import load_part_features
from src.tasks.person_tracking.strongsort import (
    InvalidPrediction,
    StrongSort,
    StrongSortConfig,
)
from src.tasks.person_tracking.strongsort_offline import AFLink, gaussian_interpolation
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.evaluation.metrics import CameraPrediction, evaluate
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256

OLD = (*VARIANTS, LAB, STRONG, NATIVE)
STRONG_POSE = f'strongsort_pp_pose__{CLIP}'
DEEP_AF = f'deep_ocsort_pose_aflink_gsi__{CLIP}'
NEW = (STRONG_POSE, DEEP_AF)
PROTOCOL = CODE / 'knowledge/runs/run-i964-tracker-hybrids-r10-20260930/protocol-addendum.md'


def setup(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _, features, source = inputs(args)
    prior = json.loads((args.previous / 'comparison.json').read_text())
    previous = json.loads(checked(prior['identity']).read_text())
    for key, path in (('features', args.features / 'features.json'), ('native', args.kpr / 'features.json'),
                      ('plan', args.features / 'plan.json'), ('aflink', args.aflink)):
        if dual_sha256(path) != previous[key]['sha256']:
            raise ValueError(f'Run 9 input changed: {key}')
    if previous['selection'] != asdict(LinkingConfig()):
        raise ValueError('Fixed court selection changed')
    native = json.loads(checked(previous['native']).read_text())
    identity = {k: previous[k] for k in ('plan', 'features', 'native', 'aflink', 'side', 'selection', 'association')}
    identity.update(protocol=record_file(PROTOCOL), previous=record_file(args.previous / 'comparison.json'),
        previous_identity=prior['identity'], strongsort_pose=asdict(StrongSortConfig(pose_weight=.15)),
        deep_ocsort_pose=asdict(DeepOCSortPoseConfig()), association_encoders=[CLIP], variants=[*OLD, *NEW],
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip(),
        code={str(p.relative_to(CODE)): dual_sha256(p) for folder in ('src/tasks/person_tracking', 'src/tasks/player_association')
              for p in sorted((CODE / folder).rglob('*.py'))},
        benchmarks={name: record_file(CODE / 'tests/benchmarks' / name) for name in
                    ('person_tracking_hybrids.py', 'person_tracking_linking.py', 'person_tracking_matrix.py')})
    destination = args.report / 'identity.json'
    if destination.exists():
        if json.loads(destination.read_text()) != identity:
            raise ValueError('Run identity changed; retain outputs and use a new directory')
    else:
        write_json_atomic(destination, identity)
    return features, native, source


def assert_tracks(raw: CameraTracks, origins: np.ndarray, record: dict[str, Any]) -> None:
    with np.load(checked(record), allow_pickle=False) as saved:
        for key, value in (('track_ids', raw.track_ids), ('boxes', raw.boxes_xyxy), ('observed', raw.observed), ('origins', origins)):
            np.testing.assert_array_equal(value, saved[key], err_msg=f'Historical {key} changed: {record["path"]}')


def reproduce_scores(args: argparse.Namespace) -> None:
    """Re-match saved boxes/IDs and rescore saved association, without copying scores."""
    _, _, source = setup(args)
    previous = json.loads((args.previous / 'comparison.json').read_text())
    identity = json.loads((args.report / 'identity.json').read_text())
    sides = json.loads(checked(identity['side']).read_text())
    audit: list[dict[str, Any]] = []
    for record in previous['records']:
        variant, clip = record['variant'], record['clip']
        if variant not in OLD:
            raise ValueError('Unexpected historical condition')
        result = json.loads(checked(record['result']).read_text())
        side = next(s for s in sides['clips'] if s['clip_id'] == clip)
        turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
        sources = {r['camera']: r for r in source['inputs'] if r['clip'] == clip}
        labels = ClipLabels.load(Path(next(iter(sources.values()))['label_path']))
        all_units: list[dict[str, Any]] = []
        all_groups: list[dict[str, Any]] = []
        raw_tracks = {}
        for cam, entry in result['cameras'].items():
            camera = calibration(sources[cam], turns[cam])
            raw, origins = read_tracks(entry['tracking'], camera)
            with np.load(checked(entry['arrays']), allow_pickle=False) as arrays:
                for key, value in (('track_ids', raw.track_ids), ('boxes', raw.boxes_xyxy), ('observed', raw.observed)):
                    np.testing.assert_array_equal(value, arrays[key])
                group = CameraTracks(camera, (1920, 1080), np.arange(len(arrays['group_boxes']), dtype=np.int64),
                                     arrays['group_boxes'], arrays['group_observed'])
                local, unknown = tracking_units(raw, arrays['selected'], labels)
                grouped, group_unknown = tracking_units(group, group.observed, labels)
            if score_units(local) != entry['metrics'] or score_units(grouped) != entry['group_metrics']:
                raise ValueError(f'Historical camera metrics changed: {variant}/{clip}/{cam}')
            if unknown != entry['unlabelled_selected_boxes'] or group_unknown != entry['group_unlabelled_selected_boxes']:
                raise ValueError('Historical unknown boxes changed')
            if 'gsi' in entry['tracking']:
                with np.load(checked(entry['tracking']['gsi']), allow_pickle=False) as gsi:
                    np.testing.assert_array_equal(gsi['observed'], raw.observed)
                    np.testing.assert_array_equal(gsi['origins'], origins)
                    assert not (gsi['interpolated'] & raw.observed).any()
                    assert (origins[gsi['interpolated']] == -1).all()
            all_units.extend(local)
            all_groups.extend(grouped)
            raw_tracks[cam] = raw
        for key, values in (('units', all_units), ('group_units', all_groups)):
            with gzip.open(checked(result[key]), 'rt') as stream:
                saved_units = [json.loads(line) for line in stream]
            if values != saved_units:
                raise ValueError(f'Historical unit matching changed: {variant}/{clip}/{key}')
        if score_units(all_units) != result['metrics'] or score_units(all_groups) != result['group_metrics']:
            raise ValueError('Historical clip metrics changed')
        for encoder, association in result['association'].items():
            if association['status'] != 'ok':
                continue
            with np.load(checked(association['arrays']), allow_pickle=False) as arrays:
                predictions = {cam: CameraPrediction(raw.track_ids, raw.boxes_xyxy, raw.observed, arrays[cam])
                               for cam, raw in raw_tracks.items()}
            if evaluate(labels, predictions) != association['metrics']:
                raise ValueError(f'Historical pair evaluation changed: {variant}/{clip}/{encoder}')
        # Preserve historical selections, solver decisions and their evidence references.
        write_json_atomic(args.report / 'evaluation' / variant / clip / 'result.json', result)
        audit.append({'variant': variant, 'clip': clip, 'source': record['result'], 'raw_group_units_exact': True,
                      'camera_clip_metrics_exact': True, 'decided_association_metrics_exact': True,
                      'retained_undecided': [e for e, a in result['association'].items() if a['status'] != 'ok']})
        print(f'reproduced scores {variant}/{clip}', flush=True)
    table = summarize(args, OLD)
    if table != previous['table']:
        raise ValueError('Historical full stratified table differs')
    write_json_atomic(args.report / 'reproduction.json', {'status': 'ok', 'strata_exact': len(table), 'records': audit,
        'scope': 'Saved tracking/selection/group and association IDs; re-match all raw/group units and rescore all decided association encoders. Historical undecided decisions retained; no detector/feature/old baseline/solver rerun.'})


def reproduce_strongsort(args: argparse.Namespace) -> None:
    features, native, source = setup(args)
    identity = json.loads((args.report / 'identity.json').read_text())
    sides = json.loads(checked(identity['side']).read_text())
    for variant, encoder in ((STRONG, CLIP), (NATIVE, KPR)):
        for record in source['inputs']:
            key = f"{record['clip']}/{record['camera']}"
            target = args.report / 'regression' / variant / f'{key}.json'
            if target.exists():
                continue
            previous = json.loads((args.previous / 'tracks' / variant / f'{key}.json').read_text())
            frames = load_part_features(checked(native['records'][key]))[0] if encoder == KPR else load_features(checked(features['records'][CLIP][key]))[0]
            tracker = StrongSort()
            assignments = [tracker.update(frame) for frame in frames]
            side = next(s for s in sides['clips'] if s['clip_id'] == record['clip'])
            turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
            camera = calibration(record, turns[record['camera']])
            raw, origins = scatter_tracks(camera, frames, assignments, (1920, 1080))
            assert_tracks(raw, origins, previous['online'])
            write_json_atomic(target, {'status': 'ok', 'source': previous['online'], 'arrays_exact': True,
                                       'identity_sha256': dual_sha256(args.report / 'identity.json')})
            print(f'reproduced online {variant}/{key}', flush=True)


def track(args: argparse.Namespace) -> None:
    features, _, source = setup(args)
    identity = json.loads((args.report / 'identity.json').read_text())
    sides = json.loads(checked(identity['side']).read_text())
    af = AFLink(args.aflink)
    for variant in NEW:
        for record in source['inputs']:
            if psutil.virtual_memory().available < 6 * 1024**3:
                raise RuntimeError('Require >=6 GiB available RAM')
            key = f"{record['clip']}/{record['camera']}"
            target = args.report / 'tracks' / variant / f'{key}.json'
            if target.exists():
                checked(json.loads(target.read_text())['arrays'])
                continue
            side = next(s for s in sides['clips'] if s['clip_id'] == record['clip'])
            turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
            camera = calibration(record, turns[record['camera']])
            frames = load_features(checked(features['records'][CLIP][key]))[0]
            start = time.perf_counter()
            result: dict[str, Any] = {'status': 'ok', 'method': variant, 'encoder': CLIP, 'source_detection_rows': True}
            try:
                trackers = [StrongSort(StrongSortConfig(pose_weight=.15)) if variant == STRONG_POSE else
                            DeepOCSortPose(record['video']['fps']) for _ in range(2)]
                assignments = []
                for frame in frames:
                    first, second = [tracker.update(frame) for tracker in trackers]
                    np.testing.assert_array_equal(first.track_ids, second.track_ids)
                    np.testing.assert_array_equal(first.detection_rows, second.detection_rows)
                    assignments.append(first)
                raw, origins = scatter_tracks(camera, frames, assignments, (1920, 1080))
                result['deterministic_repeat_equal'] = True
                if variant == DEEP_AF:
                    prior = json.loads((args.previous / 'evaluation' / DEEP / record['clip'] / 'result.json').read_text())
                    reference = prior['cameras'][record['camera']]['tracking']['arrays']
                    assert_tracks(raw, origins, reference)
                    result['online_run9_exact'] = reference
                result['online'] = save_tracks(target.with_suffix('.online.npz'), raw, origins)
                roots, result['link_candidates'] = af.links(raw.boxes_xyxy, raw.observed)
                result['links'] = {str(k): v for k, v in roots.items() if k != v}
                raw, origins = merge_tracks(raw, origins, roots)
                smooth = gaussian_interpolation(raw.boxes_xyxy, raw.observed)
                np.testing.assert_array_equal(smooth.observed, raw.observed)
                assert not (smooth.interpolated & raw.observed).any()
                assert (origins[smooth.interpolated] == -1).all()
                path = target.with_suffix('.gsi.npz')
                with path.open('xb') as out:
                    np.savez_compressed(out, boxes=smooth.boxes, observed=smooth.observed,
                                        interpolated=smooth.interpolated, origins=origins, track_ids=raw.track_ids)
                result.update(gsi=record_file(path), interpolated_boxes=int(smooth.interpolated.sum()))
            except InvalidPrediction as error:
                result.update(status='failed', reason=str(error))
                raw = CameraTracks(camera, (1920, 1080), np.empty(0, np.int64),
                                   np.zeros((0, len(frames), 4), np.float32), np.zeros((0, len(frames)), bool))
                origins = np.full(raw.observed.shape, -1, np.int64)
            result.update(arrays=save_tracks(target.with_suffix('.npz'), raw, origins),
                          elapsed_seconds=time.perf_counter() - start,
                          identity_sha256=dual_sha256(args.report / 'identity.json'), track_count=len(raw.track_ids))
            write_json_atomic(target, result)
            print(f'{variant}/{key}: {result["status"]}, {len(raw.track_ids)} IDs', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'features', 'kpr', 'aflink', 'previous', 'report'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--phase', choices=('reproduce', 'regression', 'track', 'evaluate', 'report'), required=True)
    args = parser.parse_args()
    if psutil.virtual_memory().available < 6 * 1024**3:
        raise RuntimeError('Require >=6 GiB available RAM')
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    np.random.seed(0)
    torch.manual_seed(0)
    if args.phase == 'reproduce':
        reproduce_scores(args)
    elif args.phase == 'regression':
        reproduce_strongsort(args)
    elif args.phase == 'track':
        track(args)
    elif args.phase == 'evaluate':
        prepared = setup(args)
        for variant in NEW:
            assess_variant(args, variant, prepared=prepared, association_encoders=(CLIP,))
        summarize(args, (*OLD, *NEW))
    else:
        setup(args)
        summarize(args, (*OLD, *NEW))


if __name__ == '__main__':
    main()
