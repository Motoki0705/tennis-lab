"""One immutable old/new dev batch after the named config/evidence were pushed."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from association_recalibration_features import (  # type: ignore[import-not-found]
    CODE,
    checked,
)
from person_selection_cpu import calibration  # type: ignore[import-not-found]
from person_tracking_linking import read_tracks  # type: ignore[import-not-found]

from src.tasks.person_tracking.archive import load_features
from src.tasks.person_tracking.court_linking import (
    LinkingConfig,
    select_linked_candidates,
)
from src.tasks.person_tracking.evaluation import (
    score_units,
    stratified_scores,
    tracking_units,
)
from src.tasks.person_tracking.feature_tracks import sampled_appearance
from src.tasks.person_tracking.linked_timeline import linked_timeline
from src.tasks.person_tracking.sequence import TrackingConfig
from src.tasks.player_association.association.associate import (
    AssociationUndecided,
    CameraTracks,
    associate,
)
from src.tasks.player_association.association.config import (
    DEFAULT_CONFIG,
    load_association_config,
)
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.evaluation.metrics import CameraPrediction, evaluate
from src.tasks.player_detection.evaluation.person_sources import DEV_CLIPS, write_csv
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.definition import file_identity


def require_pushed(commit: str, config: Path, bundle: Path) -> None:
    """Read no labels until the exact accepted fit and named YAML are in origin."""
    remote = 'origin/campaign930/i964-2-tracking'
    subprocess.run(['git', 'merge-base', '--is-ancestor', commit, remote], cwd=CODE, check=True)
    for path in (config, bundle / 'fit.json', bundle / 'identity.json', bundle / 'pre-dev.json'):
        blob = subprocess.check_output(['git', 'show', f'{commit}:{path.relative_to(CODE)}'], cwd=CODE)
        if hashlib.sha256(blob).hexdigest() != file_identity(path)['sha256']:
            raise ValueError(f'Uncommitted calibration evidence/config: {path}')
    fit = json.loads((bundle / 'fit.json').read_text())
    gate = json.loads((bundle / 'pre-dev.json').read_text())
    if fit['status'] != 'accepted' or gate['dev_scoring_batches'] != 0 or gate['dev_labels_opened']:
        raise ValueError('Dev is closed for withheld or already-scored calibration')
    checked(json.loads((bundle / 'identity.json').read_text())['base_config'])
    if gate['proposed_config']['sha256'] != file_identity(config)['sha256']:
        raise ValueError('Named config differs from the accepted fit')


def project_ids(raw: CameraTracks, mapping: np.ndarray, ids: np.ndarray) -> np.ndarray:
    if mapping.shape != ids.shape or ids.shape[1] != raw.observed.shape[1]:
        raise ValueError('Association and group-origin timelines differ')
    frame_ids: np.ndarray = np.full(raw.observed.shape, -1, np.int64)
    for row in range(len(ids)):
        at = np.flatnonzero((ids[row] >= 0) & (mapping[row] >= 0))
        if (mapping[row, at] >= len(raw.track_ids)).any():
            raise ValueError('Group points outside raw tracks')
        if (frame_ids[mapping[row, at], at] >= 0).any():
            raise ValueError('Multiple identities assigned to one raw observation')
        frame_ids[mapping[row, at], at] = ids[row, at]
    if ((frame_ids >= 0) & ~raw.observed).any():
        raise ValueError('Identity assigned to synthetic/unobserved data')
    return frame_ids


def pooled(results: list[dict[str, Any]]) -> dict[str, Any]:
    """All clips including undecided: abstentions are explicit -1 predictions."""
    metrics = [r['metrics'] for r in results]
    output: dict[str, Any] = {'decided_clips': sum(r['status'] == 'ok' for r in results),
                              'total_clips': len(results)}
    for field in ('pairs', 'exclusion'):
        counts = {k: sum(m[field][k] for m in metrics) for k in ('tp', 'fp', 'fn')}
        denominator = 2 * counts['tp'] + counts['fp'] + counts['fn']
        output[field] = {**counts, 'f1': 2 * counts['tp'] / denominator if denominator else None}
    output['id_switch'] = {key: sum(m['id_switch'][key] for m in metrics)
                           for key in ('true', 'predicted', 'matched')}
    correct = sum(m['group_accuracy']['frames_correct'] for m in metrics)
    frames = sum(m['group_accuracy']['frames_scored'] for m in metrics)
    output['group_accuracy'] = {'frames_correct': correct, 'frames_scored': frames,
                                'accuracy': correct / frames if frames else None}
    output['coverage'] = {camera: {key: sum(m['coverage'][camera][key] for m in metrics)
                                   for key in metrics[0]['coverage'][camera]}
                          for camera in ('cam0', 'cam1', 'cam2')}
    output['stops'] = [{'clip': r['clip'], 'reason': r['reason']} for r in results if r['status'] != 'ok']
    return output


def run(args: argparse.Namespace) -> None:
    report: Path = args.report
    if report.exists():
        raise FileExistsError('Dev scoring is one immutable batch; do not rerun or refit')
    require_pushed(args.config_commit, args.config, args.bundle)
    tracks = json.loads(args.tracks.read_text())
    track_identity = json.loads(checked(tracks['identity']).read_text())
    reuse = json.loads(checked(track_identity['reuse']).read_text())
    old_identity = json.loads(checked(reuse['old_tracking']).read_text())
    if track_identity['profile'] != TrackingConfig().identity() or old_identity['profile'] != track_identity['profile']:
        raise ValueError('The fixed production tracker changed')
    for path, sha in old_identity['code'].items():
        checked({'path': str(CODE / path), 'sha256': sha})
    feature_plan = json.loads(checked(old_identity['plan']).read_text())
    sources = json.loads(checked(feature_plan['sources']).read_text())
    sides = json.loads(checked(reuse['side_source']).read_text())
    keys = {f'{c}/cam{i}' for c in DEV_CLIPS for i in range(3)}
    if tracks['status'] != 'ok' or set(tracks['records']) != keys \
            or {f'{r["clip"]}/{r["camera"]}' for r in sources['inputs']} != keys:
        raise ValueError('Dev must use exactly the fixed four clips and twelve cameras')
    configs = {'old': load_association_config(players_per_side=1),
               'fitted': load_association_config(args.config, players_per_side=1)}
    write_json_atomic(report / 'identity.json', {
        'config_commit': args.config_commit, 'configs': {'old': file_identity(DEFAULT_CONFIG), 'fitted': file_identity(args.config)},
        'fit': file_identity(args.bundle / 'fit.json'), 'tracks': file_identity(args.tracks),
        'reuse': track_identity['reuse'], 'sources': feature_plan['sources'], 'benchmark': file_identity(Path(__file__)),
        'protocol': 'one batch / no refit / all undecided retained as -1 / reserved clips closed'})
    results: dict[str, list[dict[str, Any]]] = {'old': [], 'fitted': []}
    all_units: list[dict[str, Any]] = []
    all_groups: list[dict[str, Any]] = []
    identity_units: dict[str, list[dict[str, Any]]] = {'old': [], 'fitted': []}
    prior_differences = []
    for clip in DEV_CLIPS:
        records = sorted((r for r in sources['inputs'] if r['clip'] == clip), key=lambda r: r['camera'])
        label_path = Path(records[0]['label_path'])
        checked({'path': str(label_path), 'sha256': records[0]['label_sha256']})
        labels = ClipLabels.load(label_path)
        side = next(s for s in sides['clips'] if s['clip_id'] == clip)
        if not side['annotation']['decided']:
            raise ValueError('Frozen dev side is not decided')
        turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
        raw_cameras, groups, mappings, camera_units = [], [], [], []
        audit = {}
        for record in records:
            camera, key = record['camera'], f'{clip}/{record["camera"]}'
            saved = tracks['records'][key]
            if saved['features'] != reuse['dev'][key]['reuse_features']:
                raise ValueError('Dev projected feature identity differs')
            descriptor = json.loads(checked(saved['track']).read_text())
            expected_identity = tracks['identity']['sha256'] if saved['action'] == 'cpu_retracked' else reuse['old_tracking']['sha256']
            if descriptor['status'] != 'ok' or descriptor['identity_sha256'] != expected_identity:
                raise ValueError('Incomplete or mismatched dev tracking descriptor')
            checked(descriptor['gsi'])
            raw, origins = read_tracks(descriptor, calibration(record, turns[camera]))
            features, _ = load_features(checked(saved['features']))
            raw = replace(raw, appearance=tuple(sampled_appearance(raw, origins, features)))
            mask, selection = select_linked_candidates(raw, record['video']['fps'], LinkingConfig(), configs['old'].footpoints)
            group, mapping = linked_timeline(raw, selection)
            local, unknown = tracking_units(raw, mask, labels)
            grouped_units, group_unknown = tracking_units(group, group.observed, labels)
            all_units.extend(local)
            all_groups.extend(grouped_units)
            camera_units.append(grouped_units)
            raw_cameras.append(raw)
            groups.append(group)
            mappings.append(mapping)
            audit[camera] = {'selection': selection, 'tracking': descriptor, 'raw_metrics': score_units(local),
                             'group_metrics': score_units(grouped_units), 'unlabelled_selected': unknown,
                             'unlabelled_group': group_unknown, 'action': saved['action']}
        write_json_atomic(report / clip / 'tracking.json', audit)
        for name, config in configs.items():
            result: dict[str, Any] = {'clip': clip, 'status': 'ok'}
            try:
                association = associate(groups, records[0]['video']['fps'], config)
            except AssociationUndecided as error:
                ids = [np.full(g.observed.shape, -1, np.int64) for g in groups]
                result.update(status='undecided', reason=error.reason, diagnostics=error.diagnostics)
            else:
                ids = association.player_ids
                result['diagnostics'] = association.diagnostics
            predictions: dict[str, CameraPrediction] = {}
            arrays: dict[str, Any] = {}
            for raw, group, mapping, frame_ids, units in zip(raw_cameras, groups, mappings, ids, camera_units, strict=True):
                predicted = project_ids(raw, mapping, frame_ids)
                camera = raw.camera.camera_id
                predictions[camera] = CameraPrediction(raw.track_ids, raw.boxes_xyxy, raw.observed, predicted)
                arrays[camera] = predicted
                for unit in units:
                    identity = -1 if unit['track_id'] is None else int(
                        frame_ids[int(np.flatnonzero(group.track_ids == unit['track_id'])[0]), unit['frame']])
                    identity_units[name].append({**unit, 'track_id': None if identity < 0 else identity})
            result['metrics'] = evaluate(labels, predictions)
            path = report / clip / f'{name}.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            result['arrays'] = file_identity(path)
            write_json_atomic(report / clip / f'{name}.json', result)
            results[name].append(result)
        prior = Path(reuse['old_tracking']['path']).parent / 'evaluation/strongsort_pp_pose_merge_off__clipreid_vitb16_market1501' / clip / 'result.json'
        historical = json.loads(prior.read_text())['association']['clipreid_vitb16_market1501']
        current = results['old'][-1]
        prior_differences.append({'clip': clip, 'historical': file_identity(prior),
                                 'same_status': historical['status'] == current['status'],
                                 'same_metrics': historical.get('metrics') == current.get('metrics'),
                                 'input_note': 'Two cameras were retracked after the approved nine-row production mask projection; run11 remains a distinct input reference.'})
        print(f'{clip}: old={results["old"][-1]["metrics"]["pairs"]} fitted={results["fitted"][-1]["metrics"]["pairs"]}', flush=True)
    for name, units in (('raw', all_units), ('group', all_groups), *identity_units.items()):
        with gzip.open(report / f'{name}-units.jsonl.gz', 'xt') as stream:
            for unit in units:
                stream.write(json.dumps(unit) + '\n')
        write_csv(report / f'{name}-camera-near-far.csv', stratified_scores(units))
    write_json_atomic(report / 'comparison.json', {
        'status': 'ok', 'summaries': {name: pooled(rows) for name, rows in results.items()},
        'records': {name: [{'clip': r['clip'], 'status': r['status'], 'metrics': r['metrics'],
                           'reason': r.get('reason')} for r in rows] for name, rows in results.items()},
        'tracking': {'raw': score_units(all_units), 'group': score_units(all_groups)},
        'historical_baseline_comparison': prior_differences, 'dev_scoring_batches': 1,
        'refit_after_dev': False, 'default_changed': False, 'reserved_clips_opened': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('report', 'config', 'bundle', 'tracks'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--config-commit', required=True)
    run(parser.parse_args())
