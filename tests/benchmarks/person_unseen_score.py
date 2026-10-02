"""Score immutable unseen predictions once, after independently reviewing box labels.

CPU only; never opens video or calls a model, fitting routine, or association.
The labels manifest maps exactly the three reserved clip IDs to labels.json.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from association_recalibration_dev import pooled  # type: ignore[import-not-found]
from person_unseen import checked, claim  # type: ignore[import-not-found]
from person_unseen_freeze import CAMERAS, CLIPS  # type: ignore[import-not-found]

from src.tasks.person_tracking.evaluation import (
    score_units,
    stratified_scores,
    tracking_units,
)
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_association.evaluation.metrics import CameraPrediction, evaluate
from src.tasks.player_detection.evaluation.person_sources import write_csv
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.tennis_scene.pipeline.definition import file_identity


def score(report: Path, labels_manifest: Path) -> None:
    destination = report / 'scoring'
    if destination.exists():
        raise FileExistsError('Unseen scoring is one batch; never tune or rescore')
    completed = json.loads((report / 'complete.json').read_text())
    if completed['status'] != 'ok' or set(completed['clips']) != set(CLIPS):
        raise ValueError('Require all three completed clips, including undecided predictions')
    plan = json.loads((report / 'plan.json').read_text())
    labelled = json.loads(labels_manifest.read_text())
    if set(labelled) != set(CLIPS):
        raise ValueError('Labels must cover exactly the three reserved clips')
    inputs = {}
    for record in plan['records']:
        clip = record['clip']
        label_path = Path(labelled[clip]).resolve()
        labels = ClipLabels.load(label_path)
        if labels.clip_id != clip or labels.num_frames != record['source']['videos'][0]['num_frames'] \
                or set(labels.cameras) != set(CAMERAS):
            raise ValueError('Labels differ from the frozen input axes')
        prediction = json.loads(checked(completed['clips'][clip]['prediction']).read_text())
        checked(prediction['arrays'])
        inputs[clip] = (labels, prediction, record)
    # No metric is computed before this durable, exclusive scoring claim.
    claim(destination / 'attempt.json', {'scoring_batch': 1, 'plan': file_identity(report / 'plan.json'),
        'completed': file_identity(report / 'complete.json'), 'labels_manifest': file_identity(labels_manifest),
        'labels': {c: file_identity(Path(labelled[c]).resolve()) for c in CLIPS}, 'retune_allowed': False})
    all_units: dict[str, list[dict[str, Any]]] = {'raw': [], 'group': [], 'associated': []}
    results = []
    for clip in CLIPS:
        labels, prediction, record = inputs[clip]
        metrics, predictions = {}, {}
        with np.load(checked(prediction['arrays']), allow_pickle=False) as saved:
            for camera in CAMERAS:
                # Frozen tracking_units reads only camera_id and the 2D track axes.
                # This evaluation-only adapter has NO synthetic calibration/side.
                raw = cast(CameraTracks, SimpleNamespace(camera=SimpleNamespace(camera_id=camera),
                    track_ids=saved[f'{camera}_track_ids'], boxes_xyxy=saved[f'{camera}_boxes'],
                    observed=saved[f'{camera}_observed']))
                group = cast(CameraTracks, SimpleNamespace(camera=SimpleNamespace(camera_id=camera),
                    track_ids=saved[f'{camera}_group_track_ids'], boxes_xyxy=saved[f'{camera}_group_boxes'],
                    observed=saved[f'{camera}_group_observed']))
                raw_units, raw_unknown = tracking_units(raw, saved[f'{camera}_selected'], labels)
                group_units, group_unknown = tracking_units(group, group.observed, labels)
                assigned = saved[f'{camera}_group_ids']
                identity_units = []
                for unit in group_units:
                    identity = -1 if unit['track_id'] is None else int(
                        assigned[int(np.flatnonzero(group.track_ids == unit['track_id'])[0]), unit['frame']])
                    identity_units.append({**unit, 'track_id': None if identity < 0 else identity})
                for name, units in (('raw', raw_units), ('group', group_units), ('associated', identity_units)):
                    all_units[name].extend(units)
                predictions[camera] = CameraPrediction(raw.track_ids, raw.boxes_xyxy, raw.observed, saved[f'{camera}_ids'])
                metrics[camera] = {'raw': score_units(raw_units), 'group': score_units(group_units),
                                   'associated': score_units(identity_units),
                                   'raw_unlabelled_selected': raw_unknown, 'group_unlabelled': group_unknown}
        result = {'clip': clip, 'status': prediction['status'], 'reason': prediction.get('reason'),
                  'metrics': evaluate(labels, predictions), 'tracking': metrics}
        results.append(result)
        write_json_atomic(destination / clip / 'result.json', result)
    for name, units in all_units.items():
        with gzip.open(destination / f'{name}-units.jsonl.gz', 'xt') as stream:
            for unit in units:
                stream.write(json.dumps(unit) + '\n')
        write_csv(destination / f'{name}-camera-near-far.csv', stratified_scores(units))
    write_json_atomic(destination / 'summary.json', {
        'status': 'ok', 'scoring_batches': 1, 'refit_or_retune': False, 'association': pooled(results),
        'tracking': {name: score_units(units) for name, units in all_units.items()},
        'clips': results,
        'scope': 'same dev partial-reference units/IoU=.5; raw IDs after selection, linked groups, final associated IDs; unknown separate',
        'side_convention': 'annotation-ball i932 view_half_turns from frozen plan; no court_side evaluation',
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    args = parser.parse_args()
    score(args.report.resolve(), args.labels.resolve())
