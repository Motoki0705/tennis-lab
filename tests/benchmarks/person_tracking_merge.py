"""Run-11 CPU comparison through the shared production tracking entrypoint."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil  # type: ignore[import-untyped]
import torch
from person_selection_cpu import calibration  # type: ignore[import-not-found]
from person_tracking_hybrids import (  # type: ignore[import-not-found]
    STRONG_POSE,
    assert_tracks,
)
from person_tracking_linking import (  # type: ignore[import-not-found]
    CLIP,
    CODE,
    assess_variant,
    checked,
    inputs,
    record_file,
    save_tracks,
    summarize,
)

from src.tasks.person_tracking.archive import load_features
from src.tasks.person_tracking.court_linking import LinkingConfig
from src.tasks.person_tracking.duplicate_boxes import merge_person_boxes
from src.tasks.person_tracking.feature_tracks import (
    evidence_appearance,
    sampled_appearance,
)
from src.tasks.person_tracking.sequence import TrackingConfig, track_sequence
from src.tasks.person_tracking.strongsort import InvalidPrediction
from src.tasks.person_tracking.strongsort_offline import AFLink
from src.tasks.player_association.appearance.sampling import CropSamplingConfig
from src.tasks.player_association.association.associate import CameraTracks
from src.tasks.player_association.evaluation.labels import ClipLabels
from src.tasks.player_detection.evaluation.person_sources import write_csv
from src.tennis_scene.pipeline.artifacts import write_json_atomic
from src.utils.checksum import dual_sha256
from src.utils.geometry.bbox import pairwise_iou

OFF = f'strongsort_pp_pose_merge_off__{CLIP}'
ON = f'strongsort_pp_pose_merge_on__{CLIP}'
VARIANTS = (OFF, ON)
PROTOCOL = CODE / 'knowledge/runs/run-i964-default-merge-r11-20260930/protocol-addendum.md'


def setup(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _, features, source = inputs(args)
    previous = json.loads((args.previous / 'identity.json').read_text())
    for name, path in (('features', args.features / 'features.json'), ('aflink', args.aflink)):
        if dual_sha256(path) != previous[name]['sha256']:
            raise ValueError(f'Run 10 input changed: {name}')
    if previous['selection'] != asdict(LinkingConfig()):
        raise ValueError('Fixed court selection changed')
    identity = {k: previous[k] for k in ('plan', 'features', 'side', 'selection', 'association', 'aflink')}
    identity.update(protocol=record_file(PROTOCOL), previous=record_file(args.previous / 'comparison.json'),
        profile=TrackingConfig().identity(), variants=VARIANTS,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip(),
        code={str(p.relative_to(CODE)): dual_sha256(p) for folder in ('src/tasks/person_tracking', 'src/tasks/player_association',
                                                                  'src/tennis_scene/pipeline')
              for p in sorted((CODE / folder).rglob('*.py'))}, benchmark=record_file(Path(__file__)))
    target = args.report / 'identity.json'
    if target.exists():
        if json.loads(target.read_text()) != json.loads(json.dumps(identity)):
            raise ValueError('Run identity changed; choose a new immutable output directory')
    else:
        write_json_atomic(target, identity)
    return features, {}, source


def track(args: argparse.Namespace) -> None:
    features, _, source = setup(args)
    identity = json.loads((args.report / 'identity.json').read_text())
    sides = json.loads(checked(identity['side']).read_text())
    af = AFLink(args.aflink)
    for variant in VARIANTS:
        for record in source['inputs']:
            if psutil.virtual_memory().available < 6 * 1024**3:
                raise RuntimeError('Require >=6 GiB available RAM')
            key = f"{record['clip']}/{record['camera']}"
            target = args.report / 'tracks' / variant / f'{key}.json'
            if target.exists():
                checked(json.loads(target.read_text())['arrays'])
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            frames = load_features(checked(features['records'][CLIP][key]))[0]
            retained = []
            records: list[dict[str, Any]] = []
            for frame in frames:
                keep, merges = merge_person_boxes(frame.frame, frame.rows, frame.boxes, frame.scores, enabled=variant == ON)
                retained.append(replace(frame, rows=frame.rows[keep], boxes=frame.boxes[keep], scores=frame.scores[keep],
                                        poses=frame.poses[keep], embeddings=frame.embeddings[keep], appearance_valid=frame.appearance_valid[keep]))
                records.extend(asdict(r) for r in merges)
            merge_path = target.with_suffix('.merges.json')
            write_json_atomic(merge_path, {'clip': record['clip'], 'camera': record['camera'],
                'input_boxes': sum(len(f.rows) for f in frames), 'dropped_boxes': len(records),
                'changed_frames': len({r['frame'] for r in records}), 'records': records})
            side = next(s for s in sides['clips'] if s['clip_id'] == record['clip'])
            turns = dict(zip(side['camera_ids'], side['annotation']['view_half_turns'], strict=True))
            camera = calibration(record, turns[record['camera']])
            start = time.perf_counter()
            result: dict[str, Any] = {'status': 'ok', 'method': variant, 'encoder': CLIP,
                                     'source_detection_rows': True, 'merges': record_file(merge_path)}
            try:
                sequence = track_sequence(retained, fps=record['video']['fps'], config=TrackingConfig(), aflink=af)
                raw = CameraTracks(camera, (1920, 1080), sequence.track_ids, sequence.boxes, sequence.observed)
                origins = sequence.evidence.detection_rows
                # The standard selection path reuses exactly the same sampled features as evaluation.
                cached = evidence_appearance(raw.boxes_xyxy, raw.observed, sequence.evidence, raw.image_size, CropSamplingConfig())
                reference = sampled_appearance(raw, origins, frames)
                for left, right in zip(cached, reference, strict=True):
                    np.testing.assert_array_equal(left.frames, right.frames)
                    np.testing.assert_array_equal(left.embeddings, right.embeddings)
                result['production_sampling_exact'] = True
                assert sequence.reconstruction is not None
                smooth = sequence.reconstruction
                assert not (smooth.interpolated & raw.observed).any()
                assert (origins[smooth.interpolated] == -1).all()
                if variant == OFF:
                    previous = json.loads((args.previous / 'tracks' / STRONG_POSE / f'{key}.json').read_text())
                    assert_tracks(raw, origins, previous['arrays'])
                    with np.load(checked(previous['gsi']), allow_pickle=False) as prior:
                        for key_name, value in (('boxes', smooth.boxes), ('observed', smooth.observed), ('interpolated', smooth.interpolated)):
                            np.testing.assert_array_equal(value, prior[key_name])
                    result['run10_arrays_and_gsi_exact'] = True
                path = target.with_suffix('.gsi.npz')
                with path.open('xb') as out:
                    np.savez_compressed(out, boxes=smooth.boxes, observed=smooth.observed,
                                        interpolated=smooth.interpolated, origins=origins, track_ids=raw.track_ids)
                result.update(gsi=record_file(path), interpolated_boxes=int(smooth.interpolated.sum()),
                              source_track_ids=sequence.source_track_ids, link_candidates=sequence.link_candidates)
            except InvalidPrediction as error:
                result.update(status='failed', reason=str(error))
                raw = CameraTracks(camera, (1920, 1080), np.empty(0, np.int64),
                                   np.zeros((0, len(frames), 4), np.float32), np.zeros((0, len(frames)), bool))
                origins = np.full(raw.observed.shape, -1, np.int64)
            result.update(arrays=save_tracks(target.with_suffix('.npz'), raw, origins),
                          elapsed_seconds=time.perf_counter() - start, track_count=len(raw.track_ids),
                          identity_sha256=dual_sha256(args.report / 'identity.json'))
            write_json_atomic(target, result)
            print(f'{variant}/{key}: {result["status"]}, merged {len(records)}', flush=True)


def audit(args: argparse.Namespace) -> None:
    features, _, source = setup(args)
    counts, suspects, audit_rows = [], [], []
    for record in source['inputs']:
        key = f"{record['clip']}/{record['camera']}"
        tracking = json.loads((args.report / 'tracks' / ON / f'{key}.json').read_text())
        merges = json.loads(checked(tracking['merges']).read_text())
        counts.append({k: v for k, v in merges.items() if k != 'records'})
        labels = ClipLabels.load(Path(record['label_path']))
        reference = labels.cameras[record['camera']]
        frames = load_features(checked(features['records'][CLIP][key]))[0]
        for merge in merges['records']:
            frame = frames[merge['frame']]
            positions = [int(np.flatnonzero(frame.rows == merge[k])[0]) for k in ('kept_row', 'dropped_row')]
            at = reference.at(frame.frame)
            valid = reference.person_index[at] >= 0
            people, boxes = reference.person_index[at][valid], reference.boxes_xyxy[at][valid]
            overlap = pairwise_iou(frame.boxes[positions], boxes)
            matched = [{int(person): float(overlap[i, people == person].max()) for person in np.unique(people)
                        if overlap[i, people == person].max() >= .5} for i in range(2)]
            best = [max(m, key=m.__getitem__) if m else None for m in matched]
            distinct_best = all(p is not None for p in best) and best[0] != best[1]
            drop_only = sorted(set(matched[1]) - set(matched[0]))
            row = {**merge, 'clip': record['clip'], 'camera': record['camera'],
                   'kept_box': frame.boxes[positions[0]].tolist(), 'dropped_box': frame.boxes[positions[1]].tolist(),
                   'matched_people': [{labels.people[p].person_id: v for p, v in m.items()} for m in matched],
                   'distinct_best': distinct_best, 'drop_only_people': [labels.people[p].person_id for p in drop_only]}
            audit_rows.append(row)
            if distinct_best or drop_only:
                suspects.append(row)
    write_csv(args.report / 'merge_counts.csv', counts)
    write_json_atomic(args.report / 'merge_audit.json', {'counts': counts, 'suspects': suspects, 'records': audit_rows,
        'limitation': 'Partial historical box labels cannot rule out unlabelled second people. Suspects require visual review.'})
    print(f'{len(audit_rows)} merges; {len(suspects)} distinct-person label candidates', flush=True)


def evaluate(args: argparse.Namespace) -> None:
    prepared = setup(args)
    for variant in VARIANTS:
        assess_variant(args, variant, prepared=prepared, association_encoders=(CLIP,))
    table = summarize(args, VARIANTS)
    previous = json.loads((args.previous / 'comparison.json').read_text())
    prior = [{k: v for k, v in row.items() if k != 'variant'} for row in previous['table'] if row['variant'] == STRONG_POSE]
    current = [{k: v for k, v in row.items() if k != 'variant'} for row in table if row['variant'] == OFF]
    if current != prior:
        raise ValueError('Merge-off metrics differ from run 10')
    for record in previous['records']:
        if record['variant'] == STRONG_POSE:
            result = json.loads((args.report / 'evaluation' / OFF / record['clip'] / 'result.json').read_text())
            saved = json.loads(checked(record['result']).read_text())
            if result['association'][CLIP]['metrics'] != saved['association'][CLIP]['metrics']:
                raise ValueError('Merge-off association metrics differ from run 10')
    write_json_atomic(args.report / 'verification.json', {'run10_off_raw_group_all_strata_exact': True,
        'run10_off_pair_metrics_exact': True, 'shared_production_entrypoint': 'track_sequence', 'strata': len(table)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'features', 'aflink', 'previous', 'report'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--phase', choices=('track', 'evaluate', 'audit'), required=True)
    args = parser.parse_args()
    if psutil.virtual_memory().available < 6 * 1024**3:
        raise RuntimeError('Require >=6 GiB available RAM')
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    {'track': track, 'evaluate': evaluate, 'audit': audit}[args.phase](args)


if __name__ == '__main__':
    main()
