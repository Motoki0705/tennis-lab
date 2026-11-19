"""Exhaustive frozen-candidate accounting; diagnostic classes are not new gates."""
from collections import Counter, defaultdict
import csv
import json
import cv2
import numpy as np
from scipy.spatial.distance import cdist

from common import BASE, CONDITIONS, WORLD, correct, errors, source_hash, source_rows, truth, write_json

CATEGORIES = ('different_region_or_orientation', 'coherent_offset', 'localized_extrapolation', 'localized_supported_deformation', 'distributed_deformation')


def error_pattern(candidate, camera):
    predicted = np.asarray(candidate['points']); reference, mask = truth(camera)
    delta = predicted[mask] - reference[mask]; distance = np.linalg.norm(delta, axis=1)
    shift = np.median(delta, axis=0); centered = np.linalg.norm(delta - shift, axis=1)
    p = predicted[[0, 1, 3, 2]].astype(np.float32); r = reference[[0, 1, 3, 2]].astype(np.float32)
    intersection = float(cv2.intersectConvexConvex(p, r)[0]) if cv2.isContourConvex(p) else 0.
    iou = intersection / max(abs(cv2.contourArea(p)) + abs(cv2.contourArea(r)) - intersection, 1e-8)
    flipped = cv2.contourArea(p, oriented=True) * cv2.contourArea(r, oriented=True) <= 0
    removed = []
    for count in (1, 2, 3):
        order = np.argsort(distance); kept = distance[order[:-count]]
        if kept.mean() <= 10 and kept.max() <= 25:
            removed = np.flatnonzero(mask)[order[-count:]].tolist(); break
    selected = np.flatnonzero(candidate['selected'])
    outside = []
    if len(selected) >= 3:
        hull = cv2.convexHull(WORLD[selected].astype(np.float32))
        outside = [i for i in range(14) if cv2.pointPolygonTest(hull, tuple(WORLD[i]), False) < 0]
    if flipped or iou < .5: label = 'different_region_or_orientation'
    elif centered.mean() <= 10 and centered.max() <= 25: label = 'coherent_offset'
    elif removed and all(i in outside for i in removed): label = 'localized_extrapolation'
    elif removed: label = 'localized_supported_deformation'
    else: label = 'distributed_deformation'
    return {'pattern': label, 'mean_error_px': float(distance.mean()), 'max_error_px': float(distance.max()),
        'oracle_translation_xy': shift.tolist(), 'after_oracle_translation_mean_px': float(centered.mean()),
        'after_oracle_translation_max_px': float(centered.max()), 'court_polygon_iou': iou,
        'localized_indices': removed, 'outside_selected_world_hull': outside}


def gate_reasons(candidate, condition):
    if condition != 'legacy':
        reasons = list(candidate['quality']['reasons'])
        if not candidate['new_accepted'] and not reasons: reasons = ['geometry_or_visible_area_gate']
        return reasons
    points = np.asarray(candidate['points']); reasons = []
    if not ((points[:, 0] >= 0) & (points[:, 0] <= 1919) & (points[:, 1] >= 0) & (points[:, 1] <= 1079)).all(): reasons.append('all14_inside')
    if candidate['inliers'] < 8: reasons.append('inliers_below8')
    if candidate['status'] != 'ok': reasons.append('geometry_status')
    if not reasons and not candidate['legacy_accepted']: reasons.append('convex_or_area')
    return reasons


def main():
    out = BASE / 'candidate-analysis'; false_accepts = []; rescue = []; aggregate = {}; examples = {}; video = {}
    all_rows = source_rows()
    expected = json.loads((BASE.parent / 'court-reference-white-lines-20261116/evaluation/summary.json').read_text())
    assert source_hash() == expected['frozen_prediction_files_sha256']
    for camera in ('cam0', 'cam1', 'cam2', 'cam3'):
        aggregate[camera] = {}; video[camera] = {}
        for condition in CONDITIONS:
            counts = Counter(); patterns = Counter(); recovery = Counter(); reasons = Counter(); candidate_reasons = Counter(); accepted = []
            selected_rows = [r for _, r in all_rows if r['camera'] == camera]
            for row in selected_rows:
                pool = row['current_candidates' if condition == 'postprocess_and_acceptance' else 'legacy_candidates']
                c = row['conditions'][condition]; is_correct = correct(c['points'], camera)
                label = 'TP' if c['accepted'] and is_correct else 'FP' if c['accepted'] else 'FN' if is_correct else 'TN'; counts[label] += 1
                candidate = pool[c['region_index']]; assert candidate['region_index'] == c['region_index']
                if c['accepted']:
                    accepted.append({'key': row['key'], 'region_index': c['region_index'], 'correct': is_correct, 'points': c['points'], 'score': c['quality']['score']})
                if label == 'FP':
                    pattern = error_pattern(candidate, camera); patterns[pattern['pattern']] += 1
                    entry = {'key': row['key'], 'camera': camera, 'condition': condition, 'region_index': c['region_index'], **pattern}
                    false_accepts.append(entry)
                    examples.setdefault(camera, {}).setdefault(condition, {}).setdefault(pattern['pattern'], []).append(entry)
                good = [p for p in pool if correct(p['points'], camera)]
                gate = 'legacy_accepted' if condition == 'legacy' else 'new_accepted'
                passing = [p for p in good if p[gate]]
                original_available = any(correct(p['points'], camera) for p in row['legacy_candidates'])
                if good:
                    failure = 'recovered' if label == 'TP' else 'ranking_loss' if passing else 'all_correct_candidates_gate_rejected'
                    recovery[failure] += 1
                    all_reasons = set()
                    for p in good:
                        if p[gate]: continue
                        rr = gate_reasons(p, condition); all_reasons.update(rr); candidate_reasons.update(rr)
                    if failure == 'all_correct_candidates_gate_rejected': reasons.update(all_reasons)
                else: failure = 'no_correct_candidate'
                rescue.append({'key': row['key'], 'camera': camera, 'condition': condition, 'selected_label': label,
                    'legacy_pool_has_correct': original_available, 'correct_candidates': len(good), 'correct_gate_pass_candidates': len(passing),
                    'reason': failure, 'selected_region': c['region_index'], 'correct_regions': [p['region_index'] for p in good],
                    'correct_gate_pass_regions': [p['region_index'] for p in passing],
                    'correct_candidate_rejection_reasons': sorted({x for p in good if not p[gate] for x in gate_reasons(p, condition)})})
            for label in ('TP', 'FN', 'FP', 'TN'): assert counts[label] == expected['summary'][camera]['provisional'][condition][label]
            aggregate[camera][condition] = {'confusion': dict(counts), 'false_accept_patterns': dict(patterns), 'recovery': dict(recovery),
                'all_good_rejected_reason_images_overlapping': dict(reasons), 'correct_rejected_candidate_reasons_overlapping': dict(candidate_reasons)}
            if accepted:
                positions = np.asarray([a['points'] for a in accepted]); costs = cdist(positions.reshape(len(positions), -1), positions.reshape(len(positions), -1))
                choices = {'quality_min': min(range(len(accepted)), key=lambda i: accepted[i]['score']),
                    'medoid': int(np.argmin(costs.sum(axis=1)))}
                video[camera][condition] = {name: {k: accepted[i][k] for k in ('key', 'region_index', 'correct', 'score')} for name, i in choices.items()}
                video[camera][condition]['oracle_correct_accepted_exists'] = any(a['correct'] for a in accepted)
    # Freeze classification definitions; they describe error shape, never establish a causal label by themselves.
    definitions = {'priority': CATEGORIES, 'different_region_or_orientation': 'polygon IoU < 0.5 or opposite corner orientation; investigate semantic mismatch visually',
        'coherent_offset': 'subtracting the median 2D error vector makes mean<=10 and max<=25 (GT-only diagnostic)',
        'localized_extrapolation': 'removing at most 3 highest-error points restores agreed tolerance; all removed points outside selected support hull in court coordinates',
        'localized_supported_deformation': 'same sparse-error condition, but not confined to extrapolated support',
        'distributed_deformation': 'remaining nonuniform errors; cause not inferred from residual shape',
        'not_an_acceptance_rule': True, 'unsupported_causal_claims': 'neither a translated H nor low LINE residual proves correct semantic correspondence'}
    write_json(out / 'summary.json', {'source_sha256': source_hash(), 'reference_approved': True, 'definitions': definitions, 'by_camera': aggregate, 'video_shared_counterfactual': video})
    write_json(out / 'false-accepts.json', false_accepts); write_json(out / 'recovery.json', rescue)
    representative = []
    for camera, conditions in examples.items():
        for condition, classes in conditions.items():
            for label, entries in classes.items():
                ordered = sorted(entries, key=lambda x: x['mean_error_px']); representative.append(ordered[len(ordered) // 2])
    write_json(out / 'representatives.json', representative)
    for name, rows in [('false-accepts', false_accepts), ('recovery', rescue)]:
        with (out / (name + '.csv')).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader()
            writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows)
    for camera, conditions in aggregate.items():
        for condition, result in conditions.items(): print(camera, condition, result['false_accept_patterns'], result['recovery'])
    print('FP records', len(false_accepts), 'recovery records', len(rescue))


if __name__ == '__main__': main()
