"""Join the frozen audits into bounded tables; keep incompatible replays explicit."""
from collections import Counter, defaultdict
import csv
import itertools
import json
import numpy as np
from common import BASE, OLD, CONDITIONS, distribution, write_json


def main():
    out = BASE / 'report'; out.mkdir(exist_ok=True)
    assert (BASE / 'regression-paired/complete.json').exists()
    groups = defaultdict(list); case_rows = []; source_replay_mismatches = []; seed_failures = Counter(); cycles = Counter()
    for path in sorted((BASE / 'regression-paired').glob('*/*.json')):
        d = json.loads(path.read_text()); case = d['case']; variants = {r['variant']: r for r in d['results']}
        if case['observation_filter_inactive']:
            variants.update({name.replace('o1', 'o0'): row for name, row in list(variants.items())})
        for name, result in variants.items():
            groups[case['dataset'], case['group'], name].append(result)
        replay_old, replay_new = variants['o0s0v0'], variants['o1s1v1']
        mismatch = {name: case['frozen_status'][name] != value['status'] for name, value in [('legacy', replay_old), ('current', replay_new)]}
        if any(mismatch.values()): source_replay_mismatches.append({'key': case['key'], 'frozen': case['frozen_status'], 'paired_replay': {'legacy': replay_old['status'], 'current': replay_new['status']}})
        if case['group'] == 'lost':
            reproduced = replay_old['valid'] and not replay_new['valid']
            removals = [name for name, key in [('O', 'o0s1v1'), ('S', 'o1s0v1'), ('V', 'o1s1v0')] if variants[key]['valid']]
        elif case['group'] == 'gained':
            reproduced = not replay_old['valid'] and replay_new['valid']
            removals = [name for name, key in [('O', 'o0s1v1'), ('S', 'o1s0v1'), ('V', 'o1s1v0')] if not variants[key]['valid']]
        else: reproduced = replay_old['valid'] and replay_new['valid']; removals = []
        cause = '+'.join(removals) if reproduced and removals else 'joint_or_redundant_effect' if reproduced else 'transition_not_reproduced'
        old = json.loads((OLD / (case['dataset'] + '_legacy') / case['key'] / 'geometry.json').read_text())['candidates'][0]
        meta = next(r for r in json.loads((OLD / (case['dataset'] + '_raw') / 'manifest.json').read_text())['records'] if r['key'] == case['key'])
        gt = np.array(meta['gt_xy']); valid = np.array(meta['gt_valid'], bool)
        old_error = np.linalg.norm(np.array(old['projected'])[valid] - gt[valid], axis=1) if old['matrix'] is not None else np.array([])
        case_rows.append({'key': case['key'], 'dataset': case['dataset'], 'group': case['group'],
            'frozen_legacy_status': case['frozen_status']['legacy'], 'frozen_current_status': case['frozen_status']['current'],
            'replay_legacy_status': replay_old['status'], 'replay_current_status': replay_new['status'],
            'transition_reproduced': reproduced, 'single_factor_removal_explanation': cause,
            'old_mean_px': float(old_error.mean()) if len(old_error) else None,
            'old_max_px': float(old_error.max()) if len(old_error) else None,
            'mean_px_without_S': variants['o1s0v1']['mean_error_px'], 'valid_without_S': variants['o1s0v1']['valid'],
            'valid_without_O': variants['o0s1v1']['valid'], 'valid_without_V': variants['o1s1v0']['valid']})
        if case['dataset'] == 'val' and case['group'] == 'lost':
            seed_failures.update(r['reason'] for r in replay_new['seed_outcomes'])
            for end in replay_new['seed_outcomes']:
                if end['reason'] != 'selection_did_not_stabilize': continue
                trace = [r for r in replay_new['trace'] if r['seed'] == end['seed']]
                field = trace[-1]['field']; trace = [r for r in trace if r['field'] == field]
                sets = [tuple(r['fixed']) for r in trace] + [tuple(trace[-1]['updated'])]
                cycles['revisited_subset' if len(set(sets)) < len(sets) else 'all_subsets_distinct'] += 1
    table = []
    for key, rows in sorted(groups.items()):
        table.append({'dataset': key[0], 'cohort': key[1], 'variant': key[2], 'cases': len(rows),
            'valid': sum(r['valid'] for r in rows), 'statuses': dict(Counter(r['status'] for r in rows)),
            'valid_case_mean_error_px': distribution([r['mean_error_px'] for r in rows if r['valid']]),
            'solve_seconds': distribution([r['seconds'] for r in rows])})
    causes = {}
    for dataset in ('val', 'test'):
        causes[dataset] = {group: dict(Counter(r['single_factor_removal_explanation'] for r in case_rows if r['dataset'] == dataset and r['group'] == group)) for group in ('lost', 'gained', 'unchanged_control')}
    write_json(out / 'regression-summary.json', {'paired_LINE_evidence_frozen': True, 'variants': table, 'causes': causes,
        'val_lost_seed_failure_endings': dict(seed_failures), 'val_selection_nonconvergence': dict(cycles),
        'frozen_to_replay_status_mismatches': source_replay_mismatches,
        'limits': '221 cases: all 114 real-val losses, all 45 synthetic losses / 30 gains, 16 unchanged-success controls each. Full validation of a proposed implementation is not performed.'})
    with (out / 'regression-cases.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(case_rows)
    # Visibility is a separately inspected subset, not inferred from model confidence.
    with (BASE / 'raw-analysis/keypoint-observations.csv').open() as f: observations = list(csv.DictReader(f))
    labels = json.loads((BASE / 'occlusion-review/annotations.json').read_text())
    label_map = {(r['camera'], r['clip'], r['kp']): r['visibility_label'] for r in labels['points']}
    matched = []
    for row in observations:
        clip = int(row['key'].split('/clip_')[1].split('/')[0]); key = row['camera'], clip, int(row['kp'])
        if key not in label_map: continue
        for scope in ('full_frame', 'postprocess_and_acceptance'):
            if scope in row['scopes'].split('|'):
                matched.append({'key': row['key'], 'camera': row['camera'], 'kp': int(row['kp']), 'scope': scope,
                    'label': label_map[key], 'raw_valid': row['raw_valid'] == 'True', 'reference_inside_crop': row['reference_inside_crop'] == 'True',
                    'error_px': float(row['error_px']), 'score': float(row['score'])})
    visibility_stats = []
    for scope in ('full_frame', 'postprocess_and_acceptance'):
        for label in sorted(set(label_map.values())):
            rows = [r for r in matched if r['scope'] == scope and r['label'] == label]
            selected = [r for r in rows if r['raw_valid'] and r['reference_inside_crop']]
            visibility_stats.append({'scope': scope, 'label': label, 'points': len(rows), 'valid_in_crop': len(selected),
                'error_px': distribution([r['error_px'] for r in selected])})
    write_json(out / 'visibility-summary.json', {'annotation': labels['annotation'], 'stats': visibility_stats})
    with (out / 'visibility-observations.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(matched[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(matched)
    cases = json.loads((BASE / 'raw-analysis/cases.json').read_text()); raw_table = []
    for camera in ('cam0', 'cam1', 'cam2', 'cam3'):
        for scope in ('is_full_frame', 'is_selected_C'):
            rows = [r for r in cases if r['camera'] == camera and r[scope]]
            raw_table.append({'camera': camera, 'scope': scope, 'cases': len(rows),
                'existing_KP_only_correct': sum(r['existing_KP_only_correct'] for r in rows),
                'raw_frame_mean_error_px': distribution([r['raw_visible_error']['mean'] for r in rows]),
                'LINE_frame_median_error_px': distribution([r['line']['gt_to_nearest_unlabelled_ridge_px']['median'] for r in rows if r['line']['status'] == 'measured']),
                'LINE_frame_p95_error_px': distribution([r['line']['gt_to_nearest_unlabelled_ridge_px']['p95'] for r in rows if r['line']['status'] == 'measured'])})
    write_json(out / 'raw-head-summary.json', raw_table)
    print(json.dumps({'causes': causes, 'val_seed_failures': dict(seed_failures), 'nonconvergence': dict(cycles), 'status_mismatches': len(source_replay_mismatches)}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
