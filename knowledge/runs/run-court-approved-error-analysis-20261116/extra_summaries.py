"""Summarize observation factors, the untuned RGB pilot, and fixed-video selection."""
from collections import Counter, defaultdict
import csv
import json
import numpy as np
from common import BASE, OLD, CONDITIONS, correct, distribution, errors, source_rows, write_json


def main():
    out = BASE / 'report'
    tuples = defaultdict(Counter); point_counts = Counter(); observation_rows = []
    for path in sorted((BASE / 'observation-ablation').glob('*.json')):
        if path.name == 'complete.json': continue
        row = json.loads(path.read_text()); modes = {r['mode']: r for r in row['results']}
        pattern = '/'.join(str(int(modes[n]['valid'])) for n in ('neither', 'boundary_only', 'duplicate_only', 'both'))
        tuples[row['group']][pattern] += 1
        observation_rows.append({'key': row['key'], 'group': row['group'], **{n: modes[n]['status'] for n in modes}})
        for point in row['removed_point_evidence']:
            for reason in ('boundary', 'duplicate'):
                if point[reason]:
                    point_counts[row['group'], reason, 'all'] += 1
                    if point['gt_valid'] and point['within_existing_homography_threshold']: point_counts[row['group'], reason, 'valid_close_GT'] += 1
    write_json(out / 'observation-summary.json', {'cohort': '42 cases with O necessary in the preceding paired intervention',
        'success_tuple_order': ['neither', 'boundary_only', 'duplicate_only', 'both'],
        'success_tuples': {k: dict(v) for k, v in tuples.items()}, 'point_counts': {'/'.join(k): v for k, v in point_counts.items()},
        'limits': 'Each four-way comparison freezes its own LINE evidence. Four loss cases no longer reproduce the prior O effect; do not force an attribution.'})
    with (out / 'observation-cases.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(observation_rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(observation_rows)
    pilot = json.loads((BASE / 'rgb-pilot/results.json').read_text()); pilot_summary = {}
    for camera in ('all', 'cam0', 'cam1', 'cam2', 'cam3'):
        rows = [r for r in pilot if camera == 'all' or r['camera'] == camera]
        pilot_summary[camera] = {'cases': len(rows), 'before_correct': sum(r['old_correct'] for r in rows),
            'fitted': sum(r['matrix'] is not None for r in rows), 'after_correct': sum(r['correct'] for r in rows),
            'correct_lost': sum(r['old_correct'] and not r['correct'] for r in rows),
            'wrong_to_correct': sum(not r['old_correct'] and r['correct'] for r in rows),
            'previously_rejected_now_correct_geometry': sum(not r['old_accepted'] and r['correct'] for r in rows),
            'before_frame_mean_error_px': distribution([r['old_mean_px'] for r in rows]),
            'after_frame_mean_error_px': distribution([r['mean_error_px'] for r in rows if r['matrix'] is not None]),
            'fit_seconds': distribution([r['fit_seconds'] for r in rows]), 'decode_seconds': distribution([r['decode_seconds'] for r in rows])}
    write_json(out / 'rgb-summary.json', {'not_independent_validation': True, 'no_acceptance_rule_implemented': True, 'by_camera': pilot_summary})
    with (out / 'rgb-cases.csv').open('w', newline='') as f:
        keys = ['key', 'camera', 'old_accepted', 'old_correct', 'old_mean_px', 'status', 'correct', 'mean_error_px', 'max_error_px', 'fit_seconds', 'decode_seconds']
        w = csv.DictWriter(f, fieldnames=keys, lineterminator='\n'); w.writeheader(); w.writerows({k: r[k] for k in keys} for r in pilot)
    rows = [r for _, r in source_rows()]; video = []
    for condition in CONDITIONS:
        for camera in ('cam0', 'cam1', 'cam2', 'cam3'):
            candidates = [r for r in rows if r['camera'] == camera and r['conditions'][condition]['accepted']]
            for subset, predicate in [('all69', lambda n: True), ('first_third', lambda n: n <= 22), ('middle_third', lambda n: 23 <= n <= 45), ('last_third', lambda n: n >= 46), ('even', lambda n: n % 2 == 0), ('odd', lambda n: n % 2 == 1)]:
                available = [r for r in candidates if predicate(int(r['key'].split('/clip_')[1].split('/')[0]))]
                if not available:
                    video.append({'condition': condition, 'camera': camera, 'subset': subset, 'status': 'no_accepted_candidate', 'correct': False}); continue
                chosen = min(available, key=lambda r: (r['conditions'][condition]['quality']['score'], r['key']))
                selected = chosen['conditions'][condition]; e = errors(selected['points'], camera)
                video.append({'condition': condition, 'camera': camera, 'subset': subset, 'status': 'selected', 'available': len(available),
                    'key': chosen['key'], 'score': selected['quality']['score'], 'mean_error_px': float(e.mean()), 'max_error_px': float(e.max()), 'correct': correct(selected['points'], camera)})
    write_json(out / 'video-selection.json', {'selection_does_not_read_reference': True, 'method': 'minimum frozen quality cost among already-accepted clip outputs; deterministic key tie-break',
        'four_cameras_not_276_independent_trials': True, 'subsets_predeclared_without_tuning': True, 'results': video})
    geometry_times = {}
    for mode in ('legacy', 'current'):
        values = [json.loads(path.read_text())['seconds'] for path in (OLD / mode / 'meiji_4cam').glob('**/geometry.json')]
        geometry_times[mode] = distribution(values)
    write_json(out / 'timings.json', {'existing_meiji4_cpu_seconds_per_image_all_regions': geometry_times,
        'factorial_a1': json.loads((BASE / 'regression-factorial/complete.json').read_text()),
        'factorial_paired': json.loads((BASE / 'regression-paired/complete.json').read_text()),
        'observation_ablation': json.loads((BASE / 'observation-ablation/complete.json').read_text()),
        'rgb_pilot': json.loads((BASE / 'rgb-pilot/complete.json').read_text()),
        'new_RGB_scope': 'one model-selected initializer per image; compare to total existing region search only with this scope difference explicit'})
    print('RGB', pilot_summary['all'])
    print('C sharing', Counter((r['subset'], r['correct']) for r in video if r['condition'] == 'postprocess_and_acceptance'))


if __name__ == '__main__': main()
