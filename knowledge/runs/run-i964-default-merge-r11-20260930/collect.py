"""Promote run-11 evidence after auditing source rows, synthetic masks and video."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

BUNDLE = Path(__file__).resolve().parent
CODE = BUNDLE.parents[2]
RUN = Path('/home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/tracker_matrix/i964-default-merge-r11-20260930')
sys.path[:0] = [str(CODE), str(CODE / 'tests/benchmarks')]

from person_tracking_matrix import checked, record_file
from person_tracking_merge import CLIP, OFF, ON


def main():
    identity = json.loads((RUN / 'identity.json').read_text())
    code = {**identity['code'], 'tests/benchmarks/person_tracking_merge.py': identity['benchmark']['sha256']}
    for name, digest in code.items():
        blob = subprocess.check_output(['git', 'show', identity['commit'] + ':' + name], cwd=CODE)
        assert hashlib.sha256(blob).hexdigest() == digest, name
    comparison = json.loads((RUN / 'comparison.json').read_text())
    audit = json.loads((RUN / 'merge_audit.json').read_text())
    features = json.loads(checked(identity['features']).read_text())
    summary, changes = {}, []
    for record in comparison['records']:
        result = json.loads(checked(record['result']).read_text())
        counts = summary.setdefault(record['variant'], {'camera_conditions': 0, 'raw_observations': 0,
            'selected_observations': 0, 'synthetic': 0, 'aflink_merges': 0})
        for cam, camera in result['cameras'].items():
            tracking = camera['tracking']
            assert tracking['status'] == 'ok' and tracking['production_sampling_exact']
            if record['variant'] == OFF:
                assert tracking['run10_arrays_and_gsi_exact']
            key = record['clip'] + '/' + cam
            with np.load(checked(tracking['arrays']), allow_pickle=False) as f, np.load(checked(camera['arrays']), allow_pickle=False) as g:
                raw, group = dict(f), dict(g)
            with np.load(checked(features['records'][CLIP][key]), allow_pickle=False) as f:
                source = dict(f)
            np.testing.assert_array_equal(raw['origins'] >= 0, raw['observed'])
            for frame in range(raw['observed'].shape[1]):
                at = np.flatnonzero(raw['observed'][:, frame])
                rows = raw['origins'][at, frame]
                assert ((rows >= source['offsets'][frame]) & (rows < source['offsets'][frame + 1])).all()
                np.testing.assert_array_equal(raw['boxes'][at, frame], source['boxes'][rows])
            assert not (group['selected'] & ~raw['observed']).any()
            mapping = group['group_origins']
            assert len(mapping) <= 6
            np.testing.assert_array_equal(mapping >= 0, group['group_observed'])
            for i in range(len(mapping)):
                at = np.flatnonzero(mapping[i] >= 0)
                np.testing.assert_array_equal(group['group_boxes'][i, at], raw['boxes'][mapping[i, at], at])
            with np.load(checked(tracking['gsi']), allow_pickle=False) as gsi:
                np.testing.assert_array_equal(gsi['observed'], raw['observed'])
                np.testing.assert_array_equal(gsi['origins'], raw['origins'])
                assert not (gsi['interpolated'] & raw['observed']).any()
                assert (gsi['origins'][gsi['interpolated']] == -1).all()
                counts['synthetic'] += int(gsi['interpolated'].sum())
            counts['camera_conditions'] += 1
            counts['raw_observations'] += int(raw['observed'].sum())
            counts['selected_observations'] += int(group['selected'].sum())
            counts['aflink_merges'] += sum(len(ids) - 1 for ids in tracking['source_track_ids'])
            if record['variant'] == ON:
                old = json.loads((RUN / 'evaluation' / OFF / record['clip'] / 'result.json').read_text())['cameras'][cam]
                with np.load(checked(old['tracking']['arrays']), allow_pickle=False) as f, np.load(checked(old['arrays']), allow_pickle=False) as g:
                    before, selected = dict(f), g['selected']
                merges = [m for m in audit['records'] if m['clip'] == record['clip'] and m['camera'] == cam]
                changed_selected_frames = []
                for frame in range(len(source['offsets']) - 1):
                    left = set(before['origins'][selected[:, frame], frame].tolist())
                    right = set(raw['origins'][group['selected'][:, frame], frame].tolist())
                    if left != right:
                        changed_selected_frames.append(frame)
                changes.append({'clip': record['clip'], 'camera': cam,
                    'arrays_equal_off_on': {k: bool(np.array_equal(before[k], raw[k])) for k in raw},
                    'dropped_rows_emitted_in_off': sum(bool((before['origins'][:, m['frame']] == m['dropped_row']).any()) for m in merges),
                    'selected_source_row_change_frames': changed_selected_frames})
    review = json.loads((RUN / 'review.json').read_text())
    plan = json.loads(checked(identity['plan']).read_text())
    sources = json.loads(checked(plan['sources']).read_text())
    expected, screenshots = 0, []
    for w in review['windows']:
        video = next(r['video'] for r in sources['inputs'] if r['clip'] == w['clip'])
        changes_per_frame = np.zeros(video['num_frames'], np.int64)
        for m in audit['records']:
            if m['clip'] == w['clip']:
                changes_per_frame[m['frame']] += 1
        width = min(len(changes_per_frame), round(5 * video['fps']))
        sums = np.convolve(changes_per_frame, np.ones(width, np.int64), mode='valid')
        assert w['start'] == int(sums.argmax()) and w['end'] == w['start'] + width
        assert w['merged_boxes'] == int(sums[w['start']])
        captured = False
        for frame in range(w['start'], w['end']):
            if changes_per_frame[frame]:
                if not captured:
                    screenshots.append(expected)
                    captured = True
                expected += 3
            elif (frame - w['start']) % 4 == 0:
                expected += 1
    capture = cv2.VideoCapture(str(checked(review['video'])))
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape == (800, 1920, 3)
        if count in screenshots:
            cv2.imwrite(str(RUN / f'review-frame-{count}.jpg'), frame)
        count += 1
    capture.release()
    assert count == expected
    verification = {**json.loads((RUN / 'verification.json').read_text()), 'status': 'ok',
        'execution_commit': identity['commit'], 'code_files_match_commit': len(code), 'summary': summary,
        'off_on_changes': changes, 'merge_count': len(audit['records']), 'label_second_person_candidates': len(audit['suspects']),
        'video': {**record_file(RUN / 'review.mp4'), 'frames_read': count, 'windows_rule_verified': True,
                  'screenshot_frames': screenshots}}
    files = {str(p.relative_to(RUN)): record_file(p) for p in sorted(RUN.rglob('*')) if p.is_file()}
    verification['output_bytes'] = sum(r['bytes'] for r in files.values())
    (BUNDLE / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    (BUNDLE / 'output_manifest.json').write_text(json.dumps(files, indent=2) + '\n')
    for name in ('identity.json', 'comparison.json', 'comparison.csv', 'association.csv', 'availability.csv',
                 'correspondence.csv', 'recommendation.json', 'report.md', 'review.json', 'merge_counts.csv', 'merge_audit.json', 'merge_contacts.json'):
        shutil.copyfile(RUN / name, BUNDLE / name)
    shutil.copytree(RUN / 'merge_contacts', BUNDLE / 'merge_contacts', dirs_exist_ok=True)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
