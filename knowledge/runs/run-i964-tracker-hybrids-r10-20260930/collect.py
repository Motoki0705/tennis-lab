"""Audit and promote run 10: exact rows, masks, historical scores and video windows."""
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
RUN = Path('/home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/tracker_matrix/i964-hybrids-r10-20260930')
sys.path[:0] = [str(CODE), str(CODE / 'tests/benchmarks')]

from person_tracking_hybrids import DEEP, DEEP_AF, NEW, OLD
from person_tracking_matrix import checked, record_file
from person_tracking_matrix_video import load_result, recommendation, states


def main():
    identity = json.loads((RUN / 'identity.json').read_text())
    code = dict(identity['code'])
    for name, record in identity['benchmarks'].items():
        code['tests/benchmarks/' + name] = record['sha256']
    for name, digest in code.items():
        blob = subprocess.check_output(['git', 'show', identity['commit'] + ':' + name], cwd=CODE)
        assert hashlib.sha256(blob).hexdigest() == digest, name
    previous = json.loads(checked(identity['previous']).read_text())
    comparison = json.loads((RUN / 'comparison.json').read_text())
    assert [r for r in comparison['table'] if r['variant'] in OLD] == previous['table']
    assert len({r['variant'] for r in comparison['table']}) == 11
    regression = [json.loads(p.read_text()) for p in sorted((RUN / 'regression').rglob('*.json'))]
    assert len(regression) == 24
    for record in regression:
        assert record['status'] == 'ok' and record['arrays_exact']
        assert record['identity_sha256'] == record_file(RUN / 'identity.json')['sha256']
        checked(record['source'])
    reproduction = json.loads((RUN / 'reproduction.json').read_text())
    assert reproduction['status'] == 'ok' and reproduction['strata_exact'] == 144
    features = json.loads(checked(identity['features']).read_text())
    verified, repeated, deep_exact = 0, 0, 0
    counts = {}
    for record in comparison['records']:
        result = json.loads(checked(record['result']).read_text())
        for cam, camera in result['cameras'].items():
            tracking = camera['tracking']
            with np.load(checked(tracking['arrays']), allow_pickle=False) as raw_file, np.load(checked(camera['arrays']), allow_pickle=False) as group_file:
                raw, group = dict(raw_file), dict(group_file)
            for field in ('boxes', 'observed', 'track_ids'):
                np.testing.assert_array_equal(raw[field], group[field])
            assert not (group['selected'] & ~raw['observed']).any()
            mapping = group['group_origins']
            assert len(mapping) <= 6
            np.testing.assert_array_equal(mapping >= 0, group['group_observed'])
            for index in range(len(mapping)):
                at = np.flatnonzero(mapping[index] >= 0)
                np.testing.assert_array_equal(group['group_boxes'][index, at], raw['boxes'][mapping[index, at], at])
            if tracking['source_detection_rows']:
                key = record['clip'] + '/' + cam
                with np.load(checked(features['records']['clipreid_vitb16_market1501'][key]), allow_pickle=False) as source:
                    source_boxes, offsets = source['boxes'], source['offsets']
                np.testing.assert_array_equal(raw['origins'] >= 0, raw['observed'])
                for frame in range(raw['observed'].shape[1]):
                    at = np.flatnonzero(raw['observed'][:, frame])
                    rows = raw['origins'][at, frame]
                    assert ((rows >= offsets[frame]) & (rows < offsets[frame + 1])).all()
                    np.testing.assert_array_equal(raw['boxes'][at, frame], source_boxes[rows])
            if 'gsi' in tracking:
                with np.load(checked(tracking['gsi']), allow_pickle=False) as saved:
                    gsi = dict(saved)
                np.testing.assert_array_equal(gsi['observed'], raw['observed'])
                np.testing.assert_array_equal(gsi['origins'], raw['origins'])
                assert not (gsi['interpolated'] & gsi['observed']).any()
                assert (gsi['origins'][gsi['interpolated']] == -1).all()
                active = gsi['observed'] | gsi['interpolated']
                assert np.isfinite(gsi['boxes']).all()
                assert (gsi['boxes'][active][:, 2:] > gsi['boxes'][active][:, :2]).all()
                count = counts.setdefault(record['variant'], {'aflink_merges': 0, 'all_synthetic': 0, 'selected_track_synthetic': 0})
                count['aflink_merges'] += len(tracking['links'])
                count['all_synthetic'] += int(gsi['interpolated'].sum())
                count['selected_track_synthetic'] += int((gsi['interpolated'] & group['selected'].any(1)[:, None]).sum())
            if record['variant'] in NEW:
                assert tracking['deterministic_repeat_equal']
                repeated += 1
                if record['variant'] == DEEP_AF:
                    with np.load(checked(tracking['online']), allow_pickle=False) as new, np.load(checked(tracking['online_run9_exact']), allow_pickle=False) as old:
                        assert set(new.files) == set(old.files)
                        for field in new.files:
                            np.testing.assert_array_equal(new[field], old[field])
                    deep_exact += 1
            verified += 1
        for evidence in result['association'].values():
            if evidence['status'] == 'ok':
                checked(evidence['arrays'])
        for field in ('units', 'group_units'):
            checked(result[field])
    review = json.loads((RUN / 'review.json').read_text())
    best_new = recommendation([r for r in comparison['table'] if r['variant'] in NEW])
    assert review['candidate'] == best_new and review['baseline'] == DEEP
    plan = json.loads(checked(identity['plan']).read_text())
    source = json.loads(checked(plan['sources']).read_text())
    for window in review['windows']:
        clip = window['clip']
        before = states(*load_result(RUN, DEEP, clip))
        after = states(*load_result(RUN, best_new, clip))
        video = next(r['video'] for r in source['inputs'] if r['clip'] == clip)
        differences = np.zeros(video['num_frames'], np.int64)
        assert set(before) == set(after)
        for key in before:
            differences[key[1]] += before[key] != after[key]
        width = min(len(differences), round(5 * video['fps']))
        sums = np.convolve(differences, np.ones(width, np.int64), mode='valid')
        assert window['start'] == int(sums.argmax())
        assert window['end'] == window['start'] + width
        assert window['differing_player_units'] == int(sums[window['start']])
    capture = cv2.VideoCapture(str(checked(review['video'])))
    count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        assert frame.shape == (800, 1920, 3)
        if count in (1, 76, 151, 226):
            cv2.imwrite(str(RUN / f'review-frame-{count}.jpg'), frame)
        count += 1
    capture.release()
    assert count == 300
    files = {str(p.relative_to(RUN)): record_file(p) for p in sorted(RUN.rglob('*')) if p.is_file()}
    verification = {'status': 'ok', 'execution_commit': identity['commit'], 'code_files_match_commit': len(code),
        'camera_conditions_verified': verified, 'historical_strata_exact': 144, 'strongsort_online_exact': len(regression),
        'new_online_repeats_equal': repeated, 'deep_online_run9_exact': deep_exact, 'gsi': counts,
        'best_new': best_new, 'video': {**record_file(RUN / 'review.mp4'), 'frames_read': count, 'windows_rule_verified': True},
        'output_bytes': sum(r['bytes'] for r in files.values())}
    (BUNDLE / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    (BUNDLE / 'output_manifest.json').write_text(json.dumps(files, indent=2) + '\n')
    for name in ('identity.json', 'comparison.json', 'comparison.csv', 'association.csv', 'availability.csv',
                 'correspondence.csv', 'recommendation.json', 'report.md', 'reproduction.json', 'review.json'):
        shutil.copyfile(RUN / name, BUNDLE / name)
    shutil.copytree(RUN / 'regression', BUNDLE / 'regression', dirs_exist_ok=True)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
