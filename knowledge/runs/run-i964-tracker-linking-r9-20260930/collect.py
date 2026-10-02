"""Verify and promote the immutable run-9 output evidence, without re-running inference."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

BASE = Path('/home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/tracker_matrix')
RUN = BASE / 'i964-linking-r9-20260930-v3'
DIAG = BASE / 'i964-linking-r9-20260930-v2'
BUNDLE = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(record):
    path = Path(record['path'])
    assert sha(path) == record['sha256'], path
    return path


def main():
    identity = json.loads((RUN / 'identity.json').read_text())
    code = dict(identity['code'])
    code['tests/benchmarks/person_tracking_linking.py'] = identity['benchmark']['sha256']
    for name, digest in code.items():
        blob = subprocess.check_output(['git', 'show', identity['commit'] + ':' + name], cwd=BUNDLE.parents[2])
        assert hashlib.sha256(blob).hexdigest() == digest, name
    comparison = json.loads((RUN / 'comparison.json').read_text())
    features = json.loads(checked(identity['features']).read_text())
    old = json.loads(checked(identity['previous']).read_text())
    old_table = {(r['variant'], r['camera'], r['near_far']): r for r in old['table']}
    for row in comparison['table']:
        key = row['variant'], row['camera'], row['near_far']
        if key in old_table:
            assert all(row[k] == v for k, v in old_table[key].items()), key
    choice = json.loads((RUN / 'kpr_method.json').read_text())
    checked(choice['comparison'])
    checked(identity['aflink'])
    verified, repeated, links, gsi_counts = 0, 0, {}, {}
    for record in comparison['records']:
        result = json.loads(checked(record['result']).read_text())
        for cam, camera in result['cameras'].items():
            tracking = camera['tracking']
            with np.load(checked(tracking['arrays']), allow_pickle=False) as raw_file, np.load(checked(camera['arrays']), allow_pickle=False) as group_file:
                raw, group = dict(raw_file), dict(group_file)
                for field in ('boxes', 'observed', 'track_ids'):
                    assert np.array_equal(raw[field], group[field])
                assert not (group['selected'] & ~raw['observed']).any()
                mapping = group['group_origins']
                assert len(mapping) <= 6 and np.array_equal(mapping >= 0, group['group_observed'])
                for index in range(len(mapping)):
                    at = np.flatnonzero(mapping[index] >= 0)
                    assert np.array_equal(group['group_boxes'][index, at], raw['boxes'][mapping[index, at], at])
                if tracking['source_detection_rows']:
                    key = record['clip'] + '/' + cam
                    with np.load(checked(features['records']['clipreid_vitb16_market1501'][key]), allow_pickle=False) as source:
                        source_boxes, offsets = source['boxes'], source['offsets']
                        assert np.array_equal(raw['origins'] >= 0, raw['observed'])
                        for frame in range(raw['observed'].shape[1]):
                            at = np.flatnonzero(raw['observed'][:, frame])
                            rows = raw['origins'][at, frame]
                            assert ((rows >= offsets[frame]) & (rows < offsets[frame + 1])).all()
                            assert np.array_equal(raw['boxes'][at, frame], source_boxes[rows])
                if 'gsi' in tracking:
                    with np.load(checked(tracking['gsi']), allow_pickle=False) as gsi_file:
                        gsi = dict(gsi_file)
                        assert np.array_equal(gsi['observed'], raw['observed'])
                        assert np.array_equal(gsi['origins'], raw['origins'])
                        assert not (gsi['observed'] & gsi['interpolated']).any()
                        assert (gsi['origins'][gsi['interpolated']] == -1).all()
                        active = gsi['observed'] | gsi['interpolated']
                        assert np.isfinite(gsi['boxes']).all()
                        assert (gsi['boxes'][active][:, 2:] > gsi['boxes'][active][:, :2]).all()
                        counts = gsi_counts.setdefault(record['variant'], {'all': 0, 'selected_tracks': 0})
                        counts['all'] += int(gsi['interpolated'].sum())
                        counts['selected_tracks'] += int((gsi['interpolated'] & group['selected'].any(1)[:, None]).sum())
                    links[record['variant']] = links.get(record['variant'], 0) + len(tracking['links'])
            repeated += bool(tracking.get('deterministic_repeat_equal')) and record['variant'] not in {r['variant'] for r in old['table']}
            verified += 1
        for evidence in result['association'].values():
            if evidence['status'] == 'ok':
                checked(evidence['arrays'])
        for field in ('units', 'group_units'):
            checked(result[field])
    videos = {}
    for filename, expected_shape in [('review.mp4', (800, 1920, 3)), ('cam1-diagnosis.mp4', (880, 1920, 3))]:
        path = RUN / filename
        capture = cv2.VideoCapture(str(path))
        count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape == expected_shape
            if filename == 'review.mp4' and count in (1, 76, 151, 226):
                cv2.imwrite(str(RUN / f'review-frame-{count}.jpg'), frame)
            count += 1
        capture.release()
        assert count == (300 if filename == 'review.mp4' else 225), (filename, count)
        videos[filename] = {'sha256': sha(path), 'bytes': path.stat().st_size, 'frames': count}
    files = {str(p.relative_to(RUN)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(RUN.rglob('*')) if p.is_file()}
    (BUNDLE / 'verification.json').write_text(json.dumps({'status': 'ok', 'camera_conditions_verified': verified,
        'new_online_repeats_equal': repeated, 'historical_strata_equal': len(old_table), 'aflink_merges': links,
        'execution_code_files_verified_against_commit': len(code), 'execution_commit': identity['commit'],
        'gsi_synthetic_counts': gsi_counts, 'videos': videos, 'run_bytes': sum(v['bytes'] for v in files.values())}, indent=2) + '\n')
    (BUNDLE / 'output_manifest.json').write_text(json.dumps(files, indent=2) + '\n')
    for name in ('identity.json', 'comparison.json', 'comparison.csv', 'association.csv', 'availability.csv',
                 'correspondence.csv', 'recommendation.json', 'kpr_method.json', 'base-comparison.json', 'report.md',
                 'review.json', 'cam1-diagnosis.json'):
        shutil.copyfile(RUN / name, BUNDLE / name)
    shutil.copyfile(DIAG / 'diagnosis.json', BUNDLE / 'diagnosis.json')
    shutil.copytree(DIAG / 'diagnosis', BUNDLE / 'diagnosis', dirs_exist_ok=True)
    print((BUNDLE / 'verification.json').read_text())


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
