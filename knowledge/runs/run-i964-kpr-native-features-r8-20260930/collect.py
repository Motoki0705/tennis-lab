"""CPU-only integrity collection; run from this checkout with its venv Python."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/kamimura/projects/tennis-lab')
RUN = ROOT / 'outputs/person_tracking/evaluate/dev_features/i964-kpr-r8-20260930'
JOB = '1790735986295857060_1473474_i964-kpr-native-features-r8-20260930'
BUNDLE = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(record):
    path = Path(record['path'])
    assert sha(path) == record['sha256'], path
    if 'bytes' in record:
        assert path.stat().st_size == record['bytes'], path
    return path


def main():
    state = (ROOT / f'.training_queue/state/{JOB}.state').read_text()
    assert 'state=done\n' in state
    assert (ROOT / f'.training_queue/done/{JOB}.job').is_file()
    repro = ROOT / f'.training_queue/repro/{JOB}'
    assert not (repro / 'git_status.txt').read_text().strip()
    identity = json.loads((repro / 'run.json').read_text())
    assert identity['commit'] == '935cb0d55175214abef807d44d3f654ebb743038'
    plan = json.loads((RUN / 'plan.json').read_text())
    manifest = json.loads((RUN / 'features.json').read_text())
    assert manifest['status'] == 'ok'
    assert manifest['plan_sha256'] == sha(RUN / 'plan.json')
    for key in ('weights', 'parent', 'parent_plan', 'parity', 'entry_code', 'reservation'):
        checked(plan[key])
    for path, digest in plan['port_files'].items():
        checked({'path': path, 'sha256': digest})
    clips = ('video_000/clip_000', 'video_000/clip_007', 'video_001/clip_001', 'video_002/clip_013')
    expected = {f'{clip}/cam{cam}' for clip in clips for cam in range(3)}
    assert set(manifest['records']) == expected == set(plan['inputs'])
    records = {}
    for key, record in manifest['records'].items():
        with np.load(checked(record), allow_pickle=False) as data, np.load(checked(plan['inputs'][key]), allow_pickle=False) as parent:
            count, frames = record['detections'], record['frames']
            assert set(data.files) == {'rows', 'boxes', 'scores', 'poses', 'offsets', 'part_embeddings', 'part_visibility', 'metadata'}
            for field in ('rows', 'boxes', 'scores', 'poses', 'offsets'):
                assert data[field].dtype == parent[field].dtype
                assert np.array_equal(data[field], parent[field]), (key, field)
            assert np.array_equal(data['rows'], np.arange(count))
            assert data['offsets'].shape == (frames + 1,)
            assert data['offsets'][0] == 0 and data['offsets'][-1] == count
            assert (np.diff(data['offsets']) >= 0).all()
            parts, visible = data['part_embeddings'], data['part_visibility']
            assert parts.shape == (count, 6, 512) and parts.dtype == np.float32
            assert visible.shape == (count, 6) and visible.dtype == np.bool_
            assert all(np.isfinite(data[k]).all() for k in ('boxes', 'scores', 'poses', 'part_embeddings'))
            error = float(np.max(np.abs(np.linalg.norm(parts, axis=-1)[visible] - 1)))
            assert error < 1e-4
            meta = json.loads(str(data['metadata']))
            parent_meta = json.loads(str(parent['metadata']))['provenance']
            assert meta['schema'] == 'person_kpr_native_features_v1'
            assert meta['parent'] == plan['inputs'][key]
            assert meta['plan_sha256'] == manifest['plan_sha256']
            assert meta['weight_sha256'] == plan['weights']['sha256']
            assert meta['source'] == parent_meta['source']
            checked(meta['source'])
            with np.load(checked(parent_meta['detection']), allow_pickle=False) as detection:
                for field in ('boxes', 'scores', 'offsets'):
                    assert np.array_equal(data[field], detection[field]), (key, field, 'source detection')
            assert int(visible.sum()) == record['visible_parts']
            records[key] = {**record, 'parent_fields_bitwise_equal': True, 'source_detections_equal': True,
                            'finite': True, 'visibility_dtype': 'bool', 'visible_norm_max_error': error,
                            'rows_without_visible_parts': int((~visible.any(axis=1)).sum())}
    assert sum(r['detections'] for r in records.values()) == plan['rows'] == 40531
    assert sum(r['frames'] for r in records.values()) == plan['frames'] == 10491
    result = {'status': 'ok', 'job': JOB, 'queue_state': state, 'repro_commit': identity['commit'],
              'manifest_sha256': sha(RUN / 'features.json'), 'plan_sha256': manifest['plan_sha256'],
              'rows': 40531, 'camera_frames': 10491, 'archives_verified': 12,
              'elapsed_seconds': manifest['elapsed_seconds'],
              'peak_allocated_bytes': manifest['peak_allocated_bytes'],
              'peak_reserved_bytes': manifest['peak_reserved_bytes'],
              'output_bytes': sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),
              'weight': plan['weights'], 'records': records}
    (BUNDLE / 'collection.json').write_text(json.dumps(result, indent=2) + '\n')
    for name in ('plan.json', 'features.json'):
        (BUNDLE / name).write_bytes((RUN / name).read_bytes())
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'weight')}, indent=2))


if __name__ == '__main__':
    main()
