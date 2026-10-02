"""CPU read-back audit; run from the checkout with PYTHONPATH=. .venv/bin/python."""
import json
from pathlib import Path

import numpy as np

from src.tasks.person_tracking.archive import load_features
from src.tasks.player_detection.evaluation.far_archive import DetectionArchive
from src.utils.checksum import dual_sha256

ROOT = Path('/home/kamimura/projects/tennis-lab')
REPORT = ROOT / 'outputs/person_tracking/evaluate/dev_features/i964-features-r7-20260930'
JOB = '1790728513515020890_17831_i964-coco-person-features-r7-20260930'
BUNDLE = Path(__file__).resolve().parent


def main() -> None:
    manifest = json.loads((REPORT / 'features.json').read_text())
    plan = json.loads((REPORT / 'plan.json').read_text())
    assert manifest['status'] == 'ok'
    assert dual_sha256(REPORT / 'plan.json') == manifest['plan_sha256']
    assert (ROOT / '.training_queue/done' / f'{JOB}.job').exists()
    state = (ROOT / '.training_queue/state' / f'{JOB}.state').read_text()
    assert 'state=done\n' in state
    records = []
    for item in plan['inputs']:
        source = DetectionArchive.load(item['detection'])
        previous = None
        for encoder, entries in manifest['records'].items():
            record = entries[item['key']]
            path = Path(record['path'])
            assert dual_sha256(path) == record['sha256']
            assert path.stat().st_size == record['bytes']
            frames, provenance = load_features(path)
            assert len(frames) == record['frames'] == len(source.milliseconds)
            assert provenance['detection'] == item['detection']
            assert provenance['source'] == item['video']
            assert provenance['plan_sha256'] == manifest['plan_sha256']
            assert provenance['models']['pose'] == plan['models']['pose']
            assert provenance['models']['appearance'] == plan['models'][encoder]
            arrays = {k: np.concatenate([getattr(f, k) for f in frames])
                      for k in ('rows', 'boxes', 'scores', 'poses', 'embeddings', 'appearance_valid')}
            assert np.array_equal(arrays['rows'], np.arange(len(source.scores)))
            assert np.array_equal(arrays['boxes'], source.boxes)
            assert np.array_equal(arrays['scores'], source.scores)
            assert np.array_equal(np.cumsum([0, *[len(f.rows) for f in frames]]), source.offsets)
            assert len(arrays['rows']) == record['detections']
            assert int(arrays['appearance_valid'].sum()) == record['appearance_valid']
            if previous is not None:
                assert np.array_equal(arrays['poses'], previous)
            previous = arrays['poses']
            records.append({'encoder': encoder, 'key': item['key'], **record,
                            'source_row_box_score_offsets_equal': True,
                            'contract_read_back_ok': True, 'pose_shared_bitwise': True})
    totals = {encoder: sum(r['detections'] for r in records if r['encoder'] == encoder)
              for encoder in manifest['records']}
    assert set(totals.values()) == {40531} and len(records) == 24
    evidence = {'status': 'ok', 'job': JOB, 'queue_state': state, 'exit_code': 0,
                'exit_evidence': 'queue done state/file and worker done line; worker writes done only for rc=0 (training_queue.sh:1142-1144); success log omits exit_code',
                'manifest_sha256': dual_sha256(REPORT / 'features.json'),
                'plan_sha256': manifest['plan_sha256'], 'rows_per_encoder': totals,
                'frames_per_encoder': 10491, 'records': records,
                'elapsed_seconds': manifest['elapsed_seconds'],
                'peak_allocated_bytes': manifest['peak_allocated_bytes'],
                'peak_reserved_bytes': manifest['peak_reserved_bytes'],
                'output_bytes': sum(p.stat().st_size for p in REPORT.rglob('*') if p.is_file()),
                'limitations': 'GPU peaks are torch allocator measurements, not whole-device telemetry; CPU/RSS peak was not logged'}
    for name in ('plan.json', 'features.json'):
        (BUNDLE / name).write_bytes((REPORT / name).read_bytes())
    (BUNDLE / 'collection.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({k: v for k, v in evidence.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
