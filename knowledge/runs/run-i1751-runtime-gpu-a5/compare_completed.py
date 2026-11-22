"""CPU comparison of all published component arrays and final scene fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Fixed before running either implementation. Discrete values must be exact.
ATOL, RTOL = 1e-6, 1e-5


def arrays(left, right):
    result = {'shape_main': list(left.shape), 'shape_runtime': list(right.shape),
              'dtype_main': str(left.dtype), 'dtype_runtime': str(right.dtype)}
    if left.shape != right.shape or left.dtype != right.dtype:
        return {**result, 'exact': False, 'close': False}
    exact = (np.array_equal(left, right, equal_nan=True)
             if left.dtype.kind in 'fc' else np.array_equal(left, right))
    if left.dtype.kind in 'fc':
        close = bool(np.allclose(left, right, rtol=RTOL, atol=ATOL, equal_nan=True))
        finite = np.isfinite(left) & np.isfinite(right)
        max_abs = float(np.max(np.abs(left[finite].astype(np.float64) - right[finite].astype(np.float64)), initial=0))
        different = int(np.count_nonzero(~((left == right) | (np.isnan(left) & np.isnan(right)))))
    else:
        close, max_abs = bool(exact), 0.0 if exact else None
        different = int(np.count_nonzero(left != right))
    return {**result, 'exact': bool(exact), 'close': close, 'max_abs_diff': max_abs,
            'different_elements': different, 'elements': int(left.size)}


def compare_arrays(root):
    stores = [root / variant / 'store' for variant in ('main', 'runtime')]
    indexes = [json.loads((store / 'scene.json').read_text()) for store in stores]
    keys = [set(index['artifacts']) for index in indexes]
    report = {'tolerance': {'float_atol': ATOL, 'float_rtol': RTOL, 'discrete': 'exact'},
              'same_nodes': keys[0] == keys[1], 'missing_runtime': sorted(keys[0] - keys[1]),
              'extra_runtime': sorted(keys[1] - keys[0]), 'components': {}, 'scene': {}}
    all_exact, all_close = True, True
    for node in sorted(keys[0] & keys[1]):
        paths = [store / index['artifacts'][node]['path'] for store, index in zip(stores, indexes, strict=True)]
        documents = [json.loads(path.read_text()) for path in paths]
        names = [set(doc['arrays']) for doc in documents]
        item = {'same_array_names': names[0] == names[1],
                'same_payload': documents[0]['payload'] == documents[1]['payload'], 'arrays': {}}
        for name in sorted(names[0] & names[1]):
            values = [np.load(path.parent / name, allow_pickle=False, mmap_mode='r') for path in paths]
            item['arrays'][name] = arrays(*values)
        item['arrays_exact'] = item['same_array_names'] and all(a['exact'] for a in item['arrays'].values())
        item['arrays_close'] = item['same_array_names'] and all(a['close'] for a in item['arrays'].values())
        report['components'][node] = item
        all_exact &= item['arrays_exact']
        all_close &= item['arrays_close']
    scene_paths = [root / variant / 'scene.npz' for variant in ('main', 'runtime')]
    report['both_scenes_exist'] = all(path.is_file() for path in scene_paths)
    if report['both_scenes_exist']:
        with np.load(scene_paths[0], allow_pickle=False) as left, np.load(scene_paths[1], allow_pickle=False) as right:
            report['same_scene_fields'] = set(left.files) == set(right.files)
            for name in sorted(set(left.files) & set(right.files)):
                report['scene'][name] = arrays(left[name], right[name])
        all_exact &= report['same_scene_fields'] and all(a['exact'] for a in report['scene'].values())
        all_close &= report['same_scene_fields'] and all(a['close'] for a in report['scene'].values())
    report['all_arrays_exact'] = bool(all_exact)
    report['all_arrays_close'] = bool(all_close)
    report['numerical_acceptance'] = bool(report['both_scenes_exist'] and report['same_nodes'] and all_close)
    report['measurements'] = {variant: json.loads((root / variant / 'measurement.json').read_text())
                              for variant in ('main', 'runtime')}
    for variant in ('main', 'runtime'):
        health = json.loads((root / 'health' / (variant + '.json')).read_text())
        report['measurements'][variant].update({k: v for k, v in health.items() if k.startswith('device_memory_') or k == 'driver'})
    return report


def json_values(left, right):
    """Preserve JSON structure/discrete types and apply the pinned float policy."""
    differences = {}
    def visit(a, b, path):
        if type(a) is not type(b):
            differences[path] = {'main': a, 'runtime': b, 'close': False}
            return
        if isinstance(a, dict):
            if a.keys() != b.keys():
                differences[path] = {'keys_main': sorted(a), 'keys_runtime': sorted(b), 'close': False}
            for key in sorted(a.keys() & b.keys()):
                visit(a[key], b[key], path + '/' + key)
        elif isinstance(a, list):
            if len(a) != len(b):
                differences[path] = {'length_main': len(a), 'length_runtime': len(b), 'close': False}
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + '/' + str(i))
        elif a != b:
            close = isinstance(a, float) and bool(np.isclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True))
            differences[path] = {'main': a, 'runtime': b, 'close': close}
    visit(left, right, '')
    return {'exact': not differences, 'close': all(d['close'] for d in differences.values()),
            'differences': differences}


def compare(root):
    report = compare_arrays(root)
    report['array_numerical_acceptance'] = report['numerical_acceptance']
    stores = [root / name / 'store' for name in ('main', 'runtime')]
    indexes = [json.loads((store / 'scene.json').read_text()) for store in stores]
    for node, item in report['components'].items():
        documents = [json.loads((store / index['artifacts'][node]['path']).read_text())
                     for store, index in zip(stores, indexes)]
        item['payload_comparison'] = json_values(documents[0]['payload'], documents[1]['payload'])
    report['all_payloads_exact'] = all(item['payload_comparison']['exact'] for item in report['components'].values())
    report['all_payloads_close'] = all(item['payload_comparison']['close'] for item in report['components'].values())
    metadata = [json.loads((root / name / 'scene.metadata.json').read_text()) for name in ('main', 'runtime')]
    report['scene_metadata'] = json_values(*metadata)
    report['numerical_acceptance'] &= report['all_payloads_close'] and report['scene_metadata']['close']
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.root)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {'components', 'scene', 'measurements'}}, indent=2))
    raise SystemExit(0 if result['numerical_acceptance'] else 1)
