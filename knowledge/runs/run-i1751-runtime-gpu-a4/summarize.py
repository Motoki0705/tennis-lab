"""Recompute the small acceptance summary from retained replay evidence."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.tennis_scene.archive import load_scene_result  # noqa: E402


def summarize(root):
    comparison = json.loads((root / 'comparison.json').read_text())
    summary = {'tolerance': comparison['tolerance'], 'same_nodes': comparison['same_nodes'],
               'both_scenes_exist': comparison['both_scenes_exist'],
               'all_arrays_exact': comparison['all_arrays_exact'],
               'all_arrays_close': comparison['all_arrays_close'],
               'numerical_acceptance': comparison['numerical_acceptance'],
               'node_count': len(comparison['components']), 'variants': {},
               'scene_fields': len(comparison['scene']), 'differing_arrays': {},
               'differing_payload_nodes': [], 'same_component_array_sha256': True,
               'component_arrays': 0}
    for node, result in comparison['components'].items():
        if not result['same_payload']:
            summary['differing_payload_nodes'].append(node)
        for name, result_array in result['arrays'].items():
            if not result_array['exact']:
                summary['differing_arrays'][node + '/' + name] = result_array
    indexes = [json.loads((root / name / 'store/scene.json').read_text())
               for name in ('main', 'runtime')]
    for node in sorted(set(indexes[0]['artifacts']) & set(indexes[1]['artifacts'])):
        docs = [json.loads((root / name / 'store' / index['artifacts'][node]['path']).read_text())
                for name, index in zip(('main', 'runtime'), indexes, strict=True)]
        if set(docs[0]['arrays']) != set(docs[1]['arrays']):
            summary['same_component_array_sha256'] = False
        for name in set(docs[0]['arrays']) & set(docs[1]['arrays']):
            summary['component_arrays'] += 1
            summary['same_component_array_sha256'] &= docs[0]['arrays'][name]['sha256'] == docs[1]['arrays'][name]['sha256']
    for name in ('main', 'runtime'):
        m = comparison['measurements'][name]
        item = {field: m.get(field) for field in (
            'commit', 'status', 'frames', 'scene_status', 'torch', 'cuda', 'python', 'gpu',
            'gpu_total_bytes', 'wall_seconds', 'backend_load_counts', 'peak_allocated_bytes',
            'peak_reserved_bytes', 'device_memory_peak_mib', 'device_memory_min_mib',
            'device_memory_sampling_ms', 'device_memory_samples', 'allocated_end_bytes',
            'allocated_after_gc_bytes', 'reserved_after_gc_bytes', 'script_sha256')}
        if (root / name / 'scene.npz').is_file():
            scene = load_scene_result(root / name / 'scene.npz')
            item.update(archive_contract_validated=True,
                        validity=scene.metadata['validity_statistics'])
        else:
            item['archive_contract_validated'] = False
        summary['variants'][name] = item
    runtime = comparison['measurements']['runtime']
    active, max_active, pairs = set(), 0, Counter()
    failed_events = []
    for event in runtime['runtime_events']:
        pairs[event['action'], event['model']] += 1
        if event['status'] != 'complete':
            failed_events.append(event)
        if event['action'] == 'load' and event['status'] == 'complete':
            active.add(event['instance'])
            max_active = max(max_active, len(active))
        elif event['action'] == 'unload':
            active.discard(event['instance'])
    summary.update(runtime_loads={model: count for (action, model), count in pairs.items() if action == 'load'},
                   runtime_unloads={model: count for (action, model), count in pairs.items() if action == 'unload'},
                   runtime_max_owned_models=max_active, runtime_unclosed_instances=sorted(active),
                   runtime_failed_events=failed_events)
    summary['same_environment'] = all(summary['variants']['main'][field] == summary['variants']['runtime'][field]
                                      for field in ('torch', 'cuda', 'python', 'gpu', 'script_sha256'))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.root)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
