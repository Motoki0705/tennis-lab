"""Audit completed 640 metadata/hashes without opening any rally NPZ arrays."""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from src.tasks.ball_refiner.refiner_3d.diffusion.experiment import preflight
from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import sha256


def main() -> None:
    bundle = Path(__file__).resolve().parent
    output = bundle / 'final-audit'
    output.mkdir()
    experiment_path = bundle.parent / 'run-i936-pilot512-t128-r14-s936/plan.json'
    experiment = json.loads(experiment_path.read_text())
    launch = json.loads((bundle / 'launch.json').read_text())
    source = Path(experiment['dataset'])
    manifest = json.loads((source / 'manifest.json').read_text())
    assert sha256(Path(launch['plan'])) == launch['plan_sha256']
    assert manifest['input_hashes'] == launch['input_hashes']
    # Same gate as the requeued job, but no write to the job's reserved paths.
    gate = preflight(experiment)
    assert not Path(experiment['output']).exists()
    assert not Path(experiment['preflight_output']).exists()
    assert len(manifest['rallies']) == 640 and manifest['failures'] == []
    assert all((r['fps_numerator'], r['fps_denominator'], r['native_hz']) == (60000, 1001, 240)
               for r in manifest['rallies'])
    assert all(r['components_per_frame'] == 125 for r in manifest['rallies'])
    files = {str(p.relative_to(source)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
             for p in sorted(source.rglob('*')) if p.is_file()}
    assert sum(p.endswith('.npz') for p in files) == 640
    assert manifest['npz_bytes'] == sum(r['bytes'] for p, r in files.items() if p.endswith('.npz'))
    control = json.loads((bundle.parent / 'run-i936-anchored-dev-comparison-r12-s936/control/manifest.json').read_text())
    control_path = Path(control['dataset'])
    assert sha256(control_path / 'manifest.json') == control['audit']['manifest_sha256']
    control_files = {}
    for row in control['audit']['rallies']:
        name = row['rally_id'] + '.npz'
        digest = sha256(control_path / name)
        assert digest == row['npz_sha256']
        control_files[name] = digest
    assert len(control_files) == 96
    log = Path(launch['log'])
    messages = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
    complete_ids = [r['completed'] for r in messages if 'completed' in r]
    assert len(complete_ids) == 640 and set(complete_ids) == {r['rally_id'] for r in manifest['rallies']}
    report = {
        'collected_at': datetime.now().astimezone().isoformat(), 'status': 'complete',
        'manifest_sha256': sha256(source / 'manifest.json'), 'counts': manifest['counts'],
        'completed': 640, 'failures': [], 'frames': manifest['total_frames'],
        'all_npz_json_hashes': files, 'generation_inputs_unchanged': manifest['input_hashes'],
        'run12_plan_sha256': sha256(Path(launch['plan'])), 'run12_expanded_plan_exact_match': True,
        'log_sha256': sha256(log), 'log_unique_completions': len(set(complete_ids)),
        'fixed_hybrid': True, 'components_per_frame': 125, 'fps': [60000, 1001],
        'unassessed_frames': manifest['unassessed_frames'],
        'integration_convergence_assessed': False, 'test_arrays_read': 0, 'all_arrays_read': 0,
        'old_959_control_manifest_sha256': control['audit']['manifest_sha256'],
        'old_959_control_npz_sha256': control_files,
        'resources': {
            'generation_elapsed_seconds': manifest['elapsed_seconds'],
            'sum_rally_seconds': manifest['sum_rally_seconds'],
            'sum_simulation_seconds': sum(r['simulation_seconds'] for r in manifest['rallies']),
            'sum_triangulation_seconds': sum(r['triangulation_seconds'] for r in manifest['rallies']),
            'maximum_worker_rss_bytes': max(r['peak_worker_rss_kib'] for r in manifest['rallies']) * 1024,
            'ram_scope': 'max individual worker ru_maxrss; aggregate concurrent RSS and controller peak were not logged',
            'npz_bytes': manifest['npz_bytes'], 'all_file_bytes': sum(r['bytes'] for r in files.values()),
            'generation_manifest_mtime': datetime.fromtimestamp((source / 'manifest.json').stat().st_mtime).astimezone().isoformat(),
        },
        'same_plan_requeue_preflight': gate,
        'original_experiment_plan_sha256': sha256(experiment_path),
        'reserved_training_outputs_untouched': True,
    }
    shutil.copy2(source / 'manifest.json', output / 'generation-manifest.json')
    shutil.copy2(log, output / 'generation.log')
    (output / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'completed', 'frames', 'manifest_sha256', 'resources')}, indent=2))
    print('requeue preflight passed; fixed val', gate['primary_val_frames'], 'frames; no arrays opened')


if __name__ == '__main__':
    main()
