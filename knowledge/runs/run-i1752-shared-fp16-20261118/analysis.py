"""CPU-only summaries; retain raw recorder, per-clip nodes and conditional projections."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'knowledge/runs/run-video-batch-runtime-20261117'))
import doubles_analysis as base_analysis
from doubles_analysis import RUNTIME_MODELS, clip_summary, lifetimes, overhead_estimates, stage_estimates

# Shared-side verification reads the same video artifact per clip, like shared court.
# This changes only this experiment process, not the historical estimator/source.
base_analysis.PER_CLIP = {*base_analysis.PER_CLIP, "court_side"}
from doubles_acceptance import CLIPS, sha


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def report(root, destination):
    destination.mkdir(exist_ok=True)
    entry = json.loads((root / 'batch/entry.json').read_text())
    progress_path = Path(entry.get('progress') or next((root / 'batch/progress').glob('*/progress.json')))
    progress = json.loads(progress_path.read_text())
    recording_path = progress_path.with_name('recording.json')
    recording = json.loads(recording_path.read_text())
    terminal = json.loads((root / 'terminal.json').read_text())
    manifest = json.loads((root / 'input-manifest.json').read_text())
    frames = {clip_id: next(row['num_frames'] for row in manifest['selected_records'] if row['clip_id'] == clip_id)
              for clip_id in CLIPS}
    clips = [clip_summary(root / 'datasets/batch', clip_id) for clip_id in CLIPS]
    owned_events = [event for event in recording['events'] if event['attributes'].get('model') in RUNTIME_MODELS]
    stages = stage_estimates(progress['nodes'], owned_events, frames, 46228, 69, 4)
    overhead = overhead_estimates(recording, stages, 3, 69, frames, 46228)
    missing_estimates = [name for name, item in stages.items() if item['estimated_69clip_seconds'] is None]
    fixed_nodes = sum(item['estimated_69clip_seconds'] for item in stages.values() if item['estimated_69clip_seconds'] is not None)
    projection = [fixed_nodes + sum(item['estimated_69clip_seconds_range'][i] for item in overhead['stages']
                                  if item['estimated_69clip_seconds_range'] is not None)
                  + overhead['estimated_69clip_remaining_seconds_range'][i] for i in (0, 1)]
    cli_unrecorded = max(0, entry['entry_wall_seconds'] - recording['total_wall_seconds'])
    phase = terminal['batch']
    phase_seconds = (datetime.fromisoformat(phase['finished_at_utc']) - datetime.fromisoformat(phase['started_at_utc'])).total_seconds()
    startup_seconds = max(0, phase_seconds - entry['entry_wall_seconds'])
    projection = [value + startup_seconds + cli_unrecorded * (1 if i == 0 else 69 / 3)
                  for i, value in enumerate(projection)]
    lifetime = lifetimes(entry['runtime_events'])
    result = {'measurement_commit': terminal['commit'], 'queue_job': terminal['queue_job'],
        'all_three_completed': all(row['status'] == 'generated' for row in progress['clips'].values()),
        'identity_counts_ok': all(all(row['sides'][side]['count'] == 2 for side in ('-1', '1')) for row in clips if row['marker_exists'])
                              and all(row['marker_exists'] for row in clips),
        'body_exports_present': all(row.get('body_exports_present', False) for row in clips),
        'cuda_anomaly': (root / 'anomaly.json').exists(),
        'entry_seconds': entry['entry_wall_seconds'], 'recorder_seconds': recording['total_wall_seconds'],
        'guard_phase_seconds': phase_seconds, 'startup_and_guard_seconds': startup_seconds,
        'cli_before_recorder_seconds': cli_unrecorded,
        'global_python_profiler': entry['global_python_profiler'],
        'resources': recording['resources'], 'nvml_health': terminal['batch'],
        'measurement_errors': recording['measurement_errors'],
        'lifetime': lifetime, 'all_recorded_models': recording['models'], 'clips': clips, 'outcomes': progress['clips'],
        'node_statuses': dict(Counter(row['status'] for nodes in progress['nodes'].values() for row in nodes.values())),
        'stages': stages, 'overhead': overhead, 'full69_missing_stage_estimates': missing_estimates,
        'full69_conditional_hours_range': None if missing_estimates else [value / 3600 for value in projection],
        'full69_executed': False,
        'timing_limits': 'C1 synchronization, resource sampling, heartbeat and cold compile remain included. No global sys.setprofile. Projections are conditional, not causal precision speedups or confidence bounds.'}
    write(destination / 'summary.json', result)
    write(destination / 'progress.json', progress)
    write(destination / 'resources-and-models.json', {key: recording[key] for key in
          ('resources', 'models', 'batch_shapes', 'timing', 'stage_wall_seconds', 'stage_inclusive_seconds', 'measurement_errors')})
    rows = ['clip_id,node,status,seconds']
    for clip, nodes in progress['nodes'].items():
        rows.extend(f'{clip},{name},{row["status"]},{row["seconds"]}' for name, row in nodes.items())
    (destination / 'clip-stage-seconds.csv').write_text('\n'.join(rows) + '\n')
    write(destination / 'raw-artifacts.json', {'files': [
        {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in
        (recording_path, root / 'batch.log', root / 'batch/entry.json', root / 'health/batch-memory.csv')],
        'predictions': str(root / 'datasets/batch'), 'retained': True})
    print(json.dumps({k: result[k] for k in ('all_three_completed', 'identity_counts_ok', 'body_exports_present',
                      'cuda_anomaly', 'entry_seconds', 'full69_conditional_hours_range')}, indent=2))


def reference_report(root, destination):
    sys.path.insert(0, str(REPO / 'knowledge/runs/run-i1751-runtime-gpu-a5'))
    from compare_completed import compare
    from src.tennis_scene.pipeline.storage.scene_index import annotation_scene_path

    comparison = root / 'comparison'
    for variant, phase, dataset in (
        ('runtime', root, root / 'datasets/batch'),
        ('main', root / 'reference', root / 'datasets/single'),
    ):
        directory = comparison / variant
        directory.mkdir(parents=True, exist_ok=True)
        store = dataset / 'videos/video_000/clips/clip_027/annotations/tennis_scene'
        archive = annotation_scene_path(store, json.loads((store / 'annotation.json').read_text()))
        for link, target in ((directory / 'store', store), (directory / 'scene.npz', archive),
                             (directory / 'scene.metadata.json', archive.with_suffix('.metadata.json'))):
            if not link.exists():
                link.symlink_to(target)
        write(directory / 'measurement.json', json.loads((phase / 'batch/entry.json').read_text()))
        (comparison / 'health').mkdir(exist_ok=True)
        write(comparison / 'health' / f'{variant}.json', json.loads((phase / 'terminal.json').read_text())['batch'])
    full = compare(comparison)
    result = {key: full[key] for key in ('numerical_acceptance', 'all_arrays_exact', 'all_payloads_exact',
              'all_payloads_close', 'scene_metadata', 'same_nodes', 'tolerance')}
    result.update(reference_clip='video_000/clip_027', components=len(full['components']),
        arrays=sum(len(item['arrays']) for item in full['components'].values()) + len(full['scene']),
        runtime_lifetimes=lifetimes(json.loads((root / 'reference/batch/entry.json').read_text())['runtime_events']))
    write(destination / 'comparison-completed.json', full)
    write(destination / 'comparison-summary.json', result)
    print(json.dumps({key: result[key] for key in ('numerical_acceptance', 'all_arrays_exact', 'components', 'arrays')}, indent=2))
    return result['numerical_acceptance']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--destination', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    if args.reference:
        raise SystemExit(0 if reference_report(args.root.resolve(), args.destination.resolve()) else 1)
    report(args.root.resolve(), args.destination.resolve())
