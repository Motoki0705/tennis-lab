"""Collect only existing generation/verification evidence, without GPU work."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path('/home/kamimura/projects/tennis-lab/data/ball_refiner/i936-dev-h-r9-s936')
LOGS = Path('/home/kamimura/projects/tennis-lab/outputs/ball_refiner/generate/h-dev/r9-s936')
VERIFY = Path('/home/kamimura/projects/tennis-lab/outputs/ball_refiner/analyze/h-dev/r9-s936')
manifest = json.loads((DATA / 'manifest.json').read_text())
result = json.loads((LOGS / 'result.json').read_text())
verification = json.loads((VERIFY / 'report.json').read_text())
assert manifest['status'] == result['status'] == verification['status'] == 'complete'
resources = [json.loads(line) for line in (LOGS / 'resources.jsonl').read_text().splitlines()]
time_fields = {}
for line in (LOGS / 'time.txt').read_text().splitlines():
    for label in ('User time (seconds)', 'System time (seconds)', 'Percent of CPU this job got', 'Maximum resident set size (kbytes)'):
        if label + ':' in line:
            time_fields[label] = line.split(label + ':', 1)[1].strip()
summary = {'generation': {'status': manifest['status'], 'wall_seconds': result['elapsed_seconds'],
    'generator_wall_seconds': manifest['elapsed_seconds'], 'cpu': time_fields,
    'minimum_available_host_bytes': min(row['available_bytes'] for row in resources),
    'data_bytes': verification['dataset_bytes'], 'npz_bytes': manifest['npz_bytes'],
    'splits': {}, 'components': verification['components'],
    'failed_rallies': manifest['failures'], 'unassessed_frames': manifest['unassessed_frames'],
    'nonconverged_frames': manifest['nonconverged_frames'],
    'sum_simulation_seconds': sum(r['simulation_seconds'] for r in manifest['rallies']),
    'sum_triangulation_seconds': sum(r['triangulation_seconds'] for r in manifest['rallies']),
    'component_methods': dict(sum((Counter(r['component_method_counts']) for r in manifest['rallies']), Counter())),
    'physics_proposals': sum(len(r['physics_proposals']) for r in manifest['rallies']),
    'zero_weight_components': sum(r['float32_zero_weight_components'] for r in manifest['rallies']),
    'clipped_2d_component_means': sum(r['clipped_component_means'] for r in manifest['rallies']),
    'all_camera_gap_frames': sum(r['all_camera_occluded_frames'] for r in manifest['rallies'])},
    'verification': {key: verification[key] for key in ('status','manifest_sha256','elapsed_seconds','frames','components','presence_mass_max_error','weight_sum_max_error','samples','coverage')},
    'real_pilot_reference': {'source': 'https://github.com/Motoki0705/tennis-lab/blob/89cf0737/knowledge/runs/run-i935-detector-only-mixed-e9-s42-r18-20260930/comparison.md',
        'precision': 'rounded to 3 decimals in the run19 table; not recomputed here',
        'observed': {'count': 23007, 'hdr50_90_95': [.456,.873,.908]},
        'gap': {'count': 5077, 'hdr50_90_95': [.565,.852,.907]}}}
for split in ('train','val','test'):
    rows = [r for r in manifest['rallies'] if r['split']==split]
    summary['generation']['splits'][split] = {'rallies': len(rows),'frames':sum(r['frames'] for r in rows),
        'min_frames': min(r['frames'] for r in rows),'max_frames':max(r['frames'] for r in rows),
        'npz_bytes':sum(r['npz_bytes'] for r in rows)}
for name in ('result.json','time.txt','resources.jsonl','generation.log'):
    (ROOT/name).write_bytes((LOGS/name).read_bytes())
(ROOT/'verification-report.json').write_bytes((VERIFY/'report.json').read_bytes())
(ROOT/'generation-manifest.json').write_bytes((DATA/'manifest.json').read_bytes())
hdr = ROOT / 'hdr'
hdr.mkdir(exist_ok=True)
for path in sorted(VERIFY.glob('*-hdr.npz')):
    (hdr/path.name).write_bytes(path.read_bytes())
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2))
