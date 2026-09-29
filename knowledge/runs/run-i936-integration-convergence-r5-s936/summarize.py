"""Validate coverage of all fixed frames and summarize capped integration."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--audit', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
source_manifest = args.source / 'manifest.json'
source = json.loads(source_manifest.read_text())
source_sha = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
progress = [json.loads(p.read_text()) for p in sorted(args.audit.glob('progress-*.json'))]
if any(p['status'] != 'complete' or p['source_manifest_sha256'] != source_sha or p['failures'] for p in progress):
    raise ValueError('Incomplete or mixed-source convergence audit')
rows = [r for p in progress for r in p['records']]
expected = {(r['rally_id'], i) for r in source['rallies'] for i in range(r['frames'])}
ids = [(r['rally_id'], r['frame']) for r in rows]
if len(set(ids)) != len(ids) or set(ids) != expected:
    raise ValueError('Missing/duplicated/substituted smoke frames')
arrays = []
chunk_hashes = {}
for path in sorted(args.audit.glob('*.npz')):
    with np.load(path, allow_pickle=False) as z:
        np.linalg.cholesky(z['covariance'])
        if z['weights'].shape != (len(z['frames']), 64) or not np.allclose(z['weights'].sum(-1), 1):
            raise ValueError('Missing component or mass')
        if not all(np.isfinite(z[key]).all() for key in z.files):
            raise ValueError('Nonfinite audit artifact')
        arrays.extend((z['weights'] * ~z['component_converged']).sum(-1).tolist())
    chunk_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
if len(arrays) != len(rows):
    raise ValueError('Audit NPZ/frame count mismatch')
summary = {
    'source_manifest_sha256': source_sha, 'source_rallies': [{k: r[k] for k in ('rally_id', 'seed', 'frames', 'npz_sha256')} for r in source['rallies']],
    'rule': progress[0]['settings'], 'frames': len(rows), 'nonconverged_frames': sum(not r['converged'] for r in rows),
    'nonconverged_rate': np.mean([not r['converged'] for r in rows]), 'nll_only_nonconverged_frames': sum(r['history'][-1]['nll_delta_nat'] > progress[0]['settings']['nll_tolerance_nat'] for r in rows),
    'worker_seconds': sum(r['seconds'] for r in rows), 'batch_wall_seconds': sum(p['elapsed_seconds'] for p in progress),
    'capped_component_probability_mass': {'mean': float(np.mean(arrays)), 'p50': float(np.median(arrays)), 'p95': float(np.quantile(arrays, .95))},
    'boundary_val_00003_frame_130': next(r for r in rows if r['rally_id'] == 'val-00003' and r['frame'] == 130),
    'rallies': [], 'artifact_sha256': chunk_hashes,
}
for identity in sorted({r['rally_id'] for r in rows}):
    selected = [r for r in rows if r['rally_id'] == identity]
    summary['rallies'].append({'rally_id': identity, 'frames': len(selected), 'nonconverged_frames': sum(not r['converged'] for r in selected), 'worker_seconds': sum(r['seconds'] for r in selected), 'nll_delta_p50': float(np.median([r['history'][-1]['nll_delta_nat'] for r in selected])), 'nll_delta_p95': float(np.quantile([r['history'][-1]['nll_delta_nat'] for r in selected], .95))})
for key in ('nll_delta_nat', 'max_log_evidence_delta_nat', 'max_mean_delta', 'max_covariance_relative_delta'):
    values = [r['history'][-1][key] for r in rows]
    summary[key] = {'p50': float(np.median(values)), 'p95': float(np.quantile(values, .95)), 'max': float(np.max(values))}
args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('artifact_sha256', 'source_rallies', 'rallies')}, indent=2))
