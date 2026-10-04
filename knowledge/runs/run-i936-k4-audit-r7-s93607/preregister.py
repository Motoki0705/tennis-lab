"""Select once using only input identity and observation/event masks."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    source = Path('/home/kamimura/projects/tennis-lab/data/ball_refiner/synthetic-3d-i936-dev-r5')
    strata: dict[str, list[dict[str, object]]] = defaultdict(list)
    files = {}
    for path in sorted(source.glob('*.npz')):
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata = json.loads(path.with_suffix('.json').read_text())
        if sha != metadata['npz_sha256']:
            raise ValueError('Source hash mismatch')
        with np.load(path, allow_pickle=False) as data:
            observed = (~(data['occlusion_mask'] | data['out_of_frame_mask'])).sum(0)
            events = data['event_region_mask']
            for frame, (views, event) in enumerate(zip(observed, events, strict=True)):
                key = f'views{views}-' + ('event' if event else 'away')
                strata[key].append({'rally_id': path.stem, 'frame': frame, 'observed_views': int(views), 'near_event': bool(event), 'stratum': key})
        files[path.stem] = {'path': str(path), 'sha256': sha, 'frames': len(observed), 'seed': metadata['seed'], 'physics_proposals': metadata['physics_proposals']}
    rng = np.random.default_rng(93607)
    selected = []
    counts = {}
    for key, frames in sorted(strata.items()):
        indices = sorted(rng.choice(len(frames), min(50, len(frames)), replace=False).tolist())
        selected.extend(frames[i] for i in indices)
        counts[key] = {'population': len(frames), 'sample': len(indices)}
    plan = yaml.safe_load((root / 'src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml').read_text())
    output = Path(__file__).with_name('sample.json')
    result = {
        'schema': 1, 'seed': 93607, 'baseline_commit': '32cc5d02766dade2370ede2a9007dd73b42c8421',
        'selection_rule': 'sorted strata; PCG64(93607); min(50, population) without replacement; no result or GT inspection',
        'stratum_definition': 'views=sum(~(occlusion|out_of_frame)); near_event=saved event_region_mask (hit/bounce +/-5 frames)',
        'scope': 'all nine completed rallies of stopped dev-r5; completion-biased train-only sample, not full generator population',
        'reporting': 'unweighted balanced sample AND population-stratum-weighted estimates; failures count as nonconverged; all 125 components; 4 workers, native threads=1; mean/p50/p95/max worker seconds plus actual wall',
        'targets': {'overall_converged': .95, 'all_missing_converged': .90, 'mean_seconds_per_frame': .6},
        'projection': {'frames_per_rally': 450, 'rallies': [96, 640], 'workers': 4, 'budget_margin': 1.2},
        'settings': plan['degradation'], 'strata': counts, 'sources': files, 'frames': selected,
    }
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'frames': len(selected), 'strata': counts, 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == '__main__':
    main()
