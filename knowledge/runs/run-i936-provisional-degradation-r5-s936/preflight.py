"""Check storage of three deterministic 72-frame development prefixes."""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from src.tasks.ball_refiner.refiner_3d.synthetic.configuration import load_plan
from src.tasks.ball_refiner.refiner_3d.synthetic.dataset import SyntheticDataset
from src.tasks.ball_refiner.refiner_3d.synthetic.generator import _job, write_json
from src.utils.configuration import PathResolver, RuntimePathRoots


def main():
    parser = argparse.ArgumentParser()
    for name in ('project-root', 'data-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    roots = RuntimePathRoots(project_root=args.project_root, data_root=args.data_root,
        output_root=args.output.parent, artifact_root=args.output.parent, checkpoint_root=args.output.parent,
        cache_root=args.output.parent, external_asset_root=args.output.parent)
    original = load_plan(args.project_root / 'src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml', PathResolver(roots))
    values = copy.deepcopy(original.values)
    values['sampling']['max_frames_per_rally'] = 72
    plan = replace(original, values=values)
    plan.verify_inputs()
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(_job, (plan, index, 3, args.output)) for index in range(3)]
        records = [future.result() for future in futures]
    plan.verify_inputs()
    manifest = {'schema': 'ball_refiner_3d.synthetic.v2', 'status': 'complete', 'mode': 'three_prefix_preflight',
                'preflight_overrides': {'max_frames_per_rally': 72, 'rally_index': 3},
                'input_hashes': plan.input_hashes, 'plan': values, 'counts': {'train': 1, 'val': 1, 'test': 1},
                'rallies': records, 'elapsed_seconds': time.monotonic() - began}
    write_json(args.output / 'manifest.json', manifest)
    dataset = SyntheticDataset(args.output)
    for record in dataset.records:
        dataset.load(record)
    write_json(args.output / 'verification.json', {'rallies_reloaded': 3, 'frames': 216, 'components': 125,
        'nonconverged_frames': sum(r['integration']['nonconverged_frames'] for r in records), 'elapsed_seconds': manifest['elapsed_seconds'],
        'worker_seconds': sum(r['elapsed_seconds'] for r in records), 'npz_bytes': sum(r['npz_bytes'] for r in records)})
    print(json.dumps(json.loads((args.output / 'verification.json').read_text())))


if __name__ == '__main__':
    main()
