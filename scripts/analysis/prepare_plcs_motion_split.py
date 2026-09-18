"""Create an immutable PLCS dataset version with source-motion-disjoint splits."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from src.tasks.base.generate_dataset import resolve_court_keypoint_contract
from src.tasks.plcs.generate_dataset.io.dataset_io import PLCSDatasetWriter
from src.tennis_scene.generate_dataset.manifest import file_sha256
from src.utils.io import save_json_atomic


def prepare(source: Path, destination: Path, *, seed: int) -> None:
    source, destination = source.resolve(), destination.resolve()
    if source == destination or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Source and versioned destination must be separate directories')
    identity = {'source': str(source), 'source_meta_sha256': file_sha256(source / 'meta.json'),
                'source_scenes_meta_sha256': file_sha256(source / 'scenes_meta.json'),
                'split_group': 'motion_source', 'seed': seed, 'val_ratio': .1, 'test_ratio': .1}
    receipt = destination / 'dataset_version.json'
    if destination.exists():
        if receipt.is_file() and json.loads(receipt.read_text()) == identity:
            print(f'Using verified split version {destination}')
            return
        raise ValueError(f'Destination is incomplete or has another recipe: {destination}')
    staging = destination.with_name(destination.name + '.building')
    if staging.exists():
        raise ValueError(f'Incomplete previous split preparation: {staging}')
    meta = json.loads((source / 'meta.json').read_text())
    writer = PLCSDatasetWriter(staging, court_keypoint_contract=resolve_court_keypoint_contract('physical_v1'))
    for entry in source.iterdir():
        if entry.name in {'train.txt', 'val.txt', 'test.txt', 'split_info.json'}:
            continue
        target = staging / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, copy_function=os.link, dirs_exist_ok=True)
        else:
            shutil.copyfile(entry, target)
    source_records = {record['scene_id']: record for record in json.loads((source / 'scenes_meta.json').read_text())}
    if set(source_records) != {record['scene_id'] for record in meta['scenes']}:
        raise ValueError('Scene inventory and motion-source metadata disagree')
    writer.scene_records = [{**record, 'motion_source': source_records[record['scene_id']]['motion_source']} for record in meta['scenes']]
    writer.save_motion_group_splits(val_ratio=.1, test_ratio=.1, seed=seed)
    save_json_atomic(identity, staging / receipt.name)
    staging.rename(destination)
    print((destination / 'split_info.json').read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not args.source.is_absolute() or not args.destination.is_absolute():
        parser.error('source and destination must be explicit absolute paths')
    prepare(args.source, args.destination, seed=args.seed)


if __name__ == '__main__':
    main()
