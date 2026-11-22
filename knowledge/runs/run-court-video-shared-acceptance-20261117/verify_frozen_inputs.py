"""Read inherited raw/LINE/RGB bytes against run73, writing only new evidence."""
import argparse
import json
from pathlib import Path
import time

from build import save, sha


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    metadata_path = Path('knowledge/runs/run-court-fixed-support-rgb-20261117/artifact-inventories.json')
    metadata = json.loads(metadata_path.read_text())
    report = {'source_inventory_metadata_sha256': sha(metadata_path), 'groups': {}}
    prefix = Path('outputs/tennis_scene/evaluate')
    for group in ('input-raw', 'input-val_raw', 'input-test_raw', 'line-evidence', 'rgb'):
        record = metadata[group]
        index = args.prior_output / Path(record['inventory_path']).relative_to(prefix)
        assert sha(index) == record['jsonl_sha256']
        root = args.prior_output / Path(record['root']).relative_to(prefix)
        count = size = 0
        for line in index.read_text().splitlines():
            item = json.loads(line)
            path = root / item['path']
            before = path.stat()
            assert before.st_size == item['bytes'] and sha(path) == item['sha256'], path
            assert path.stat().st_mtime_ns == before.st_mtime_ns, path
            count += 1; size += item['bytes']
        assert (count, size) == (record['files'], record['bytes'])
        report['groups'][group] = {'files': count, 'bytes': size, 'all_hashes_match': True, 'inventory_sha256': sha(index)}
        print(group, count, flush=True)
    report['seconds'] = time.monotonic() - started
    save(args.output, report)
