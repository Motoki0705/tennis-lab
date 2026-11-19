"""Reuse run 75's exact paired replay without changing historical evidence."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    destination = args.output / 'replay-source'
    destination.mkdir(parents=True, exist_ok=False)
    original = Path('knowledge/runs/run-court-legacy-default-20261117')
    for name in ('replay.py', 'legacy_regions.py', 'baseline-source.json'):
        shutil.copyfile(original / name, destination / name)
    spec = importlib.util.spec_from_file_location('paired_legacy_replay', destination / 'replay.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = args.output
    module.SAVED = args.prior_output / 'court-fixed-support-rgb-20261117'
    module.RAW = args.prior_output / 'court-acceptance-implementation-20261116'
    sys.argv = [str(destination / 'replay.py'), '--workers', str(args.workers)]
    module.main()
