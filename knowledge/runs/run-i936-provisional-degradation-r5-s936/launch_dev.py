"""Detach the explicitly bounded 96-rally CPU development generation."""
import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
for name in ('project-root', 'data-root', 'dataset', 'run-output'):
    parser.add_argument('--' + name, type=Path, required=True)
parser.add_argument('--expected-seconds', type=float, required=True)
args = parser.parse_args()
if not all(p.is_absolute() for p in (args.project_root, args.data_root, args.dataset, args.run_output)):
    parser.error('Use absolute paths')
plan_path = args.project_root / 'src/tasks/ball_refiner/refiner_3d/dataset_plan.yaml'
plan = yaml.safe_load(plan_path.read_text())
if plan['counts']['dev_rallies'] != {'train': 64, 'val': 16, 'test': 16} or plan['simulation']['workers'] != 4:
    raise ValueError('This launch permits exactly 96 rallies and four CPU workers')
if args.expected_seconds <= 0 or args.dataset.exists() or args.run_output.exists():
    raise ValueError('Need a positive estimate and fresh output paths')
mem = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line}
if mem['MemAvailable'] < 6 * 1024 ** 3:
    raise RuntimeError('Less than 6 GiB available; generation was not launched')
head = subprocess.check_output(['git', '-C', str(args.project_root), 'rev-parse', 'HEAD'], text=True).strip()
subprocess.run(['git', '-C', str(args.project_root), 'diff', '--quiet', 'HEAD'], check=True)
args.run_output.mkdir(parents=True, exist_ok=False)
log = args.run_output / 'generation.log'
command = ['setsid', 'nohup', 'env', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=',
    'nice', '-n', '10', str(args.project_root / '.venv/bin/python'), '-u', '-m', 'src.tasks.ball_refiner.scripts.generate_synthetic_3d',
    '--project-root', str(args.project_root), '--data-root', str(args.data_root), '--plan', str(plan_path),
    '--output', str(args.dataset), '--mode', 'dev']
started = datetime.now(timezone.utc)
with log.open('xb') as stream:
    process = subprocess.Popen(command, cwd=args.project_root, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
record = {'schema': 'i936.detached_dev_generation.v1', 'status': 'launched', 'pid': process.pid, 'commit': head,
    'started_at': started.isoformat(), 'expected_seconds': args.expected_seconds,
    'expected_finish_jst': (started + timedelta(seconds=args.expected_seconds)).astimezone(timezone(timedelta(hours=9))).isoformat(),
    'command': shlex.join(command), 'cwd': str(args.project_root), 'dataset': str(args.dataset),
    'progress': str(args.dataset / 'manifest.json'), 'per_rally_progress': str(args.dataset / '*.progress.json'),
    'log': str(log), 'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(), 'available_ram_bytes_at_launch': mem['MemAvailable']}
(args.run_output / 'pid').write_text(str(process.pid) + '\n')
(args.run_output / 'job.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
