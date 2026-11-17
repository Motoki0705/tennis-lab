"""One guarded main/runtime pair. Stop after any failed process or health anomaly."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from health_guard import run_variant, utc, write

HERE = Path(__file__).resolve().parent
CHECKOUTS = {
    'main': (Path('/home/kamimura/projects/tennis-lab/.claude/worktrees/scene-inference-runtime-baseline'),
             '18a2fea05291eeaccb6d1dbfc81160e26c4080dd'),
    'runtime': (Path('/home/kamimura/projects/tennis-lab/.claude/worktrees/scene-inference-runtime'),
                '1bef628a9c2c5761d241ddc5c1b98c0f89625b35'),
}


def execute_pair(root, checkouts, run=run_variant):
    outcomes = {}
    for name, (checkout, _) in checkouts.items():
        print('Starting ' + name, flush=True)
        outcome = run(root, name,
            [str(checkout / '.venv/bin/python'), '-u', str(root / 'benchmark.py'),
             '--directory', str(root / name), '--pin', str(root / 'inputs.json')], checkout)
        outcomes[name] = outcome
        print(f'{name}: {outcome["status"]}, exit={outcome.get("returncode")}', flush=True)
        if outcome['status'] != 'complete':
            break  # No second GPU process after a failure, CUDA-related or otherwise.
    return outcomes


def main():
    if not os.environ.get('TENNIS_RUN_ID'):
        raise RuntimeError('Run through the shared training queue')
    terminal = {'queue_job': os.environ['TENNIS_RUN_ID'], 'status': 'running',
                'started_at_utc': utc(), 'outcomes': {}}
    try:
        for name, (checkout, expected) in CHECKOUTS.items():
            actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
            if actual != expected:
                raise ValueError(f'{name} HEAD changed: {actual}')
            subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'src'], cwd=checkout, check=True)
        names = ('benchmark.py', 'compare.py', 'inputs.json', 'run_pair.py', 'health_guard.py')
        write(HERE / 'scripts.json', {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in names})
        terminal['outcomes'] = execute_pair(HERE, CHECKOUTS)
        if len(terminal['outcomes']) == 2 and all(item['status'] == 'complete' for item in terminal['outcomes'].values()):
            comparison = subprocess.run([sys.executable, str(HERE / 'compare.py'), str(HERE),
                                         '--report', str(HERE / 'comparison.json')])
            terminal['comparison_returncode'] = comparison.returncode
            terminal['status'] = 'complete' if comparison.returncode == 0 else 'failed'
        else:
            terminal['status'] = 'failed'
    except BaseException as exc:
        terminal.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        terminal['finished_at_utc'] = utc()
        terminal['anomaly'] = str(HERE / 'anomaly.json') if (HERE / 'anomaly.json').exists() else None
        write(HERE / 'terminal.json', terminal)
    return 0 if terminal['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
