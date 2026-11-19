"""Queue-owned sequential processes, with a terminal file for event monitoring."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKOUTS = {
    'main': (Path('/home/kamimura/projects/tennis-lab/.claude/worktrees/scene-inference-runtime-baseline'),
             '486aa1ab09be857909d0fcf8c2b616c75a53f5b0'),
    'runtime': (Path('/home/kamimura/projects/tennis-lab/.claude/worktrees/scene-inference-runtime'),
                'ab638fb2ab825e84275863552da3f47473e390ed'),
}


def main():
    if not os.environ.get('TENNIS_RUN_ID'):
        raise RuntimeError('Run through the shared training queue')
    terminal = {'queue_job': os.environ['TENNIS_RUN_ID'], 'status': 'running', 'returncodes': {}}
    try:
        for variant, (checkout, expected) in CHECKOUTS.items():
            actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip()
            if actual != expected:
                raise ValueError(f'{variant} HEAD changed: {actual}')
            subprocess.run(['git', 'diff', '--exit-code', 'HEAD', '--', 'src'], cwd=checkout, check=True)
        manifest = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                    for name in ('benchmark.py', 'compare.py', 'inputs.json', 'run_pair.py')}
        (HERE / 'scripts.json').write_text(json.dumps(manifest, indent=2) + '\n')
        for variant, (checkout, _) in CHECKOUTS.items():
            print('Starting ' + variant, flush=True)
            with (HERE / (variant + '.log')).open('x') as log:
                process = subprocess.run([str(checkout / '.venv/bin/python'), '-u', str(HERE / 'benchmark.py'),
                                          '--directory', str(HERE / variant), '--pin', str(HERE / 'inputs.json')],
                                         cwd=checkout, stdout=log, stderr=subprocess.STDOUT)
            terminal['returncodes'][variant] = process.returncode
            print(f'{variant} exit={process.returncode}', flush=True)
        comparison = subprocess.run([sys.executable, str(HERE / 'compare.py'), str(HERE),
                                     '--report', str(HERE / 'comparison.json')])
        terminal['returncodes']['comparison'] = comparison.returncode
        terminal['status'] = 'complete' if all(code == 0 for code in terminal['returncodes'].values()) else 'failed'
    except BaseException as exc:
        terminal.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        temporary = HERE / 'terminal.json.tmp'
        temporary.write_text(json.dumps(terminal, indent=2) + '\n')
        temporary.replace(HERE / 'terminal.json')
    return 0 if terminal['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
