"""Enforce the owned pilot session's deadline, independently of the conversation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

CLI = Path('.agents/skills/tennis-colab/scripts/colab.py')
STATE_ROOT = Path.home() / '.local/state/tennis-lab/colab'
TERMINAL = {'completed', 'failed', 'cancelled', 'abandoned'}


def elapsed(created_at: str, *, now: datetime | None = None) -> float:
    created = datetime.fromisoformat(created_at)
    if created.tzinfo is None:
        raise ValueError('Session creation time must include a timezone')
    return ((now or datetime.now(UTC)) - created).total_seconds()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def call(session: str, *argv: str) -> dict[str, Any]:
    result = subprocess.run([sys.executable, str(CLI), argv[0], '--session', session, *argv[1:]],
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError(f'Colab {argv[0]} failed: {result.stderr[-1000:]}')
    return cast(dict[str, Any], json.loads(result.stdout))


def run(args: argparse.Namespace) -> None:
    state_path = STATE_ROOT / args.session / 'state.json'
    if args.output.exists():
        raise FileExistsError('Use an absent guard receipt')
    state = json.loads(state_path.read_text())
    created_at = state['created_at']
    if state['session'] != args.session or state['commit'] != args.bootstrap_commit:
        raise ValueError('Owned session identity differs from the declared bootstrap')
    audit: dict[str, Any] = dict(schema_version=1, session=args.session, created_at=created_at,
                                stop_work_after_seconds=12600, hard_limit_seconds=14400, actions=[])
    write(args.output, audit)
    while elapsed(created_at) < 12600:
        state = json.loads(state_path.read_text())
        if state['phase'] == 'stopped':
            audit.update(result='already_stopped', observed_at=datetime.now(UTC).isoformat(),
                         elapsed_seconds=elapsed(created_at))
            write(args.output, audit)
            return
        time.sleep(min(30, max(0, 12600 - elapsed(created_at))))
    # No new job is submitted here. Only this exact owned session is touched.
    state = json.loads(state_path.read_text())
    for job in state.get('jobs', {}):
        status = call(args.session, 'status', '--job-id', job)
        actual = status['jobs'][0]
        if actual.get('status') not in TERMINAL:
            audit['actions'].append(call(args.session, 'cancel', '--job-id', job))
            write(args.output, audit)
    for job in state.get('jobs', {}):
        while True:
            status = call(args.session, 'status', '--job-id', job)
            actual = status['jobs'][0]
            if actual.get('status') in TERMINAL and actual.get('drive_saved_at'):
                break
            if actual.get('status') == 'unknown':
                audit['actions'].append(call(args.session, 'recover', '--job-id', job))
            elif actual.get('status') == 'save_failed':
                audit['actions'].append(call(args.session, 'save', '--job-id', job))
            if elapsed(created_at) > 13800:
                raise RuntimeError('Guard could not obtain a saved terminal job before the stop reserve')
            time.sleep(10)
    audit['actions'].append(call(args.session, 'stop'))
    doctor = subprocess.run([sys.executable, str(CLI), 'doctor', '--remote'],
                            capture_output=True, text=True, check=True, timeout=120)
    audit.update(result='stopped_by_deadline_guard', doctor=json.loads(doctor.stdout),
                 observed_at=datetime.now(UTC).isoformat(), elapsed_seconds=elapsed(created_at))
    write(args.output, audit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--bootstrap-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
