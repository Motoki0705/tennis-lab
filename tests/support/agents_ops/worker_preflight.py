"""Completed implementation fixtures for tests of result/review authorization.

These tests do not exercise planning delivery. Seed its explicit skip and a
terminal offline runner record instead of adding agent turns to every case.
The public planning/runner connection is covered by the worker e2e tests.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

SEED = """
import hashlib, os, sys
from pathlib import Path
from lib.agent_settings import DEFAULT_CONFIG, resolve
from lib.identity import session_identity
from parent.github import GitHub
from parent.session import ParentSession
from parent.state import Store, state_directory
from parent import worker_preflight as phase

child = int(sys.argv[1])
store = Store(state_directory())
with store.locked():
    session = ParentSession(GitHub('o/r', store=store), store, 1, session_identity(dict(os.environ)))
    task = session.row['tasks'][str(child)]
    task['preflight'] = phase.new(task['delegation_id'])
    value, text = phase.prepare_turn(session, child, 'offline fixture', resume=False,
        skip_reason='結果・レビューの検証fixtureにある自明な1行変更。')
    directory = store.directory / 'fixture-workers' / task['delegation_id']
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = phase.commit_turn(task, value, directory)
    prompt_sha = hashlib.sha256(text.encode()).hexdigest()
    value['turns'][-1]['prompt_sha256'] = prompt_sha
    settings = resolve(DEFAULT_CONFIG, 'codex', None, None, None, 'write')
    contract = {'task_id': task['delegation_id'], 'cwd': task['worktree'],
        'branch': task['branch'], 'mode': 'write', 'settings': settings}
    record = {'schema_version': 1, 'run_id': 'fixture-worker', 'result': 'completed',
        'turn_end_reason': 'completed', 'exit_code': 0, 'task_success': None,
        'contract': contract, 'cleanup_complete': True, 'runner': None, 'process': None,
        'prompt_sha256': prompt_sha, 'runtime': {'session_id': 'offline-worker'}}
    Store.write_json(directory / 'result.json', record)
    Store.write_json(directory / 'invocation.json', {'schema_version': 1, 'contract': contract})
    task['worker_runs'] = [{'delegation_id': task['delegation_id'], 'run_dir': str(directory),
        'kind': 'registered', 'recorded_at': '2026-11-06T00:00:00+00:00', 'preflight': snapshot}]
    task['worker_ended'] = {'delegation_id': task['delegation_id'], 'evidence': 'offline fixture ended'}
    store.save()
"""


def completed_implementation(p: Any, child: int = 10) -> None:
    result = subprocess.run(
        [sys.executable, "-c", SEED, str(child)],
        cwd=p.repo,
        env=p.env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
