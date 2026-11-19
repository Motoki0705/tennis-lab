"""A fixture's Git work must finish before its directory can be replaced."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tests.e2e.agents_ops.test_parent_session import ParentEnv, run_git
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.support.agents_ops.directory_seed import DirectorySeed

OBSERVE_GIT = """
import json, os, subprocess, sys
from pathlib import Path
from lib.opsproc import enable_child_subreaper

# Scope adoption to this observer; never reap another test's worker process.
enable_child_subreaper()
repo, trace = map(Path, sys.argv[1:])
subprocess.run(
    ['git', 'commit', '--allow-empty', '-m', 'maintenance lifecycle'],
    cwd=repo, env={**os.environ, 'GIT_TRACE2_EVENT': str(trace)},
    check=True, capture_output=True,
)
graph_at_return = (repo / '.git/objects/info/commit-graphs/commit-graph-chain').is_file()
adopted = []
while True:
    try:
        pid, status = os.waitpid(-1, 0)
    except ChildProcessError:
        break
    adopted.append({'pid': pid, 'status': status})
print(json.dumps({'adopted': adopted, 'graph_at_return': graph_at_return}))
"""


def test_seed_restore_waits_for_git_maintenance(
    parent_env: ParentEnv, tmp_path_factory: pytest.TempPathFactory
) -> None:
    p = parent_env
    # Force real useful maintenance, independent of repository size or host policy.
    for key, value in (
        ("maintenance.auto", "true"),
        ("maintenance.gc.enabled", "false"),
        ("maintenance.commit-graph.enabled", "true"),
        ("maintenance.commit-graph.auto", "-1"),
    ):
        run_git(p.repo, "config", key, value)
    host_config = p.root / "host-git-config"
    host_config.write_text("[maintenance]\n\tautoDetach = true\n[gc]\n\tautoDetach = true\n")
    trace = p.root / "git-trace.jsonl"
    result = subprocess.run(
        [sys.executable, "-c", OBSERVE_GIT, str(p.repo), str(trace)],
        env={**p.env, "GIT_CONFIG_GLOBAL": str(host_config), "GIT_CONFIG_NOSYSTEM": "1"},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    observed = json.loads(result.stdout)
    assert not observed["adopted"], observed
    assert observed["graph_at_return"], observed
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(e.get("name") == "maintenance" for e in events)
    assert not any(e.get("label") == "detach" for e in events)
    assert not (p.repo / ".git/objects/maintenance.lock").exists()

    cache = tmp_path_factory.mktemp("maintenance-seed")
    original = cache / "source"
    original.mkdir()
    (original / "replacement").write_text("ready\n")
    seed = DirectorySeed.capture(original, cache / "snapshot")
    seed.restore(p.root)
    assert (p.root / "replacement").read_text() == "ready\n"
    assert not p.repo.exists()
