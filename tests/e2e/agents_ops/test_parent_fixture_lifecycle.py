"""A fixture's Git work must finish before its directory can be replaced."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.e2e.agents_ops.test_issue_supplement import (
    controller_seed as controller_seed,
)
from tests.e2e.agents_ops.test_parent_session import OPS, ParentEnv, run_git
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


def _assert_git_maintenance_finished(repo: Path, env: dict[str, str]) -> None:
    # Force real useful maintenance, independent of repository size or host policy.
    for key, value in (
        ("maintenance.auto", "true"),
        ("maintenance.gc.enabled", "false"),
        ("maintenance.commit-graph.enabled", "true"),
        ("maintenance.commit-graph.auto", "-1"),
    ):
        run_git(repo, "config", key, value)
    host_config = repo.parent / "host-git-config"
    host_config.write_text(
        "[maintenance]\n\tautoDetach = true\n[gc]\n\tautoDetach = true\n"
    )
    trace = repo.parent / "git-trace.jsonl"
    result = subprocess.run(
        [sys.executable, "-c", OBSERVE_GIT, str(repo), str(trace)],
        env={
            **env,
            "PYTHONPATH": str(OPS),
            "GIT_CONFIG_GLOBAL": str(host_config),
            "GIT_CONFIG_NOSYSTEM": "1",
        },
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
    assert not (repo / ".git/objects/maintenance.lock").exists()


def test_seed_restore_waits_for_git_maintenance(
    parent_env: ParentEnv, tmp_path_factory: pytest.TempPathFactory
) -> None:
    p = parent_env
    _assert_git_maintenance_finished(p.repo, p.env)

    cache = tmp_path_factory.mktemp("maintenance-seed")
    original = cache / "source"
    original.mkdir()
    (original / "replacement").write_text("ready\n")
    seed = DirectorySeed.capture(original, cache / "snapshot")
    seed.restore(p.root)
    assert (p.root / "replacement").read_text() == "ready\n"
    assert not p.repo.exists()


def test_controller_seed_waits_for_git_maintenance(
    controller_seed: DirectorySeed, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    controller_seed.restore(repo)
    _assert_git_maintenance_finished(repo, dict(os.environ))
    # Older Git only detaches garbage collection. Check both repo-local settings.
    for key in ("maintenance.autoDetach", "gc.autoDetach"):
        detaches = run_git(repo, "config", "--local", "--default=true", "--get", key)
        assert detaches == "false", key

    seed = DirectorySeed.capture(repo, tmp_path / "snapshot")
    graph = seed.snapshot / ".git/objects/info/commit-graphs/commit-graph-chain"
    assert graph.is_file()
