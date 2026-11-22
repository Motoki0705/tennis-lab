"""Run the Actions entry points on pinned Git event and knowledge fixtures."""

from __future__ import annotations

import importlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"
REPORT = "knowledge/reports/record.md"
SOURCE = "src/example.py"
CHECKS = ("graph", "repro")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=ci@example.test", *args],
        cwd=root,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


def _write(root: Path, relative: str, content: str = "Test fixture\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", message)
    return _git(root, "rev-parse", "HEAD")


def _merge(root: Path, head: str) -> tuple[str, str]:
    _git(root, "switch", "-q", "main")
    base = _git(root, "rev-parse", "HEAD")
    _git(root, "merge", "--no-ff", "-qm", "GitHub test merge", head)
    return base, _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-qb", "main")
    _write(repo, REPORT)
    _write(repo, SOURCE)
    _commit(repo, "Common ancestor")
    return repo


@pytest.fixture
def knowledge_ci(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(OPS))
    return importlib.import_module("lib.knowledge_ci")


@pytest.fixture(scope="module")
def workflows() -> dict[str, Any]:
    return {
        name: yaml.safe_load((ROOT / f".github/workflows/{name}.yml").read_text())
        for name in ("ci", "knowledge")
    }


def _actions_env(
    root: Path, event: str, head: str, base: str, merge: str
) -> dict[str, str]:
    payload = (
        {"pull_request": {"number": 12, "head": {"sha": head}, "base": {"sha": base}}}
        if event == "pull_request"
        else {"ref": "refs/heads/main", "before": base, "after": merge}
    )
    path = root.parent / "event.json"
    path.write_text(json.dumps(payload))
    return {
        **os.environ,
        "GITHUB_EVENT_NAME": event,
        "GITHUB_SHA": merge,
        "GITHUB_EVENT_PATH": str(path),
        "GITHUB_OUTPUT": str(root.parent / "output"),
        "PYTHONPATH": str(OPS),
    }


def _scope(
    repo: Path, step: dict[str, Any], env: dict[str, str]
) -> tuple[dict[str, str], str]:
    output = Path(env["GITHUB_OUTPUT"])
    output.unlink(missing_ok=True)
    command = shlex.split(step["run"])
    result = subprocess.run(
        [sys.executable, *command[1:]],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return dict(
        line.split("=", 1) for line in output.read_text().splitlines()
    ), result.stdout


def test_workflows_only_route_record_prs_to_the_light_checks(
    repo: Path,
    workflows: dict[str, Any],
) -> None:
    _git(repo, "switch", "-qc", "topic")
    (repo / REPORT).rename(repo / "knowledge/reports/renamed.md")
    for path in (
        "knowledge/nodes/fixture/.sequence",
        "knowledge/nodes/fixture/000001-run-example.md",
        "knowledge/runs/run-example/repro.sh",
        "knowledge/runs/run-example/repro.py",
        "knowledge/Papers/paper-example/paper.md",
        "knowledge/citation_map/papers/cite-example.md",
        "knowledge/summary.md",
    ):
        _write(repo, path)
    head = _commit(repo, "Records only")
    _git(repo, "switch", "-q", "main")
    _write(repo, SOURCE, "Base-only changes are not part of the PR.\n")
    _commit(repo, "Advance main")
    base, merge = _merge(repo, head)
    ci_steps = workflows["ci"]["jobs"]["scope"]["steps"]
    knowledge = workflows["knowledge"]["jobs"]
    library_steps = knowledge["knowledge"]["steps"]
    ci_scope = next(step for step in ci_steps if step.get("id") == "knowledge")
    library_scope = next(step for step in library_steps if step.get("id") == "scope")
    pr = _actions_env(repo, "pull_request", head, base, merge)
    expected = {
        "records_only": "true",
        "pr_number": "12",
        "head_sha": head,
        "base_sha": base,
        "merge_sha": merge,
    }
    for step in (ci_scope, library_scope):
        assert _scope(repo, step, pr)[0] == expected
    push = _actions_env(repo, "push", head, base, merge)
    for step in (ci_scope, library_scope):
        output, log = _scope(repo, step, push)
        assert output == {"records_only": "false"}
        assert "push uses ordinary CI" in log

    assert knowledge["knowledge-webui"]["needs"] == ["knowledge"]
    assert knowledge["knowledge-webui"]["if"] == (
        "${{ !cancelled() && needs.knowledge.outputs.records_only != 'true' }}"
    )
    assert knowledge["knowledge"]["outputs"]["records_only"] == (
        "${{ steps.scope.outputs.records_only }}"
    )
    assert all(
        step.get("if") == "steps.scope.outputs.records_only != 'true'"
        for step in library_steps
        if "-m pytest" in step.get("run", "")
    )
    triggers = workflows["knowledge"].get("on", workflows["knowledge"].get(True))
    assert triggers["push"]["branches"] == ["main"]
    assert "knowledge/**" in triggers["push"]["paths"]


@pytest.mark.parametrize(
    "path",
    [
        SOURCE,
        "knowledge/webui/main.js",
        "knowledge/registry/deployments.yaml",
        ".agents/skills/knowledge-control/scripts/kg_validate.py",
        ".agents/skills/knowledge-control/scripts/kg_citation_reading.schema.json",
        "knowledge/README.md",
        "knowledge/run-storage-exceptions.json",
        ".github/workflows/ci.yml",
        "knowledge/reports-backup/copy.md",
    ],
)
def test_mixed_diff_reports_the_outside_path(
    repo: Path,
    knowledge_ci: Any,
    path: str,
) -> None:
    _git(repo, "switch", "-qc", "topic")
    _write(repo, REPORT, "Changed report\n")
    _write(repo, path, "Changed out-of-scope file\n")
    head = _commit(repo, "Mixed diff")
    base, merge = _merge(repo, head)
    decision = knowledge_ci.classify_pr(
        repo, head_sha=head, base_sha=base, merge_sha=merge
    )
    assert not decision.records_only
    assert f"non-record path {path!r}" in decision.reason


@pytest.mark.parametrize(
    "operation", ["move-out", "move-in", "delete-outside", "symlink"]
)
def test_diff_checks_both_rename_paths_and_file_type(
    repo: Path,
    knowledge_ci: Any,
    operation: str,
) -> None:
    _git(repo, "switch", "-qc", "topic")
    if operation == "move-out":
        _git(repo, "mv", REPORT, "src/report.md")
        outside = "src/report.md"
    elif operation == "move-in":
        _git(repo, "mv", SOURCE, "knowledge/reports/example.py")
        outside = SOURCE
    elif operation == "delete-outside":
        _git(repo, "rm", SOURCE)
        outside = SOURCE
    else:
        (repo / REPORT).unlink()
        (repo / REPORT).symlink_to("../../src/example.py")
        outside = REPORT
    head = _commit(repo, operation)
    base, merge = _merge(repo, head)
    decision = knowledge_ci.classify_pr(
        repo, head_sha=head, base_sha=base, merge_sha=merge
    )
    assert not decision.records_only
    assert outside in decision.reason


def test_unpinned_or_unavailable_git_evidence_requests_ordinary_ci(
    repo: Path,
    workflows: dict[str, Any],
    knowledge_ci: Any,
) -> None:
    _git(repo, "switch", "-qc", "topic")
    _write(repo, REPORT, "Changed report\n")
    head = _commit(repo, "Records")
    base, merge = _merge(repo, head)
    env = _actions_env(repo, "pull_request", head, "", merge)
    scope = workflows["ci"]["jobs"]["scope"]["steps"][-1]
    output, log = _scope(repo, scope, env)
    assert output == {"records_only": "false"}
    assert "PR base must be a full commit SHA" in log

    stale = knowledge_ci.classify_pr(
        repo, head_sha=head, base_sha=head, merge_sha=merge
    )
    assert not stale.records_only and "parents do not match" in stale.reason
    missing = knowledge_ci.classify_pr(
        repo, head_sha=head, base_sha=base, merge_sha="9" * 40
    )
    assert not missing.records_only and "rev-list" in missing.reason
    shallow = repo.parent / "shallow"
    _git(repo, "clone", "--depth=2", repo.as_uri(), str(shallow))
    incomplete = knowledge_ci.classify_pr(
        shallow, head_sha=head, base_sha=base, merge_sha=merge
    )
    assert not incomplete.records_only and "full Git history" in incomplete.reason


def test_record_topic_cannot_exempt_an_outside_change_in_the_test_merge(
    repo: Path,
    knowledge_ci: Any,
) -> None:
    _git(repo, "switch", "-qc", "topic")
    _write(repo, REPORT, "Changed report\n")
    head = _commit(repo, "Records")
    base, _ = _merge(repo, head)
    _write(repo, SOURCE, "Unexpected change from a merge\n")
    _git(repo, "add", SOURCE)
    tree = _git(repo, "write-tree")
    merge = _git(
        repo, "commit-tree", tree, "-p", base, "-p", head, "-m", "Overbroad merge"
    )
    decision = knowledge_ci.classify_pr(
        repo, head_sha=head, base_sha=base, merge_sha=merge
    )
    assert not decision.records_only
    assert f"test merge diff changes non-record path {SOURCE!r}" in decision.reason


def _node(repo: Path, sequence: int, slug: str) -> None:
    _write(repo, "knowledge/nodes/fixture/.sequence", f"{sequence}\n")
    _write(
        repo,
        f"knowledge/nodes/fixture/{sequence:06d}-run-{slug}.md",
        f"---\nid: run-{slug}\ntype: run\ntask: fixture\nsequence: {sequence}\n"
        "recorded_at: '2026-11-22'\ntitle: Test run\nstatus: done\nissue: 12\n"
        "summary_impact: none\n---\n\nThis synthetic record checks the knowledge CI.\n",
    )


@pytest.mark.parametrize(
    ("fault", "failed_check", "diagnostic"),
    [
        (None, None, ""),
        ("collision", "graph", "new sequence must exceed base allocation 2"),
        ("deletion", "graph", "deletion record size/SHA-256 mismatch with base blob"),
        ("reference", "repro", "not in the run bundle: missing.py"),
    ],
)
def test_light_checks_validate_the_merged_tree_and_record_the_outcomes(
    repo: Path,
    workflows: dict[str, Any],
    knowledge_ci: Any,
    fault: str | None,
    failed_check: str | None,
    diagnostic: str,
) -> None:
    _node(repo, 1, "existing")
    _write(repo, "knowledge/summary.md", "A substantive judgement for the fixture.\n")
    for name, contents in {
        "tasks.yaml": "tasks: {fixture: {status: active, note: Test task}}\n",
        "relations.yaml": "relations: {related: {description: A related record}}\n",
        "deployments.yaml": "configs: []\ndeployments: []\n",
    }.items():
        _write(repo, f"knowledge/registry/{name}", "schema_version: 1\n" + contents)
    old_array = "knowledge/runs/run-existing/old.npy"
    if fault == "deletion":
        _write(repo, old_array, "data")
    fork = _commit(repo, "Existing library")
    _git(repo, "switch", "-qc", "topic")
    _node(repo, 2, "new")
    _write(
        repo, "knowledge/runs/run-new/check.py", "raise RuntimeError('do not run')\n"
    )
    reference = "missing.py" if fault == "reference" else "check.py"
    _write(
        repo, "knowledge/runs/run-new/repro.sh", f'python "$SCRIPT_DIR/{reference}"\n'
    )
    if fault == "deletion":
        (repo / old_array).unlink()
        _write(
            repo,
            "knowledge/runs/run-existing/artifact-manifest.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": "run-existing",
                    "files": [
                        {
                            "path": "old.npy",
                            "size_bytes": 4,
                            "sha256": "0" * 64,
                            "reason": "raw-data-events-archives",
                            "removed_from_commit": fork,
                        }
                    ],
                }
            ),
        )
    head = _commit(repo, "New record with reproduction scripts")
    if fault == "collision":
        _git(repo, "switch", "-q", "main")
        _write(repo, "knowledge/nodes/fixture/.sequence", "2\n")
        _commit(repo, "Main allocates sequence 2 without retaining a node")
    base, merge = _merge(repo, head)
    env = _actions_env(repo, "pull_request", head, base, merge)
    steps = {
        step["id"]: step
        for step in workflows["knowledge"]["jobs"]["knowledge"]["steps"]
        if "id" in step
    }
    scope, _ = _scope(repo, steps["scope"], env)
    assert scope["records_only"] == "true"
    assert steps["graph"]["env"]["BASE_REF"].startswith(
        "${{ steps.scope.outputs.base_sha ||"
    )
    env.update(BASE_REF=scope["base_sha"], KNOWLEDGE_DIR=str(repo / "knowledge"))

    outcomes = dict.fromkeys(CHECKS, "skipped")
    for step in CHECKS:
        command = steps[step]["run"].replace(
            ".venv/bin/python", shlex.quote(sys.executable)
        )
        command = command.replace(
            ".agents/skills/knowledge-control/scripts",
            shlex.quote(str(ROOT / ".agents/skills/knowledge-control/scripts")),
        )
        check = subprocess.run(
            ["bash", "-e", "-c", command],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert check.returncode == (1 if step == failed_check else 0), (
            check.stdout + check.stderr
        )
        outcomes[step] = "success" if check.returncode == 0 else "failure"
        if check.returncode:
            assert diagnostic in check.stdout
            break

    env.update(
        RUNNER_TEMP=str(repo.parent),
        GITHUB_RUN_ID="142",
        GITHUB_RUN_ATTEMPT="3",
        KG_PR_NUMBER=scope["pr_number"],
        KG_HEAD_SHA=scope["head_sha"],
        KG_BASE_SHA=scope["base_sha"],
        KG_MERGE_SHA=scope["merge_sha"],
        KG_VALIDATE_OUTCOME=outcomes["graph"],
        KG_REPRO_PATHS_OUTCOME=outcomes["repro"],
    )
    command = steps["result"]["run"].replace("python3", shlex.quote(sys.executable), 1)
    subprocess.run(
        ["bash", "-e", "-c", command],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        timeout=10,
    )
    result = (repo.parent / "knowledge-records.json").read_bytes()
    data = json.loads(result)
    assert (
        data["merge_sha"] == merge
        and data["base_sha"] == base
        and data["head_sha"] == head
    )
    assert data["conclusion"] == ("failure" if fault else "success")
    assert data["checks"] == {
        "kg_validate": outcomes["graph"],
        "kg_repro_paths": outcomes["repro"],
    }
    assert (
        "artifact_name=knowledge-records-142-3\n"
        in Path(env["GITHUB_OUTPUT"]).read_text()
    )
    if fault is None:
        run = {
            "event": "pull_request",
            "path": ".github/workflows/knowledge.yml",
            "head_sha": head,
            "status": "completed",
            "conclusion": "success",
            "id": 142,
            "run_attempt": 3,
        }
        assert (
            knowledge_ci.verify_result(
                result, run, pr_number=12, head_sha=head, base_sha=base, merge_sha=merge
            )
            is None
        )
