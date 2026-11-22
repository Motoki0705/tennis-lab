"""Run AutoDream against real Git history and the fake GitHub publisher."""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import tomllib
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tests.support.agents_ops.shared_memory import (
    CURRENT,
    LAYERS,
    MEMORY,
    OLD,
    OPS,
    SUMMARY,
    Repository,
    add,
    delete,
    sync_documents,
    update,
    write_entry,
)
from tests.support.agents_ops.shared_memory import memory_repo as memory_repo
from tests.support.d_capacity import CONFIG_ENV
from tests.support.fake_d_capacity.sitecustomize import BYTES_FILE

autodream = importlib.import_module("shared_memory.autodream")
publish = importlib.import_module("shared_memory.publish")

NOW = datetime(2026, 1, 2, 0, 0, tzinfo=UTC)
JOB = OPS / "shared_memory/autodream_job.py"


def _state_dir(repo: Repository) -> Path:
    directory: Path = repo.tmp
    return directory / "autodream-state"


def _state(repo: Repository) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(
        (_state_dir(repo) / "autodream_state.json").read_text()
    )
    return result


def _run(
    repo: Repository, runner: Callable[..., str], when: datetime = NOW, **kwargs: Any
) -> dict[str, Any]:
    result: dict[str, Any] = autodream.run(
        repo.source,
        state_dir=_state_dir(repo),
        runner=runner,
        clock=lambda: when,
        **kwargs,
    )
    return result


def _plan(
    *operations: dict[str, object], repair_index: bool = False
) -> Callable[..., str]:
    raw = json.dumps(
        {
            "summary": "確認済みの観察を統合する。",
            "operations": operations,
            "repair_index": repair_index,
        }
    )
    return lambda *args, **kwargs: raw


def _commit_main(repo: Repository, message: str) -> str:
    repo.git("add", "-A")
    repo.git("commit", "-qm", message)
    repo.git("push", "origin", "HEAD:refs/heads/main")
    head: str = repo.bare("rev-parse", "main")
    return head


def _job(
    repo: Repository, *options: str, launch: bool = True, unit: bool = False
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(JOB),
        "run",
        "--state-dir",
        str(_state_dir(repo)),
        *options,
    ]
    if unit:
        for relative, target in (
            (".venv", Path(sys.prefix)),
            (".agents/ops/checkout_job.py", OPS / "checkout_job.py"),
            (".agents/ops/shared_memory/autodream_job.py", JOB),
        ):
            link = repo.source / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(target)
        service = JOB.parent / "systemd/tennis-lab-memory-autodream.service"
        exec_start = next(
            line.removeprefix("ExecStart=")
            for line in service.read_text().splitlines()
            if line.startswith("ExecStart=")
        )
        command = [
            *shlex.split(
                exec_start.replace("%h/projects/tennis-lab", str(repo.source))
            ),
            "--state-dir",
            str(_state_dir(repo)),
            *options,
        ]
    elif launch:
        command[:0] = [
            sys.executable,
            str(OPS / "checkout_job.py"),
            "--repo",
            str(repo.source),
            "--",
        ]
    return subprocess.run(
        command,
        cwd=repo.source,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def test_all_layers_apply_add_update_and_delete_at_observed_version(
    memory_repo: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = memory_repo
    specs = []
    expected_digests = {}
    for layer in LAYERS:
        revised = f"{MEMORY}/{layer}/revise.md"
        retired = f"{MEMORY}/{layer}/obsolete.md"
        write_entry(
            repo.source,
            revised,
            title=f"{layer} original",
            summary=f"{layer} initial.",
        )
        write_entry(
            repo.source,
            retired,
            title=f"{layer} workaround",
            summary=f"{layer} superseded.",
            applies_to="all",
            source="Published trial notes",
            citations=[],
            evidence="Archived outcome confirms this workaround was superseded.",
            body="A previous handoff used a second manually inspected queue before approval.",
        )
        for path in (revised, retired):
            file = repo.source / path
            file.write_bytes(file.read_bytes().replace(b"\n", b"\r\n"))
            expected_digests[path] = sha256(file.read_bytes()).hexdigest()
        removal = delete(layer, "obsolete")
        if layer == "decision":
            removal["expected_sha256"] = expected_digests[retired]
        specs.extend(
            [
                update(
                    "revise",
                    layer=layer,
                    title=f"{layer} checked",
                    summary=f"{layer} corrected.",
                ),
                removal,
                add(layer, "addition", title=f"{layer} added", summary=f"{layer} new."),
            ]
        )
    sync_documents(repo.source)
    base = _commit_main(repo, "Seed entries for all layers")
    original = publish.publish_operations

    def check_versions(
        source: Path, batch: list[dict[str, object]], **kwargs: Any
    ) -> Any:
        assert {
            f"{MEMORY}/{op['layer']}/{op['slug']}.md": op["expected_sha256"]
            for op in batch
            if op["op"] != "add"
        } == expected_digests
        assert all("expected_sha256" not in op for op in batch if op["op"] == "add")
        return original(source, batch, **kwargs)

    monkeypatch.setattr(publish, "publish_operations", check_versions)
    before = repo.snapshot()
    dedicated = f"{MEMORY}/ops-report/SUMMARY.md"
    result = _run(repo, _plan(*specs), force=True)
    status = _state(repo)
    assert status["last_outcome"] == "merged"
    assert status["last_trigger"]["reasons"] == ["forced"]
    assert status["last_pr"] == 17
    assert status["last_merge_sha"] == repo.bare("rev-parse", "main")
    assert result["publication"]["base"] == base
    assert status["last_consolidated_sha"] == status["last_merge_sha"]
    assert status["last_completed_at"] == NOW.isoformat()
    assert status["last_error"] is None
    assert len(repo.transactions()) == 1
    assert repo.snapshot() == before
    assert repo.bare("show", f"{base}:{dedicated}") == repo.bare(
        "show", f"main:{dedicated}"
    )
    summary = repo.bare("show", f"main:{SUMMARY}")
    for layer in LAYERS:
        index = repo.bare("show", f"main:{MEMORY}/{layer}/index.md")
        assert f"[{layer} checked](revise.md)" in index
        assert f"[{layer} added](addition.md)" in index
        assert "obsolete.md" not in index
        assert f"- {layer} corrected." in summary
        assert f"- {layer} new." in summary
        assert f"- {layer} superseded." not in summary
    assert len(result["publication"]["operations"]) == 12


def test_concurrent_reverification_rejects_deletion(memory_repo: Repository) -> None:
    repo = memory_repo
    _run(repo, _plan(), force=True)
    before = _state(repo)

    def concurrent(*args: Any, **kwargs: Any) -> str:
        repo.parallel_commit("refreshed-entry", ref="main", target=OLD)
        return _plan(delete())()

    with pytest.raises(ValueError, match="Prune & Index: .*remote main changed"):
        _run(repo, concurrent, NOW + timedelta(hours=1), force=True)
    result = _state(repo)
    assert result["last_outcome"] == "failed"
    assert "replan AutoDream" in result["last_error"]
    assert result["last_completed_at"] == before["last_completed_at"]
    assert result["last_consolidated_sha"] == before["last_consolidated_sha"]
    assert "Reverified" in repo.bare("show", f"main:{OLD}")
    repo.assert_no_publication()


def test_unrelated_main_change_also_requires_replanning(
    memory_repo: Repository,
) -> None:
    repo = memory_repo

    def concurrent(*args: Any, **kwargs: Any) -> str:
        (repo.source / "unrelated.md").write_text("# Unrelated change\n")
        _commit_main(repo, "Change another part of main")
        return _plan(update(summary="Fresh evidence."))()

    with pytest.raises(ValueError, match="Prune & Index: .*remote main changed"):
        _run(repo, concurrent, force=True)
    assert _state(repo)["last_outcome"] == "failed"
    repo.assert_no_publication()


def test_deletion_refused_when_the_duplicate_it_relies_on_disappears(
    memory_repo: Repository,
) -> None:
    repo = memory_repo
    shared = {
        "summary": "The evidence is the same in both observations.",
        "citations": [],
    }
    write_entry(repo.source, CURRENT, **shared)
    write_entry(repo.source, OLD, **shared)
    sync_documents(repo.source)
    base = _commit_main(repo, "Two entries preserve the same observation")
    previously_seen = repo.bare("show", f"{base}:{CURRENT}")

    def concurrent(*args: Any, **kwargs: Any) -> str:
        (repo.source / OLD).unlink()
        sync_documents(repo.source)
        _commit_main(repo, "Remove the copy the earlier plan relied on")
        return _plan(
            delete("workflow", "current", reason="同じ観察を decision に残す。")
        )()

    with pytest.raises(ValueError, match="remote main changed.*replan"):
        _run(repo, concurrent, force=True)
    assert _state(repo)["pending_publication"] is None
    assert repo.bare("ls-tree", "main", "--", OLD) == ""
    assert repo.bare("show", f"main:{CURRENT}") == previously_seen
    repo.assert_no_publication()


@pytest.mark.parametrize("request_repair", [False, True])
def test_generated_drift_repairs_only_the_index_files(
    memory_repo: Repository, request_repair: bool
) -> None:
    repo = memory_repo
    _run(repo, _plan(), force=True)
    original = repo.bare("show", f"main:{SUMMARY}")
    (repo.source / SUMMARY).write_text("Wrong generated document.\n")
    base = _commit_main(repo, "Drift generated summary")
    result = _run(repo, _plan(repair_index=request_repair), NOW + timedelta(hours=1))
    assert _state(repo)["last_trigger"]["reasons"] == ["generated_drift"]
    assert result["publication"]["base"] == base
    assert _state(repo)["last_consolidated_sha"] == repo.bare("rev-parse", "main")
    assert result["publication"]["operations"] == ()
    assert result["publication"]["files"] == {SUMMARY: "modified"}
    assert result["publication"]["index_repair_reason"]
    assert repo.bare("show", f"main:{SUMMARY}") == original


def test_noop_sets_completion_then_a_skip_only_records_the_check(
    memory_repo: Repository,
) -> None:
    repo = memory_repo
    base = repo.bare("rev-parse", "main")
    result = _run(repo, _plan(), force=True)
    assert result["last_outcome"] == "noop"
    assert result["last_completed_at"] == NOW.isoformat()
    assert result["last_consolidated_sha"] == base
    assert result["publication"] is None

    def unexpected(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("planner called with unmet thresholds")

    later = NOW + timedelta(hours=167)
    result = _run(repo, unexpected, later)
    status = _state(repo)
    assert result["last_outcome"] == "skipped_trigger"
    assert status["last_run_at"] == later.isoformat()
    assert status["last_completed_at"] == NOW.isoformat()
    assert status["last_consolidated_sha"] == base
    assert status["last_trigger"] == {
        "reasons": [],
        "elapsed_seconds": 167 * 3600,
        "main_commits": 0,
        "memory_commits": 0,
        "audit_problems": 0,
        "broken_citations": 0,
        "stale_entries": 0,
        "limits_exceeded": 0,
        "generated_drift": 0,
    }
    repo.assert_no_publication()


def test_history_uses_first_parent_and_reports_changed_evidence(
    memory_repo: Repository,
) -> None:
    repo = memory_repo
    write_entry(repo.source, CURRENT, summary="Same observation.")
    sync_documents(repo.source)
    _commit_main(repo, "Observe a citation before consolidation")
    _run(repo, _plan(), force=True)
    repo.git("switch", "-qc", "feature")
    write_entry(
        repo.source,
        f"{MEMORY}/gotcha/same.md",
        summary="Same observation.",
        title="A similar observation",
    )
    sync_documents(repo.source)
    repo.git("add", "-A")
    repo.git("commit", "-qm", "Add shared evidence on feature")
    (repo.source / "reference.md").write_text(
        "# Current heading\n\nA revised reference.\n"
    )
    (repo.source / "AGENTS.md").write_text("# Rules\n")
    (repo.source / ".agents/README.md").write_text("# Agent rules\n")
    (repo.source / ".agents/skills/memory/SKILL.md").write_text(
        "# Skill\n\n## 形式\n\n新しい手順\n"
    )
    repo.git("add", "-A")
    repo.git("commit", "-qm", "Promote a workflow and revise its evidence")
    repo.git("switch", "-q", "callers-branch")
    (repo.source / f"{MEMORY}/ops-report/SUMMARY.md").write_text("Dedicated update.\n")
    _commit_main(repo, "Update dedicated memory")
    (repo.source / "unrelated.md").write_text("# Unrelated\n")
    _commit_main(repo, "Update unrelated file")
    repo.git("merge", "-q", "--no-ff", "feature", "-m", "Merge feature")
    repo.git("push", "origin", "HEAD:refs/heads/main")
    result = _run(
        repo,
        _plan(),
        NOW + timedelta(days=7),
        dry_run=True,
        thresholds=autodream.Thresholds(main_commits=4, memory_commits=1),
    )
    trigger, signals = result["last_trigger"], result["signals"]
    assert trigger["reasons"] == ("memory_commits",)
    assert trigger["main_commits"] == 3
    assert trigger["memory_commits"] == 1
    assert signals["changed_citations"] == {
        CURRENT: ["reference.md"],
        f"{MEMORY}/gotcha/same.md": ["reference.md"],
    }
    assert signals["promotion_paths"] == [
        ".agents/README.md",
        ".agents/skills/memory/SKILL.md",
        "AGENTS.md",
    ]
    assert signals["duplicate_candidates"] == [(f"{MEMORY}/gotcha/same.md", CURRENT)]
    assert _state(repo)["last_completed_at"] == NOW.isoformat()
    repo.assert_no_publication()


def test_removed_citation_is_an_immediate_repair_signal(
    memory_repo: Repository,
) -> None:
    repo = memory_repo
    write_entry(repo.source, CURRENT)
    sync_documents(repo.source)
    _commit_main(repo, "Create a cited observation")
    _run(repo, _plan(), force=True)
    (repo.source / "reference.md").rename(repo.source / "replacement.md")
    _commit_main(repo, "Move the cited reference")
    repair = _plan(
        update(
            citations=[{"path": "replacement.md", "symbol": "#current-heading"}],
            body="See [the current reference](../../../replacement.md#current-heading).",
        )
    )
    result = _run(repo, repair, NOW + timedelta(hours=1), dry_run=True)
    assert result["last_trigger"]["reasons"] == ("broken_citations",)
    assert result["signals"]["changed_citations"] == {CURRENT: ["reference.md"]}
    assert result["publication"]["files"] == {CURRENT: "modified"}
    assert _state(repo)["last_completed_at"] == NOW.isoformat()
    repo.assert_no_publication()


def test_entry_over_limit_is_an_immediate_repair_signal(
    memory_repo: Repository,
) -> None:
    repo = memory_repo
    _run(repo, _plan(), force=True)
    entry = repo.source / CURRENT
    with entry.open("a", encoding="utf-8") as stream:
        stream.write("Repeated evidence.\n" * 41)
    _commit_main(repo, "Exceed the entry line budget")
    result = _run(
        repo,
        _plan(update(body="Rechecked and summarized the evidence.")),
        NOW + timedelta(hours=1),
        dry_run=True,
    )
    assert result["last_trigger"]["reasons"] == ("limits_exceeded",)
    assert result["last_trigger"]["limits_exceeded"] == 1
    assert result["publication"]["files"] == {CURRENT: "modified"}
    assert _state(repo)["last_completed_at"] == NOW.isoformat()
    repo.assert_no_publication()


def test_missing_citations_triggers_even_when_indexes_match_and_age_is_short(
    memory_repo: Repository,
) -> None:
    repo = memory_repo
    _run(repo, _plan(), force=True)
    path = f"{MEMORY}/gotcha/incomplete.md"
    write_entry(repo.source, path, citations=[])
    entry = repo.source / path
    entry.write_text(entry.read_text().replace("citations: []\n", ""))
    _commit_main(repo, "Add incomplete entry without changing the generated documents")
    result = _run(
        repo,
        _plan(
            {
                "op": "update",
                "layer": "gotcha",
                "slug": "incomplete",
                "reason": "引用欄を復元する。",
                "citations": [],
            }
        ),
        NOW + timedelta(hours=1),
        dry_run=True,
    )
    trigger = result["last_trigger"]
    assert trigger["reasons"] == ("broken_citations",)
    assert trigger["generated_drift"] == 0 and trigger["memory_commits"] == 1
    assert result["signals"]["audit"]["broken_citations"] == [
        {
            "path": path,
            "code": "citation",
            "message": "missing frontmatter fields: citations",
        }
    ]
    assert result["last_outcome"] == "dry_run"
    repo.assert_no_publication()


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (lambda *a, **kw: "not JSON", "strict JSON"),
        (
            lambda *a, **kw: (
                '{"summary":"Checked","operations":[],"repair_index":true,"other":false}'
            ),
            "JSON Schema",
        ),
        (_plan(delete("ops-report")), "JSON Schema"),
        (_plan(add(slug="../../escape")), "slug must be kebab-case"),
        (
            _plan(add(citations=[{"path": f"{MEMORY}/ops-report/SUMMARY.md"}])),
            "dedicated ops-report",
        ),
        (
            _plan(update(citations=[{"path": "missing.md"}])),
            "target file does not exist",
        ),
        (_plan(update(body="password=fixture-only")), "credential assignment"),
        (_plan(delete(expected_sha256="f" * 64)), "expected_sha256 differs"),
    ],
)
def test_rejected_plan_records_failure_before_publication(
    memory_repo: Repository, runner: Callable[..., str], message: str
) -> None:
    repo = memory_repo
    _run(repo, _plan(), force=True)
    old = _state(repo)
    with pytest.raises(ValueError, match=message):
        _run(repo, runner, NOW + timedelta(hours=1), force=True)
    status = _state(repo)
    assert status["last_outcome"] == "failed"
    assert message in status["last_error"]
    assert "fixture-only" not in status["last_error"]
    assert status["last_completed_at"] == old["last_completed_at"]
    assert status["last_consolidated_sha"] == old["last_consolidated_sha"]
    repo.assert_no_publication()


def test_unresolved_stale_entry_cannot_be_confirmed_as_noop(
    memory_repo: Repository,
) -> None:
    repo = memory_repo
    previous = datetime(2026, 4, 1, tzinfo=UTC)
    _run(repo, _plan(), previous, force=True)
    with pytest.raises(ValueError, match=r"\[stale\] last verified 91 days ago"):
        _run(repo, _plan(), previous + timedelta(days=1))
    status = _state(repo)
    assert status["last_completed_at"] == previous.isoformat()
    assert status["last_trigger"]["reasons"] == ["stale_entries"]
    assert status["last_outcome"] == "failed"
    repo.assert_no_publication()


def test_checkout_launcher_and_lock_contention(memory_repo: Repository) -> None:
    repo = memory_repo
    direct = _job(repo, "--force", launch=False)
    assert direct.returncode == 75
    assert "use checkout_job.py" in direct.stderr
    assert not _state_dir(repo).exists()
    _run(repo, _plan(), force=True)
    state = _state_dir(repo) / "autodream_state.json"
    before = state.read_bytes()
    with (_state_dir(repo) / "autodream.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        overlap = _job(repo, "--force")
    assert overlap.returncode == 1
    assert "AutoDream lock contention" in overlap.stderr
    assert before == state.read_bytes()
    repo.assert_no_publication()


def test_fetch_failure_exits_nonzero_and_keeps_completion(
    memory_repo: Repository,
) -> None:
    repo = memory_repo
    _run(repo, _plan(), force=True)
    completed = _job(repo, "--force", "--remote", "missing")
    assert completed.returncode == 1
    assert "Orient: git ls-remote failed" in completed.stderr
    status = _state(repo)
    assert status["last_outcome"] == "failed"
    assert status["last_completed_at"] == NOW.isoformat()
    assert status["last_trigger"] is None
    repo.assert_no_publication()


def _prepare_runnable_main(repo: Repository) -> str:
    today = date.today().isoformat()
    for path in (CURRENT, OLD):
        write_entry(repo.source, path, created=today, last_verified=today, citations=[])
    (repo.source / "escaped.md").unlink()
    sync_documents(repo.source)
    return _commit_main(repo, "Refresh verified timestamps")


def _install_fake_codex(
    repo: Repository, monkeypatch: pytest.MonkeyPatch, scenario: str = "success"
) -> None:
    plan = repo.tmp / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "summary": "引用先を再検証した。",
                "repair_index": False,
                "operations": [
                    {
                        "op": "update",
                        "layer": "decision",
                        "slug": "old",
                        "reason": "根拠を追加",
                        "expected_sha256": None,
                        "title": None,
                        "summary": None,
                        "applies_to": None,
                        "source": None,
                        "created": None,
                        "last_verified": None,
                        "evidence": None,
                        "citations": [
                            {"path": "reference.md", "symbol": None, "note": None}
                        ],
                        "body": None,
                    }
                ],
            }
        )
    )
    settings = {
        "FAKE_SCENARIO": scenario,
        "FAKE_STRUCTURED_OUTPUT_FILE": str(plan),
        "FAKE_ARGS": str(repo.tmp / "args"),
        "FAKE_CWD": str(repo.tmp / "cwd"),
        "FAKE_CHILD": str(repo.tmp / "agent-child"),
    }
    codex = repo.tmp / "bin/codex"
    codex.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\nimport os\nimport sys\n"
        "assert Path('reference.md').read_text().startswith('# Current heading')\n"
        "assert not Path('.agents/memory/ops-report').exists()\n"
        "assert not Path('.git').exists()\n"
        "assert 'AUTODREAM_SOURCE_MARKER' not in os.environ\n"
        "assert 'PYTHONPATH' not in os.environ and 'GIT_WORK_TREE' not in os.environ\n"
        "assert not (Path(os.environ['CODEX_HOME']) / 'AGENTS.md').exists()\n"
        f"Path({str(repo.tmp / 'agent-pid')!r}).write_text(str(os.getpid()))\n"
        f"os.environ.update({settings!r})\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from fake_agent_cli import main\nmain()\n"
    )
    codex.chmod(0o700)
    # AutoDream gives the real runner an allowlisted environment. Inject only
    # its capacity probe, then remove the bootstrap path before Codex is launched.
    probe = repo.tmp / "autodream-runner-probe"
    probe.mkdir(exist_ok=True)
    (probe / "sitecustomize.py").write_text(
        "import os, sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(OPS)!r})\n"
        "from lib import d_capacity\n"
        "sys.path.pop(0)\n"
        f"d_capacity.CONFIG_PATH = Path({os.environ[CONFIG_ENV]!r})\n"
        "d_capacity.measure_available_bytes = "
        f"lambda: int(Path({os.environ[BYTES_FILE]!r}).read_text())\n"
        "del os.environ['PYTHONPATH']\n"
    )
    bash = codex.with_name("bash")
    bash.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = {shlex.quote(str(OPS / "lib/agent_exec.sh"))} ]; then\n'
        f"    export PYTHONPATH={shlex.quote(str(probe))}\n"
        'fi\nexec /bin/bash "$@"\n'
    )
    bash.chmod(0o700)
    host_home = repo.tmp / "host-codex"
    host_home.mkdir(exist_ok=True)
    (host_home / "AGENTS.md").write_text("Host-only instructions.\n")
    monkeypatch.setenv("CODEX_HOME", str(host_home))
    monkeypatch.setenv("AUTODREAM_SOURCE_MARKER", str(repo.source))


@pytest.mark.parametrize("scenario", ["success", "quota"])
def test_job_runs_read_only_on_the_pinned_archive(
    memory_repo: Repository, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    repo = memory_repo
    base = _prepare_runnable_main(repo)
    # The current checkout differs from remote main.
    (repo.source / "reference.md").write_text("Uncommitted change.\n")
    checkout_before = repo.snapshot()
    _install_fake_codex(repo, monkeypatch, scenario)

    completed = _job(repo, "--force", "--dry-run")
    assert completed.returncode == (0 if scenario == "success" else 1), completed.stderr
    argv = (repo.tmp / "args").read_text().splitlines()
    assert "--sandbox" not in argv
    assert 'default_permissions="memory_autodream"' in argv
    permission = next(arg for arg in argv if arg.startswith("permissions="))
    rules = tomllib.loads(permission)["permissions"]["memory_autodream"]
    assert rules == {
        "filesystem": {
            ":minimal": "read",
            (repo.tmp / "cwd").read_text(): "read",
            str(repo.tmp / "bin/codex"): "read",
        },
        "network": {"enabled": False},
    }
    assert "--strict-config" in argv and "--ignore-user-config" in argv
    assert argv[argv.index("--model") + 1] == "gpt-6.2-sol"
    assert 'model_reasoning_effort="high"' in argv
    assert 'service_tier="default"' in argv
    assert "--skip-git-repo-check" in argv and "--output-schema" in argv
    assert not Path((repo.tmp / "cwd").read_text()).parent.exists()
    assert repo.snapshot() == checkout_before
    status = _state(repo)
    assert status["last_completed_at"] is None
    assert status["last_consolidated_sha"] is None
    if scenario == "success":
        assert status["last_outcome"] == "dry_run"
        result = json.loads(completed.stdout)
        assert result["base"] == base
        assert result["validation"] == "passed"
        assert result["signals"]["audit"]["stale"] == []
        assert result["publication"]["files"] == {OLD: "modified"}
        assert result["plan"]["operations"] == [
            {
                "op": "update",
                "layer": "decision",
                "slug": "old",
                "reason": "根拠を追加",
                "expected_sha256": sha256((repo.source / OLD).read_bytes()).hexdigest(),
                "citations": [{"path": "reference.md"}],
            }
        ]
    else:
        assert status["last_outcome"] == "failed"
        assert "read-only agent failed" in status["last_error"]
        logs = list((_state_dir(repo) / "agent-runs").glob("*/result.json"))
        assert len(logs) == 1
        assert "usage limit exceeded" in logs[0].read_text()
    repo.assert_no_publication()


@pytest.mark.parametrize("refusal", ["server_refused", "main_advanced"])
def test_definite_publication_refusal_clears_intent_without_rebase(
    memory_repo: Repository, monkeypatch: pytest.MonkeyPatch, refusal: str
) -> None:
    repo = memory_repo
    previous = _run(repo, _plan(), force=True)
    base = repo.bare("rev-parse", "main")
    if refusal == "main_advanced":
        wanted = repo.parallel_commit("new-entry")
        case: dict[str, object] = {"advance_to": wanted}
    else:
        wanted = base
        case = {"conflicts": [409]}
    monkeypatch.setenv("MEMORY_TEST_CASE", json.dumps(case))
    monkeypatch.setenv(
        "MEMORY_TEST_INTENT_FILE", str(_state_dir(repo) / "autodream_state.json")
    )

    with pytest.raises(ValueError, match="replan AutoDream"):
        _run(
            repo,
            _plan(update(summary="Rechecked evidence.")),
            NOW + timedelta(hours=1),
            force=True,
        )
    status = _state(repo)
    assert status["last_outcome"] == "failed"
    assert status["pending_publication"] is None
    assert status["last_completed_at"] == previous["last_completed_at"]
    assert status["last_consolidated_sha"] == base
    assert repo.bare("rev-parse", "main") == wanted
    assert len(repo.transactions()) == 1
    assert len(repo.git_calls("push")) == 1


@pytest.mark.parametrize(
    ("fault", "landed"),
    [
        ("response_lost_after_merge", True),
        ("response_lost_before_merge", False),
        ("signal_after_merge", True),
        ("signal_before_merge", False),
    ],
)
def test_next_run_reconciles_pending_publication_before_planning(
    memory_repo: Repository, monkeypatch: pytest.MonkeyPatch, fault: str, landed: bool
) -> None:
    repo = memory_repo
    prior = _run(repo, _plan(), force=True)
    base = _prepare_runnable_main(repo)
    _install_fake_codex(repo, monkeypatch)
    monkeypatch.setenv("MEMORY_TEST_CASE", json.dumps({fault: True}))
    monkeypatch.setenv(
        "MEMORY_TEST_INTENT_FILE", str(_state_dir(repo) / "autodream_state.json")
    )
    result = _job(repo, "--force")
    assert result.returncode != 0, result.stdout
    error = (
        "HTTP 503" if fault.startswith("response_lost") else "interrupted by SIGTERM"
    )
    assert error in result.stderr
    failed = _state(repo)
    intent = failed["pending_publication"]
    assert failed["last_outcome"] == "failed"
    assert failed["last_completed_at"] == prior["last_completed_at"]
    assert failed["last_consolidated_sha"] == prior["last_consolidated_sha"]
    assert intent["base"] == base and intent["operation"] == "batch"
    assert intent["pr"] == 17 and intent["stage"] == "advancing"
    assert repo.bare("rev-parse", "main") == (intent["head"] if landed else base)
    assert len(repo.transactions()) == 1

    monkeypatch.delenv("MEMORY_TEST_CASE")
    # Recovery may not start a second model, branch or transaction.
    (repo.tmp / "bin/codex").write_text("#!/bin/sh\nexit 93\n")
    recovered = _job(repo, "--force")
    status = _state(repo)
    assert len(repo.transactions()) == 1
    assert sum(call["method"] == "POST" for call in repo.api_calls()) == 2
    assert len(list((_state_dir(repo) / "agent-runs").glob("*/result.json"))) == 1
    if landed:
        assert recovered.returncode == 0, recovered.stderr
        assert json.loads(recovered.stdout)["recovered"] is True
        assert status["last_outcome"] == "merged"
        assert status["last_pr"] == 17
        assert (
            status["last_merge_sha"]
            == status["last_consolidated_sha"]
            == intent["head"]
        )
        assert status["last_completed_at"] > failed["last_completed_at"]
        assert status["pending_publication"] is None
        assert repo.bare("for-each-ref", f"refs/heads/{intent['branch']}") == ""
    else:
        assert recovered.returncode != 0
        assert "not confirmed on main" in recovered.stderr
        assert "manual reconciliation required" in status["last_error"]
        assert status["last_outcome"] == "failed"
        assert status["last_completed_at"] == prior["last_completed_at"]
        assert status["pending_publication"] == intent
        assert len(repo.git_calls("push")) == 1


def test_confirmed_merge_advances_state_even_if_branch_cleanup_fails(
    memory_repo: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = memory_repo
    monkeypatch.setenv(
        "MEMORY_TEST_CASE", json.dumps({"change_branch_after_merge": True})
    )
    result = _run(repo, _plan(update(summary="A verified change.")), force=True)
    assert result["last_outcome"] == "merged"
    status = _state(repo)
    assert (
        status["last_merge_sha"]
        == status["last_consolidated_sha"]
        == repo.bare("rev-parse", "main")
    )
    assert status["last_completed_at"] == NOW.isoformat()
    assert status["last_error"] is None and status["pending_publication"] is None
    assert (
        status["branch_cleanup_error"] == result["publication"]["branch_cleanup_error"]
    )
    assert "shared-memory/" in status["branch_cleanup_error"]
    assert repo.bare("for-each-ref", "refs/heads/shared-memory")


def test_recovery_after_merge_and_cleanup_precede_the_completion_record(
    memory_repo: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = memory_repo
    prior = _run(repo, _plan(), force=True)
    save = autodream.write_state

    def stop_after_merge(path: Path, record: Any) -> None:
        if record.last_outcome == "merged":
            raise OSError("simulated stop before recording completion")
        save(path, record)

    with monkeypatch.context() as writing:
        writing.setattr(autodream, "write_state", stop_after_merge)
        with pytest.raises(OSError, match="before recording completion"):
            _run(
                repo,
                _plan(update(summary="Preserve the verified change.")),
                NOW + timedelta(hours=1),
                force=True,
            )
    persisted = _state(repo)
    assert persisted["last_outcome"] == "publishing"
    assert persisted["last_completed_at"] == prior["last_completed_at"]
    assert persisted["pending_publication"]["head"] == repo.bare("rev-parse", "main")
    assert repo.bare("for-each-ref", "refs/heads/shared-memory") == ""

    def unexpected(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("recovery started another plan")

    recovered = _run(repo, unexpected, NOW + timedelta(hours=2), force=True)
    assert recovered["recovered"] is True and recovered["last_outcome"] == "merged"
    assert (
        recovered["pending_publication"] is None
        and recovered["branch_cleanup_error"] is None
    )
    assert recovered["last_consolidated_sha"] == repo.bare("rev-parse", "main")
    assert len(repo.transactions()) == 1 and len(repo.git_calls("push")) == 2


@pytest.mark.parametrize("unit", [False, True], ids=["manual", "service-exec"])
def test_hung_agent_times_out_and_releases_the_job_resources(
    memory_repo: Repository, monkeypatch: pytest.MonkeyPatch, unit: bool
) -> None:
    repo = memory_repo
    _prepare_runnable_main(repo)
    _install_fake_codex(repo, monkeypatch, "detached")
    completed = _job(repo, "--force", "--runner-timeout", "0.75", unit=unit)
    assert completed.returncode == 1
    assert "reason=timeout" in completed.stderr
    status = _state(repo)
    assert (
        status["last_outcome"] == "failed" and "reason=timeout" in status["last_error"]
    )
    assert status["last_completed_at"] is None
    assert not Path((repo.tmp / "cwd").read_text()).parent.exists()
    assert not list((_state_dir(repo) / "agent-runs").glob(".codex-*"))
    for name in ("agent-pid", "agent-child"):
        pid = int((repo.tmp / name).read_text())
        assert not Path(f"/proc/{pid}").exists()
    for path in (
        _state_dir(repo) / "autodream.lock",
        repo.source / ".git/ops-code.lock",
    ):
        with path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    repo.assert_no_publication()


def test_outer_watchdog_reaps_a_detached_child_of_a_hung_runner_wrapper(
    memory_repo: Repository, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = memory_repo
    _prepare_runnable_main(repo)
    _install_fake_codex(repo, monkeypatch)
    wrapper = repo.tmp / "hung-wrapper.sh"
    child_pid = repo.tmp / "wrapper-child"
    observed_cwd = repo.tmp / "wrapper-cwd"
    wrapper.write_text(
        f"pwd > {shlex.quote(str(observed_cwd))}\n"
        f"setsid bash -c 'echo $$ > \"$1\"; exec sleep 60' bash {shlex.quote(str(child_pid))} &\n"
        "wait\n"
    )
    driver = repo.tmp / "job-with-detached-runner.py"
    driver.write_text(
        "import sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(OPS)!r})\n"
        "from lib.checkout_lock import require_checkout_job\nrequire_checkout_job()\n"
        "from shared_memory import autodream_job, autodream_plan\n"
        f"autodream_plan.AGENT_EXEC = Path({str(wrapper)!r})\n"
        "autodream_plan.RUNNER_CLEANUP_ALLOWANCE = 0\n"
        "raise SystemExit(autodream_job.main())\n"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(OPS / "checkout_job.py"),
            "--repo",
            str(repo.source),
            "--",
            sys.executable,
            str(driver),
            "run",
            "--force",
            "--runner-timeout",
            "0.5",
            "--state-dir",
            str(_state_dir(repo)),
        ],
        cwd=repo.source,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == 1
    assert "timed out after 0.5s" in _state(repo)["last_error"]
    assert not Path(f"/proc/{int(child_pid.read_text())}").exists()
    assert not Path(observed_cwd.read_text().strip()).parent.exists()
    for path in (
        _state_dir(repo) / "autodream.lock",
        repo.source / ".git/ops-code.lock",
    ):
        with path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    repo.assert_no_publication()


def test_codex_tool_cannot_read_source_checkout_or_other_host_files(
    tmp_path: Path,
) -> None:
    if shutil.which("codex") is None:
        pytest.skip("offline kernel sandbox test requires the installed Codex CLI")
    snapshot, original, codex_home = (
        tmp_path / name for name in ("pinned", "current", "codex-state")
    )
    for path in (snapshot, original, codex_home):
        path.mkdir()
    (snapshot / "reference.md").write_text("pinned-readable-marker\n")
    dedicated = original / ".agents/memory/ops-report/secret.md"
    dedicated.parent.mkdir(parents=True)
    uncommitted = original / "uncommitted.md"
    for path in (dedicated, uncommitted):
        path.write_text("host-only-marker\n")
    command = (
        "set -e; /usr/bin/cat reference.md; "
        + " ".join(
            f"if /usr/bin/cat {shlex.quote(str(path))}; then exit 53; fi;"
            for path in (dedicated, uncommitted)
        )
        + " if /usr/bin/touch probe-write; then exit 54; fi; echo sandbox-isolated"
    )
    requests: list[dict[str, Any]] = []

    class Responses(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            assert self.path == "/v1/responses"
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            response = {"id": f"resp-{len(requests)}"}
            item: dict[str, object]
            if len(requests) == 1:
                item = {
                    "type": "function_call",
                    "call_id": "probe",
                    "name": "exec_command",
                    "arguments": json.dumps(
                        {"cmd": command, "login": False, "workdir": str(snapshot)}
                    ),
                }
            else:
                item = {
                    "type": "message",
                    "role": "assistant",
                    "id": "msg-probe",
                    "content": [{"type": "output_text", "text": "probe complete"}],
                }
            events = [
                {"type": "response.created", "response": response},
                {"type": "response.output_item.done", "item": item},
                {
                    "type": "response.completed",
                    "response": {
                        **response,
                        "usage": {
                            "input_tokens": 12,
                            "output_tokens": 4,
                            "total_tokens": 16,
                            "input_tokens_details": None,
                            "output_tokens_details": None,
                        },
                    },
                },
            ]
            body = "".join(
                f"data: {json.dumps(event)}\n\n" for event in events
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    settings = importlib.import_module("lib.agent_settings")
    pins = settings.resolve(
        settings.DEFAULT_CONFIG, "codex", "memory_autodream", None, None, "read-only"
    )
    argv = settings.command(
        pins, "read-only", snapshot, tmp_path / "final.md", None, []
    )
    with ThreadingHTTPServer(("127.0.0.1", 0), Responses) as server:
        argv[-1:-1] = [
            "-c",
            'model_provider="fixture"',
            "-c",
            f'model_providers={{fixture={{name="fixture",base_url="http://127.0.0.1:{server.server_port}/v1",wire_api="responses",supports_websockets=false}}}}',
        ]
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            completed = subprocess.run(
                argv,
                cwd=snapshot,
                input="Run the fixed filesystem check.",
                text=True,
                env={
                    "PATH": os.defpath,
                    "LANG": "C.UTF-8",
                    "CODEX_HOME": str(codex_home),
                },
                capture_output=True,
                check=False,
                timeout=12,
            )
        finally:
            server.shutdown()
            worker.join(timeout=2)
    assert completed.returncode == 0, completed.stderr
    assert len(requests) == 2
    result = next(
        row["output"]
        for row in requests[1]["input"]
        if row.get("call_id") == "probe" and row["type"] == "function_call_output"
    )
    assert "pinned-readable-marker" in result and "sandbox-isolated" in result
    assert "host-only-marker" not in result
    assert not (snapshot / "probe-write").exists()
