"""Run AutoDream against real Git history and the fake GitHub publisher."""

from __future__ import annotations

import fcntl
import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
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

autodream = importlib.import_module("shared_memory.autodream")
publish = importlib.import_module("shared_memory.publish")

NOW = datetime(2026, 1, 2, 0, 0, tzinfo=UTC)
JOB = OPS / "shared_memory/autodream_job.py"


def _state_dir(repo: Repository) -> Path:
    return repo.tmp / "autodream-state"


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
    return repo.bare("rev-parse", "main")


def _job(
    repo: Repository, *options: str, launch: bool = True
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(JOB),
        "run",
        "--state-dir",
        str(_state_dir(repo)),
        *options,
    ]
    if launch:
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
    assert status["last_consolidated_sha"] == base
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

    with pytest.raises(
        ValueError, match="Prune & Index: .*entry changed since it was inspected"
    ):
        _run(repo, concurrent, NOW + timedelta(hours=1), force=True)
    result = _state(repo)
    assert result["last_outcome"] == "failed"
    assert "entry changed since it was inspected" in result["last_error"]
    assert result["last_completed_at"] == before["last_completed_at"]
    assert result["last_consolidated_sha"] == before["last_consolidated_sha"]
    assert "Reverified" in repo.bare("show", f"main:{OLD}")
    repo.assert_no_publication()


def test_publication_fails_if_new_main_introduces_a_stale_entry(
    memory_repo: Repository,
) -> None:
    repo = memory_repo
    now = datetime(2026, 4, 1, tzinfo=UTC)

    def concurrent(*args: Any, **kwargs: Any) -> str:
        write_entry(
            repo.source,
            f"{MEMORY}/gotcha/earlier.md",
            last_verified="2025-12-31",
            created="2025-12-31",
            title="Earlier report",
        )
        sync_documents(repo.source)
        _commit_main(repo, "Add an observation over 90 days old")
        return _plan(update(summary="Fresh evidence."))()

    with pytest.raises(ValueError, match=r"Prune & Index: [\s\S]*\[stale\]"):
        _run(repo, concurrent, now, force=True)
    assert _state(repo)["last_outcome"] == "failed"
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
    assert _state(repo)["last_consolidated_sha"] == base
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


@pytest.mark.parametrize("scenario", ["success", "quota"])
def test_job_runs_read_only_on_the_pinned_archive(
    memory_repo: Repository, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    repo = memory_repo
    today = date.today().isoformat()
    for path in (CURRENT, OLD):
        write_entry(repo.source, path, created=today, last_verified=today, citations=[])
    (repo.source / "escaped.md").unlink()
    sync_documents(repo.source)
    base = _commit_main(repo, "Refresh verified timestamps")
    # The current checkout differs from remote main.
    (repo.source / "reference.md").write_text("Uncommitted change.\n")
    checkout_before = repo.snapshot()
    codex = repo.tmp / "bin/codex"
    codex.write_text(
        f"#!{sys.executable}\n"
        "from pathlib import Path\nimport sys\n"
        "assert Path('reference.md').read_text().startswith('# Current heading')\n"
        "assert not Path('.agents/memory/ops-report').exists()\n"
        "assert not Path('.git').exists()\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        "from fake_agent_cli import main\nmain()\n"
    )
    codex.chmod(0o700)
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
    for key, value in {
        "CODEX_HOME": str(repo.tmp / "codex-home"),
        "FAKE_SCENARIO": scenario,
        "FAKE_STRUCTURED_OUTPUT_FILE": str(plan),
        "FAKE_ARGS": str(repo.tmp / "args"),
        "FAKE_CWD": str(repo.tmp / "cwd"),
    }.items():
        monkeypatch.setenv(key, value)

    completed = _job(repo, "--force", "--dry-run")
    assert completed.returncode == (0 if scenario == "success" else 1), completed.stderr
    argv = (repo.tmp / "args").read_text().splitlines()
    assert argv[argv.index("--sandbox") + 1] == "read-only"
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
