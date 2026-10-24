"""Published memory uses actual ordinary review records and the existing gate."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops import test_parent_session as parents
from tests.e2e.agents_ops.test_daily_ci_gate import call_gate, connect_gate
from tests.e2e.agents_ops.test_merge_gate import GateEnv, make_gate_env
from tests.e2e.agents_ops.test_ops_report_memory import ROOT, _update
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env
from tests.e2e.agents_ops.test_parent_session import run_git


@pytest.fixture
def memory_gate(
    parent_env: Any, monkeypatch: pytest.MonkeyPatch
) -> tuple[GateEnv, Path]:
    p = parent_env
    for key, value in p.env.items():
        monkeypatch.setenv(key, value)
    memory = importlib.import_module("opsreport.memory")
    publisher = importlib.import_module("opsreport.memory_pr")
    journal_path = (
        Path(p.env["XDG_STATE_HOME"]) / "tennis-lab-agents/ops-report/memory.json"
    )

    def setup(p: Any, **kwargs: Any) -> list[int]:
        shutil.copytree(ROOT / memory.DIRECTORY, p.repo / memory.DIRECTORY)
        run_git(p.repo, "add", memory.DIRECTORY)
        run_git(p.repo, "commit", "-m", "memory baseline")
        base = run_git(p.repo, "rev-parse", "HEAD")
        report = (
            "<!-- ops-report:v2 slot=20261012 since=2026-10-09T08:00:00+09:00 "
            "until=2026-10-12T08:00:00+09:00 proposals=0 status=complete -->\n"
            "<!-- ops-report:proposals:begin -->\n<!-- ops-report:proposals:end -->\n"
        )
        store = publisher.Store(p.repo, journal_path)
        row = {
            "state": "pending",
            "stage": "review_handoff",
            "error": None,
            "base_sha": base,
            "files": memory.read_tree(p.repo, base),
            "update": _update(),
            "publication": {
                "title": "report",
                "body_sha256": hashlib.sha256(report.encode()).hexdigest(),
                "author": "operator",
            },
            "report_url": "https://github.com/o/r/issues/90",
            "head": None,
            "pr_url": "https://github.com/o/r/pull/100",
            "work_issue": "https://github.com/o/r/issues/1",
            "validation": None,
            "work_issue_create_attempted": True,
        }
        expected = memory.apply_update(
            row["files"], row["update"], source=row["report_url"], today=date.today()
        )
        branch = "ops-report-memory/20261012"
        row["head"] = store._commit("20261012", branch, row, expected)
        store._verify_commit(row, expected)
        row["validation"] = store._validation(row, expected)
        store.records["20261012"] = row
        store.save()
        run_git(p.root, "clone", "--bare", str(p.repo), str(p.root / "remote.git"))
        run_git(p.repo, "remote", "add", "origin", str(p.root / "remote.git"))
        run_git(p.repo, "fetch", "origin")
        state = p.state()
        state["issues"]["1"].update(
            title="memory review",
            labels=[{"name": "ai-proposed"}],
            body=store._work_body("20261012", row),
        )
        state["issues"]["90"] = {
            **state["issues"]["1"],
            "number": 90,
            "id": 10090,
            "body": report,
            "title": "report",
            "labels": [{"name": "ops-report"}],
        }
        pr = p.pr(100, 1, row["head"], branch)
        pr.update(body=store._pr_body("20261012", row), html_url=row["pr_url"])
        state["prs"] = [pr]
        p.save(state)
        p.agreed()  # Existing fixture authorization; exporter never emits agreement.
        p.cli(
            "record-result",
            "--child",
            "1",
            "--evidence-file",
            p.note("固定memoryの実diffとローカル検証を確認。"),
        )
        return [100]

    # Extend the same fake REST service with gh's view projection used by publisher.
    gh = p.root / "bin/gh"
    shebang, body = gh.read_text().split("\n", 1)
    intercept = """
import json, os, sys
args = sys.argv[1:]
if args[:2] in (["issue", "view"], ["pr", "view"]):
    state = json.load(open(os.environ["PARENT_GH_STATE"]))
    n = int(args[2].rsplit("/", 1)[-1])
    if args[0] == "issue":
        raw = state["issues"][str(n)]
        value = {**raw, "state": raw["state"].upper(), "url": f"https://github.com/o/r/issues/{n}", "author": {"login": "operator"}}
    else:
        raw = next(p for p in state["prs"] if p["number"] == n)
        value = {"number": n, "url": raw["html_url"], "state": "MERGED" if raw.get("merged_at") else raw["state"].upper(),
                 "body": raw["body"], "headRefName": raw["head"]["ref"], "headRefOid": raw["head"]["sha"],
                 "baseRefName": "main", "baseRefOid": raw["base"]["sha"], "isDraft": raw["draft"],
                 "isCrossRepository": False, "mergeable": "MERGEABLE", "statusCheckRollup": []}
    print(json.dumps({k: value[k] for k in args[args.index("--json") + 1].split(",")}))
    raise SystemExit(0)
"""
    gh.write_text(shebang + "\n" + intercept + body)
    monkeypatch.setattr(parents, "review_setup", setup)
    gate = make_gate_env(p, monkeypatch, "standalone")
    return gate, journal_path


@pytest.mark.parametrize(
    "case", ["success", "no_review", "contents", "head", "missing_journal"]
)
def test_memory_uses_ordinary_gate_without_ci(
    memory_gate: tuple[GateEnv, Path], case: str
) -> None:
    g, journal = memory_gate
    if case == "no_review":
        (g.review.run_dir / "result.json").unlink()
    elif case == "contents":
        value = json.loads(journal.read_text())
        value["records"]["20261012"]["update"]["summary"] = "unpublished change"
        journal.write_text(json.dumps(value))
    elif case == "head":
        state = g.review.p.state()
        state["prs"][0]["head"]["sha"] = state["prs"][0]["base"]["sha"]
        g.review.p.save(state)
    elif case == "missing_journal":
        journal.unlink()
    response = g.cli("merge")
    assert (response.returncode == 0) == (case == "success"), (
        response.stdout + response.stderr
    )
    assert len(g.merge_calls()) == int(case == "success")


def test_memory_normal_merge_obeys_same_daily_freeze(
    memory_gate: tuple[GateEnv, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    g, _ = memory_gate
    module, api, clock = connect_gate(g, monkeypatch)
    code, result, error = call_gate(module, capsys)
    assert code == 1, error
    assert "daily_ci_normal_merge_frozen" in result["results"][0]["reasons"]
    assert not g.merge_calls()
    clock.set("2026-10-25T09:00:00+09:00")
    api.expire()
    code, result, error = call_gate(module, capsys)
    assert code == 0, error
    assert result["results"][0]["action"] == "merged"


def test_memory_publication_is_an_unowned_dispatcher_review_candidate(
    memory_gate: tuple[GateEnv, Path],
) -> None:
    g, _ = memory_gate
    p = g.review.p
    # Read the production snapshot and its report closure through the fake REST API.
    module = importlib.import_module("dispatcher.collect")
    schema = importlib.import_module("dispatcher.schema")
    gh = importlib.import_module("parent.github").GitHub("o/r")
    store = importlib.import_module("parent.state").Store(
        p.root / "empty-dispatcher-registry"
    )
    state = p.state()
    state["comments"]["1"] = []  # Newly published handoff, before parent assignment.
    for issue in state["issues"].values():
        issue["assignees"] = []
    p.save(state)
    with store.locked():
        snapshot = module.collect(gh, store, p.repo, seeds=[1])
    assert 90 in snapshot["issues"]["1"]["source_reports"]
    assert "90" in snapshot["issues"]
    assert schema.guard(snapshot, 1) is None
    # No handoff may spawn a worker in the gap between issue and PR publication.
    original = snapshot["pulls"]
    snapshot["pulls"] = []
    assert schema.guard(snapshot, 1)["disposition"] == "information_needed"
    snapshot["pulls"] = original
    snapshot["pulls"][0]["head"]["sha"] = "f" * 40
    assert schema.guard(snapshot, 1)["disposition"] == "existing_owner"
    snapshot["issues"]["1"]["labels"].append({"name": "needs-human"})
    assert schema.guard(snapshot, 1)["disposition"] == "human_hold"
