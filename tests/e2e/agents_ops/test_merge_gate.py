"""Explicit merge gate against actual parent-run records, fake gh and real git."""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.agents_ops.test_parent_session import (
    OPS,
    ParentEnv,
    ReviewEnv,
    make_review_env,
    review_policy_module,
    run_git,
)
from tests.e2e.agents_ops.test_parent_session import (
    parent_env as parent_env,
)
from tests.support.agents_ops.directory_seed import DirectorySeed
from tests.support.agents_ops.worker_preflight import completed_implementation

SCRIPT = OPS / "review/merge_gate.py"
SEEDS = pytest.StashKey[dict[str | None, "GateSeed"]]()


def ci_rollup(state: str) -> list[dict[str, str]]:
    if state == "absent":
        return []
    name, status, conclusion = {
        "pending": ("Python", "QUEUED", ""),
        "failure": ("Python", "COMPLETED", "FAILURE"),
        "cancelled": ("Python", "COMPLETED", "CANCELLED"),
        "skipped": ("Python", "COMPLETED", "SKIPPED"),
        "neutral": ("Python", "COMPLETED", "NEUTRAL"),
        "timed_out": ("Python", "COMPLETED", "TIMED_OUT"),
        "success": ("Python", "COMPLETED", "SUCCESS"),
        "unrelated": ("Labeler", "COMPLETED", "SUCCESS"),
    }[state]
    return [
        {
            "__typename": "CheckRun",
            "name": name,
            "workflowName": "CI",
            "status": status,
            "conclusion": conclusion,
        }
    ]


def assert_unclassified_agreement_comment(
    response: subprocess.CompletedProcess[str], comment_id: int
) -> None:
    assert response.returncode == 2, response.stderr
    result = json.loads(response.stderr)
    assert result["decision"] == "blocked"
    assert result["reasons"] == ["input_or_collection_error"]
    assert result["agent_invocations"] == 0
    error = result["error"]
    assert "unclassified agreement comments" in error
    assert f"ID={comment_id}" in error
    assert f"https://github.com/o/r/issues/1#issuecomment-{comment_id}" in error
    assert "agreement-ignore-comments" in error and "checking → agree" in error


@dataclass
class GateEnv:
    review: ReviewEnv
    policy: Any

    def cli(self, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        assert args == ("merge",)
        return self.review.p.cli(
            "merge-continue",
            "--run-id",
            self.review.request["review_run_id"],
            "--instruction-file",
            self.review.p.note("Continue this accepted progression"),
            "--wait",
            extra=env,
            ok=None,
        )

    def merge_calls(self) -> list[list[str]]:
        return [c for c in self.review.p.calls() if c[:2] == ["pr", "merge"]]


@dataclass(frozen=True)
class GateSeed:
    directory: DirectorySeed
    template: GateEnv

    @classmethod
    def capture(cls, gate: GateEnv, snapshot: Path) -> GateSeed:
        # Keep metadata separate from the first test's mutable dataclasses, too.
        review = replace(
            gate.review,
            p=replace(gate.review.p, env=gate.review.p.env.copy()),
            request=copy.deepcopy(gate.review.request),
        )
        return cls(
            DirectorySeed.capture(review.p.root, snapshot), replace(gate, review=review)
        )

    def restore(self, p: ParentEnv, monkeypatch: pytest.MonkeyPatch) -> GateEnv:
        p.env["AGENT_AUTO_MERGE"] = "1"
        self.directory.restore(p.root)
        run_git(p.repo, "worktree", "repair")
        review = self.template.review
        original = self.directory.original
        root: Path = p.root

        def relocated(path: Path) -> Path:
            return root / path.relative_to(original)

        # Restoring the immutable seed relocates absolute runtime paths. Bind
        # that freshly restored proof before any case injects its own changes.
        directory = relocated(review.run_dir)
        execution_path = directory / "execution.json"
        execution = json.loads(execution_path.read_text())
        if "stages" in execution:
            runtime = json.loads((directory / "agent-runtime/result.json").read_text())
            execution["stages"]["agent"]["runtime_sha256"] = (
                self.template.policy.digest(runtime)
            )
            execution_path.write_text(json.dumps(execution))

        return GateEnv(
            replace(
                review,
                p=p,
                request=copy.deepcopy(review.request),
                request_path=relocated(review.request_path),
                fake=relocated(review.fake),
                output=relocated(review.output),
                calls_path=relocated(review.calls_path),
                clock_script=relocated(review.clock_script)
                if review.clock_script
                else None,
            ),
            review_policy_module(monkeypatch),
        )


@pytest.fixture
def gate_env(
    parent_env: ParentEnv,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> GateEnv:
    # Each worker/shard prepares its own immutable seeds through the real CLI.
    # Only private copies reach tests; case-level sharding remains independent.
    seeds: dict[str | None, GateSeed] = request.config.stash.setdefault(SEEDS, {})
    variant = getattr(request, "param", None)
    if variant in {"parent_self_worker", "parent_self_worker_rejected_resume"}:
        # A saved worker binds hashes to its absolute worktree/contract. Reusing
        # that runtime in another temporary root would alter its identity.
        return make_gate_env(parent_env, monkeypatch, variant)
    if variant not in seeds:
        gate = make_gate_env(parent_env, monkeypatch, variant)
        cache = request.getfixturevalue("tmp_path_factory").mktemp("merge-gate-seed")
        seeds[variant] = GateSeed.capture(gate, cache / "snapshot")
        return gate
    return seeds[variant].restore(parent_env, monkeypatch)


def make_gate_env(
    parent_env: ParentEnv, monkeypatch: pytest.MonkeyPatch, variant: str | None = None
) -> GateEnv:
    def configure(p: ParentEnv) -> None:
        if variant in {"research_dependency", "preflight_dependency"}:
            p.child("research")
        state = p.state()
        if variant in {"dependency", "research_dependency", "preflight_dependency"}:
            parent_policy = importlib.import_module("parent.policy")
            dependency = (
                13 if variant in {"research_dependency", "preflight_dependency"} else 11
            )
            if variant in {"research_dependency", "preflight_dependency"}:
                state["issues"]["13"].update(state="closed", state_reason="completed")
            state["issues"]["10"]["body"] = parent_policy.replace_section(
                state["issues"]["10"]["body"], "依存関係", f"#{dependency} の完了後。"
            )
        elif variant == "stacked":
            state["prs"][0]["base"]["ref"] = "codex/lower"
            run_git(p.repo, "branch", "codex/lower")
        elif variant == "non_main_default":
            state["default_branch"] = "trunk"
            for pr in state["prs"]:
                pr["base"]["ref"] = "trunk"
                pr["base"]["repo"]["default_branch"] = "trunk"
            run_git(p.repo, "branch", "-m", "main", "trunk")
        elif variant == "publication_paths":
            prefix = ".agents/logs/parents/parent-1/"
            event = prefix + "activity-public/event-existing.json"
            readme = prefix + "README.md"
            for name, text in {
                event: '{"fixture": "original"}\n',
                readme: "# Generated navigation\n",
            }.items():
                target = p.repo / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text)
            run_git(p.repo, "add", "--force", "--", event, readme)
            run_git(p.repo, "commit", "-m", "existing public log fixtures")
            base = run_git(p.repo, "rev-parse", "HEAD")
            worker = prefix + "child-10/d-abc/0001-initial/"
            # Freeze the actual log-only diffs before review, so CI absence is
            # the sole rejection reason, rather than an obsolete review HEAD.
            changes = [
                {event: '{"fixture": "edited"}\n'},
                {readme: "# Generated navigation\nHand-written addition\n"},
                {
                    prefix + "activity-public/event-new.json": "{}\n",
                    prefix + "activity-public/README.md": "# Navigation\n",
                    worker + "instruction.md": "Instruction\n",
                    worker + "final.md": "Final\n",
                    worker + "meta.json": "{}\n",
                },
            ]
            for index, (pr, files) in enumerate(
                zip(state["prs"], changes, strict=True)
            ):
                tree = p.repo / f".claude/worktrees/pr-{index}"
                run_git(tree, "reset", "--hard", base)
                for name, text in files.items():
                    target = tree / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text)
                run_git(tree, "add", "--force", "--", *files)
                run_git(tree, "commit", "-m", f"ordinary log PR {index}")
                pr["base"]["sha"] = base
                pr["head"].update(
                    sha=run_git(tree, "rev-parse", "HEAD"), ref=f"codex/pr-{index}"
                )
        p.save(state)
        if variant in {
            "parent_self",
            "parent_self_worker",
            "parent_self_worker_rejected_resume",
        }:
            if variant != "parent_self":
                from tests.e2e.agents_ops.test_manual_worker_runs import manual

                prs = state["prs"]
                prs[0]["head"]["ref"] = "codex/pr-0"
                state["prs"] = []
                p.save(state)
                p.cli(
                    "delegate",
                    "--child",
                    "1",
                    "--agent",
                    "codex",
                    "--branch",
                    "codex/pr-0",
                    "--worktree",
                    str(p.repo / ".claude/worktrees/pr-0"),
                )
                manual(
                    p,
                    "run-worker",
                    "--child",
                    "1",
                    "--prompt-file",
                    p.note("実装・検証する。"),
                )
                if variant == "parent_self_worker_rejected_resume":
                    rejected = manual(
                        p,
                        "run-worker",
                        "--child",
                        "1",
                        "--resume",
                        "--prompt-file",
                        p.note("同じworkerで追加確認する。"),
                        ok=False,
                        MANUAL_ADMISSION_FAILURE="1",
                    )
                    assert rejected.returncode == 2
                    assert (
                        json.loads(rejected.stdout)["launch_state"]
                        == "rejected_before_start"
                    )
                state = p.state()
                state["prs"] = prs
                p.save(state)
            p.cli(
                "record-result",
                "--child",
                "1",
                "--evidence-file",
                p.note("自身の受入条件・差分・テストを確認した。"),
            )
        if variant == "preflight_dependency":
            # Reserve before publishing the PRs, then provide terminal offline
            # worker records. Review/merge still use their real public CLIs.
            prs = state["prs"]
            for index, pr in enumerate(prs):
                pr["head"]["ref"] = f"codex/pr-{index}"
            state["prs"] = []
            p.save(state)
            for child in (10, 11):
                p.cli(
                    "delegate",
                    "--child",
                    str(child),
                    "--agent",
                    "codex",
                    "--branch",
                    f"codex/pr-{child - 10}",
                    "--worktree",
                    str(p.repo / f".claude/worktrees/pr-{child - 10}"),
                )
            state = p.state()
            state["prs"] = prs
            p.save(state)
            for child in (10, 11):
                completed_implementation(p, child)
                p.cli(
                    "record-result", "--child", str(child), "--evidence-file", p.note()
                )

    # Only scenarios that exercise interactions between PRs need three targets.
    review_env = make_review_env(
        parent_env,
        independent=variant
        in {
            "independent",
            "dependency",
            "research_dependency",
            "preflight_dependency",
            "publication_paths",
        },
        standalone=variant == "standalone",
        parent_self=variant
        in {"parent_self", "parent_self_worker", "parent_self_worker_rejected_resume"},
        pr_count=3
        if variant
        in {
            "multiple",
            "independent",
            "dependency",
            "research_dependency",
            "preflight_dependency",
            "publication_paths",
        }
        else 1,
        shared=variant == "shared",
        before_review=configure,
    )
    p = review_env.p
    policy = review_policy_module(monkeypatch)
    state = p.state()
    for pr in state["prs"]:
        pr.update(
            labels=[],
            mergeable=True,
            statusCheckRollup=[
                {
                    "__typename": "CheckRun",
                    "name": "Python",
                    "workflowName": "CI",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                }
            ],
        )
    p.save(state)
    for name in ("codex", "claude"):
        (p.root / "bin" / name).write_text(
            f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(p.root / 'unexpected-agent.txt')!r}).write_text('called')\nraise SystemExit(97)\n"
        )
    gate = GateEnv(review_env, policy)
    return gate
