"""Small offline runner receipts and a GitHub substitute backed by real Git."""

from __future__ import annotations

import copy
import importlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops"))
journal: Any = importlib.import_module("worker_logs.journal")
schema: Any = importlib.import_module("worker_logs.schema")
sources: Any = importlib.import_module("worker_logs.sources")
Publisher: Any = importlib.import_module("worker_logs.publication").Publisher


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, stderr=subprocess.PIPE, text=True
    ).strip()


def record_source() -> tuple[dict[str, Any], dict[str, Any]]:
    source = {
        "repo": "o/r",
        "parent": 1,
        "issue": 10,
        "public_key": "a" * 32,
        "sequence": 1,
        "phase": "planning",
        "kind": "initial",
        "decision": None,
        "instruction": "\n親の具体指示。tests/example.py:12を確認する。\r\n",
        "instruction_kind": "task",
        "additional": [],
        "issue_sha256": "b" * 64,
        "head_at_start": "c" * 40,
        "captured_at": "2026-11-07T01:00:00+00:00",
        "worktree": "/private/checkout",
        "private": ["private-session-value"],
        "standard": {
            "commit": "d" * 40,
            "path": ".agents/ops/parent/worker_prompt.md",
            "blob": "e" * 40,
        },
    }
    record = {
        "schema_version": 1,
        "run_id": "private-run-value",
        "result": "completed",
        "turn_end_reason": "completed",
        "cleanup_complete": True,
        "started_at": 1794013200,
        "finished_at": 1794013201,
        "contract": {
            "settings": {
                "agent": "codex",
                "profile": None,
                "selected": {"model": "gpt-6.1-sol", "effort": "max"},
            }
        },
        "runtime": {
            "session_id": "private-session-value",
            "turn_id": "private-turn-value",
            "effective": {
                k: {"status": "observed", "value": v}
                for k, v in {"model": "gpt-6.1-sol", "effort": "max"}.items()
            },
        },
    }
    final = "確認結果: tests/example.py:12。https://github.com/o/r/pull/42\n"
    return source, {
        "final": final,
        "final_sha256": schema.sha(final.encode()),
        "record": record,
    }


class FakeGitHub:
    def __init__(self, root: Path, upstream: Path) -> None:
        self.repo = "o/r"
        self.root, self.upstream = root, upstream
        self.rows: dict[int, dict[str, Any]] = {}
        self.calls: list[str] = []
        self.failure: str | None = None
        self.mergeable: bool | None = True
        self.change_head_on_second_get = False
        self.gets = 0
        self.pause_reads = False
        self.lock = threading.RLock()

    def repository(self) -> dict[str, Any]:
        return {
            "full_name": self.repo,
            "clone_url": str(self.upstream),
            "ssh_url": None,
        }

    def main(self) -> str:
        return git(self.upstream, "rev-parse", "main")

    def get(self, number: int) -> dict[str, Any]:
        with self.lock:
            if self.pause_reads:
                raise journal.PublicationError("github_failed", "injected GET boundary")
            self.gets += 1
            pr = self.rows[number]
            if self.change_head_on_second_get and self.gets == 2:
                pr["head"]["sha"] = "f" * 40
            return copy.deepcopy(pr)

    def files(self, number: int) -> list[dict[str, Any]]:
        head = self.rows[number]["head"]["sha"]
        fields = git(
            self.upstream,
            "diff",
            "--name-status",
            "--no-renames",
            "-z",
            f"main...{head}",
        ).split("\0")
        return [
            {
                "filename": fields[i + 1],
                "status": {"A": "added", "M": "modified", "D": "removed"}[fields[i]],
            }
            for i in range(0, len(fields) - 1, 2)
        ]

    def find(self, branch: str) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(r) for r in self.rows.values() if r["head"]["ref"] == branch
        ]

    def create(self, branch: str, title: str, body: str) -> dict[str, Any]:
        with self.lock:
            self.calls.append("create")
            n = len(self.rows) + 100
            self.rows[n] = {
                "number": n,
                "html_url": f"https://github.com/o/r/pull/{n}",
                "body": body,
                "title": title,
                "draft": False,
                "changed_files": 3,
                "state": "open",
                "merged": False,
                "mergeable": self.mergeable,
                "merge_commit_sha": None,
                "head": {
                    "ref": branch,
                    "sha": git(self.upstream, "rev-parse", branch),
                    "repo": {"full_name": self.repo},
                },
                "base": {
                    "ref": "main",
                    "sha": self.main(),
                    "repo": {"full_name": self.repo},
                },
            }
            if self.failure == "create_response_lost":
                self.failure = None
                raise journal.PublicationError(
                    "github_failed", "private-server-response"
                )
            if self.failure == "closed_unmerged":
                self.rows[n]["state"] = "closed"
            return self.get_without_count(n)

    def get_without_count(self, number: int) -> dict[str, Any]:
        return copy.deepcopy(self.rows[number])

    def merge(self, number: int, head: str) -> None:
        with self.lock:
            self.calls.append("merge")
            pr = self.rows[number]
            assert pr["head"]["sha"] == head
            tree = self.root.parent / f"merge-{number}"
            git(self.root.parent, "clone", str(self.upstream), str(tree))
            git(tree, "config", "user.name", "fixture")
            git(tree, "config", "user.email", "fixture@example.invalid")
            git(tree, "merge", "--no-ff", head, "-m", "merge worker log")
            commit = git(tree, "rev-parse", "HEAD")
            git(tree, "push", "origin", "main")
            pr.update(state="closed", merged=True, merge_commit_sha=commit)
            if self.failure == "merge_response_lost":
                self.failure = None
                raise journal.PublicationError(
                    "github_failed", "private-merge-response"
                )


class LogEnv:
    def __init__(self, root: Path, monkeypatch: Any) -> None:
        self.root = root / "repo"
        self.root.mkdir()
        git(self.root, "init", "-b", "main")
        git(self.root, "config", "user.name", "fixture")
        git(self.root, "config", "user.email", "fixture@example.invalid")
        self.standard = b"standard worker fixture\n"
        template = self.root / ".agents/ops/parent/worker_prompt.md"
        template.parent.mkdir(parents=True)
        template.write_bytes(self.standard)
        (self.root / "code.py").write_text("value = 1\n")
        git(self.root, "add", ".")
        git(self.root, "commit", "-qm", "seed")
        self.base = git(self.root, "rev-parse", "HEAD")
        self.upstream = root / "upstream.git"
        git(root, "clone", "--bare", str(self.root), str(self.upstream))
        git(self.root, "remote", "add", "origin", str(self.upstream))
        git(self.root, "fetch", "origin")
        self.state = root / "state"
        self.github = FakeGitHub(self.root, self.upstream)
        monkeypatch.setattr(sources, "TEMPLATE", template)
        self.sessions: list[Any] = []

    def turn(
        self,
        *,
        parent: int = 1,
        child: int = 10,
        instruction: str = "\n実際に送った指示\r\n",
        phase: str = "planning",
    ) -> tuple[Path, dict[str, Any]]:
        tree = self.root / f".claude/worktrees/worker-{parent}-{child}"
        branch = f"codex/worker-{parent}-{child}"
        git(self.root, "worktree", "add", "-b", branch, str(tree), self.base)
        directory = self.state / f"runner-{parent}-{child}"
        snapshot = {
            "phase": phase,
            "lineage_id": f"private-lineage-{parent}-{child}",
            "turn_id": f"private-turn-{parent}-{child}",
            "previous_turn_id": None,
            "decision_id": None,
        }
        turn = {
            "snapshot": snapshot,
            "source": str(directory),
            "head_at_start": self.base,
            "delegation_id": f"private-delegation-{parent}-{child}",
            "instruction_source": {
                "text": instruction,
                "kind": "task",
                "additional": [],
                "issue_sha256": schema.sha(b"issue body"),
            },
        }
        task = {
            "worktree": str(tree),
            "branch": branch,
            "delegation_id": turn["delegation_id"],
            "preflight": {
                "lineage_id": snapshot["lineage_id"],
                "turns": [turn],
                "decisions": [],
            },
        }
        session = SimpleNamespace(
            row={
                "session": "codex:private-parent-session",
                "tasks": {str(child): task},
            },
            store=SimpleNamespace(directory=self.state),
            gh=SimpleNamespace(repo="o/r"),
            parent=parent,
        )
        prompt = "standard\n" + str(tree) + "\n" + instruction
        path = sources.freeze(session, child, prompt, self.standard)
        self.sessions.append(session)
        source, receipt = record_source()
        record = receipt["record"]
        record["contract"].update(
            task_id=task["delegation_id"], cwd=str(tree), branch=branch, mode="write"
        )
        record["prompt_sha256"] = schema.sha(prompt.encode())
        record["run_id"] = directory.name
        directory.mkdir(parents=True)
        (directory / "prompt.md").write_bytes(prompt.encode())
        (directory / "final.md").write_text(receipt["final"])
        (directory / "result.json").write_text(json.dumps(record))
        proof = {
            "record": record,
            "summary": {
                "runner_alive": False,
                "agent_alive": False,
                "agent_group_alive": False,
            },
        }
        return path, proof

    def ready(self, **kwargs: Any) -> Path:
        path, proof = self.turn(**kwargs)
        result = sources.receive(path, proof)
        assert result["state"] == "ready", result
        return path

    def publisher(self, path: Path) -> Any:
        return Publisher(self.root, path, self.github)

    def main_files(self) -> list[str]:
        return git(self.upstream, "ls-tree", "-r", "--name-only", "main").splitlines()
