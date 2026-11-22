"""Disposable Git main and fake GitHub for the shared memory publishing tests."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"
sys.path.insert(0, str(OPS))
checks = importlib.import_module("shared_memory.validate")

MEMORY = ".agents/memory"
SUMMARY = f"{MEMORY}/SUMMARY.md"
CURRENT = f"{MEMORY}/workflow/current.md"
OLD = f"{MEMORY}/decision/old.md"
LAYERS = ("workflow", "decision", "environment", "gotcha")


def add(
    layer: str = "workflow", slug: str = "new", **fields: object
) -> dict[str, object]:
    return {
        "op": "add",
        "layer": layer,
        "slug": slug,
        "title": "New observation",
        "summary": "New summary.",
        "applies_to": "codex",
        "source": "Test observation",
        "evidence": "Check the documented behavior again.",
        "citations": [{"path": "reference.md", "symbol": "#current-heading"}],
        "body": "See [the reference](../../../reference.md#current-heading).",
        **fields,
    }


def update(slug: str = "current", **fields: object) -> dict[str, object]:
    return {
        "op": "update",
        "layer": "workflow",
        "slug": slug,
        "reason": "根拠を再確認",
        **fields,
    }


def delete(
    layer: str = "decision", slug: str = "old", **fields: object
) -> dict[str, object]:
    return {
        "op": "delete",
        "layer": layer,
        "slug": slug,
        "reason": "観察の役目を終えた",
        **fields,
    }


def write_entry(repo: Path, path: str, **fields: object) -> None:
    metadata = {**add(), **fields}
    body = metadata.pop("body")
    for key in ("op", "layer", "slug"):
        metadata.pop(key)
    metadata.setdefault("created", "2026-01-01")
    metadata.setdefault("last_verified", "2026-01-01")
    header = yaml.safe_dump(metadata, sort_keys=False)
    (repo / path).write_text(f"---\n{header}---\n\n{body}\n", encoding="utf-8")


def sync_documents(repo: Path) -> None:
    edits, problems = checks.plan_sync(repo)
    assert not problems, problems
    for path, data in edits.items():
        (repo / path).write_bytes(data)


@dataclass(frozen=True)
class Repository:
    tmp: Path
    source: Path
    origin: Path
    real_git: str

    def git(self, *args: str, cwd: Path | None = None) -> str:
        return subprocess.check_output(
            [
                self.real_git,
                "-c",
                "user.name=Tester",
                "-c",
                "user.email=tests@example.test",
                *args,
            ],
            cwd=cwd or self.source,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()

    def bare(self, *args: str) -> str:
        return self.git(f"--git-dir={self.origin}", *args)

    def cli(
        self, *args: str, input: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "shared_memory", *args, "--repo", str(self.source)],
            cwd=self.tmp,
            input=input,
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )

    def batch(
        self, specs: list[dict[str, object]], *args: str
    ) -> subprocess.CompletedProcess[str]:
        return self.cli("batch", "--spec", "-", *args, input=json.dumps(specs))

    def api_calls(self) -> list[dict[str, Any]]:
        return [
            json.loads(line) for line in (self.tmp / "api-log").read_text().splitlines()
        ]

    def transactions(self) -> list[list[dict[str, Any]]]:
        return [
            row["data"]["variables"]["input"]["refUpdates"]
            for row in self.api_calls()
            if row["path"] == "graphql"
        ]

    def git_calls(self, name: str) -> list[str]:
        return [
            line
            for line in (self.tmp / "git-log").read_text().splitlines()
            if f" {name} " in line
        ]

    def assert_no_publication(self) -> None:
        assert self.api_calls() == []
        assert self.git_calls("push") == []
        assert self.bare("for-each-ref", "refs/heads/shared-memory") == ""

    def snapshot(self) -> tuple[str, str, str, bytes, bytes]:
        return (
            self.git("status", "--porcelain", "--branch"),
            self.git("rev-parse", "HEAD", "refs/remotes/origin/main"),
            self.git("worktree", "list", "--porcelain"),
            (self.source / ".git/index").read_bytes(),
            (self.source / ".git/FETCH_HEAD").read_bytes(),
        )

    def parallel_commit(
        self, kind: str, *, ref: str = "parallel", target: str = CURRENT
    ) -> str:
        other = self.tmp / "other"
        self.git("clone", str(self.origin), str(other))
        if kind == "new-entry":
            write_entry(
                other, f"{MEMORY}/workflow/parallel.md", summary="Parallel observation."
            )
            sync_documents(other)
        elif kind == "updated-summary":
            write_entry(
                other, CURRENT, title="Current observation", summary="New summary."
            )
            sync_documents(other)
        elif kind == "refreshed-entry":
            path = other / target
            before = path.read_text()
            refreshed = before.replace(
                "last_verified: '2026-01-01'", f"last_verified: {date.today()}"
            )
            assert refreshed != before
            path.write_text(refreshed + "\nReverified against the latest behavior.\n")
            sync_documents(other)
        else:
            assert kind == "removed-anchor"
            (other / "reference.md").write_text("# Changed heading\n")
        self.git("add", "-A", cwd=other)
        self.git("commit", "-qm", "Parallel change", cwd=other)
        self.git("push", "origin", f"HEAD:refs/heads/{ref}", cwd=other)
        return self.git("rev-parse", "HEAD", cwd=other)


@pytest.fixture
def memory_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Repository:
    real_git = shutil.which("git")
    assert real_git
    for key in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM"):
        monkeypatch.setenv(key, os.devnull)
    source, origin = tmp_path / "source", tmp_path / "origin.git"
    source.mkdir()
    fixture = Repository(tmp_path, source, origin, real_git)
    fixture.git("init", "-q", "-b", "main")
    fixture.git("init", "-q", "--bare", "-b", "main", str(origin))
    fixture.git("config", "user.name", "Tester")
    fixture.git("config", "user.email", "tests@example.test")
    fixture.git("remote", "add", "origin", str(origin))
    for layer in (*LAYERS, "ops-report"):
        (source / MEMORY / layer).mkdir(parents=True)
    (source / MEMORY / "ops-report/SUMMARY.md").write_text("Independent contents.\n")
    skill = source / ".agents/skills/memory/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# 共有memory\n\n## 形式\n")
    (source / "reference.md").write_text("# Current heading\n\nVerifiable detail.\n")
    (source / "escaped.md").symlink_to("../outside.md")
    write_entry(
        source,
        CURRENT,
        title="Current observation",
        summary="Current summary.",
        citations=[],
    )
    write_entry(
        source,
        OLD,
        title="Old observation",
        summary="Old summary.",
        applies_to="all",
        source="Earlier trial of annotation review",
        evidence="Compare the archived settings in the decision record.",
        citations=[],
        body="A former workaround for reviewer handoff required a second pass.\n\nThe current workflow keeps the pending decisions in the same queue.",
    )
    sync_documents(source)
    fixture.git("add", "-A")
    fixture.git("commit", "-qm", "Seed shared memory")
    fixture.git("push", "origin", "main")
    fixture.git("switch", "-qc", "callers-branch")
    (source / ".git/FETCH_HEAD").write_text("caller fetch marker\n")

    tools = tmp_path / "bin"
    tools.mkdir()
    git_wrapper = tools / "git"
    git_wrapper.write_text(
        "#!/bin/bash\n"
        'case " $* " in\n'
        '  *" remote get-url --push "*) printf "%s\\n" "${MEMORY_TEST_PUSH_URL:-git@github.com:test-owner/tennis-memory.git}"; exit 0;;\n'
        '  *" remote get-url "*) printf "%s\\n" "git@github.com:test-owner/tennis-memory.git"; exit 0;;\n'
        '  *" mktree "*) if [[ -n "${MEMORY_TEST_SIGNAL_READY:-}" ]]; then\n'
        '    printf "%s\\n" "$$" > "$MEMORY_TEST_SIGNAL_READY"\n'
        '    read -r unused < "$MEMORY_TEST_SIGNAL_BLOCK"\n'
        "    exit 97\n"
        "  fi;;\n"
        '  *" fetch "*|*" push "*) printf "%s\\n" "$*" >> "$MEMORY_TEST_GIT_LOG";;\n'
        "esac\n"
        'exec "$MEMORY_TEST_GIT" "$@"\n'
    )
    git_wrapper.chmod(0o755)
    gh = tools / "gh"
    helper = ROOT / "tests/e2e/agents_ops/fake_shared_memory_gh.py"
    gh.write_text(f"#!{sys.executable}\n" + helper.read_text(encoding="utf-8"))
    gh.chmod(0o755)
    (tmp_path / "api-log").touch()
    (tmp_path / "git-log").touch()
    for key, value in {
        "MEMORY_TEST_GIT": real_git,
        "MEMORY_TEST_ORIGIN": str(origin),
        "MEMORY_TEST_PR": str(tmp_path / "pr.json"),
        "MEMORY_TEST_API_LOG": str(tmp_path / "api-log"),
        "MEMORY_TEST_GIT_LOG": str(tmp_path / "git-log"),
        "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(OPS),
    }.items():
        monkeypatch.setenv(key, value)
    return fixture
