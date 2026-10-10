"""Fixtures for .agents/ops tests: a fake ``gh`` with on-disk state, a fake
``agent_exec.sh`` and a throwaway git repository."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
WEEKLY_REPORT = ROOT / ".agents/ops/weekly-report/weekly_report.py"
NOW = "2026-10-12T08:00:00+09:00"  # Monday of ISO week 2026-W42

FAKE_GH = r"""
import json, os, sys

args = sys.argv[1:]
with open(os.environ["FAKE_GH_CALLS"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(args) + "\n")
fail = os.environ.get("FAKE_GH_FAIL")
if fail and " ".join(args).startswith(fail):
    print(f"fake gh: injected failure for {fail!r}", file=sys.stderr)
    sys.exit(1)
path = os.environ["FAKE_GH_STATE"]
with open(path, encoding="utf-8") as fh:
    state = json.load(fh)


def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default


def opts(name):
    return [args[i + 1] for i, a in enumerate(args) if a == name]


def emit(rows):
    fields = opt("--json").split(",")
    rows = rows[: int(opt("--limit", "30"))]
    print(json.dumps([{k: r[k] for k in fields} for r in rows]))


def gh_issue(r):
    return {**r, "labels": [{"name": n} for n in r["labels"]], "author": {"login": "bot"}}


def save():
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh)


head = args[:2]
if head == ["issue", "list"]:
    rows = state["issues"]
    st = opt("--state", "open")
    if st != "all":
        rows = [r for r in rows if r["state"].lower() == st]
    if opt("--label"):
        rows = [r for r in rows if opt("--label") in r["labels"]]
    emit([gh_issue(r) for r in sorted(rows, key=lambda r: -r["number"])])
elif head == ["issue", "view"]:
    issue = next(r for r in state["issues"] if r["number"] == int(args[2]))
    print(json.dumps({k: gh_issue(issue)[k] for k in opt("--json").split(",")}))
elif head == ["issue", "create"]:
    number = state["next_number"]
    state["next_number"] += 1
    with open(opt("--body-file"), encoding="utf-8") as fh:
        body = fh.read()
    state["issues"].append({
        "number": number, "title": opt("--title"), "body": body, "labels": opts("--label"),
        "state": "OPEN", "createdAt": "2026-10-12T00:00:00Z", "updatedAt": "2026-10-12T00:00:00Z",
        "url": f"https://github.com/o/r/issues/{number}",
    })
    save()
    print(f"https://github.com/o/r/issues/{number}")
elif head == ["issue", "edit"]:
    issue = next(r for r in state["issues"] if r["number"] == int(args[2]))
    with open(opt("--body-file"), encoding="utf-8") as fh:
        issue["body"] = fh.read()
    save()
elif head == ["pr", "list"]:
    emit(state["prs"] if opt("--state") == "open" else state["merged_prs"])
elif head == ["run", "list"]:
    emit(state["runs"])
elif head == ["label", "list"]:
    emit([{"name": n} for n in state["labels"]])
elif head == ["label", "create"]:
    state["labels"].append(args[2])
    save()
else:
    print(f"fake gh: unsupported {args}", file=sys.stderr)
    sys.exit(2)
"""

FAKE_AGENT_EXEC = r"""#!/usr/bin/env bash
set -euo pipefail
out="" prompt="" agent=""
while (($#)); do
  case "$1" in
    --out) out="$2"; shift 2 ;;
    --prompt-file) prompt="$2"; shift 2 ;;
    --agent) agent="$2"; shift 2 ;;
    *) shift ;;
  esac
done
cp "$prompt" "$FAKE_AGENT_PROMPT_COPY"
echo "$agent" > "$FAKE_AGENT_USED"
if [[ -n "${FAKE_AGENT_FAIL:-}" ]]; then
  echo "fake agent: quota exhausted" >&2
  exit 3
fi
cp "$FAKE_AGENT_OUTPUT" "$out"
"""

AGENT_OUTPUT: dict[str, Any] = {
    "summary": "- 今週は小さな修正が中心。\n- 放置PRが1件ある。",
    "proposals": [
        {
            "category": "code-debt",
            "priority": "high",
            "title": "big.py を分割する",
            "detail": "big.py が肥大化している。\n- [ ] これはチェックボックスではなく本文",
            "evidence": "big.py 300行",
        },
        {
            "category": "control-rules",
            "priority": "low",
            "title": "週次レポートのプロンプトを改善する",
            "detail": "提案数の目安を見直す。",
            "evidence": ".agents/ops/weekly-report/prompt.md",
            "options": ["5件に絞る", "上限なし"],
        },
    ],
    "stale_pr_notes": {"7": "rebaseして仕上げを推奨。差分が小さい。"},
}


def base_state() -> dict[str, Any]:
    return {
        "next_number": 100,
        "labels": ["ai-proposed"],
        "issues": [
            {
                "number": 50,
                "title": "古い提案",
                "body": "x",
                "labels": ["ai-proposed"],
                "state": "OPEN",
                "createdAt": "2026-08-01T00:00:00Z",
                "updatedAt": "2026-08-02T00:00:00Z",
                "url": "https://github.com/o/r/issues/50",
            },
        ],
        "prs": [
            {
                "number": 7,
                "title": "放置されたPR",
                "headRefName": "claude/old",
                "author": {"login": "me"},
                "createdAt": "2026-09-01T00:00:00Z",
                "updatedAt": "2026-09-20T00:00:00Z",
                "isDraft": False,
                "labels": [],
                "url": "https://github.com/o/r/pull/7",
            },
            {
                "number": 8,
                "title": "新しいPR",
                "headRefName": "codex/new",
                "author": {"login": "me"},
                "createdAt": "2026-10-11T00:00:00Z",
                "updatedAt": "2026-10-11T00:00:00Z",
                "isDraft": True,
                "labels": [{"name": "needs-human"}],
                "url": "https://github.com/o/r/pull/8",
            },
        ],
        "merged_prs": [
            {
                "number": 6,
                "title": "merged",
                "headRefName": "x",
                "mergedAt": "2026-10-10T00:00:00Z",
            },
            {
                "number": 5,
                "title": "old",
                "headRefName": "y",
                "mergedAt": "2026-09-01T00:00:00Z",
            },
        ],
        "runs": [
            {
                "databaseId": 1,
                "workflowName": "CI",
                "conclusion": "failure",
                "status": "completed",
                "headBranch": "main",
                "event": "push",
                "createdAt": "2026-10-11T00:00:00Z",
                "url": "https://github.com/o/r/actions/runs/1",
            },
            {
                "databaseId": 2,
                "workflowName": "CI",
                "conclusion": "success",
                "status": "completed",
                "headBranch": "main",
                "event": "push",
                "createdAt": "2026-09-01T00:00:00Z",
                "url": "https://github.com/o/r/actions/runs/2",
            },
        ],
    }


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def make_repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    (path / "big.py").write_text("x = 1\n" * 300, encoding="utf-8")
    (path / "small.py").write_text("# TODO: tidy\ny = 2\n", encoding="utf-8")
    (path / "AGENTS.md").write_text("# rules\n", encoding="utf-8")
    (path / "knowledge").mkdir()
    (path / "knowledge/summary.md").write_text("# summary\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(
        path,
        "-c", "user.name=t", "-c", "user.email=t@example.com",
        "commit", "-q", "-m", "init", "--date", "2026-10-09T00:00:00+09:00",
    )  # fmt: skip
    return path


def _install(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@dataclass
class OpsEnv:
    tmp: Path
    repo: Path
    env: dict[str, str]
    state_file: Path
    calls_file: Path

    @property
    def state(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.state_file.read_text(encoding="utf-8"))
        return data

    def write_state(self, state: dict[str, Any]) -> None:
        self.state_file.write_text(json.dumps(state), encoding="utf-8")

    def calls(self) -> list[list[str]]:
        if not self.calls_file.exists():
            return []
        return [
            json.loads(ln)
            for ln in self.calls_file.read_text(encoding="utf-8").splitlines()
        ]

    def set_agent_output(self, payload: dict[str, Any] | str) -> None:
        text = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False)
        )
        Path(self.env["FAKE_AGENT_OUTPUT"]).write_text(text, encoding="utf-8")

    def run(self, *args: str, **extra_env: str) -> subprocess.CompletedProcess[str]:
        cmd = [sys.executable, str(WEEKLY_REPORT), *args]
        cmd += ["--repo-root", str(self.repo), "--log-dir", str(self.tmp / "logs")]
        if args[0] == "report":
            cmd += ["--base-ref", "main", "--now", NOW]
        return subprocess.run(
            cmd,
            env={**self.env, **extra_env},
            text=True,
            capture_output=True,
            check=False,
        )

    def run_dirs(self) -> list[Path]:
        return sorted((self.tmp / "logs").iterdir())


@pytest.fixture
def agent_output() -> dict[str, Any]:
    """A valid agent final message (fresh deep copy per test)."""
    copied: dict[str, Any] = json.loads(json.dumps(AGENT_OUTPUT))
    return copied


@pytest.fixture
def ops_env(tmp_path: Path) -> Iterator[OpsEnv]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _install(bin_dir / "gh", f"#!{sys.executable}\n{FAKE_GH}")
    fake_exec = tmp_path / "fake_agent_exec.sh"
    _install(fake_exec, FAKE_AGENT_EXEC)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("WEEKLY_REPORT_", "FAKE_"))
    }
    env.update(
        {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_GH_STATE": str(tmp_path / "gh_state.json"),
            "FAKE_GH_CALLS": str(tmp_path / "gh_calls.jsonl"),
            "FAKE_AGENT_OUTPUT": str(tmp_path / "agent_output.json"),
            "FAKE_AGENT_PROMPT_COPY": str(tmp_path / "prompt_seen.md"),
            "FAKE_AGENT_USED": str(tmp_path / "agent_used.txt"),
            "WEEKLY_REPORT_AGENT_EXEC": str(fake_exec),
        }
    )
    ops = OpsEnv(
        tmp=tmp_path,
        repo=make_repo(tmp_path / "repo"),
        env=env,
        state_file=tmp_path / "gh_state.json",
        calls_file=tmp_path / "gh_calls.jsonl",
    )
    ops.write_state(base_state())
    ops.set_agent_output(AGENT_OUTPUT)
    yield ops
