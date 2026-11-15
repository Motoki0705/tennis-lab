"""Real CLI bootstraps with only their first project import replaced by a pipe wait."""

from __future__ import annotations

import os
import select
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OPS = ROOT / ".agents/ops"


@dataclass(frozen=True)
class Entry:
    name: str
    source: str
    dependency: str
    module: str | None = None

    def command(self, repo: Path) -> list[str]:
        if self.module is not None:
            return [sys.executable, "-m", self.module]
        source = repo / ".agents/ops" / self.source
        return ["bash" if source.suffix == ".sh" else sys.executable, str(source)]

    @property
    def description(self) -> str:
        if self.module is not None:
            return f"python -m {self.module}"
        return "lib/agent_runtime.py" if self.source.endswith(".sh") else self.source


ENTRIES = (
    Entry("parent", "parent/__main__.py", "parent/cli_impl.py", "parent"),
    Entry("parent-cli", "parent/cli.py", "parent/cli_impl.py", "parent.cli"),
    Entry("parent-review", "review/parent_review.py", "review/parent_review_impl.py"),
    Entry("merge-gate", "review/merge_gate.py", "review/merge_gate_impl.py"),
    Entry("agent-exec", "lib/agent_exec.sh", "lib/agent_runtime_impl.py"),
    Entry("agent-runtime", "lib/agent_runtime.py", "lib/agent_runtime_impl.py"),
    Entry(
        "explain-html",
        "explain_html/__main__.py",
        "explain_html/cli_impl.py",
        "explain_html",
    ),
    Entry("ci-watch", "ci_watch/__main__.py", "ci_watch/cli.py", "ci_watch"),
    Entry(
        "review-evaluation",
        "review/evaluation/__main__.py",
        "review/evaluation/cli_impl.py",
        "review.evaluation",
    ),
    Entry(
        "issue-supplement",
        "issue_supplement/__main__.py",
        "issue_supplement/cli.py",
        "issue_supplement",
    ),
    Entry(
        "project-progress",
        "project_progress/__main__.py",
        "project_progress/cli_impl.py",
        "project_progress",
    ),
    Entry(
        "retrieval-server",
        "lib/review_tools/mcp_server.py",
        "lib/review_tools/mcp_server_impl.py",
        "lib.review_tools.mcp_server",
    ),
    Entry(
        "review-server",
        "lib/review_tools/review_server.py",
        "lib/review_tools/review_server_impl.py",
        "lib.review_tools.review_server",
    ),
    Entry(
        "report-server",
        "ops-report/opsreport/mcp_server.py",
        "ops-report/opsreport/mcp_server_impl.py",
        "opsreport.mcp_server",
    ),
    Entry(
        "reproduction-server",
        "ops-report/opsreport/reproduction_server.py",
        "ops-report/opsreport/reproduction_server_impl.py",
        "opsreport.reproduction_server",
    ),
    Entry("dispatch-worker", "dispatcher/worker.py", "dispatcher/worker_impl.py"),
)
PARENT = ENTRIES[0]
PROJECT_WAIT = "import sys\nprint('project-import', flush=True)\nsys.stdin.read()\nraise SystemExit(0)\n"


def install_cli(repo: Path, entry: Entry = PARENT, project: str = PROJECT_WAIT) -> None:
    """Copy the actual bootstrap and its package initializers, not a second CLI."""
    ops = repo / ".agents/ops"
    sources = ["lib/checkout_lock.py", entry.source]
    if entry.source.endswith(".sh"):
        sources.append("lib/agent_runtime.py")
    for relative in [*sources, entry.dependency]:
        target = ops / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        for directory in (OPS / relative).parents:
            if directory == OPS:
                break
            initializer = directory / "__init__.py"
            if initializer.exists():
                destination = ops / initializer.relative_to(OPS)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(initializer, destination)
        if relative in sources:
            shutil.copyfile(OPS / relative, target)
    (ops / entry.dependency).write_text(project)
    if entry.source.endswith(".sh"):
        (repo / ".venv").symlink_to(ROOT / ".venv", target_is_directory=True)


def environment(repo: Path) -> dict[str, str]:
    return {
        **{
            key: value
            for key, value in os.environ.items()
            if not key.startswith("TENNIS_LAB_CHECKOUT_")
        },
        "PYTHONPATH": os.pathsep.join(
            str(repo / path) for path in (".agents/ops", ".agents/ops/ops-report")
        ),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


@contextmanager
def running_cli(
    repo: Path,
    entry: Entry = PARENT,
    *,
    cwd: Path | None = None,
    arguments: tuple[str, ...] = (),
) -> Iterator[subprocess.Popen[str]]:
    with subprocess.Popen(
        [*entry.command(repo), *arguments],
        cwd=cwd or repo,
        env=environment(repo),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            yield process
        finally:
            try:
                process.communicate(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=5)


def line(process: subprocess.Popen[str], *, stderr: bool = False) -> str:
    stream = process.stderr if stderr else process.stdout
    assert stream is not None
    assert select.select([stream], [], [], 8)[0], "CLI handshake timed out"
    result = stream.readline().strip()
    assert isinstance(result, str)
    assert result, f"CLI exited before handshake (exit={process.poll()})"
    return result
