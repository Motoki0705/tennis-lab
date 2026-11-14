"""Measure parent commands against fake GitHub; never write to real GitHub.

Run this script from the implementation checkout, passing an unchanged baseline
checkout to --checkout to compare against its CLI/fixtures. Results are private
PARENT_METRICS_FILE JSONL. The HTTP revision uses the real transport and a local
HTTP bridge whose backend runs the same fake-gh scenarios as the baseline.

Example:
    .venv/bin/python tests/benchmarks/agents_ops/parent_http.py \
        --checkout /path/to/baseline-worktree --output /tmp/parent-before
    .venv/bin/python tests/benchmarks/agents_ops/parent_http.py \
        --checkout . --output /tmp/parent-after
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SUPPORT = Path(__file__).resolve().parents[2] / "support/agents_ops"


def run(checkout: Path, output: Path, repeats: int) -> None:
    # tests/ deliberately has no __init__.py; don't import an unrelated installed
    # package named tests when this diagnostic runs outside pytest's import mode.
    package = ModuleType("tests")
    package.__path__ = [str(checkout / "tests")]
    sys.modules["tests"] = package
    sys.path[:0] = [str(checkout), str(checkout / ".agents/ops")]
    module = importlib.import_module("tests.e2e.agents_ops.test_parent_session")
    use_http = (checkout / ".agents/ops/issues/transport.py").exists()
    output.mkdir(parents=True, exist_ok=False)
    metric_file = output / "metrics.jsonl"
    for repeat in range(repeats):
        directory = output / str(repeat)
        directory.mkdir()
        with pytest.MonkeyPatch.context() as patch:

            def environment(directory: Path) -> Any:
                p = module.parent_env.__wrapped__(directory, patch)
                if use_http:
                    p.env["AGENTS_TEST_GITHUB_HTTP"] = "1"
                    p.env["PYTHONPATH"] = os.pathsep.join(
                        [str(SUPPORT / "http_bootstrap"), str(checkout / ".agents/ops")]
                    )
                return p

            p = environment(directory)
            extra = {"PARENT_METRICS_FILE": str(metric_file)}
            p.cli("register", extra=extra)
            p.cli("checking")
            p.cli(
                "agree",
                "--spec",
                p.spec(
                    {
                        "purpose": "benchmark",
                        "scope": "fake repo only",
                        "behavior": "same inputs",
                        "acceptance": "requests measured",
                        "human_confirmation": "この条件で合意します。",
                    }
                ),
            )
            p.child("first")
            p.cli("status", extra=extra)
            p.cli("note", "--body-file", p.note("比較用の固定メモ。"), extra=extra)
            worktree = p.worktree("benchmark")
            p.cli(
                "delegate",
                "--child",
                "10",
                "--agent",
                "codex",
                "--branch",
                "codex/benchmark",
                "--worktree",
                str(worktree),
                extra=extra,
            )
            review_directory = directory / "review"
            review_directory.mkdir()
            p = environment(review_directory)
            module.review_setup(p, pr_count=1)
            prepared = json.loads(
                p.cli("review-prepare", *p.review_flags(), "--prs", "100").stdout
            )
            p.cli(
                "review-request",
                *p.review_flags(),
                "--run-id",
                prepared["review_run_id"],
                "--instruction-file",
                p.note("変更の契約を確認する。"),
                extra=extra,
            )
    print(metric_file.read_text(), end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    run(args.checkout.resolve(), args.output.resolve(), args.repeats)


if __name__ == "__main__":
    main()
