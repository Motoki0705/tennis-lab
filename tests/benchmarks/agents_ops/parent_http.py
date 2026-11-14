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
    requests_file = output / "requests.jsonl"
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

            extra = {"PARENT_METRICS_FILE": str(metric_file)}

            def measure(
                p: Any, *args: str, extra: dict[str, str] = extra, repeat: int = repeat
            ) -> Any:
                offset = len(p.calls())
                result = p.cli(*args, extra=extra)
                calls = []
                for call in p.calls()[offset:]:
                    if call[0] != "api":
                        continue
                    endpoint = next(
                        a
                        for a in call
                        if a in {"user", "graphql"} or a.startswith("repos/")
                    )
                    method = (
                        call[call.index("--method") + 1]
                        if "--method" in call
                        else "POST"
                    )
                    calls.append({"method": method, "endpoint": endpoint})
                with requests_file.open("a") as stream:
                    stream.write(
                        json.dumps(
                            {"command": args[0], "repeat": repeat, "requests": calls}
                        )
                        + "\n"
                    )
                return result

            p = environment(directory)
            measure(p, "register", "--summary", "計測対象")
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
            measure(p, "status")
            measure(p, "note", "--body-file", p.note("比較用の固定メモ。"))
            worktree = p.worktree("benchmark")
            measure(
                p,
                "delegate",
                "--child",
                "10",
                "--agent",
                "codex",
                "--branch",
                "codex/benchmark",
                "--worktree",
                str(worktree),
            )
            review_directory = directory / "review"
            review_directory.mkdir()
            p = environment(review_directory)
            module.review_setup(p, pr_count=1)
            prepared = json.loads(
                p.cli("review-prepare", *p.review_flags(), "--prs", "100").stdout
            )
            measure(
                p,
                "review-request",
                *p.review_flags(),
                "--run-id",
                prepared["review_run_id"],
                "--instruction-file",
                p.note("変更の契約を確認する。"),
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
