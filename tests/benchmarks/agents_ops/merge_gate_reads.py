"""Measure merge-gate reads with fake gh, real git and PARENT_METRICS_FILE.

Run this script in the implementation checkout with --checkout pointing at
either revision. Output contains endpoint counts, call stacks and metrics;
no real GitHub or agent is called. See review/README.md for the protocol.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest


def run(checkout: Path, output: Path, repeats: int, peers: int, mode: str) -> None:
    package = ModuleType("tests")
    package.__path__ = [
        str(checkout / "tests"),
        str(Path(__file__).resolve().parents[2]),
    ]
    sys.modules["tests"] = package
    sys.path[:0] = [str(checkout), str(checkout / ".agents/ops")]
    module = importlib.import_module("tests.e2e.agents_ops.test_parent_session")
    gates = importlib.import_module("tests.e2e.agents_ops.test_merge_gate")
    support = importlib.import_module("tests.support.agents_ops.merge_gate_reads")
    use_http = (checkout / ".agents/ops/issues/transport.py").exists()
    bootstrap = checkout / "tests/support/agents_ops/http_bootstrap"
    if use_http and not (bootstrap / "sitecustomize.py").is_file():
        raise ValueError("HTTP checkout requires its local fake GitHub bootstrap")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for repeat in range(repeats):
        directory = output / str(repeat)
        directory.mkdir()
        with pytest.MonkeyPatch.context() as patch:
            p = module.parent_env.__wrapped__(directory, patch)
            if use_http:
                # Reuse the HTTP benchmark's local bridge, never the live API.
                p.env["AGENTS_TEST_GITHUB_HTTP"] = "1"
                p.env["PYTHONPATH"] = os.pathsep.join(
                    [str(bootstrap), str(checkout / ".agents/ops")]
                )
            gate = gates.make_gate_env(p, patch, "independent")
            support.add_unrelated_parents(p, peers)
            offset = len(p.calls())
            metrics = directory / "metrics.jsonl"
            trace = directory / "read-trace.jsonl"
            # The baseline gate has no metrics wrapper. Instrument it here so
            # both revisions use the same counter and fake-backend comparison.
            driver = directory / "measure.py"
            driver.write_text(
                "import inspect, json, os, sys\nfrom pathlib import Path\n"
                f"sys.path.insert(0, {str(checkout / '.agents/ops/review')!r})\n"
                "import merge_gate\nfrom parent import metrics\nfrom issues.cli import GitHub\n"
                "original = GitHub.get\n"
                "def traced(self, endpoint, **kwargs):\n"
                "    frames = inspect.stack()\n"
                "    callers = [f.function for f in frames if '/.agents/ops/' in f.filename]\n"
                "    with open(os.environ['PARENT_READ_TRACE_FILE'], 'a') as stream:\n"
                "        stream.write(json.dumps({'endpoint': endpoint, 'callers': callers}) + '\\n')\n"
                "    return original(self, endpoint, **kwargs)\n"
                "GitHub.get = traced\n"
                "if hasattr(merge_gate, '_main'):\n"
                "    code = merge_gate.main(sys.argv[1:])\n"
                "else:\n"
                "    with metrics.command('merge_gate.merge', Path(os.environ['PARENT_METRICS_FILE'])) as row:\n"
                "        code = merge_gate.main(sys.argv[1:])\n"
                "        row['exit_code'] = code\n"
                "raise SystemExit(code)\n"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(driver),
                    "--repo",
                    "o/r",
                    "--parent",
                    "1",
                    "merge",
                ],
                cwd=p.repo,
                env={
                    **p.env,
                    "PARENT_METRICS_FILE": str(metrics),
                    "PARENT_READ_TRACE_FILE": str(trace),
                    "AGENT_AUTO_MERGE": "1" if mode == "merge" else "0",
                    "PARENT_MERGE_ADVANCE_BASE": "1",
                },
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
            (directory / "stdout.json").write_text(result.stdout)
            (directory / "stderr.txt").write_text(result.stderr)
            assert result.returncode == 0, result.stderr
            calls = p.calls()[offset:]
            endpoints = support.api_endpoints(calls)
            (measured,) = [
                json.loads(line) for line in metrics.read_text().splitlines()
            ]
            assert measured["github_requests"] == len(endpoints)
            assert measured["exit_code"] == 0
            assert len(gate.review.calls()) == 1
            record = {
                "repeat": repeat,
                "mode": mode,
                "peers": peers,
                "metrics": measured,
                "endpoint_counts": support.endpoint_counts(endpoints),
                "exact_counts": dict(Counter(endpoints)),
                "pr_view_calls": sum(c[:2] == ["pr", "view"] for c in calls),
                "merge_calls": sum(c[:2] == ["pr", "merge"] for c in calls),
            }
            (directory / "measurement.json").write_text(json.dumps(record, indent=2))
            records.append(record)
            print(json.dumps(record), flush=True)
    (output / "measurements.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--peers", type=int, default=20)
    parser.add_argument("--mode", choices=["dry-run", "merge"], default="dry-run")
    args = parser.parse_args()
    if args.repeats < 1 or args.peers < 0:
        parser.error("repeats must be positive and peers nonnegative")
    run(
        args.checkout.resolve(),
        args.output.resolve(),
        args.repeats,
        args.peers,
        args.mode,
    )
