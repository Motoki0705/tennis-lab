"""Different review requests must serialize overlapping PRs, not a whole parent."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
from dataclasses import replace

import pytest

from tests.e2e.agents_ops.test_parent_session import (
    REVIEW_SCRIPT,
    ReviewEnv,
    review_policy_module,
)
from tests.e2e.agents_ops.test_parent_session import (
    independent_pair_review_env as independent_pair_review_env,
)
from tests.e2e.agents_ops.test_parent_session import parent_env as parent_env


def subset_request(r: ReviewEnv, n: int) -> ReviewEnv:
    manifest = json.loads(
        r.p.cli(
            "review-prepare",
            *r.p.review_flags(),
            "--prs",
            str(n),
            "--reason-file",
            r.p.note("並行要求の照合"),
        ).stdout
    )
    request = json.loads(
        r.p.cli(
            "review-request",
            "--run-id",
            manifest["review_run_id"],
            "--instruction-file",
            r.p.note("選んだPRだけを確認する"),
        ).stdout
    )
    path = r.p.root / (request["review_run_id"] + ".json")
    path.write_text(json.dumps(request))
    output = json.loads(r.output.read_text())
    output["prs"] = [p for p in output["prs"] if p["number"] == n]
    output_path = path.with_suffix(".output.json")
    output_path.write_text(json.dumps(output))
    return replace(r, request=request, request_path=path, output=output_path)


def test_interrupted_execution_blocks_a_new_run_after_lease_release(
    independent_pair_review_env: ReviewEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r = independent_pair_review_env
    a, b = subset_request(r, 100), subset_request(r, 100)
    policy = review_policy_module(monkeypatch)
    a.run_dir.joinpath("execution.json").write_text(
        json.dumps(
            {
                "request_sha256": policy.digest(
                    json.loads((a.run_dir / "request.json").read_text())
                ),
                "status": "running",
                "agent_invocations": 1,
            }
        )
    )
    response = b.run(ok=False)
    assert response.returncode == 2 and "interrupted" in response.stderr
    assert len(r.calls()) == 0 and not (b.run_dir / "execution.json").exists()


@pytest.mark.parametrize("overlap", [True, False], ids=["same-pr", "independent-pr"])
def test_distinct_requests_cannot_start_overlapping_prs(
    independent_pair_review_env: ReviewEnv, overlap: bool
) -> None:
    r = independent_pair_review_env
    a = subset_request(r, 100)
    b = subset_request(r, 100 if overlap else 101)
    ready, release = r.p.root / "ready.pipe", r.p.root / "release.pipe"
    for path in (ready, release):
        os.mkfifo(path)
    # The first fake reviewer has actually entered before the second CLI starts.
    # Pipes coordinate process boundaries without a timing-dependent sleep.
    marker = 'if os.environ.get("REVIEW_AGENT_FAIL"):'
    barrier = """if os.environ.get("REVIEW_BARRIER"):
    with open(os.environ["REVIEW_READY"], "w") as signal:
        signal.write("ready\\n")
    with open(os.environ["REVIEW_RELEASE"]) as signal:
        signal.read(1)
"""
    r.fake.write_text(r.fake.read_text().replace(marker, barrier + marker))
    ready_fd = os.open(ready, os.O_RDWR | os.O_NONBLOCK)
    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                str(a.clock_script or REVIEW_SCRIPT),
                "run",
                "--request",
                str(a.request_path),
                "--agent-exec",
                str(a.fake),
            ],
            cwd=r.p.repo,
            env={
                **r.p.env,
                "REVIEW_AGENT_CALLS": str(r.calls_path),
                "REVIEW_AGENT_OUTPUT": str(a.output),
                "REVIEW_BARRIER": "1",
                "REVIEW_READY": str(ready),
                "REVIEW_RELEASE": str(release),
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(ready_fd, selectors.EVENT_READ)
                assert selector.select(timeout=30), "first reviewer never entered"
                assert os.read(ready_fd, 100) == b"ready\n"
            response = b.run(ok=not overlap)
        finally:
            # RDWR cannot hang if the first process failed before opening the FIFO.
            release_fd = os.open(release, os.O_RDWR | os.O_NONBLOCK)
            os.write(release_fd, b"x")
            try:
                stdout, stderr = proc.communicate(timeout=30)
            finally:
                os.close(release_fd)
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate()
        assert proc.returncode == 0, stdout + stderr
        if overlap:
            assert response.returncode == 2
            assert "PR" in response.stderr and "active" in response.stderr
        assert len(r.calls()) == (1 if overlap else 2)
    finally:
        os.close(ready_fd)
