"""Disposable local git, simulated test execution and real process-limit checks.

The fake runner never certifies Linux isolation or launches pytest/agents/gh.
All subprocess fixtures and locks live under pytest's temporary directory.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / ".agents/ops/ops-report"
sys.path.insert(0, str(PACKAGE))
repro: Any = importlib.import_module("opsreport.reproduction")
isolation: Any = importlib.import_module("opsreport.repro_isolation")
ledger: Any = importlib.import_module("opsreport.ledger")
server: Any = importlib.import_module("opsreport.reproduction_server")
transport: Any = importlib.import_module("opsreport.mcp_server")
SLOT = "20261024"
AT = "2026-10-24T00:00:00+00:00"
NODE = "tests/test_cpu.py::test_truth"


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"},
    )
    return result.stdout.strip()


class FakeRunner:
    def __init__(self) -> None:
        self.jobs: list[Any] = []
        self.dirty: str | None = None
        self.exit_code = 0
        self.reason = "exited"
        self.fail = False
        self.started: threading.Event | None = None
        self.release: threading.Event | None = None

    def inspect(self) -> dict[str, Any]:
        return {"enabled": True, "backend": "fake-test-only", "simulated": True}

    def run(self, job: Any) -> Any:
        import fcntl

        # Simulate the wrapper's lock contract, without invoking another pytest
        # while this test suite owns the real shared pytest lock.
        with job.pytest_lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.jobs.append(job)
            assert job.argv[:2] == ["/usr/bin/bash", str(job.checkout / repro.WRAPPER)]
            assert job.nodeids == (NODE,)
            assert (
                job.checkout / "tests/test_cpu.py"
            ).read_text() == "def test_truth():\n    assert True\n"
            job.log.write_text("fake CPU test\n1 passed\n")
            if self.started:
                self.started.set()
                assert self.release is not None and self.release.wait(10)
            if self.dirty == "tracked":
                (job.checkout / "tests/test_cpu.py").write_text("dirty\n")
            if self.dirty == "ignored":
                (job.checkout / "ignored.out").write_text("must retain\n")
            if self.dirty == "head":
                git(
                    job.checkout,
                    "-c",
                    "user.name=t",
                    "-c",
                    "user.email=t@example.invalid",
                    "commit",
                    "--allow-empty",
                    "-qm",
                    "changed",
                )
            if self.fail:
                raise OSError("fake runner failed after writing its log")
            return isolation.ProcessResult(
                self.exit_code, self.reason, 0.01, job.log.stat().st_size
            )


@dataclass
class Scenario:
    repo: Path
    base: str
    path: Path
    book: Path
    lock: Path
    runner: FakeRunner
    client: Any

    def manifests(self) -> list[dict[str, Any]]:
        return [json.loads(p.read_bytes()) for p in self.path.glob("*/result.json")]

    def book_now(self) -> Any:
        return ledger.Ledger.load(self.book, slot=SLOT, base_sha=self.base)


@pytest.fixture
def scenario(tmp_path: Path) -> Scenario:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    for name, value in {
        "tests/test_cpu.py": "def test_truth():\n    assert True\n",
        "tests/test_gpu.py": "def test_gpu():\n    raise RuntimeError('GPU forbidden')\n",
        ".gitignore": "ignored.out\n",
        repro.WRAPPER: (ROOT / repro.WRAPPER).read_text(),
    }.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    (repo / "tests/test_link.py").symlink_to(tmp_path / "secret.py")
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    base = git(repo, "rev-parse", "HEAD")
    (repo / "tests/test_cpu.py").write_text("later committed version\n")
    git(repo, "add", ".")
    git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "commit",
        "-qm",
        "later",
    )
    (repo / "tests/test_cpu.py").write_text(
        "dirty source; must not execute or modify\n"
    )
    (repo / "untracked.txt").write_text("preserve\n")
    state = tmp_path / "private"
    state.mkdir(mode=0o700)
    book_path = state / "ledger.json"
    book = ledger.Ledger.create(
        book_path, slot=SLOT, base_sha=base, goal="investigate", at=AT
    )
    book.record_task(
        {
            "id": "task",
            "question": "Can the suspicion be reproduced?",
            "scope": ["tests"],
            "origin": {"kind": "goal", "id": SLOT},
            "agent_id": None,
            "status": "planned",
            "started_at": None,
            "finished_at": None,
            "conclusion": "",
            "evidence": [],
            "next_questions": [],
            "unconfirmed": [],
        },
        at=AT,
    )
    book.record_finding(
        {
            "id": "finding",
            "task_id": "task",
            "kind": "ops",
            "status": "suspected",
            "conclusion": "A suspicion requiring an isolated test",
            "evidence": [],
            "counterevidence": [],
            "details": {},
        },
        at=AT,
    )
    runner = FakeRunner()
    runs = state / "runs"
    lock = state / "pytest.lock"
    client = repro.Reproduction(
        repo, base, SLOT, runs, book_path, [NODE], runner=runner, pytest_lock=lock
    )
    return Scenario(repo, base, runs, book_path, lock, runner, client)


def request(client: Any, **extra: Any) -> Any:
    return client.request({"finding_id": "finding", "nodeids": [NODE], **extra})


def test_pinned_clone_log_evidence_and_clean_removal(scenario: Scenario) -> None:
    before = scenario.client._source_state()
    registrations = git(scenario.repo, "worktree", "list", "--porcelain")
    result = request(scenario.client)
    assert result.data["status"] == "passed"
    assert result.data["cleanup"] == "removed-clean"
    assert result.data["source_before"] == result.data["source_after"] == before
    assert not Path(result.data["checkout"]).exists()
    assert scenario.client._source_state() == before
    assert (scenario.repo / "tests/test_cpu.py").read_text().startswith("dirty source")
    assert (scenario.repo / "untracked.txt").read_text() == "preserve\n"
    assert git(scenario.repo, "worktree", "list", "--porcelain") == registrations
    job = scenario.runner.jobs[0]
    assert job.pytest_lock == scenario.lock
    assert "-n" not in job.argv and "--numprocesses" not in job.argv
    evidence = scenario.book_now().snapshot["evidence"][result.data["evidence_id"]]
    assert evidence["reference"] == result.data["reference"]
    assert evidence["target_version"] == scenario.base
    finding = scenario.book_now().snapshot["findings"]["finding"]
    assert finding["status"] == "suspected"
    assert finding["evidence"] == [result.data["evidence_id"]]
    log = scenario.client.log(result.data["run_id"], line_count=1)
    assert log.pagination["has_more"]
    assert "fake CPU test" in log.data["log"]["text"]
    assert log.pagination["next"]["run_id"] == result.data["run_id"]


@pytest.mark.parametrize("dirty", ["tracked", "ignored", "head"])
def test_dirty_checkout_is_retained(scenario: Scenario, dirty: str) -> None:
    scenario.runner.dirty = dirty
    with pytest.raises(repro.RetrievalError, match="checkout retained") as failure:
        request(scenario.client)
    assert failure.value.code == "CHECKOUT_DIRTY"
    manifest = scenario.manifests()[0]
    assert manifest["cleanup"] == "retained-dirty-or-head-changed"
    assert Path(manifest["checkout"]).is_dir()
    assert scenario.book_now().snapshot["evidence"]


@pytest.mark.parametrize(
    "reason,code", [("exited", 1), ("timeout", -9), ("log_limit", -9)]
)
def test_failure_is_not_problem_confirmation(
    scenario: Scenario, reason: str, code: int
) -> None:
    scenario.runner.reason, scenario.runner.exit_code = reason, code
    result = request(scenario.client)
    assert result.data["status"] == "failed"
    assert result.data["disposition"] == "needs-reproduction"
    assert result.data["cleanup"] == "removed-clean"
    assert scenario.book_now().snapshot["findings"]["finding"]["status"] == "suspected"


def test_runner_failure_preserves_partial_log_and_evidence(scenario: Scenario) -> None:
    scenario.runner.fail = True
    with pytest.raises(repro.RetrievalError) as failure:
        request(scenario.client)
    assert failure.value.code == "REPRODUCTION_ERROR"
    manifest = scenario.manifests()[0]
    assert manifest["status"] == "error"
    assert manifest["log_sha256"]
    assert manifest["cleanup"] == "removed-clean"
    assert (
        "fake CPU test" in scenario.client.log(manifest["run_id"]).data["log"]["text"]
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"command": "echo forbidden"},
        {"code": "print('forbidden')"},
        {"network": True},
        {"env": {"CUDA_VISIBLE_DEVICES": "0"}},
        {"timeout": 901},
        {"memory_bytes": 8 * 1024**3},
        {"base_sha": "a" * 40},
        {"nodeids": ["../tests/test_cpu.py"]},
        {"nodeids": ["/tmp/test_cpu.py"]},
        {"nodeids": ["tests/../tests/test_cpu.py"]},
        {"nodeids": ["tests/test_cpu.py;touch /tmp/forbidden"]},
        {"nodeids": ["--pyargs"]},
        {"nodeids": ["tests/test_gpu.py"]},
        {"nodeids": ["tests/test_cpu.py::test_truth[param]"]},
        {"nodeids": ["tests/test_link.py"]},
        {"nodeids": [NODE, NODE]},
        {"nodeids": []},
    ],
)
def test_scope_refusals_are_ledger_evidence(
    scenario: Scenario, arguments: dict[str, Any]
) -> None:
    with pytest.raises(repro.RetrievalError) as failure:
        request(scenario.client, **arguments)
    assert failure.value.code == "FORBIDDEN_REQUEST", failure.value.details
    assert scenario.runner.jobs == []
    assert len(scenario.book_now().snapshot["evidence"]) == 1
    manifest = scenario.manifests()[0]
    assert manifest["disposition"] == "needs-reproduction"
    assert manifest["checkout"] is None


@pytest.mark.parametrize(
    "nodeids",
    [
        ["tests/test_link.py"],
        ["tests/test_missing.py"],
        ["../tests/test_cpu.py"],
        [NODE, NODE],
        [],
    ],
)
def test_host_allowlist_must_name_committed_regular_tests(
    scenario: Scenario, nodeids: list[str]
) -> None:
    with pytest.raises(ValueError):
        repro.Reproduction(
            scenario.repo,
            scenario.base,
            SLOT,
            scenario.path,
            scenario.book,
            nodeids,
            pytest_lock=scenario.lock,
        )


def test_same_host_lock_serializes_independent_clients(scenario: Scenario) -> None:
    scenario.runner.started, scenario.runner.release = (
        threading.Event(),
        threading.Event(),
    )
    second = repro.Reproduction(
        scenario.repo,
        scenario.base,
        SLOT,
        scenario.path,
        scenario.book,
        [NODE],
        runner=scenario.runner,
        pytest_lock=scenario.lock,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(request, scenario.client)
        try:
            assert scenario.runner.started.wait(10)
            with pytest.raises(repro.RetrievalError) as failure:
                request(second)
            assert failure.value.code == "REPRODUCTION_BUSY"
            assert len(scenario.runner.jobs) == 1
        finally:
            scenario.runner.release.set()
        running.result(timeout=10)
    scenario.runner.started, scenario.runner.release = None, None
    assert request(second).data["status"] == "passed"
    assert len(scenario.runner.jobs) == 2


def test_environment_is_constructed_without_credentials(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = [
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "API_KEY",
        "SSH_AUTH_SOCK",
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
        "AWS_SECRET_ACCESS_KEY",
        "HTTP_PROXY",
        "PYTEST_ADDOPTS",
        "PYTHONPATH",
        "LD_PRELOAD",
    ]
    for name in names:
        monkeypatch.setenv(name, "must-not-propagate")
    result = request(scenario.client)
    assert not set(names) & set(result.data["environment"])
    assert "must-not-propagate" not in json.dumps(result.data)
    assert result.data["environment"]["CUDA_VISIBLE_DEVICES"] == ""


def test_default_server_never_advertises_unverified_execution(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUDIT_REPRODUCTION_ENABLED", "1")
    client = server.ReproductionServer(
        scenario.repo,
        scenario.base,
        "o/r",
        SLOT,
        scenario.path.parent / "retrieval",
        scenario.book,
        [NODE],
        pytest_lock=scenario.lock,
    )
    listed = transport.dispatch(
        client, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    )
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert {"code_file", "cpu_test_status", "cpu_test_log"} <= names
    assert "cpu_tests" not in names
    status = client.call("cpu_test_status", {})
    assert status["data"]["enabled"] is False
    assert {
        "aggregate_memory",
        "gpu",
        "credential_files",
        "network",
        "descendants",
    } <= set(status["data"]["unverified"])
    refusal = client.call("cpu_tests", {"finding_id": "finding", "nodeids": [NODE]})
    assert not refusal["ok"]
    assert refusal["error"]["code"] == "ISOLATION_UNAVAILABLE"
    assert refusal["sources"][0]["disposition"] == "needs-reproduction"
    assert scenario.book_now().snapshot["evidence"]
    assert not list(client.reproduction.state_dir.glob("*/checkout"))
    records = [
        json.loads(line) for line in client.journal.path.read_text().splitlines()
    ]
    assert any(
        row.get("receipt") == refusal["receipt"] and row.get("event") == "received"
        for row in records
    )


def test_server_can_record_simulated_run_via_real_stdio_transport(
    scenario: Scenario,
) -> None:
    client = server.ReproductionServer(
        scenario.repo,
        scenario.base,
        "o/r",
        SLOT,
        scenario.path.parent / "retrieval",
        scenario.book,
        [NODE],
        runner=scenario.runner,
        pytest_lock=scenario.lock,
    )
    source = io.BytesIO(
        (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "cpu_tests",
                        "arguments": {"finding_id": "finding", "nodeids": [NODE]},
                    },
                }
            )
            + "\n"
        ).encode()
    )
    target = io.BytesIO()
    transport.serve(client, source, target)
    message = json.loads(target.getvalue())
    response = json.loads(message["result"]["content"][0]["text"])
    assert not message["result"]["isError"] and response["ok"]
    assert response["data"]["isolation"]["simulated"]


def test_real_server_process_keeps_cpu_execution_disabled(scenario: Scenario) -> None:
    allowlist = scenario.path.parent / "cpu-tests.json"
    allowlist.write_text(json.dumps({"base_sha": scenario.base, "nodeids": [NODE]}))
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "cpu_tests",
                "arguments": {"finding_id": "finding", "nodeids": [NODE]},
            },
        },
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "opsreport.reproduction_server",
            "serve",
            "--repo",
            str(scenario.repo),
            "--base-sha",
            scenario.base,
            "--repository",
            "o/r",
            "--round-id",
            SLOT,
            "--journal-dir",
            str(scenario.path.parent / "process-retrieval"),
            "--ledger",
            str(scenario.book),
            "--cpu-tests-file",
            str(allowlist),
        ],
        input="".join(json.dumps(message) + "\n" for message in messages),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
        env={**os.environ, "PYTHONPATH": str(PACKAGE)},
    )
    assert result.returncode == 0, result.stderr
    listed, called = [json.loads(line) for line in result.stdout.splitlines()]
    assert "cpu_tests" not in {tool["name"] for tool in listed["result"]["tools"]}
    assert called["result"]["isError"] is True
    response = json.loads(called["result"]["content"][0]["text"])
    assert response["error"]["code"] == "ISOLATION_UNAVAILABLE"
    assert len(scenario.book_now().snapshot["evidence"]) == 1


@pytest.mark.parametrize("tamper", ["log", "manifest"])
def test_evidence_tampering_is_not_read_as_valid(
    scenario: Scenario, tamper: str
) -> None:
    result = request(scenario.client)
    directory = scenario.path / result.data["run_id"]
    path = directory / ("pytest.log" if tamper == "log" else "result.json")
    if tamper == "log":
        path.write_text("edited log\n")
    else:
        value = json.loads(path.read_bytes())
        value["status"] = "edited"
        path.write_text(json.dumps(value))
    with pytest.raises(repro.RetrievalError) as failure:
        scenario.client.log(result.data["run_id"])
    assert failure.value.code == "RECORD_MISMATCH"


def test_recording_failure_does_not_return_a_successful_run(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise ledger.LedgerConflictError("concurrent ledger writer")

    monkeypatch.setattr(ledger.Ledger, "record_evidence", fail)
    with pytest.raises(repro.RetrievalError) as failure:
        request(scenario.client)
    assert failure.value.code == "RECORDING_FAILED"
    assert len(scenario.manifests()) == 1
    assert not scenario.book_now().snapshot["evidence"]


def test_missing_log_is_an_explicit_incomplete_record(scenario: Scenario) -> None:
    result = request(scenario.client)
    (scenario.path / result.data["run_id"] / "pytest.log").unlink()
    with pytest.raises(repro.RetrievalError) as failure:
        scenario.client.log(result.data["run_id"])
    assert failure.value.code == "MISSING_RECORD"


def test_evidence_and_observation_use_the_same_timestamp(
    scenario: Scenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A wall-clock correction after acquisition must not make that acquisition
    # appear to be in the future relative to its own ledger event.
    times = iter(
        [
            "2026-10-24T01:00:00+00:00",
            "2026-10-24T01:01:00+00:00",
            "2026-10-24T00:59:00+00:00",
        ]
    )
    monkeypatch.setattr(repro, "now", lambda: next(times))
    result = request(scenario.client)
    evidence = scenario.book_now().snapshot["evidence"][result.data["evidence_id"]]
    assert evidence["acquired_at"] == result.data["finished_at"]


@pytest.mark.parametrize("close_output", [False, True])
def test_timeout_reaps_group_when_output_is_open_or_closed(
    tmp_path: Path, close_output: bool
) -> None:
    child = tmp_path / "child.pid"
    redirection = (
        ",stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL" if close_output else ""
    )
    script = (
        "import os,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-I','-S','-c','import time; time.sleep(30)']"
        + redirection
        + "); "
        f"open({str(child)!r},'w').write(str(p.pid)); "
        + ("os.close(1); os.close(2); " if close_output else "")
        + "time.sleep(30)"
    )
    result = isolation.run_bounded(
        [sys.executable, "-I", "-S", "-c", script],
        cwd=tmp_path,
        log=tmp_path / "process.log",
        env=isolation.clean_environment(),
        timeout=0.5,
    )
    assert result.reason == "timeout" and result.exit_code != 0
    pid = int(child.read_text())
    status = Path(f"/proc/{pid}/status")
    assert not status.exists() or "State:\tZ" in status.read_text()


def test_log_limit_kills_the_process(tmp_path: Path) -> None:
    result = isolation.run_bounded(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import os,time; os.write(1,b'x'*4096); time.sleep(30)",
        ],
        cwd=tmp_path,
        log=tmp_path / "process.log",
        env=isolation.clean_environment(),
        timeout=2,
        max_log_bytes=1024,
    )
    assert result.reason == "log_limit"
    assert result.log_bytes == 1024
    assert (tmp_path / "process.log").stat().st_size == 1024
