"""Real pytest/xdist output, failure propagation, and safe timing refresh."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.automation.ci.reporting import (
    junit_timings,
    run_tests,
    update_durations,
    write_json,
)
from src.automation.ci.sharding import (
    CI_EXCLUDED_FILES,
    partition_tests,
    read_durations,
)


@pytest.mark.parametrize("workers", [0, 2])
def test_real_pytest_reports_survive_failures_and_include_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_sample.py").write_text(
        "import pytest\n"
        "def test_pass(): pass\n"
        "def test_fail(): assert False\n"
        "@pytest.mark.skip(reason='fixture')\n"
        "def test_skip(): pass\n"
    )
    monkeypatch.setattr(
        "src.automation.ci.reporting.subprocess.check_output",
        lambda *args, **kwargs: "revision\n",
    )
    summary = tmp_path / "github-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    group = partition_tests(["tests/test_sample.py"], {}, count=1)[0]
    output = tmp_path / "reports/shard-1"
    code = run_tests(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            str(workers),
            "tests/test_sample.py",
        ],
        repo_root=tmp_path,
        shard=group,
        output=output,
    )
    assert code == 1
    report = json.loads((output / "metrics.json").read_text())
    assert report["exit_code"] == 1
    assert report["elapsed_seconds"] > 0
    assert report["seconds_by_file"]["tests/test_sample.py"] >= 0
    assert "Exit code: 1" in summary.read_text()
    with pytest.raises(ValueError, match="successful"):
        update_durations(
            tmp_path / "reports", repo_root=tmp_path, output=tmp_path / "profile.json"
        )


def test_junit_sums_parameterized_cases_and_rejects_unassigned_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "junit.xml"
    path.write_text(
        '<testsuites><testsuite><testcase file="tests/test_a.py" time="1.5"/>'
        '<testcase file="tests/test_a.py" time="2.5"/></testsuite></testsuites>'
    )
    assert junit_timings(path, ("tests/test_a.py", "tests/test_cuda.py")) == {
        "tests/test_a.py": 4.0,
        "tests/test_cuda.py": 0.0,
    }
    with pytest.raises(ValueError, match="unassigned"):
        junit_timings(path, ("tests/test_other.py",))


def _reports(tmp_path: Path) -> tuple[Path, Path]:
    excluded = tmp_path / next(iter(CI_EXCLUDED_FILES))
    excluded.parent.mkdir(parents=True)
    excluded.touch()
    root = tmp_path / "reports"
    for index in (1, 2):
        name = f"tests/test_{index}.py"
        (tmp_path / name).touch()
        write_json(
            root / f"shard-{index}/metrics.json",
            {
                "schema_version": 1,
                "index": index,
                "count": 2,
                "revision": "abc",
                "exit_code": 0,
                "files": [name],
                "seconds_by_file": {name: float(index)},
            },
        )
    return root, tmp_path / "profile.json"


def test_complete_successful_run_can_refresh_timing_profile(tmp_path: Path) -> None:
    reports, output = _reports(tmp_path)
    update_durations(reports, repo_root=tmp_path, output=output)
    assert read_durations(output) == {"tests/test_1.py": 1.0, "tests/test_2.py": 2.0}


@pytest.mark.parametrize(
    "corruption", ["missing", "revision", "duplicate", "coverage", "new_file"]
)
def test_incomplete_or_inconsistent_runs_cannot_replace_baseline(
    tmp_path: Path, corruption: str
) -> None:
    reports, output = _reports(tmp_path)
    path = reports / "shard-2/metrics.json"
    payload = json.loads(path.read_text())
    if corruption == "missing":
        path.unlink()
    elif corruption == "revision":
        payload["revision"] = "different"
    elif corruption == "duplicate":
        payload["index"] = 1
    elif corruption == "coverage":
        payload["files"] = ["tests/test_1.py"]
    else:
        (tmp_path / "tests/test_new.py").touch()
    if corruption != "missing":
        write_json(path, payload)
    output.write_text("original baseline")
    with pytest.raises(ValueError):
        update_durations(reports, repo_root=tmp_path, output=output)
    assert output.read_text() == "original baseline"
