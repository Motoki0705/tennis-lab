"""Small recorded observations test cost warnings without launching pytest suites."""

from __future__ import annotations

import copy
import html
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.ci_metrics import growth, growth_baseline

REPO = Path(__file__).resolve().parents[3]


def _snapshot(
    nodes: list[str],
    *,
    family: str = "tests/test_cost.py::test_cost",
    seconds: float = 1.0,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "complete": True,
        "source_sha": "main-sha",
        "run_id": "42",
        "attempt": "2",
        "universe_count": len(nodes),
        "requested_count": len(nodes),
        "deferred_count": 0,
        "nodeids": nodes,
        "details": [
            {
                "nodeid": node,
                "file": node.split("::", 1)[0],
                "family": family,
                "originalname": "test_cost",
                "parametrized": True,
                "callspec_id": f"value[{i}::nested]",
                "fixtures": ["shared"],
            }
            for i, node in enumerate(nodes)
        ],
        "results": [
            {
                "nodeid": node,
                "status": "passed",
                "phase_s": {"setup": 0.0, "call": seconds, "teardown": 0.0},
            }
            for node in nodes
        ],
        "fixtures": [],
    }


@pytest.mark.parametrize(
    "offset,expected", [(-0.001, False), (0, False), (0.001, True)]
)
def test_warning_boundaries_are_strict_and_use_single_fixture_executions(
    offset: float, expected: bool
) -> None:
    nodes = ["tests/test_cost.py::test_cost[value[inner::id]]"]
    current = _snapshot(nodes)
    current["results"][0]["phase_s"] = {"setup": 1 + offset, "call": 3, "teardown": 1}
    current["fixtures"] = [
        {
            "fixture": "support.shared [session]",
            "name": "shared",
            "scope": "session",
            "count": 2,
            "setup_s": 2 * (1 + offset),
            "max_setup_s": 1 + offset,
            "errors": 0,
            "skipped": 0,
            "xfailed": 0,
        }
    ]
    result = growth.analyze(current, None, baseline_reason="first run")
    assert bool(result["slow_cases"]) is expected
    assert bool(result["setup_cases"]) is expected
    assert bool(result["slow_fixtures"]) is expected
    if expected:
        assert result["slow_cases"][0]["total_s"] == pytest.approx(5.001)
        assert result["slow_fixtures"][0]["count"] == 2
    document = growth.markdown(result)
    assert "first run" in document and "never fail CI" in document
    assert "setup + call + teardown" in document


@pytest.mark.parametrize("count,expected", [(49, False), (50, False), (51, True)])
def test_expansion_uses_collected_originalname_even_with_brackets_and_separators(
    count: int, expected: bool
) -> None:
    family = "tests/test_[compat].py::TestCost::test_cost"
    nodes = [f"{family}[value[{i}::nested]]" for i in range(count)]
    current = _snapshot(nodes, family=family)
    result = growth.analyze(current, current)
    assert bool(result["expanded_functions"]) is expected
    if expected:
        assert result["expanded_functions"] == [
            {"family": family, "count": 51, "total_s": 51.0}
        ]
        assert family in growth.markdown(result)


def test_file_deltas_new_heavy_cases_and_profile_drift_keep_independent_costs() -> None:
    old = _snapshot(
        [
            "tests/test_cost.py::test_cost[old]",
            "tests/test_cost.py::test_cost[retained]",
            "tests/test_deleted.py::test_removed",
        ],
        seconds=5,
    )
    current = _snapshot(
        [
            "tests/test_cost.py::test_cost[retained]",
            "tests/test_cost.py::test_cost[new]",
            "tests/test_new.py::test_new",
        ],
        seconds=6,
    )
    result = growth.analyze(
        current, old, profile={"file_seconds": {"tests/test_cost.py": 8}}
    )
    assert result["file_changes"] == [
        {
            "file": "tests/test_cost.py",
            "before": 2,
            "after": 2,
            "added": 1,
            "removed": 1,
            "net": 0,
        },
        {
            "file": "tests/test_deleted.py",
            "before": 1,
            "after": 0,
            "added": 0,
            "removed": 1,
            "net": -1,
        },
        {
            "file": "tests/test_new.py",
            "before": 0,
            "after": 1,
            "added": 1,
            "removed": 0,
            "net": 1,
        },
    ]
    assert {row["nodeid"]: row["change"] for row in result["slow_cases"]} == {
        current["nodeids"][0]: "crossed threshold",
        current["nodeids"][1]: "new case",
        current["nodeids"][2]: "new case",
    }
    assert result["profile_drift"] == [
        {
            "file": "tests/test_new.py",
            "profile_s": None,
            "measured_s": 6.0,
            "delta_s": None,
            "delta_percent": None,
        },
        {
            "file": "tests/test_cost.py",
            "profile_s": 8,
            "measured_s": 12.0,
            "delta_s": 4.0,
            "delta_percent": 50.0,
        },
    ]
    assert "unmeasured in profile" in growth.markdown(result)


def test_partial_scope_excludes_deferred_files_from_all_comparisons() -> None:
    baseline = _snapshot(
        ["tests/test_cost.py::test_cost[old]", "tests/test_deferred.py::test_deferred"]
    )
    current = _snapshot(["tests/test_cost.py::test_cost[new]"])
    current.update(universe_count=2, requested_count=1, deferred_count=1)
    result = growth.analyze(
        current,
        baseline,
        profile={
            "file_seconds": {"tests/test_cost.py": 2, "tests/test_deferred.py": 100}
        },
    )
    assert result["file_changes"] == [
        {
            "file": "tests/test_cost.py",
            "before": 1,
            "after": 1,
            "added": 1,
            "removed": 1,
            "net": 0,
        }
    ]
    assert len(result["profile_drift"]) == 1
    document = growth.markdown(result)
    assert "R only" in document and "D is unmeasured" in document
    assert "test_deferred" not in document


def test_existing_heavy_and_setup_skips_keep_their_status_and_missing_call() -> None:
    current = _snapshot(["tests/test_cost.py::test_cost[x]"], seconds=6)
    baseline = copy.deepcopy(current)
    result = growth.analyze(current, baseline)
    assert result["slow_cases"][0]["change"] == "existing"
    current["results"][0].update(status="skipped", phase_s={"setup": 6, "teardown": 0})
    report = growth.analyze(current, None, baseline_reason="no ancestor")
    document = growth.markdown(report)
    assert "skipped" in document and "unmeasured" in document
    assert "no ancestor" in document


@pytest.mark.parametrize("fault", ["missing", "disagreement", "identity"])
def test_missing_or_inconsistent_collection_metadata_has_an_explicit_reason(
    fault: str,
) -> None:
    current = _snapshot(["tests/test_cost.py::test_cost[x]"])
    copies = [current["details"], copy.deepcopy(current["details"])]
    if fault == "missing":
        copies[0] = None
    elif fault == "disagreement":
        copies[1][0]["originalname"] = "different"
    else:
        current["nodeids"] = ["other"]
    result = growth.inventory_details(copies, current["nodeids"])
    assert result["items"] == []
    assert result["issues"]


def test_cost_constants_match_the_authoritative_policy() -> None:
    policy = (REPO / "tests/README.md").read_text()
    expansion = re.search(r"展開後 \*\*(\d+)件まで\*\*", policy)
    case = re.search(r"\| 1ケースの時間 \| \*\*(\d+)秒\*\*", policy)
    setup = re.search(r"\*\*setup が(\d+)秒\*\*", policy)
    assert expansion and case and setup
    assert (int(expansion[1]), int(case[1]), int(setup[1])) == (
        growth.EXPANSION_LIMIT,
        growth.CASE_SECONDS_LIMIT,
        growth.SETUP_SECONDS_LIMIT,
    )


def test_split_family_is_counted_once_and_cheap_fixture_repeats_stay_below_budget(
    tmp_path: Path,
) -> None:
    nodes = [f"tests/test_cost.py::test_cost[{i}]" for i in range(51)]
    snapshot = _snapshot(nodes)
    for row in snapshot["results"][:2]:
        row["phase_s"] = {"setup": 0.6, "call": 0.4, "teardown": 0}
    for index in range(2):
        directory = tmp_path / f"shard-{index}/tests"
        directory.mkdir(parents=True)
        selected = nodes[index::2]
        (directory / "inventory.json").write_text(json.dumps(selected))
        (directory / "inventory-details.json").write_text(
            json.dumps(
                {"schema_version": 1, "issues": [], "items": snapshot["details"]}
            )
        )
        (directory / "results.json").write_text(
            json.dumps(
                [row for row in snapshot["results"] if row["nodeid"] in selected]
            )
        )
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "fixtures": [
                        {
                            "fixture": "support.shared [session]",
                            "name": "shared",
                            "scope": "session",
                            "count": 1,
                            "setup_s": 0.6,
                            "max_setup_s": 0.6,
                            "errors": 0,
                            "skipped": 0,
                            "xfailed": 0,
                        }
                    ]
                }
            )
        )
    combined = growth.aggregate_snapshot(
        tmp_path,
        {
            "observations_complete": True,
            **{
                key: snapshot[key]
                for key in (
                    "source_sha",
                    "run_id",
                    "attempt",
                    "universe_count",
                    "requested_count",
                    "deferred_count",
                )
            },
        },
    )
    report = growth.analyze(combined, None)
    assert len(combined["nodeids"]) == 51
    assert report["expanded_functions"] == [
        {"family": "tests/test_cost.py::test_cost", "count": 51, "total_s": 51.0}
    ]
    assert combined["fixtures"][0]["count"] == 2
    assert combined["fixtures"][0]["setup_s"] == 1.2
    assert report["slow_fixtures"] == []


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "no_run",
        "non_ancestor",
        "expired",
        "missing_growth",
        "schema",
        "coverage_schema",
        "missing_field",
        "missing_call",
        "wrong_attempt",
        "partial",
        "corrupt_zip",
        "api_error",
    ],
)
def test_main_baseline_requires_the_latest_full_ancestor_and_explains_missing_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str | None
) -> None:
    snapshot = _snapshot(["tests/test_cost.py::test_cost[x]"])
    coverage = {
        "schema_version": 2,
        "success": True,
        "full_cpu_success": fault != "partial",
        "event": "push",
        "ref": "refs/heads/main",
        **{
            key: snapshot[key]
            for key in (
                "source_sha",
                "run_id",
                "attempt",
                "universe_count",
                "requested_count",
                "deferred_count",
            )
        },
    }
    if fault == "schema":
        snapshot["schema_version"] = 99
    elif fault == "coverage_schema":
        coverage["schema_version"] = 1
    elif fault == "missing_field":
        del snapshot["results"]
    elif fault == "missing_call":
        del snapshot["results"][0]["phase_s"]["call"]
    elif fault == "wrong_attempt":
        snapshot["attempt"] = "1"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("coverage.json", json.dumps(coverage))
        if fault != "missing_growth":
            stream.writestr("growth.json", json.dumps(snapshot))
    fetched: list[str] = []

    def api(endpoint: str) -> bytes:
        fetched.append(endpoint)
        if fault == "api_error":
            raise subprocess.CalledProcessError(1, "gh api")
        if "/workflows/" in endpoint:
            return json.dumps(
                {
                    "workflow_runs": []
                    if fault == "no_run"
                    else [
                        {
                            "id": 42,
                            "run_attempt": 2,
                            "head_sha": "main-sha",
                            "event": "push",
                            "head_branch": "main",
                            "conclusion": "success",
                        },
                        {
                            "id": 41,
                            "run_attempt": 1,
                            "head_sha": "older",
                            "event": "push",
                            "head_branch": "main",
                            "conclusion": "success",
                        },
                    ]
                }
            ).encode()
        if endpoint.endswith("/zip"):
            return b"not a zip" if fault == "corrupt_zip" else archive.getvalue()
        return json.dumps(
            {
                "artifacts": [
                    {
                        "name": "cpu-test-coverage-42-2",
                        "id": 7,
                        "expired": fault == "expired",
                    }
                ]
            }
        ).encode()

    monkeypatch.setattr(growth_baseline, "_api", api)
    monkeypatch.setattr(
        growth_baseline.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1 if fault == "non_ancestor" else 0),
    )
    baseline, reason = growth_baseline.previous_main(
        tmp_path, "owner/repo", "head", "current-run"
    )
    assert (baseline is not None) is (fault is None)
    if fault is not None:
        assert reason
        document = growth.markdown(
            growth.analyze(
                _snapshot(["tests/test_cost.py::test_cost[x]"]),
                baseline,
                baseline_reason=reason,
            )
        )
        assert (
            "Baseline comparison unavailable:" in document
            and reason in html.unescape(document)
        )
    else:
        assert reason is None and baseline == snapshot
    if fault in {
        "missing_growth",
        "schema",
        "missing_field",
        "missing_call",
        "wrong_attempt",
        "coverage_schema",
        "expired",
    }:
        assert not any("/runs/41/" in endpoint for endpoint in fetched)


@pytest.mark.parametrize("exclude_newest", ["non_ancestor", "current_run", "partial"])
def test_baseline_skips_only_proven_ineligible_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exclude_newest: str
) -> None:
    def api(endpoint: str) -> bytes:
        if "/workflows/" in endpoint:
            return json.dumps(
                {
                    "workflow_runs": [
                        {
                            "id": run,
                            "run_attempt": 1,
                            "head_sha": f"sha-{run}",
                            "event": "push",
                            "head_branch": "main",
                            "conclusion": "success",
                        }
                        for run in (43, 42)
                    ]
                }
            ).encode()
        run = 43 if "/43/" in endpoint else 42
        if endpoint.endswith("/zip"):
            snapshot = _snapshot(["tests/test_cost.py::test_cost[x]"])
            snapshot.update(source_sha=f"sha-{run}", run_id=str(run), attempt="1")
            coverage = {
                "schema_version": 2,
                "success": True,
                "full_cpu_success": run == 42,
                "event": "push",
                "ref": "refs/heads/main",
                **{
                    key: snapshot[key]
                    for key in (
                        "source_sha",
                        "run_id",
                        "attempt",
                        "universe_count",
                        "requested_count",
                        "deferred_count",
                    )
                },
            }
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("coverage.json", json.dumps(coverage))
                archive.writestr("growth.json", json.dumps(snapshot))
            return buffer.getvalue()
        return json.dumps(
            {
                "artifacts": [
                    {"name": f"cpu-test-coverage-{run}-1", "id": run, "expired": False}
                ]
            }
        ).encode()

    monkeypatch.setattr(growth_baseline, "_api", api)
    monkeypatch.setattr(
        growth_baseline.subprocess,
        "run",
        lambda command, **kwargs: SimpleNamespace(
            returncode=int(exclude_newest == "non_ancestor" and command[-2] == "sha-43")
        ),
    )
    baseline, reason = growth_baseline.previous_main(
        tmp_path,
        "owner/repo",
        "head",
        "43" if exclude_newest == "current_run" else "44",
    )
    assert baseline is not None and baseline["run_id"] == "42"
    assert reason is None
