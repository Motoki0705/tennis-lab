"""Aggregate canonical contracts for Issue #753 benchmark evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from src.tasks.blcs.benchmarks import (
    track_query_compressor,
    track_query_cswa,
    track_query_integrated,
    track_query_mhc,
)
from src.tasks.blcs.benchmarks.contracts import (
    BenchmarkContractError,
    benchmark_path_resolver,
    load_json_object,
    repository_root,
    resolve_benchmark_cli_path,
)
from src.tasks.blcs.benchmarks.track_query_integrated.provenance import (
    EXPECTED_COMPONENTS,
    EXPECTED_EVIDENCE_PATHS,
    EXPECTED_ORCHESTRATOR_SESSION_UUID,
    EXPECTED_OWNERS,
    OWNER_SESSION_SNAPSHOT_DIRECTORY,
    PACKAGE_ORDER,
    PROVENANCE_SCHEMA_VERSION,
    SESSION_META_BYTE_SCOPE,
    SESSION_META_RECORD_SCHEMA,
    ProvenancePackageSpec,
    build_serial_provenance,
    validate_serial_provenance,
)
from src.utils.models.components.ops.compressed_time_local import api

_EVIDENCE_ROOT = Path("src/tasks/blcs/benchmarks/results/issue_753")
_TEST_SESSION_UUIDS = {
    package_id: f"00000000-0000-4000-8000-{index:012d}"
    for index, package_id in enumerate(PACKAGE_ORDER, start=1)
}


def _evidence(name: str) -> dict[str, Any]:
    return load_json_object(_EVIDENCE_ROOT / f"{name}.json")


def _rebind_provenance_derived_fields(document: dict[str, Any]) -> None:
    """Refresh attacker-controlled derived fields after one adversarial edit."""
    packages = cast(list[dict[str, Any]], document["packages"])
    serial = cast(dict[str, Any], document["serial_validation"])
    serial["status"] = (
        "PASS"
        if all(
            serial[key] is True
            for key in (
                "unique_owner_tasks",
                "unique_child_sessions",
                "unique_queue_jobs",
                "job_ids_strictly_increasing",
                "no_time_overlap",
            )
        )
        else "FAIL"
    )
    payload = json.dumps(
        packages,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    document["record_bundle_sha256"] = hashlib.sha256(payload).hexdigest()


def _session_meta_record_bytes(
    *, session_uuid: str, owner_task: str, parent_thread_id: str
) -> bytes:
    record = {
        "timestamp": "2026-08-19T00:00:00.000Z",
        "type": "session_meta",
        "payload": {
            "session_id": parent_thread_id,
            "id": session_uuid,
            "parent_thread_id": parent_thread_id,
            "source": {
                "subagent": {
                    "thread_spawn": {
                        "parent_thread_id": parent_thread_id,
                        "agent_path": owner_task,
                    }
                }
            },
            "thread_source": "subagent",
            "agent_path": owner_task,
        },
    }
    return (
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def _prepare_serial_provenance_inputs(
    tmp_path: Path,
) -> tuple[
    list[ProvenancePackageSpec],
    Path,
    Path,
    Path,
    dict[str, bytes],
]:
    root = tmp_path / "repository"
    queue_root = root / ".training_queue"
    codex_home = tmp_path / "codex"
    session_root = codex_home / "sessions" / "2026" / "08" / "19"
    session_root.mkdir(parents=True)
    for directory in ("done", "repro", "logs"):
        (queue_root / directory).mkdir(parents=True)

    for package_id, evidence_path in EXPECTED_EVIDENCE_PATHS.items():
        evidence_file = root / evidence_path
        evidence_file.parent.mkdir(parents=True, exist_ok=True)
        evidence_file.write_text(
            json.dumps({"component": EXPECTED_COMPONENTS[package_id]}) + "\n",
            encoding="utf-8",
        )

    specs: list[ProvenancePackageSpec] = []
    first_records: dict[str, bytes] = {}
    previous_done: datetime | None = None
    for index, package_id in enumerate(PACKAGE_ORDER, start=1):
        session_uuid = _TEST_SESSION_UUIDS[package_id]
        owner_task = EXPECTED_OWNERS[package_id]
        first_record = _session_meta_record_bytes(
            session_uuid=session_uuid,
            owner_task=owner_task,
            parent_thread_id=EXPECTED_ORCHESTRATOR_SESSION_UUID,
        )
        session_path = session_root / f"rollout-{session_uuid}.jsonl"
        session_path.write_bytes(first_record + b'{"type":"response_item"}\n')
        first_records[package_id] = first_record

        job_id = f"{index:03d}_issue753_{package_id.lower()}"
        started = (
            datetime.now(UTC) - timedelta(seconds=1)
            if previous_done is None
            else previous_done
        )
        added = started - timedelta(microseconds=1)
        run_path = queue_root / "repro" / job_id / "run.json"
        run_path.parent.mkdir(parents=True)
        run_path.write_text(
            json.dumps(
                {
                    "run_id": job_id,
                    "name": f"issue753-{package_id.lower()}",
                    "provider": "codex",
                    "session": session_uuid,
                    "issue": "753",
                    "command": f"benchmark {package_id}",
                    "cwd": str(root),
                    "captured_at": started.isoformat(),
                }
            ),
            encoding="utf-8",
        )
        (queue_root / "logs" / f"{job_id}.log").write_text(
            "completed\n", encoding="utf-8"
        )
        done_path = queue_root / "done" / f"{job_id}.job"
        done_path.write_text(
            "".join(
                (
                    f"# name: issue753-{package_id.lower()}\n",
                    f"# added: {added.isoformat()}\n",
                    "# provider: codex\n",
                    f"# session: {session_uuid}\n",
                    "# issue: 753\n",
                )
            ),
            encoding="utf-8",
        )
        previous_done = datetime.fromtimestamp(done_path.stat().st_ctime, tz=UTC)
        specs.append(
            ProvenancePackageSpec(
                package_id=package_id,
                owner_task=owner_task,
                session_uuid=session_uuid,
                session_meta_path=session_path,
                job_id=job_id,
            )
        )
    return specs, queue_root, codex_home, root, first_records


def _build_test_serial_provenance(
    tmp_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, bytes]]:
    specs, queue_root, codex_home, root, first_records = (
        _prepare_serial_provenance_inputs(tmp_path)
    )
    document = build_serial_provenance(
        specs,
        queue_root=queue_root,
        codex_home=codex_home,
        root=root,
    )
    return document, root, first_records


def _snapshot_path(
    document: Mapping[str, Any], root: Path, package_index: int
) -> Path:
    packages = cast(list[Mapping[str, Any]], document["packages"])
    owner = cast(Mapping[str, Any], packages[package_index]["owner"])
    session_meta = cast(Mapping[str, Any], owner["session_meta"])
    return root / cast(str, session_meta["snapshot_path"])


def _replace_snapshot_bytes(
    document: dict[str, Any],
    root: Path,
    package_index: int,
    snapshot_bytes: bytes,
) -> None:
    snapshot_path = _snapshot_path(document, root, package_index)
    snapshot_path.write_bytes(snapshot_bytes)
    packages = cast(list[dict[str, Any]], document["packages"])
    owner = cast(dict[str, Any], packages[package_index]["owner"])
    session_meta = cast(dict[str, Any], owner["session_meta"])
    session_meta["sha256"] = hashlib.sha256(snapshot_bytes).hexdigest()
    _rebind_provenance_derived_fields(document)


def _synthetic_available_integrated_runs() -> list[dict[str, Any]]:
    return [
        track_query_integrated._available_run(
            case=case,
            candidate=candidate,
            dtype_name=dtype_name,
            measurement=measurement,
            parameter_count=1,
            latency={"median_ms": 1.0, "p95_ms": 1.0},
            memory={"peak_allocated_bytes": 1, "peak_reserved_bytes": 1},
            throughput={"unit": "input-frames/s", "value": 1.0},
            parity=track_query_integrated._self_parity(candidate, dtype_name),
        )
        for case in track_query_integrated.BENCHMARK_CASES
        for dtype_name in track_query_integrated.DTYPES
        for candidate in track_query_integrated.CANDIDATES
        for measurement in track_query_integrated.MEASUREMENTS
    ]


def _required_training_run(
    runs: list[dict[str, Any]], candidate: str
) -> dict[str, Any]:
    return next(
        run
        for run in runs
        if run["case"] == track_query_integrated.REQUIRED_TRAINING_CASE
        and run["dtype"] == track_query_integrated.REQUIRED_TRAINING_DTYPE
        and run["measurement"]
        == track_query_integrated.REQUIRED_TRAINING_MEASUREMENT
        and run["candidate"] == candidate
    )


def test_benchmark_path_boundaries_return_concrete_paths(tmp_path: Path) -> None:
    root = repository_root()
    resolver = benchmark_path_resolver(tmp_path)

    assert isinstance(root, Path)
    assert resolve_benchmark_cli_path(
        Path("results/evidence.json"), resolver=resolver
    ) == (tmp_path / "results/evidence.json").resolve()
    absolute = (tmp_path / "runtime.json").resolve()
    resolved_absolute = resolve_benchmark_cli_path(absolute, resolver=resolver)
    assert isinstance(resolved_absolute, Path)
    assert resolved_absolute == absolute


def test_committed_cuda_package_and_integrated_evidence_is_canonical() -> None:
    mhc = _evidence("mhc")
    compressor = _evidence("compressor")
    cswa = _evidence("cswa")
    integrated = _evidence("integrated")

    track_query_mhc.validate_mhc_evidence(mhc)
    track_query_compressor.validate_compressor_evidence(compressor)
    track_query_cswa.validate_cswa_evidence(cswa)
    track_query_integrated.validate_integrated_evidence(integrated)

    assert cast(Mapping[str, Any], mhc["decision"])["status"] == "NO-GO"
    assert cast(Mapping[str, Any], mhc["decision"])["optimized_candidate"] == (
        "custom_cuda_prototype"
    )
    assert cast(Mapping[str, Any], compressor["decision"])["status"] == "NO-GO"
    assert cast(Mapping[str, Any], compressor["decision"])[
        "optimized_candidate"
    ] is None
    assert cast(Mapping[str, Any], cswa["decision"])["status"] == "GO"
    assert cast(Mapping[str, Any], cswa["decision"])["optimized_candidate"] == (
        "cuda"
    )
    assert cast(Mapping[str, Any], integrated["decision"])["status"] == "PASS"

    validate_serial_provenance(_evidence("provenance"))


def test_stable_benchmark_evidence_is_part_of_the_git_candidate() -> None:
    evidence_paths = {
        str(_EVIDENCE_ROOT / f"{component}.json")
        for component in ("mhc", "compressor", "cswa", "integrated", "provenance")
    }
    provenance = _evidence("provenance")
    for package in cast(list[Mapping[str, Any]], provenance["packages"]):
        owner = cast(Mapping[str, Any], package["owner"])
        session_meta = cast(Mapping[str, Any], owner["session_meta"])
        evidence_paths.add(cast(str, session_meta["snapshot_path"]))

    tracked = subprocess.run(
        ["git", "ls-files", "--", *sorted(evidence_paths)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert set(tracked) == evidence_paths, (
        "Stable benchmark evidence must be tracked so the sealed candidate and "
        f"PR contain it; missing={sorted(evidence_paths - set(tracked))}"
    )


def test_integrated_matrix_covers_real_widths_contexts_and_both_temporal_paths() -> None:
    protocol = track_query_integrated.PROTOCOL
    profiles = cast(list[Mapping[str, Any]], protocol["model_profiles"])
    cases = cast(list[Mapping[str, Any]], protocol["cases"])
    cases_by_name = {cast(str, case["name"]): case for case in cases}
    admission = cast(Mapping[str, Any], protocol["required_training_admission"])

    assert {profile["name"] for profile in profiles} == {"small", "base", "large"}
    assert {case["frames"] for case in cases} >= {512, 1024, 2048}
    required_case = cases_by_name[track_query_integrated.REQUIRED_TRAINING_CASE]
    assert required_case["batch_size"] == 1
    assert required_case["num_views"] == 3
    assert required_case["frames"] == 1024
    assert "physical batch=1" in cast(str, required_case["provenance"])
    assert "accumulate_grad_batches=8" in cast(str, required_case["provenance"])
    assert "effective batch=8" in cast(str, required_case["provenance"])
    assert "maximum configured T=1024" in cast(str, required_case["provenance"])

    historic = cases_by_name["historic-diagnostic-small-b8-t512"]
    assert historic["batch_size"] == 8
    assert historic["frames"] == 512
    assert "not a configured production training shape" in cast(
        str, historic["provenance"]
    )
    assert admission == {
        "case": track_query_integrated.REQUIRED_TRAINING_CASE,
        "model_profile": "small",
        "physical_batch_size": 1,
        "accumulate_grad_batches": 8,
        "effective_batch_size": 8,
        "num_views": 3,
        "frames": 1024,
        "configured_sequence_bound": "maximum",
        "dtype": "bfloat16",
        "measurement": "forward-backward",
        "candidates": list(track_query_integrated.CANDIDATES),
    }
    assert all(case["object_path_n"] == case["batch_size"] * case["num_views"] for case in cases)
    assert all(case["query_path_n"] == case["batch_size"] * case["num_queries"] for case in cases)
    assert protocol["dtypes"] == ["float32", "bfloat16"]
    assert protocol["measurements"] == ["forward", "forward-backward"]
    assert protocol["compile"] is False
    assert protocol["dropout"] == 0.0


def test_integrated_required_training_triplet_carries_complete_metrics() -> None:
    stable = _evidence("integrated")
    capacity = cast(Mapping[str, Any], stable["environment"])[
        "device_total_memory_bytes"
    ]
    required_runs = [
        run
        for run in cast(list[Mapping[str, Any]], stable["runs"])
        if run["case"] == track_query_integrated.REQUIRED_TRAINING_CASE
        and run["dtype"] == track_query_integrated.REQUIRED_TRAINING_DTYPE
        and run["measurement"]
        == track_query_integrated.REQUIRED_TRAINING_MEASUREMENT
    ]

    assert {run["candidate"] for run in required_runs} == set(
        track_query_integrated.CANDIDATES
    )
    for run in required_runs:
        assert run["available"] is True
        assert run["status"] == "ok"
        assert cast(Mapping[str, Any], run["latency"])["median_ms"] > 0
        assert cast(Mapping[str, Any], run["throughput"])["value"] > 0
        assert (
            track_query_integrated._physical_memory_violation(
                cast(Mapping[str, object], run["memory"]), capacity
            )
            is None
        )
        expected_parity = (
            "not-applicable"
            if run["candidate"] == "A-global-only"
            else "pass"
        )
        assert cast(Mapping[str, Any], run["parity"])["status"] == expected_parity


def test_integrated_decision_accepts_complete_required_training_triplet() -> None:
    decision = track_query_integrated.decide_integrated_benchmark(
        _synthetic_available_integrated_runs(),
        device_total_memory_bytes=1024,
    )

    assert decision["status"] == "PASS"


@pytest.mark.parametrize("candidate", track_query_integrated.CANDIDATES)
def test_integrated_required_training_triplet_rejects_missing_candidate(
    candidate: str,
) -> None:
    runs = _synthetic_available_integrated_runs()
    runs.remove(_required_training_run(runs, candidate))

    decision = track_query_integrated.decide_integrated_benchmark(
        runs,
        device_total_memory_bytes=1024,
    )

    assert decision["status"] == "FAIL"
    assert "required training candidate record missing" in cast(
        str, decision["reason"]
    )


@pytest.mark.parametrize("candidate", track_query_integrated.CANDIDATES)
@pytest.mark.parametrize(
    "violation",
    (
        "missing_latency",
        "invalid_latency",
        "missing_throughput",
        "invalid_throughput",
        "missing_memory",
        "invalid_memory",
        "over_capacity",
        "invalid_parity",
    ),
)
def test_required_training_admission_rejects_invalid_metrics_and_parity(
    candidate: str,
    violation: str,
) -> None:
    runs = _synthetic_available_integrated_runs()
    run = _required_training_run(runs, candidate)
    capacity = 1024

    if violation == "missing_latency":
        run["latency"] = None
        expected = "positive finite latency"
    elif violation == "invalid_latency":
        cast(dict[str, Any], run["latency"])["p95_ms"] = float("nan")
        expected = "positive finite latency"
    elif violation == "missing_throughput":
        run["throughput"] = None
        expected = "positive finite throughput"
    elif violation == "invalid_throughput":
        cast(dict[str, Any], run["throughput"])["value"] = 0.0
        expected = "positive finite throughput"
    elif violation == "missing_memory":
        run["memory"] = None
        expected = "must carry CUDA allocator memory"
    elif violation == "invalid_memory":
        cast(dict[str, Any], run["memory"])["peak_allocated_bytes"] = True
        expected = "memory is not capacity-valid"
    elif violation == "over_capacity":
        memory = cast(dict[str, Any], run["memory"])
        memory["peak_allocated_bytes"] = capacity + 1
        memory["peak_reserved_bytes"] = capacity + 1
        expected = "memory is not capacity-valid"
    else:
        parity = cast(dict[str, Any], run["parity"])
        parity["status"] = (
            "pass" if candidate == "A-global-only" else "fail"
        )
        expected = "parity must be"

    failure = track_query_integrated._required_training_run_failure(
        run,
        candidate=candidate,
        device_total_memory_bytes=capacity,
    )

    assert failure is not None
    assert expected in failure


@pytest.mark.parametrize("candidate", track_query_integrated.CANDIDATES)
@pytest.mark.parametrize("status", ("oom", "unsupported", "unavailable"))
def test_integrated_required_training_triplet_rejects_unavailable_rows(
    candidate: str,
    status: str,
) -> None:
    tampered = copy.deepcopy(_evidence("integrated"))
    capacity = cast(Mapping[str, Any], tampered["environment"])[
        "device_total_memory_bytes"
    ]
    run = next(
        run
        for run in cast(list[dict[str, Any]], tampered["runs"])
        if run["case"] == track_query_integrated.REQUIRED_TRAINING_CASE
        and run["dtype"] == track_query_integrated.REQUIRED_TRAINING_DTYPE
        and run["measurement"]
        == track_query_integrated.REQUIRED_TRAINING_MEASUREMENT
        and run["candidate"] == candidate
    )
    run["available"] = False
    run["status"] = status
    run["latency"] = None
    run["throughput"] = None
    run["memory"] = None
    run["parity"] = track_query_integrated._not_run_parity()
    run["unavailable_reason"] = f"adversarial {status}"
    tampered["decision"] = track_query_integrated.decide_integrated_benchmark(
        cast(list[Mapping[str, Any]], tampered["runs"]),
        device_total_memory_bytes=capacity,
    )

    decision = cast(Mapping[str, Any], tampered["decision"])
    assert decision["status"] == "FAIL"
    assert f"{candidate} must be available=true,status=ok" in cast(
        str, decision["reason"]
    )
    with pytest.raises(BenchmarkContractError, match="decision must PASS"):
        track_query_integrated.validate_integrated_evidence(tampered)


def test_integrated_schema_rejects_missing_runs_and_silent_omissions() -> None:
    stable = _evidence("integrated")
    missing_run = copy.deepcopy(stable)
    cast(list[dict[str, Any]], missing_run["runs"]).pop()
    with pytest.raises(BenchmarkContractError, match="run matrix/order"):
        track_query_integrated.validate_integrated_evidence(missing_run)

    unavailable = copy.deepcopy(stable)
    run = cast(list[dict[str, Any]], unavailable["runs"])[0]
    run["available"] = False
    run["status"] = "oom"
    run["latency"] = None
    run["throughput"] = None
    run["memory"] = None
    run["parity"] = track_query_integrated._not_run_parity()
    run["unavailable_reason"] = ""
    with pytest.raises(BenchmarkContractError, match="unavailable_reason"):
        track_query_integrated.validate_integrated_evidence(unavailable)


def test_integrated_schema_recomputes_decision_instead_of_trusting_go_text() -> None:
    stable = _evidence("integrated")
    tampered = copy.deepcopy(stable)
    cast(dict[str, Any], tampered["decision"])["complete_same_shape_triplets"] = -1

    with pytest.raises(BenchmarkContractError, match="decision does not match"):
        track_query_integrated.validate_integrated_evidence(tampered)


def test_physical_memory_capacity_accepts_normal_and_exact_boundary() -> None:
    capacity = 1024
    assert (
        track_query_integrated._physical_memory_violation(
            {"peak_allocated_bytes": 512, "peak_reserved_bytes": 768}, capacity
        )
        is None
    )
    assert (
        track_query_integrated._physical_memory_violation(
            {"peak_allocated_bytes": capacity, "peak_reserved_bytes": capacity},
            capacity,
        )
        is None
    )


@pytest.mark.parametrize(
    ("allocated", "reserved", "match"),
    (
        (1025, 1025, "exceed recorded physical device capacity"),
        (1024, 1025, "exceed recorded physical device capacity"),
        (769, 768, "exceeds peak_reserved_bytes"),
    ),
)
def test_physical_memory_capacity_rejects_impossible_counters(
    allocated: int, reserved: int, match: str
) -> None:
    violation = track_query_integrated._physical_memory_violation(
        {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved}, 1024
    )
    assert violation is not None
    assert match in violation


@pytest.mark.parametrize(
    ("memory", "capacity", "match"),
    (
        (
            {"peak_allocated_bytes": 1, "peak_reserved_bytes": 1},
            0,
            "capacity must be positive",
        ),
        (
            {"peak_allocated_bytes": True, "peak_reserved_bytes": 1},
            1024,
            "non-negative integers",
        ),
        (
            {"peak_allocated_bytes": 1, "peak_reserved_bytes": -1},
            1024,
            "non-negative integers",
        ),
    ),
)
def test_physical_memory_capacity_rejects_invalid_authority_and_counter_types(
    memory: dict[str, object], capacity: int, match: str
) -> None:
    violation = track_query_integrated._physical_memory_violation(memory, capacity)

    assert violation is not None
    assert match in violation


@pytest.mark.parametrize("memory_key", ("peak_allocated_bytes", "peak_reserved_bytes"))
def test_integrated_schema_rejects_over_capacity_memory_tampering(
    memory_key: str,
) -> None:
    stable = _evidence("integrated")
    tampered = copy.deepcopy(stable)
    capacity = cast(Mapping[str, Any], tampered["environment"])[
        "device_total_memory_bytes"
    ]
    run = next(
        run
        for run in cast(list[dict[str, Any]], tampered["runs"])
        if run["available"] is True
    )
    memory = cast(dict[str, Any], run["memory"])
    if memory_key == "peak_allocated_bytes":
        memory["peak_allocated_bytes"] = capacity + 1
        memory["peak_reserved_bytes"] = capacity + 1
    else:
        memory["peak_allocated_bytes"] = min(
            memory["peak_allocated_bytes"], capacity
        )
        memory["peak_reserved_bytes"] = capacity + 1

    with pytest.raises(BenchmarkContractError, match="violates capacity"):
        track_query_integrated.validate_integrated_evidence(tampered)


def test_serial_provenance_builder_copies_only_authenticated_first_record_bytes(
    tmp_path: Path,
) -> None:
    document, root, first_records = _build_test_serial_provenance(tmp_path)

    assert document["schema_version"] == PROVENANCE_SCHEMA_VERSION
    assert document["orchestrator_session_uuid"] == (
        EXPECTED_ORCHESTRATOR_SESSION_UUID
    )
    packages = cast(list[Mapping[str, Any]], document["packages"])
    for index, package_id in enumerate(PACKAGE_ORDER):
        owner = cast(Mapping[str, Any], packages[index]["owner"])
        session_meta = cast(Mapping[str, Any], owner["session_meta"])
        expected_path = (
            f"{OWNER_SESSION_SNAPSHOT_DIRECTORY}/"
            f"{package_id}-{_TEST_SESSION_UUIDS[package_id]}.jsonl"
        )
        assert session_meta == {
            "snapshot_path": expected_path,
            "sha256": hashlib.sha256(first_records[package_id]).hexdigest(),
            "byte_scope": SESSION_META_BYTE_SCOPE,
            "record_schema": SESSION_META_RECORD_SCHEMA,
            "source_origin": (
                "sessions/2026/08/19/"
                f"rollout-{_TEST_SESSION_UUIDS[package_id]}.jsonl"
            ),
        }
        assert (root / expected_path).read_bytes() == first_records[package_id]
    validate_serial_provenance(document, root=root)


def test_serial_provenance_builder_rejects_semantically_wrong_source_owner(
    tmp_path: Path,
) -> None:
    specs, queue_root, codex_home, root, _ = _prepare_serial_provenance_inputs(
        tmp_path
    )
    first_spec = specs[0]
    wrong_record = _session_meta_record_bytes(
        session_uuid=first_spec.session_uuid,
        owner_task=EXPECTED_OWNERS["6B"],
        parent_thread_id=EXPECTED_ORCHESTRATOR_SESSION_UUID,
    )
    first_spec.session_meta_path.write_bytes(wrong_record)

    with pytest.raises(BenchmarkContractError, match="agent_path mismatch"):
        build_serial_provenance(
            specs,
            queue_root=queue_root,
            codex_home=codex_home,
            root=root,
        )


def test_serial_provenance_builder_never_overwrites_immutable_snapshot(
    tmp_path: Path,
) -> None:
    specs, queue_root, codex_home, root, _ = _prepare_serial_provenance_inputs(
        tmp_path
    )
    snapshot_path = (
        root
        / OWNER_SESSION_SNAPSHOT_DIRECTORY
        / f"6A-{_TEST_SESSION_UUIDS['6A']}.jsonl"
    )
    snapshot_path.parent.mkdir(parents=True)
    snapshot_path.write_bytes(b"preexisting-different-bytes\n")

    with pytest.raises(BenchmarkContractError, match="already differs"):
        build_serial_provenance(
            specs,
            queue_root=queue_root,
            codex_home=codex_home,
            root=root,
        )
    assert snapshot_path.read_bytes() == b"preexisting-different-bytes\n"


def test_serial_provenance_builder_rejects_symlinked_snapshot_directory(
    tmp_path: Path,
) -> None:
    specs, queue_root, codex_home, root, _ = _prepare_serial_provenance_inputs(
        tmp_path
    )
    snapshot_directory = root / OWNER_SESSION_SNAPSHOT_DIRECTORY
    snapshot_directory.parent.mkdir(parents=True, exist_ok=True)
    alternate_directory = root / "alternate-owner-sessions"
    alternate_directory.mkdir()
    snapshot_directory.symlink_to(alternate_directory, target_is_directory=True)

    with pytest.raises(BenchmarkContractError, match="must not contain symlinks"):
        build_serial_provenance(
            specs,
            queue_root=queue_root,
            codex_home=codex_home,
            root=root,
        )


def test_serial_provenance_rejects_snapshot_byte_tamper(tmp_path: Path) -> None:
    document, root, _ = _build_test_serial_provenance(tmp_path)
    _snapshot_path(document, root, 0).write_bytes(b"tampered\n")

    with pytest.raises(BenchmarkContractError, match="does not match snapshot bytes"):
        validate_serial_provenance(document, root=root)


def test_serial_provenance_rejects_other_owner_snapshot_after_bundle_rebind(
    tmp_path: Path,
) -> None:
    document, root, _ = _build_test_serial_provenance(tmp_path)
    other_owner_bytes = _snapshot_path(document, root, 1).read_bytes()
    _replace_snapshot_bytes(document, root, 0, other_owner_bytes)

    with pytest.raises(BenchmarkContractError, match="payload.id mismatch"):
        validate_serial_provenance(document, root=root)


@pytest.mark.parametrize(
    ("snapshot_bytes", "match"),
    (
        (b"not-json\n", "invalid Codex session_meta record"),
        (
            _session_meta_record_bytes(
                session_uuid=_TEST_SESSION_UUIDS["6A"],
                owner_task=EXPECTED_OWNERS["6A"],
                parent_thread_id=EXPECTED_ORCHESTRATOR_SESSION_UUID,
            )
            + b'{"type":"response_item"}\n',
            "exactly one record",
        ),
    ),
)
def test_serial_provenance_rejects_malformed_or_multiple_snapshot_records(
    tmp_path: Path, snapshot_bytes: bytes, match: str
) -> None:
    document, root, _ = _build_test_serial_provenance(tmp_path)
    _replace_snapshot_bytes(document, root, 0, snapshot_bytes)

    with pytest.raises(BenchmarkContractError, match=match):
        validate_serial_provenance(document, root=root)


def test_serial_provenance_rejects_non_session_snapshot_record(
    tmp_path: Path,
) -> None:
    document, root, first_records = _build_test_serial_provenance(tmp_path)
    record = cast(dict[str, Any], json.loads(first_records["6A"]))
    record["type"] = "response_item"
    snapshot_bytes = json.dumps(record, separators=(",", ":")).encode() + b"\n"
    _replace_snapshot_bytes(document, root, 0, snapshot_bytes)

    with pytest.raises(BenchmarkContractError, match="must be session_meta"):
        validate_serial_provenance(document, root=root)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("id", _TEST_SESSION_UUIDS["6B"], "payload.id mismatch"),
        (
            "agent_path",
            EXPECTED_OWNERS["6B"],
            "payload.agent_path mismatch",
        ),
        (
            "parent_thread_id",
            "00000000-0000-4000-8000-999999999999",
            "session_id mismatch",
        ),
    ),
)
def test_serial_provenance_rejects_session_semantic_mismatch_after_rebind(
    tmp_path: Path, field: str, value: str, match: str
) -> None:
    document, root, first_records = _build_test_serial_provenance(tmp_path)
    record = cast(dict[str, Any], json.loads(first_records["6A"]))
    payload = cast(dict[str, Any], record["payload"])
    payload[field] = value
    snapshot_bytes = json.dumps(record, separators=(",", ":")).encode() + b"\n"
    _replace_snapshot_bytes(document, root, 0, snapshot_bytes)

    with pytest.raises(BenchmarkContractError, match=match):
        validate_serial_provenance(document, root=root)


@pytest.mark.parametrize(
    ("violation", "match"),
    (
        ("thread_source", "thread_source mismatch"),
        ("spawn_agent_path", "thread_spawn.agent_path mismatch"),
        ("spawn_parent", "thread_spawn.parent_thread_id mismatch"),
    ),
)
def test_serial_provenance_rejects_nested_session_authentication_mismatch(
    tmp_path: Path,
    violation: str,
    match: str,
) -> None:
    document, root, first_records = _build_test_serial_provenance(tmp_path)
    record = cast(dict[str, Any], json.loads(first_records["6A"]))
    payload = cast(dict[str, Any], record["payload"])
    source = cast(dict[str, Any], payload["source"])
    subagent = cast(dict[str, Any], source["subagent"])
    thread_spawn = cast(dict[str, Any], subagent["thread_spawn"])
    if violation == "thread_source":
        payload["thread_source"] = "root"
    elif violation == "spawn_agent_path":
        thread_spawn["agent_path"] = EXPECTED_OWNERS["6B"]
    else:
        thread_spawn["parent_thread_id"] = (
            "00000000-0000-4000-8000-999999999999"
        )
    snapshot_bytes = json.dumps(record, separators=(",", ":")).encode() + b"\n"
    _replace_snapshot_bytes(document, root, 0, snapshot_bytes)

    with pytest.raises(BenchmarkContractError, match=match):
        validate_serial_provenance(document, root=root)


def test_serial_provenance_rejects_missing_parent_thread_id_after_rebind(
    tmp_path: Path,
) -> None:
    document, root, first_records = _build_test_serial_provenance(tmp_path)
    record = cast(dict[str, Any], json.loads(first_records["6A"]))
    payload = cast(dict[str, Any], record["payload"])
    del payload["parent_thread_id"]
    snapshot_bytes = json.dumps(record, separators=(",", ":")).encode() + b"\n"
    _replace_snapshot_bytes(document, root, 0, snapshot_bytes)

    with pytest.raises(BenchmarkContractError, match="payload keys missing"):
        validate_serial_provenance(document, root=root)


def test_serial_provenance_rejects_different_authenticated_parent_after_rebind(
    tmp_path: Path,
) -> None:
    document, root, first_records = _build_test_serial_provenance(tmp_path)
    different_parent = "00000000-0000-4000-8000-999999999999"
    for index, package_id in enumerate(PACKAGE_ORDER):
        record = cast(dict[str, Any], json.loads(first_records[package_id]))
        payload = cast(dict[str, Any], record["payload"])
        payload["session_id"] = different_parent
        payload["parent_thread_id"] = different_parent
        source = cast(dict[str, Any], payload["source"])
        subagent = cast(dict[str, Any], source["subagent"])
        thread_spawn = cast(dict[str, Any], subagent["thread_spawn"])
        thread_spawn["parent_thread_id"] = different_parent
        snapshot_bytes = json.dumps(record, separators=(",", ":")).encode() + b"\n"
        _replace_snapshot_bytes(document, root, index, snapshot_bytes)
    document["orchestrator_session_uuid"] = different_parent
    _rebind_provenance_derived_fields(document)

    with pytest.raises(BenchmarkContractError, match="expected orchestrator"):
        validate_serial_provenance(document, root=root)


@pytest.mark.parametrize(
    "snapshot_path",
    (
        "../../outside.jsonl",
        (
            "src/tasks/blcs/benchmarks/results/issue_753/owner_sessions/"
            "../outside.jsonl"
        ),
        "/tmp/outside.jsonl",
    ),
)
def test_serial_provenance_rejects_snapshot_path_traversal(
    tmp_path: Path, snapshot_path: str
) -> None:
    document, root, _ = _build_test_serial_provenance(tmp_path)
    packages = cast(list[dict[str, Any]], document["packages"])
    owner = cast(dict[str, Any], packages[0]["owner"])
    session_meta = cast(dict[str, Any], owner["session_meta"])
    session_meta["snapshot_path"] = snapshot_path
    _rebind_provenance_derived_fields(document)

    with pytest.raises(
        BenchmarkContractError, match="canonical repository-relative path"
    ):
        validate_serial_provenance(document, root=root)


@pytest.mark.parametrize("entry_kind", ("missing", "symlink", "directory"))
def test_serial_provenance_rejects_missing_symlink_or_nonregular_snapshot(
    tmp_path: Path, entry_kind: str
) -> None:
    document, root, _ = _build_test_serial_provenance(tmp_path)
    snapshot_path = _snapshot_path(document, root, 0)
    original_bytes = snapshot_path.read_bytes()
    snapshot_path.unlink()
    if entry_kind == "symlink":
        alternate_path = root / "alternate-session.jsonl"
        alternate_path.write_bytes(original_bytes)
        snapshot_path.symlink_to(alternate_path)
    elif entry_kind == "directory":
        snapshot_path.mkdir()

    match = {
        "missing": "file is missing",
        "symlink": "must not contain symlinks",
        "directory": "not a regular file",
    }[entry_kind]
    with pytest.raises(BenchmarkContractError, match=match):
        validate_serial_provenance(document, root=root)


def test_serial_provenance_rejects_symlinked_snapshot_parent(
    tmp_path: Path,
) -> None:
    document, root, _ = _build_test_serial_provenance(tmp_path)
    snapshot_directory = root / OWNER_SESSION_SNAPSHOT_DIRECTORY
    relocated_directory = snapshot_directory.with_name("relocated-owner-sessions")
    snapshot_directory.rename(relocated_directory)
    snapshot_directory.symlink_to(relocated_directory, target_is_directory=True)

    with pytest.raises(BenchmarkContractError, match="must not contain symlinks"):
        validate_serial_provenance(document, root=root)


def test_serial_provenance_rejects_session_schema_key_deviation_after_rebind(
    tmp_path: Path,
) -> None:
    document, root, first_records = _build_test_serial_provenance(tmp_path)
    record = cast(dict[str, Any], json.loads(first_records["6A"]))
    record["unexpected"] = True
    snapshot_bytes = json.dumps(record, separators=(",", ":")).encode() + b"\n"
    _replace_snapshot_bytes(document, root, 0, snapshot_bytes)

    with pytest.raises(BenchmarkContractError, match="keys mismatch"):
        validate_serial_provenance(document, root=root)


def test_serial_provenance_rejects_metadata_schema_deviation_after_rebind(
    tmp_path: Path,
) -> None:
    document, root, _ = _build_test_serial_provenance(tmp_path)
    packages = cast(list[dict[str, Any]], document["packages"])
    owner = cast(dict[str, Any], packages[0]["owner"])
    session_meta = cast(dict[str, Any], owner["session_meta"])
    session_meta["origin"] = session_meta["source_origin"]
    _rebind_provenance_derived_fields(document)

    with pytest.raises(BenchmarkContractError, match="keys mismatch"):
        validate_serial_provenance(document, root=root)


def test_session_snapshot_metadata_participates_in_record_bundle(
    tmp_path: Path,
) -> None:
    document, root, _ = _build_test_serial_provenance(tmp_path)
    packages = cast(list[dict[str, Any]], document["packages"])
    owner = cast(dict[str, Any], packages[0]["owner"])
    session_meta = cast(dict[str, Any], owner["session_meta"])
    session_meta["source_origin"] = (
        "sessions/2099/01/01/"
        f"rollout-copy-{_TEST_SESSION_UUIDS['6A']}.jsonl"
    )

    with pytest.raises(BenchmarkContractError, match="record_bundle_sha256"):
        validate_serial_provenance(document, root=root)


def test_integrated_provenance_rejects_duplicate_session_after_derived_rebind(
    tmp_path: Path,
) -> None:
    document, root, _ = _build_test_serial_provenance(tmp_path)
    packages = cast(list[dict[str, Any]], document["packages"])
    previous_owner = cast(Mapping[str, Any], packages[-2]["owner"])
    current_owner = cast(dict[str, Any], packages[-1]["owner"])
    current_owner["session_uuid"] = previous_owner["session_uuid"]
    current_owner["session_meta"] = copy.deepcopy(previous_owner["session_meta"])
    serial = cast(dict[str, Any], document["serial_validation"])
    serial["unique_child_sessions"] = False
    _rebind_provenance_derived_fields(document)

    with pytest.raises(BenchmarkContractError):
        validate_serial_provenance(document, root=root)


def test_integrated_provenance_rejects_nonincreasing_unique_jobs_after_rebind(
    tmp_path: Path,
) -> None:
    document, root, _ = _build_test_serial_provenance(tmp_path)
    packages = cast(list[dict[str, Any]], document["packages"])
    queue = cast(dict[str, Any], packages[1]["queue"])
    queue["job_id"] = "0_adversarial_unique_job"
    records = cast(dict[str, dict[str, str]], queue["records"])
    records["run_json"]["origin"] = (
        ".training_queue/repro/0_adversarial_unique_job/run.json"
    )
    records["done_job"]["origin"] = (
        ".training_queue/done/0_adversarial_unique_job.job"
    )
    records["log"]["origin"] = ".training_queue/logs/0_adversarial_unique_job.log"
    serial = cast(dict[str, Any], document["serial_validation"])
    serial["job_ids_strictly_increasing"] = False
    _rebind_provenance_derived_fields(document)

    with pytest.raises(BenchmarkContractError, match="serial execution"):
        validate_serial_provenance(document, root=root)


def test_integrated_provenance_rejects_duplicate_queue_job_after_rebind(
    tmp_path: Path,
) -> None:
    document, root, _ = _build_test_serial_provenance(tmp_path)
    packages = cast(list[dict[str, Any]], document["packages"])
    previous_queue = cast(Mapping[str, Any], packages[0]["queue"])
    current_queue = cast(dict[str, Any], packages[1]["queue"])
    duplicate_job_id = cast(str, previous_queue["job_id"])
    current_queue["job_id"] = duplicate_job_id
    records = cast(dict[str, dict[str, str]], current_queue["records"])
    records["run_json"]["origin"] = (
        f".training_queue/repro/{duplicate_job_id}/run.json"
    )
    records["done_job"]["origin"] = (
        f".training_queue/done/{duplicate_job_id}.job"
    )
    records["log"]["origin"] = f".training_queue/logs/{duplicate_job_id}.log"
    serial = cast(dict[str, Any], document["serial_validation"])
    serial["unique_queue_jobs"] = False
    serial["job_ids_strictly_increasing"] = False
    _rebind_provenance_derived_fields(document)

    with pytest.raises(BenchmarkContractError, match="serial execution"):
        validate_serial_provenance(document, root=root)


def test_integrated_provenance_rejects_time_overlap_after_derived_rebind(
    tmp_path: Path,
) -> None:
    document, root, _ = _build_test_serial_provenance(tmp_path)
    packages = cast(list[dict[str, Any]], document["packages"])
    previous_queue = cast(dict[str, Any], packages[-2]["queue"])
    current_queue = cast(Mapping[str, Any], packages[-1]["queue"])
    current_done = datetime.fromisoformat(cast(str, current_queue["done_at"]))
    previous_queue["done_at"] = (current_done + timedelta(seconds=1)).isoformat()
    serial = cast(dict[str, Any], document["serial_validation"])
    serial["no_time_overlap"] = False
    _rebind_provenance_derived_fields(document)

    with pytest.raises(BenchmarkContractError, match="serial execution"):
        validate_serial_provenance(document, root=root)


def test_integrated_provenance_rejects_evidence_and_outcome_tampering(
    tmp_path: Path,
) -> None:
    document, root, _ = _build_test_serial_provenance(tmp_path)

    evidence_tamper = copy.deepcopy(document)
    package = cast(list[dict[str, Any]], evidence_tamper["packages"])[-1]
    cast(dict[str, Any], package["stable_evidence"])["sha256"] = "0" * 64
    with pytest.raises(BenchmarkContractError, match="stable_evidence.sha256"):
        validate_serial_provenance(evidence_tamper, root=root)

    outcome_tamper = copy.deepcopy(document)
    package = cast(list[dict[str, Any]], outcome_tamper["packages"])[-1]
    cast(dict[str, Any], package["queue"])["exit_code"] = 1
    with pytest.raises(BenchmarkContractError, match="terminal outcome"):
        validate_serial_provenance(outcome_tamper, root=root)

    queue_record_tamper = copy.deepcopy(document)
    package = cast(list[dict[str, Any]], queue_record_tamper["packages"])[-1]
    queue = cast(Mapping[str, Any], package["queue"])
    records = cast(Mapping[str, Any], queue["records"])
    cast(dict[str, Any], records["run_json"])["sha256"] = "0" * 64
    with pytest.raises(BenchmarkContractError, match="record_bundle_sha256"):
        validate_serial_provenance(queue_record_tamper, root=root)


def test_integrated_provenance_rejects_wrong_owner_after_derived_rebind(
    tmp_path: Path,
) -> None:
    document, root, _ = _build_test_serial_provenance(tmp_path)
    packages = cast(list[dict[str, Any]], document["packages"])
    current_owner = cast(dict[str, Any], packages[-1]["owner"])

    assert current_owner["canonical_agent_task"] == EXPECTED_OWNERS["6D"]
    current_owner["canonical_agent_task"] = "/root/issue753_wrong_component6d"
    _rebind_provenance_derived_fields(document)
    with pytest.raises(BenchmarkContractError, match="owner is not canonical"):
        validate_serial_provenance(document, root=root)


def test_cuda_dispatch_is_explicit_and_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = RuntimeError("explicit CUDA unavailable")

    def _unavailable() -> None:
        raise sentinel

    monkeypatch.setattr(api, "_require_cuda_executor", _unavailable)
    with pytest.raises(RuntimeError, match="explicit CUDA unavailable") as captured:
        api.resolve_compressed_time_local_attention(
            "cuda",
            compression_ratio=4,
            window_radius=4,
        )
    assert captured.value is sentinel

    with pytest.raises(ValueError, match="Unsupported"):
        api.resolve_compressed_time_local_attention(
            "auto",  # type: ignore[arg-type]
            compression_ratio=4,
            window_radius=4,
        )


def test_integrated_execute_requires_separate_stable_and_runtime_paths(
    tmp_path: Path,
) -> None:
    path = tmp_path / "integrated.json"
    with pytest.raises(BenchmarkContractError, match="must differ"):
        track_query_integrated.execute(
            evidence_path=path,
            runtime_result_path=path,
            record_evidence=False,
        )
