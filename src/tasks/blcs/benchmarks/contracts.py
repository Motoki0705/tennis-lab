"""Shared machine-readable contracts for Issue #753 CUDA benchmark evidence.

Stable evidence is committed separately from fresh runtime results.  The
canonical benchmark commands always write only the latter, then validate the
stable evidence against the current source fingerprint and component-specific
semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

from src.utils.configuration import PathResolver, PathRole, RuntimePathRoots
from src.utils.paths import PROJECT_ROOT

SCHEMA_VERSION = 1
ISSUE_NUMBER = 753
DECISIONS = frozenset({"GO", "NO-GO"})
RUN_KINDS = frozenset(
    {
        "pytorch_reference",
        "cuda_prototype",
        "cuda_production",
        "architecture_baseline",
    }
)
PARITY_STATUSES = frozenset({"pass", "fail", "not-run"})


class BenchmarkContractError(ValueError):
    """Raised when benchmark evidence does not satisfy the stable contract."""


def repository_root() -> Path:
    """Return the repository root for this source-tree benchmark module."""
    return Path(PROJECT_ROOT)


def benchmark_path_resolver(root: Path | None = None) -> PathResolver:
    """Return the explicit project-root resolver for benchmark source and CLI paths."""
    project_root = repository_root() if root is None else root.resolve(strict=False)
    roots = RuntimePathRoots(
        project_root=project_root,
        data_root=project_root,
        checkpoint_root=project_root,
        artifact_root=project_root,
        output_root=project_root,
        cache_root=project_root,
        external_asset_root=project_root,
    )
    return PathResolver(roots)


def resolve_benchmark_cli_path(value: Path, *, resolver: PathResolver) -> Path:
    """Resolve one CLI path against the declared project root without using CWD."""
    if value.is_absolute():
        return Path(resolver.validate(PathRole.PROJECT, value))
    return Path(resolver.resolve(PathRole.PROJECT, value))


def utc_timestamp() -> str:
    """Return an RFC 3339 UTC timestamp suitable for evidence metadata."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def source_fingerprint(root: Path, relative_paths: Sequence[str]) -> str:
    """Hash path names and contents in a deterministic order."""
    digest = hashlib.sha256()
    resolver = benchmark_path_resolver(root)
    for relative_path in sorted(relative_paths):
        path = resolver.resolve(PathRole.PROJECT, relative_path)
        if not path.is_file():
            raise BenchmarkContractError(
                f"benchmark source file does not exist: {relative_path}"
            )
        encoded_path = relative_path.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def build_source_record(root: Path, relative_paths: Sequence[str]) -> dict[str, Any]:
    """Capture the component-scoped source identity used by one benchmark."""
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    files = sorted(relative_paths)
    return {
        "git_commit": git_commit,
        "files": files,
        "fingerprint_sha256": source_fingerprint(root, files),
    }


def build_cuda_environment() -> dict[str, Any]:
    """Capture the CUDA runtime environment after availability is established."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested, but torch CUDA is unavailable")
    device = torch.device("cuda")
    capability = torch.cuda.get_device_capability(device)
    return {
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": f"{capability[0]}.{capability[1]}",
    }


def load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object with a contract-oriented error."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BenchmarkContractError(f"evidence file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise BenchmarkContractError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(raw, dict):
        raise BenchmarkContractError(f"JSON root must be an object: {path}")
    return cast(dict[str, Any], raw)


def write_json_atomic(path: Path, document: Mapping[str, Any]) -> None:
    """Atomically write one formatted JSON result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def validate_common_evidence(
    document: Mapping[str, Any],
    *,
    component: str,
    source_files: Sequence[str],
    protocol: Mapping[str, Any],
    root: Path,
) -> None:
    """Validate the common schema and current component-scoped source identity."""
    required_keys = {
        "schema_version",
        "issue",
        "component",
        "generated_at_utc",
        "source",
        "environment",
        "protocol",
        "runs",
        "decision",
        "risks",
    }
    _require_exact_keys(document, required_keys, "evidence")
    _require_equal(document["schema_version"], SCHEMA_VERSION, "schema_version")
    _require_equal(document["issue"], ISSUE_NUMBER, "issue")
    _require_equal(document["component"], component, "component")
    _require_nonempty_string(document["generated_at_utc"], "generated_at_utc")

    source = _require_mapping(document["source"], "source")
    _require_exact_keys(
        source,
        {"git_commit", "files", "fingerprint_sha256"},
        "source",
    )
    _require_hex_digest(source["git_commit"], "source.git_commit", length=40)
    _require_hex_digest(
        source["fingerprint_sha256"],
        "source.fingerprint_sha256",
        length=64,
    )
    expected_files = sorted(source_files)
    _require_equal(source["files"], expected_files, "source.files")
    expected_fingerprint = source_fingerprint(root, expected_files)
    _require_equal(
        source["fingerprint_sha256"],
        expected_fingerprint,
        "source.fingerprint_sha256",
    )

    environment = _require_mapping(document["environment"], "environment")
    _require_exact_keys(
        environment,
        {"python", "torch", "cuda", "gpu", "compute_capability"},
        "environment",
    )
    for key in environment:
        _require_nonempty_string(environment[key], f"environment.{key}")

    _require_equal(document["protocol"], dict(protocol), "protocol")
    runs = document["runs"]
    if not isinstance(runs, list) or not runs:
        raise BenchmarkContractError("runs must be a non-empty array")
    for index, run in enumerate(runs):
        validate_run_record(_require_mapping(run, f"runs[{index}]"), index=index)

    decision = _require_mapping(document["decision"], "decision")
    if decision["status"] not in DECISIONS:
        raise BenchmarkContractError("decision.status must be GO or NO-GO")
    _require_nonempty_string(decision["reason"], "decision.reason")

    risks = document["risks"]
    if not isinstance(risks, list) or any(
        not isinstance(risk, str) or not risk for risk in risks
    ):
        raise BenchmarkContractError("risks must be an array of non-empty strings")


def validate_run_record(run: Mapping[str, Any], *, index: int) -> None:
    """Validate a reusable per-candidate benchmark run record."""
    path = f"runs[{index}]"
    _require_exact_keys(
        run,
        {
            "case",
            "candidate",
            "kind",
            "implementation",
            "available",
            "shape",
            "dtype",
            "warmup",
            "iterations",
            "latency",
            "throughput",
            "memory",
            "parity",
            "unavailable_reason",
        },
        path,
    )
    for key in ("case", "candidate", "implementation", "dtype"):
        _require_nonempty_string(run[key], f"{path}.{key}")
    if run["kind"] not in RUN_KINDS:
        raise BenchmarkContractError(f"{path}.kind is unsupported: {run['kind']!r}")
    if not isinstance(run["available"], bool):
        raise BenchmarkContractError(f"{path}.available must be boolean")
    shape = _require_mapping(run["shape"], f"{path}.shape")
    if not shape:
        raise BenchmarkContractError(f"{path}.shape must not be empty")
    for key, value in shape.items():
        if not isinstance(key, str) or not key:
            raise BenchmarkContractError(f"{path}.shape keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise BenchmarkContractError(
                f"{path}.shape.{key} must be a non-negative number"
            )
    _require_positive_int(run["warmup"], f"{path}.warmup")
    _require_positive_int(run["iterations"], f"{path}.iterations")

    parity = _require_mapping(run["parity"], f"{path}.parity")
    _require_exact_keys(
        parity,
        {
            "status",
            "forward_max_abs_error",
            "forward_mean_abs_error",
            "backward_max_abs_error",
            "backward_mean_abs_error",
            "atol",
            "rtol",
        },
        f"{path}.parity",
    )
    if parity["status"] not in PARITY_STATUSES:
        raise BenchmarkContractError(f"{path}.parity.status is invalid")

    if run["available"]:
        _require_equal(run["unavailable_reason"], None, f"{path}.unavailable_reason")
        latency = _require_mapping(run["latency"], f"{path}.latency")
        _require_exact_keys(latency, {"median_ms", "p95_ms"}, f"{path}.latency")
        _require_finite_positive(latency["median_ms"], f"{path}.latency.median_ms")
        _require_finite_positive(latency["p95_ms"], f"{path}.latency.p95_ms")
        throughput = _require_mapping(run["throughput"], f"{path}.throughput")
        _require_exact_keys(throughput, {"unit", "value"}, f"{path}.throughput")
        _require_nonempty_string(throughput["unit"], f"{path}.throughput.unit")
        _require_finite_positive(throughput["value"], f"{path}.throughput.value")
        memory = _require_mapping(run["memory"], f"{path}.memory")
        _require_exact_keys(
            memory,
            {"peak_allocated_bytes", "peak_reserved_bytes"},
            f"{path}.memory",
        )
        for key in memory:
            _require_non_negative_int(memory[key], f"{path}.memory.{key}")
        if parity["status"] == "not-run":
            raise BenchmarkContractError(
                f"{path}.parity must be measured for an available candidate"
            )
        for key in (
            "forward_max_abs_error",
            "forward_mean_abs_error",
            "backward_max_abs_error",
            "backward_mean_abs_error",
            "atol",
            "rtol",
        ):
            _require_finite_non_negative(parity[key], f"{path}.parity.{key}")
    else:
        _require_nonempty_string(
            run["unavailable_reason"], f"{path}.unavailable_reason"
        )
        for key in ("latency", "throughput", "memory"):
            _require_equal(run[key], None, f"{path}.{key}")
        _require_equal(parity["status"], "not-run", f"{path}.parity.status")
        for key in parity:
            if key != "status":
                _require_equal(parity[key], None, f"{path}.parity.{key}")


def parity_not_run() -> dict[str, Any]:
    """Return the canonical unavailable-candidate parity payload."""
    return {
        "status": "not-run",
        "forward_max_abs_error": None,
        "forward_mean_abs_error": None,
        "backward_max_abs_error": None,
        "backward_mean_abs_error": None,
        "atol": None,
        "rtol": None,
    }


def unavailable_run(
    *,
    case: str,
    candidate: str,
    kind: str,
    implementation: str,
    shape: Mapping[str, int | float],
    dtype: str,
    warmup: int,
    iterations: int,
    reason: str,
) -> dict[str, Any]:
    """Build a schema-valid unavailable candidate record."""
    return {
        "case": case,
        "candidate": candidate,
        "kind": kind,
        "implementation": implementation,
        "available": False,
        "shape": dict(shape),
        "dtype": dtype,
        "warmup": warmup,
        "iterations": iterations,
        "latency": None,
        "throughput": None,
        "memory": None,
        "parity": parity_not_run(),
        "unavailable_reason": reason,
    }


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkContractError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise BenchmarkContractError(f"{path} keys must be strings")
    return cast(Mapping[str, Any], value)


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: set[str], path: str
) -> None:
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise BenchmarkContractError(
            f"{path} keys mismatch; missing={missing}, unknown={unknown}"
        )


def _require_equal(actual: object, expected: object, path: str) -> None:
    if actual != expected:
        raise BenchmarkContractError(
            f"{path} mismatch: expected {expected!r}, got {actual!r}"
        )


def _require_nonempty_string(value: object, path: str) -> None:
    if not isinstance(value, str) or not value:
        raise BenchmarkContractError(f"{path} must be a non-empty string")


def _require_hex_digest(value: object, path: str, *, length: int) -> None:
    _require_nonempty_string(value, path)
    assert isinstance(value, str)
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise BenchmarkContractError(
            f"{path} must be {length} lowercase hex characters"
        )


def _require_positive_int(value: object, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BenchmarkContractError(f"{path} must be a positive integer")


def _require_non_negative_int(value: object, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkContractError(f"{path} must be a non-negative integer")


def _require_finite_positive(value: object, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchmarkContractError(f"{path} must be a finite positive number")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise BenchmarkContractError(f"{path} must be a finite positive number")


def _require_finite_non_negative(value: object, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchmarkContractError(f"{path} must be a finite non-negative number")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise BenchmarkContractError(f"{path} must be a finite non-negative number")
