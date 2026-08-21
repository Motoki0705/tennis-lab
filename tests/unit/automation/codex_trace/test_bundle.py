from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.automation.codex_trace.bundle import TraceBundle, TraceBundleError
from tests.support.codex_trace import write_sample_trace_bundle


def test_bundle_rejects_unknown_schema_version(tmp_path: Path) -> None:
    bundle_path = write_sample_trace_bundle(tmp_path)
    state_path = bundle_path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["schema_version"] = 99
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(TraceBundleError, match="unsupported rollout-trace"):
        TraceBundle.load(bundle_path, auto_reduce=False)


def test_bundle_rejects_payload_path_traversal(tmp_path: Path) -> None:
    bundle_path = write_sample_trace_bundle(tmp_path)
    state_path = bundle_path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["raw_payloads"]["request-1"]["path"] = "../outside.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (bundle_path.parent / "outside.json").write_text("{}", encoding="utf-8")
    bundle = TraceBundle.load(bundle_path, auto_reduce=False)

    with pytest.raises(TraceBundleError, match="escapes the trace bundle"):
        bundle.payload_json("request-1")


def test_bundle_requires_explicit_reduction_when_disabled(tmp_path: Path) -> None:
    bundle_path = tmp_path / "unreduced"
    bundle_path.mkdir()

    with pytest.raises(TraceBundleError, match="trace-reduce"):
        TraceBundle.load(bundle_path, auto_reduce=False)
