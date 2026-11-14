"""Only the canonical comment adapter supplies parent acceptance semantics."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from tests.support.agents_ops.traceability import lifecycle_receipt

capture: Any = importlib.import_module("decision_logs.capture")
lifecycle: Any = importlib.import_module("decision_logs.lifecycle")


def test_lookalike_result_payload_and_summary_do_not_supply_acceptance(
    tmp_path: Path,
) -> None:
    op = capture.Operation.begin(tmp_path, "o/r", 1, "parent.note", "record")
    with op.transaction():
        op.result(
            {
                "event": {
                    "parent": 1,
                    "generation": 1,
                    "event": "completion",
                    "state": "completed",
                },
                "body_sha256": "a" * 64,
            },
            artifact="arbitrary command response",
            status="success",
            summary="Parent completed and accepted all work.",
        )
    before = op.path.read_bytes()
    evidence, acceptance, previous = lifecycle.canonical_sources(
        lifecycle_receipt(), state_dir=tmp_path
    )
    assert acceptance is None and evidence == [] and previous is None
    assert op.path.read_bytes() == before
