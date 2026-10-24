"""Evidence sinks decouple isolated test execution from an audit ledger."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .retrieval import Journal, RetrievalError, strict_json


class EvidenceRecorder(Protocol):
    def validate_finding(self, finding_id: str) -> None: ...
    def record(self, result: dict[str, Any], reference: str) -> None: ...
    def verify_reference(self, run_id: str, reference: str) -> None: ...


class JournalRecorder:
    def __init__(self, directory: Path, run: str, sha: str) -> None:
        self.journal = Journal(
            directory, {"run": run, "sha": sha, "kind": "cpu-evidence"}
        )

    def validate_finding(self, finding_id: str) -> None:
        if not finding_id.strip():
            raise ValueError("finding_id must describe the question being tested")

    def record(self, result: dict[str, Any], reference: str) -> None:
        self.journal.append(
            {"event": "cpu-test", "run_id": result["run_id"], "reference": reference}
        )

    def verify_reference(self, run_id: str, reference: str) -> None:
        matches = [
            strict_json(line)
            for line in self.journal.path.read_text().splitlines()
            if strict_json(line).get("run_id") == run_id
        ]
        if len(matches) != 1 or matches[0]["reference"] != reference:
            raise RetrievalError(
                "RECORD_MISMATCH", "manifest differs from its recorded evidence"
            )
