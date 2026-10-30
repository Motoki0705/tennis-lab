"""Audit ledger adapter for the shared isolated CPU-test executor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lib.review_tools.reproduction import *  # noqa: F403
from lib.review_tools.reproduction import Reproduction as Executor
from lib.review_tools.retrieval import RetrievalError

from .ledger import Ledger


class LedgerRecorder:
    def __init__(self, path: Path, slot: str, sha: str) -> None:
        self.path, self.slot, self.sha = path.resolve(strict=True), slot, sha
        self.book()

    def book(self) -> Ledger:
        return Ledger.load(self.path, slot=self.slot, base_sha=self.sha)

    def validate_finding(self, finding_id: str) -> None:
        if finding_id not in self.book().snapshot["findings"]:
            raise RetrievalError(
                "UNKNOWN_FINDING", "finding_id must already exist in the pinned ledger"
            )

    def record(self, result: dict[str, Any], reference: str) -> None:
        book = self.book()
        evidence_id = "cpu-test-" + result["run_id"]
        book.record_evidence(
            {
                "id": evidence_id,
                "source_kind": "isolated-cpu-test",
                "reference": reference,
                "acquired_at": result["finished_at"],
                "target_version": self.sha,
                "pr_head": None,
            },
            at=result["finished_at"],
        )
        finding_id = result.get("finding_id")
        if finding_id in book.snapshot["findings"]:
            finding = book.snapshot["findings"][finding_id]
            finding["evidence"].append(evidence_id)
            book.record_finding(finding, at=result["finished_at"])

    def verify_reference(self, run_id: str, reference: str) -> None:
        evidence = self.book().snapshot["evidence"].get("cpu-test-" + run_id)
        if evidence is None or evidence["reference"] != reference:
            raise RetrievalError(
                "RECORD_MISMATCH", "manifest differs from immutable ledger evidence"
            )


class Reproduction(Executor):  # type: ignore[no-redef]
    def __init__(
        self,
        repo: Path,
        base_sha: str,
        slot: str,
        state_dir: Path,
        ledger_path: Path,
        cpu_nodeids: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(
            repo,
            base_sha,
            slot,
            state_dir,
            LedgerRecorder(ledger_path, slot, base_sha),
            cpu_nodeids,
            **kwargs,
        )
