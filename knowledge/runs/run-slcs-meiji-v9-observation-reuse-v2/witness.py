"""One instrumented CPU execution of the unchanged v1 observation reuse driver.

Only its dual_sha256 binding is replaced. ViTPose's three scheduled reads are
captured from the same stream; other inputs retain the original dual hash.
A failed gate is sticky: exception-handler reads cannot reopen the live weight.
A post-publication anomaly can leave already published target files; no rollback,
receipt repair, overwrite, or automatic retry is performed. Stream agreement is
an observation, not a resolution of the underlying environment anomaly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parents[3]
MAIN = Path("/home/kamimura/projects/tennis-lab")
CHECKPOINT = (
    MAIN / "third_party/GVHMR/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
)
OLD = REPO / "knowledge/runs/run-slcs-meiji-v9-observation-reuse-v1/reuse.py"
HELPER = REPO / "knowledge/runs/run-slcs-meiji-read-stream-capture-v1/capture.py"
PIN = "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc"
PHASES = ("initial_pin", "prepublication", "postpublication")


def load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Witness:
    def __init__(
        self,
        *,
        original: Callable[[Path], str],
        helper: ModuleType,
        checkpoint: Path,
        evidence: Path,
        expected: str,
        metadata: dict[str, Any],
    ) -> None:
        self.original = original
        self.helper = helper
        self.checkpoint = checkpoint.resolve()
        self.evidence = evidence
        self.expected = expected
        self.failed = False
        self.report: dict[str, Any] = {
            **metadata,
            "status": "running",
            "checkpoint": str(self.checkpoint),
            "expected_sha256": expected,
            "calls": 0,
            "captured_calls": 0,
            "skipped_after_failure": 0,
            "captures": [],
            "comparisons": [],
            "failures": [],
            "legacy_run_completed": False,
            "limitations": __doc__,
        }
        self.save()

    def save(self) -> None:
        atomic_json(self.evidence / "ledger.json", self.report)

    def reject(self, reason: str) -> None:
        self.failed = True
        self.report["status"] = "failed"
        self.report["failures"].append(reason)
        self.save()

    def __call__(self, path: Path) -> str:
        if Path(path).resolve() != self.checkpoint:
            try:
                return self.original(path)
            except Exception as error:
                self.reject(f"Original dual hash failed for {path}: {error!r}")
                raise ValueError(self.report["failures"][-1]) from error
        self.report["calls"] += 1
        if self.failed:
            self.report["skipped_after_failure"] += 1
            self.save()
            raise ValueError(
                "ViTPose read refused after earlier failure; live file not reopened"
            )
        if self.report["captured_calls"] >= 3:
            self.reject(
                "Fourth ViTPose read refused; maximum is three, live file not reopened"
            )
            raise ValueError(self.report["failures"][-1])
        index = self.report["captured_calls"]
        self.report["captured_calls"] += 1
        row: dict[str, Any] = {"phase": PHASES[index], "status": "starting"}
        self.report["captures"].append(row)
        self.save()
        try:
            capture = self.helper.read_stream(
                self.checkpoint,
                self.evidence / f"{index + 1:03d}-{PHASES[index]}.bin",
                self.expected,
            )
            row.update(capture)
            self.save()  # Persist the live-read evidence before any snapshot-only reads.
            if Path(row["snapshot"]).exists():
                self.helper.verify_snapshot(row)
            self.save()
            required = (
                "providers_agree",
                "expected_pin_matches",
                "stat_unchanged",
                "byte_count_matches",
            )
            saved = (
                "saved_providers_match_capture",
                "saved_chunks_match_capture",
                "saved_length_matches_capture",
            )
            valid = (
                row["status"] == "passed"
                and not row["errors"]
                and all(row.get("checks", {}).get(key) is True for key in required)
                and all(
                    row.get("snapshot_checks", {}).get(key) is True for key in saved
                )
                and row["hashlib_sha256"] == row["cpython_sha256"] == self.expected
            )
            for previous in self.report["captures"][:-1]:
                comparison = self.helper.compare_snapshots(previous, row)
                self.report["comparisons"].append(comparison)
                valid = (
                    valid
                    and comparison["equal"] is True
                    and comparison["different_bytes"] == 0
                    and not comparison["different_chunk_indices"]
                )
                self.save()
            if not valid:
                raise ValueError(f"{PHASES[index]} capture or snapshot gate failed")
            row["witness_gates_passed"] = True
            self.save()
            return self.expected
        except BaseException as error:
            row["witness_gates_passed"] = False
            self.reject(f"{PHASES[index]}: {error!r}")
            raise ValueError(self.report["failures"][-1]) from error

    def finish(self) -> None:
        self.report["legacy_run_completed"] = True
        if (
            self.failed
            or self.report["captured_calls"] != 3
            or not all(
                row.get("witness_gates_passed") is True
                for row in self.report["captures"]
            )
        ):
            self.reject(
                "Legacy run returned without exactly three successful witnessed reads"
            )
            raise ValueError(self.report["failures"][-1])
        self.report["status"] = "passed"
        self.save()


def run(*, source: Path, target: Path, output: Path, evidence: Path) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CUDA_VISIBLE_DEVICES must be explicitly empty")
    if output.exists() or evidence.exists():
        raise ValueError("Main output and evidence directory must both be new")
    if not output.is_relative_to(MAIN / "outputs") or not evidence.is_relative_to(
        MAIN / "outputs"
    ):
        raise ValueError("Main output and evidence must be under main outputs")
    paths = (source, target, output, evidence)
    if any(
        a == b or a.is_relative_to(b) or b.is_relative_to(a)
        for i, a in enumerate(paths)
        for b in paths[i + 1 :]
    ):
        raise ValueError(
            "Source, target, output and evidence must be separate directories"
        )
    evidence.mkdir(parents=True, exist_ok=False)  # The old driver creates main output.
    metadata: dict[str, Any] = {
        "source": str(source),
        "target": str(target),
        "output": str(output),
        "evidence": str(evidence),
        "command": sys.argv,
        "python": sys.executable,
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "scripts_sha256": {},
        "status": "initializing",
        "calls": 0,
        "captured_calls": 0,
        "skipped_after_failure": 0,
        "limitations": __doc__,
    }
    atomic_json(evidence / "ledger.json", metadata)
    try:
        old, helper = (
            load(OLD, "legacy_observation_reuse"),
            load(HELPER, "read_stream_capture"),
        )
        original = old.dual_sha256
        metadata["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        for path in (Path(__file__), OLD, HELPER):
            metadata["scripts_sha256"][str(path)] = original(path)
            atomic_json(evidence / "ledger.json", metadata)
        witness = Witness(
            original=original,
            helper=helper,
            checkpoint=CHECKPOINT,
            evidence=evidence,
            expected=PIN,
            metadata=metadata,
        )
    except BaseException as error:
        metadata.update(status="failed", failures=[f"Initialization failed: {error!r}"])
        atomic_json(evidence / "ledger.json", metadata)
        raise
    old.__dict__["dual_sha256"] = witness
    try:
        old.run(source, target, output)
        witness.finish()
    except BaseException as error:
        witness.reject(f"Legacy execution did not complete successfully: {error!r}")
        raise
    finally:
        old.__dict__["dual_sha256"] = original
        witness.save()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "target", "output", "evidence"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    run(
        **{
            name: getattr(args, name).resolve()
            for name in ("source", "target", "output", "evidence")
        }
    )


if __name__ == "__main__":
    main()
