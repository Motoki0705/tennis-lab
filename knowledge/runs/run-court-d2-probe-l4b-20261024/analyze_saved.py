"""Bind the existing QA analysis to this run's exact pushed source commit."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

PREVIOUS = Path(__file__).resolve().parents[1] / "run-court-d2-probe-l4-20261024"
sys.path.insert(0, str(PREVIOUS))

import analyze_probe as analysis  # noqa: E402


def verify_bundle(root: Path, source_commit: str) -> dict[str, Any]:
    manifest = analysis.read(root / "review-manifest.json")
    assert manifest["schema"] == "court_d2_review_bundle_v1"
    expected = {entry["path"] for entry in manifest["files"]}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert actual == expected | {"review-manifest.json"}
    for entry in manifest["files"]:
        path = root / entry["path"]
        assert path.stat().st_size == entry["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    state = analysis.read(root / "probe/state.json")
    assert state["source_commit"] == source_commit
    assert state["outcome"] == "completed"
    assert state["draw_attempts"] == state["completed_frames"] == 192
    ledger = analysis.read(root / "probe/nht-submissions.json")
    assert ledger["submitted_cameras"] == 192
    assert len(ledger["invocations"]) == 16
    assert all(
        row["outcome"] == "completed"
        and row["returncode"] == 0
        and row["camera_count"] == 12
        for row in ledger["invocations"]
    )
    selection = analysis.read(root / "probe/selection.json")
    assert isinstance(selection, dict)
    assert selection == analysis.read(analysis.FROZEN / "render-selection.json")
    for invocation, scenario in zip(
        ledger["invocations"], selection["scenarios"], strict=True
    ):
        assert invocation["camera_ids"] == [
            sample["sample_id"] for sample in scenario["samples"]
        ]
    assert (
        analysis.digest({k: v for k, v in selection.items() if k != "selection_sha256"})
        == selection["selection_sha256"]
    )
    return selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--source-commit", required=True)
    args, remaining = parser.parse_known_args()

    def verify(root: Path) -> dict[str, Any]:
        return verify_bundle(root, args.source_commit)

    with (
        patch.object(analysis, "verify_bundle", side_effect=verify),
        patch.object(sys, "argv", [sys.argv[0], *remaining]),
    ):
        analysis.main()


if __name__ == "__main__":
    main()
