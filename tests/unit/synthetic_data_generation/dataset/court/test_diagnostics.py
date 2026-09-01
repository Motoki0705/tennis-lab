"""Tests for transactional Court diagnostic directory publication."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import src.synthetic_data_generation.dataset.court.diagnostics as diagnostics_module
from src.synthetic_data_generation.dataset.court.contracts import CourtDatasetPlanAny
from src.synthetic_data_generation.dataset.court.diagnostics import (
    write_court_diagnostics,
)


def test_diagnostic_failure_exposes_no_partial_inventory_or_unrelated_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_bytes(b"belongs to another invocation")
    diagnostics_root = tmp_path / "diagnostics"

    def fail_at_occupancy_publication(
        staged_root: Path,
        **kwargs: object,
    ) -> tuple[str, ...]:
        del kwargs
        staged_root.mkdir(parents=True)
        (staged_root / "trajectory-plan.json").write_text(
            "partial",
            encoding="utf-8",
        )
        raise RuntimeError("injected occupancy publication failure")

    monkeypatch.setattr(
        diagnostics_module,
        "_write_court_diagnostics_contents",
        fail_at_occupancy_publication,
    )

    with pytest.raises(RuntimeError, match="injected occupancy"):
        write_court_diagnostics(
            diagnostics_root,
            plan=cast(CourtDatasetPlanAny, SimpleNamespace()),
            accepted_sample_ids=(),
            rejected=(),
            coverage_counts={},
            visible_by_class={},
        )

    assert unrelated.read_bytes() == b"belongs to another invocation"
    assert not diagnostics_root.exists()
    assert not tuple(tmp_path.glob(".diagnostics.*.staging"))


def test_diagnostic_publication_never_replaces_preexisting_directory(
    tmp_path: Path,
) -> None:
    diagnostics_root = tmp_path / "diagnostics"
    diagnostics_root.mkdir()
    unrelated = diagnostics_root / "unrelated.txt"
    unrelated.write_bytes(b"preexisting")

    with pytest.raises(FileExistsError, match="already exist"):
        write_court_diagnostics(
            diagnostics_root,
            plan=cast(CourtDatasetPlanAny, SimpleNamespace()),
            accepted_sample_ids=(),
            rejected=(),
            coverage_counts={},
            visible_by_class={},
        )

    assert unrelated.read_bytes() == b"preexisting"
