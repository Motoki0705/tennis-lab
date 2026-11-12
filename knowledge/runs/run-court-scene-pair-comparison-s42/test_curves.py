"""Accepted fit history excludes the unaccepted tail of a lost VM."""

# mypy: disallow-untyped-decorators=False
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "accepted_fit_curves", Path(__file__).with_name("export_curves.py")
)
assert SPEC is not None and SPEC.loader is not None
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def test_resume_uses_saved_boundary_and_excludes_old_tail() -> None:
    accepted, discarded = c.select_points(
        [[50, 9.0], [100, 7.0], [150, 99.0], [200, 88.0]], [[150, 3.0], [200, 2.0]], 100
    )
    assert accepted == [[50, 9.0], [100, 7.0], [150, 3.0], [200, 2.0]]
    assert discarded == [[150, 99.0], [200, 88.0]]


@pytest.mark.parametrize(
    "fault", ["resumed-before-boundary", "conflicting-step", "nonfinite"]
)
def test_bad_fit_history_is_rejected(fault: str) -> None:
    before, after = [[50, 9.0], [100, 7.0]], [[150, 3.0]]
    if fault == "resumed-before-boundary":
        after = [[99, 5.0]]
    elif fault == "conflicting-step":
        after = [[150, 3.0], [150, 4.0]]
    elif fault == "nonfinite":
        after = [[150, float("nan")]]
    with pytest.raises(ValueError):
        c.select_points(before, after, 100)
