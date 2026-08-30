"""End-to-end contract checks for knowledge-control convergence curves."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = ROOT / ".agents/skills/knowledge-control/scripts"


def _load_curves_module() -> ModuleType:
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "kg_curves_under_test", SCRIPTS_DIR / "kg_curves.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS_DIR))


class _FakeAccumulator:
    def __init__(self, scalar_values: dict[str, list[float]]) -> None:
        self.scalar_values = scalar_values

    def Tags(self) -> dict[str, list[str]]:
        return {"scalars": list(self.scalar_values)}

    def Scalars(self, tag: str) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(step=step, value=value)
            for step, value in enumerate(self.scalar_values[tag])
        ]


def test_select_bases_includes_pose_curves_and_prefers_current_aliases() -> None:
    curves = _load_curves_module()
    tags = {
        f"{stage}/{metric}": [1.0, 0.5]
        for stage in ("train", "val")
        for metric in (
            "loss",
            "position_error_m",
            "pos_error_m",
            "angular_error_deg",
            "ang_error_deg",
            "canonical_mpjpe_m",
            "canonical_mpjpe",
            "canonical_pck_0.1m",
        )
    }

    assert curves.select_bases(_FakeAccumulator(tags)) == [
        "loss",
        "position_error_m",
        "angular_error_deg",
        "canonical_mpjpe_m",
        "canonical_pck_0.1m",
    ]


@pytest.mark.parametrize(
    ("current", "legacy"),
    [
        ("position_error_m", "pos_error_m"),
        ("angular_error_deg", "ang_error_deg"),
        ("canonical_mpjpe_m", "canonical_mpjpe"),
    ],
)
def test_select_bases_prefers_current_without_falling_back_to_legacy_alias(
    current: str,
    legacy: str,
) -> None:
    curves = _load_curves_module()
    accumulator = _FakeAccumulator(
        {
            f"train/{current}": [0.4, 0.3],
            f"val/{current}": [0.5, 0.35],
            f"train/{legacy}": [0.4, 0.3],
            f"val/{legacy}": [0.5, 0.35],
        }
    )

    assert curves.select_bases(accumulator) == [current]


@pytest.mark.parametrize(
    "legacy",
    ["pos_error_m", "ang_error_deg", "canonical_mpjpe"],
)
def test_select_bases_reads_legacy_curve_when_current_is_absent(legacy: str) -> None:
    curves = _load_curves_module()
    accumulator = _FakeAccumulator(
        {
            f"train/{legacy}": [0.4, 0.3],
            f"val/{legacy}": [0.5, 0.35],
        }
    )

    assert curves.select_bases(accumulator) == [legacy]


def test_final_test_metrics_normalizes_legacy_pose_key_and_prefers_current(
    monkeypatch,
    tmp_path: Path,
) -> None:
    curves = _load_curves_module()
    accumulator = _FakeAccumulator(
        {
            "test/canonical_mpjpe": [0.4],
            "test/canonical_mpjpe_m": [0.3],
            "test/canonical_pck_0.1m": [0.8],
        }
    )
    monkeypatch.setattr(curves, "_accumulator", lambda _: accumulator)

    assert curves.final_test_metrics(tmp_path / "events") == {
        "canonical_mpjpe_m": 0.3,
        "canonical_pck_0.1m": 0.8,
    }


def test_fingerprint_matching_accepts_legacy_pose_key(monkeypatch, tmp_path: Path) -> None:
    curves = _load_curves_module()
    event_dir = tmp_path / "events"
    monkeypatch.setattr(
        curves,
        "final_test_metrics",
        lambda _: {
            "loss": 0.2,
            "position_error_m": 0.3,
            "canonical_mpjpe": 0.4,
        },
    )

    matched, _, matched_keys = curves.match_event_dir(
        {"loss": 0.2, "position_error_m": 0.3, "canonical_mpjpe_m": 0.4},
        [event_dir],
    )

    assert matched == event_dir
    assert matched_keys == 3
