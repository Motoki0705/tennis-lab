"""CPU geometry tests; no model inference or manual annotation access."""

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "court_crop_probe", Path(__file__).with_name("probe.py")
    )
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_union_uses_projected_points_and_clips_image() -> None:
    probe = module()
    points = np.tile([50.0, 70.0], (14, 1))
    points[0] = [-10.0, 180.0]
    assert probe.union_roi((40, 40, 80, 80), points, (100, 100)) == (0, 20, 100, 100)


@pytest.mark.parametrize(
    "roi", [(0, 0, 0, 10), (-1, 0, 10, 10), (0, 0, 101, 10), (10, 20, 5, 10)]
)
def test_invalid_roi_rejected(roi: tuple[int, int, int, int]) -> None:
    with pytest.raises(ValueError):
        module().validate_roi(roi, (100, 100))


def test_nonfinite_projection_rejected() -> None:
    with pytest.raises(ValueError):
        module().union_roi((1, 1, 10, 10), np.full((14, 2), np.nan), (100, 100))


def test_repeatability_is_exact_without_tolerance() -> None:
    probe = module()
    a = np.ones((14, 2))
    b = a.copy()
    b[0, 0] = np.nextafter(1.0, 2.0)
    assert probe.difference(a, a)["exact"]
    assert not probe.difference(a, b)["exact"]
