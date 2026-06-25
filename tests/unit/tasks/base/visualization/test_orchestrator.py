"""Unit tests for visualization orchestration parsing helpers."""

from __future__ import annotations

import pytest

from src.tasks.base.visualization.orchestrator import parse_cameras, resolve_device

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_parse_cameras_empty_returns_none(raw) -> None:
    assert parse_cameras(raw) is None


def test_parse_cameras_all_keyword() -> None:
    assert parse_cameras("all") == "all"
    assert parse_cameras("  all  ") == "all"


def test_parse_cameras_comma_string() -> None:
    assert parse_cameras("0,1,2") == [0, 1, 2]
    assert parse_cameras(" 3 , 4 ") == [3, 4]


def test_parse_cameras_iterable() -> None:
    assert parse_cameras([0, 2, 5]) == [0, 2, 5]
    assert parse_cameras((1, 3)) == [1, 3]


def test_parse_cameras_invalid_string_raises() -> None:
    with pytest.raises(ValueError):
        parse_cameras("a,b")


def test_resolve_device_explicit_passthrough() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_auto_returns_valid_device() -> None:
    import torch

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device("auto") == expected
