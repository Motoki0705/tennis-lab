"""The review CLI validates read-only roots before starting its server."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from src.tasks.player_detection.scripts import review_dataset
from src.utils.configuration import PathContractError
from src.utils.configuration.discovery import discover_runtime_boundaries
from src.utils.configuration.inventory import EXPECTED_RUNTIME_BOUNDARIES, BoundaryKind
from src.utils.paths import PROJECT_ROOT


def test_review_cli_has_one_discoverable_validated_boundary() -> None:
    module = "src.tasks.player_detection.scripts.review_dataset"
    boundary = next(item for item in EXPECTED_RUNTIME_BOUNDARIES if item.module == module)
    assert boundary.kind == BoundaryKind.ARGPARSE
    assert boundary.executable_module
    assert boundary.validator_key == review_dataset.PATH_BOUNDARY.name
    discovered = discover_runtime_boundaries(PROJECT_ROOT / "src")
    assert any(item.module == module and item.callable_name == "main" for item in discovered)
    assert {field.name for field in review_dataset.PATH_BOUNDARY.fields} == {"project_root", "data_root"}


def test_review_cli_invalid_data_root_fails_before_serving(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Server must not start before input validation")

    monkeypatch.setattr("uvicorn.run", forbidden)
    monkeypatch.setattr(sys, "argv", ["review_dataset", "--project-root", str(tmp_path)])
    with pytest.raises(PathContractError):
        review_dataset.main()


def test_review_cli_accepts_separate_data_root_and_loopback_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.tasks.player_detection.review.service import PlayerReviewService

    data = tmp_path / "external-data"
    data.mkdir()
    observed: list[tuple[str, int]] = []
    roots: list[Path] = []
    original = PlayerReviewService.__init__

    def capture_root(self: PlayerReviewService, data_root: Path) -> None:
        roots.append(data_root)
        original(self, data_root)

    def capture_server(app: Any, *, host: str, port: int) -> None:
        observed.append((host, port))

    monkeypatch.setattr(PlayerReviewService, "__init__", capture_root)
    monkeypatch.setattr("uvicorn.run", capture_server)
    monkeypatch.setattr(sys, "argv", ["review_dataset", "--project-root", str(tmp_path), "--data-root", str(data), "--port", "8895"])
    review_dataset.main()
    assert roots == [data.resolve()]
    assert observed == [("127.0.0.1", 8895)]
