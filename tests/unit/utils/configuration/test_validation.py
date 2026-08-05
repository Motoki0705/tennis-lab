"""Executable regression tests for the strict foundation validation matrix."""

import pytest

from src.utils.configuration.validation import main


def test_foundation_validation_main_rejects_every_negative_case(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main() == 0
    captured = capsys.readouterr()
    assert captured.out == (
        "Strict negative validation passed: missing-key, unknown-key, "
        "wrong-exact-type, mutually-exclusive, path-escape\n"
    )
