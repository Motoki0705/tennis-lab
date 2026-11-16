"""The public request cannot accept an empty PR target list."""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops"))
parser = importlib.import_module("parent.cli").parser


def test_empty_review_request_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parser().parse_args(
            [
                "--parent",
                "1",
                "review-request",
                "--prs",
                "--evidence-file",
                "e",
                "--instruction-file",
                "i",
            ]
        )
