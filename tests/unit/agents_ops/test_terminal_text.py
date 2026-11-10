"""Terminal text has explicit, counted transformations, including malformed input."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / ".agents/ops"))
terminal: Any = importlib.import_module("lib.review_tools.terminal_text")


@pytest.mark.parametrize(
    "raw,expected,sequences,controls",
    [
        ("テスト\tOK\n\\u001b[31m", "テスト\tOK\n\\u001b[31m", 0, 0),
        ("\x1b[31mred\x1b[0m", "red", 2, 0),
        ("\x1b[?25lhidden\x1b[1 q", "hidden", 2, 0),
        ("\x9b32mgreen\x9b0m", "green", 2, 0),
        ("\x1b]0;title\aOK", "OK", 1, 0),
        ("\x1b]8;;https://example.invalid\x1b\\link\x1b]8;;\x1b\\", "link", 2, 0),
        ("\x9d0;title\x9cOK", "OK", 1, 0),
        ("\x1b]0;title\x9cOK", "OK", 1, 0),
        ("\x1bPpayload\x1b\\OK", "OK", 1, 0),
        ("\x1bXpayload\x1b\\OK", "OK", 1, 0),
        ("\x1b^payload\x1b\\OK", "OK", 1, 0),
        ("\x1b_payload\x1b\\OK", "OK", 1, 0),
        ("\x90payload\x9cOK", "OK", 1, 0),
        ("\x98payload\x9cOK", "OK", 1, 0),
        ("\x9epayload\x9cOK", "OK", 1, 0),
        ("\x9fpayload\x9cOK", "OK", 1, 0),
        ("\x1b(Bascii\x1b7", "ascii", 2, 0),
        ("\x1b[31", "\\u001b[31", 0, 1),
        ("\x1b]title", "\\u001b]title", 0, 1),
        ("\x1bPunfinished", "\\u001bPunfinished", 0, 1),
        ("\x9dunfinished", "\\u009dunfinished", 0, 1),
        ("\x1b", "\\u001b", 0, 1),
        (
            "\0\a\b\v\f\r\x1b\x1f\x7f\x80\x85\x9f\t\n",
            "\\u0000\\u0007\\u0008\\u000b\\u000c\\u000d"
            "\\u001b\\u001f\\u007f\\u0080\\u0085\\u009f\t\n",
            0,
            12,
        ),
    ],
)
def test_terminal_text_has_visible_and_counted_transformations(
    raw: str, expected: str, sequences: int, controls: int
) -> None:
    assert terminal.sanitize_terminal_text(raw) == (expected, sequences, controls)


def test_repeated_unterminated_c1_strings_remain_visible() -> None:
    raw = "\x9d\x90" * 4096
    assert terminal.sanitize_terminal_text(raw) == (
        "\\u009d\\u0090" * 4096,
        0,
        8192,
    )
