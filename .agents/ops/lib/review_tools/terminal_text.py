"""Counted terminal-control neutralization for text received from gh."""

from __future__ import annotations

import re

# Complete ECMA-48 sequences, in their 7-bit and 8-bit forms. Do not treat an
# unterminated CSI/control string as a short ESC sequence: its ESC stays visible.
# Exclude embedded ESC/C1 from string payloads so repeated unterminated
# introducers cannot cause quadratic searches for a terminator.
ANSI_SEQUENCE = re.compile(
    r"(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]"
    r"|(?:\x1b\]|\x9d)[^\x07\x1b\x80-\x9f]*(?:\x07|\x1b\\|\x9c)"
    r"|(?:\x1b[PX^_]|[\x90\x98\x9e\x9f])[^\x1b\x80-\x9f]*(?:\x1b\\|\x9c)"
    r"|\x1b(?![\[\]PX^_])[ -/]*[0-~]"
)
CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize_terminal_text(text: str) -> tuple[str, int, int]:
    """Strip complete ANSI sequences; show remaining C0/DEL/C1 except tab/LF."""
    text, sequences = ANSI_SEQUENCE.subn("", text)
    text, controls = CONTROL_CHARACTER.subn(
        lambda match: f"\\u{ord(match[0]):04x}", text
    )
    return text, sequences, controls
