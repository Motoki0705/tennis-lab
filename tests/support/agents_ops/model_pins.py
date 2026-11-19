"""Model expectations from the repository pins, with a pre-upgrade config."""

from __future__ import annotations

import shlex
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[3] / ".agents/ops/lib/agents.conf"


def pin(name: str) -> str:
    for line in CONFIG.read_text().splitlines():
        if line.startswith(name + "="):
            return shlex.split(line, comments=True)[0].split("=", 1)[1]
    raise KeyError(name)


WORKER_DEFAULT_MODEL = pin("CODEX_MODEL")
WORKER_ALTERNATE_MODEL = "gpt-6.2-astra"  # previously selectable
REVIEW_FIXED_MODEL = pin("REVIEW_CODEX_MODEL")
REVIEW_OTHER_MODEL = "gpt-6.2-astra"

OLD_SOL = "gpt-6.1-sol"
OLD_ASTRA = "gpt-6-astra"


def previous_config(
    path: Path, *, review_mode: str = "select", worker_model: str = OLD_SOL
) -> Path:
    previous = {
        "CODEX_MODEL": worker_model,
        "CODEX_ALTERNATE_MODEL": OLD_ASTRA,
        "REVIEW_CODEX_MODEL": OLD_ASTRA,
        "REVIEW_CODEX_ALLOWED_PAIRS": f"{OLD_SOL}:xhigh,{OLD_ASTRA}:xhigh",
        "REVIEW_SELECTION_MODE": review_mode,
    }
    omit = {"CODEX_CONTINUATION_MODELS", "REVIEW_CODEX_CONTINUATION_PAIRS"}
    lines = []
    for line in CONFIG.read_text().splitlines():
        key = line.partition("=")[0]
        if key in previous:
            line = f'{key}="{previous[key]}"'
        if key not in omit:
            lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    return path
