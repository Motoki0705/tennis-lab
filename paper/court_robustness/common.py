"""Paths and byte provenance shared by the paper's reproduction scripts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PAPER_PAGES = 8
REPO = ROOT.parents[1]
MAIN = Path(
    subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        text=True,
    ).strip()
).parent


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sources() -> list[dict[str, Any]]:
    return json.loads((ROOT / "evidence/inputs.json").read_text())["images"]
