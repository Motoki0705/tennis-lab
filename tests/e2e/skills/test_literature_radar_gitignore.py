from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_literature_json_is_reincluded_from_gitignore() -> None:
    patterns = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "!knowledge/literature/**/*.json" in patterns
