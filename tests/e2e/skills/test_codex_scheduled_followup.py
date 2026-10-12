"""Exercise the CLI follow-up helper without touching a real scheduler or queue."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SKILL = ROOT / ".agents/skills/codex-scheduled-followup"


def test_packaged_skill_metadata_and_references() -> None:
    text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    metadata = yaml.safe_load(text.split("---", 2)[1])
    assert metadata["name"] == SKILL.name
    assert isinstance(metadata["description"], str) and metadata["description"].strip()
    for document in (SKILL / "SKILL.md", *sorted((SKILL / "references").glob("*.md"))):
        for link in re.findall(
            r"\[[^]]+\]\(([^)]+)\)", document.read_text(encoding="utf-8")
        ):
            if "://" not in link:
                assert (document.parent / link.split("#")[0]).is_file(), (
                    document,
                    link,
                )
    ui = yaml.safe_load((SKILL / "agents/openai.yaml").read_text(encoding="utf-8"))
    assert f"${SKILL.name}" in ui["interface"]["default_prompt"]
    assert 25 <= len(ui["interface"]["short_description"]) <= 64


@pytest.mark.parametrize("script_name", ["test_cli_heartbeat.py", "test_adopt_rollout.py"])
def test_cli_helper_regressions(tmp_path: Path, script_name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SKILL / "scripts" / script_name)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
