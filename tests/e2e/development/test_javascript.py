"""Pytest entrypoints for the existing Node and self-contained browser suites."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = sorted(
    path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").rglob("*.test.mjs")
)


@pytest.mark.parametrize("script", SCRIPTS)
def test_node_regressions(script: str) -> None:
    subprocess.run(["node", "--test", script], cwd=ROOT, check=True, timeout=120)


def test_detection_refresh_race_in_chromium() -> None:
    subprocess.run(
        ["node", "tests/e2e/tasks/detection/refresh_race_browser.cjs"],
        cwd=ROOT,
        check=True,
        timeout=120,
    )
