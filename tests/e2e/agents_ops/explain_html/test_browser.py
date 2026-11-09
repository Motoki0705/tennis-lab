"""One browser launch, five reader/editor scenarios, no external API or GPU."""

import json
import os
import subprocess
import time
from pathlib import Path

from tests.support.agents_ops.explain_html_ui import live_browser_server

REPO = Path(__file__).resolve().parents[4]


def test_browser_reader_editor_workflow(tmp_path: Path) -> None:
    setup = time.monotonic()
    with live_browser_server(tmp_path / "runtime") as live:
        setup_seconds = time.monotonic() - setup
        result = subprocess.run(
            ["node", "tests/e2e/agents_ops/explain_html_browser.cjs"],
            input=json.dumps({**live.config, "output": str(tmp_path / "browser")}),
            cwd=REPO,
            env=os.environ,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        print(f"serviceSetupSeconds={setup_seconds:.3f} {result.stdout}")
