"""CLI input, environment-rooted output and nonzero exit boundaries."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / ".agents/skills/knowledge-control/scripts/kg_explore.py"


def test_depth_zero_cli_records_unread_root_without_starting_agy(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "arxiv:2311.05237", "--role", "comparison",
                             "--max-depth", "0", "--search-id", "search-cli-zero", "--agy", str(tmp_path / "absent-agy")],
                            env={**os.environ, "KNOWLEDGE_DIR": str(tmp_path)}, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["agy_calls"] == report["read_papers"] == 0
    assert report["status"] == "complete"
    papers = list((tmp_path / "citation_map/papers").glob("*.md"))
    assert len(papers) == 1
    assert yaml.safe_load(papers[0].read_text().split("---", 2)[1])["text_status"] == "not_requested"


def test_cli_rejects_invalid_depth_before_writing(tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "arxiv:2311.05237", "--role", "comparison", "--max-depth", "-1"],
                            env={**os.environ, "KNOWLEDGE_DIR": str(tmp_path)}, capture_output=True, text=True)
    assert result.returncode == 1 and "max_depth" in result.stderr
    assert not (tmp_path / "citation_map").exists()


@pytest.mark.parametrize("wrong_model", [False, True])
def test_cli_reads_public_html_with_fake_agy_and_propagates_failure(tmp_path: Path, wrong_model: bool) -> None:
    class PaperHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(("<article><h1>CLI paper</h1><p>" + "Article body. " * 150 + "</p><h2>References</h2></article>").encode())

        def log_message(self, *args: object) -> None:
            pass

    agy = tmp_path / "agy"
    agy.write_text(f"#!{sys.executable}\n" + '''
import json, sys
from pathlib import Path
args = sys.argv[1:]
if args == ["--version"]:
    print("1.5.9")
else:
    label = "WRONG_MODEL" if WRONG_MODEL else "Gemini 4 Hydrogen (High)"
    Path(args[args.index("--log-file") + 1]).write_text('Propagating selected model override to backend: label="' + label + '"\\n')
    print(json.dumps({"event":"result", "result":{"status":"SUCCESS", "structured_output":{
        "title":"CLI paper", "year":2023, "authors":["Author"], "summary":"CLI fixture summary.", "references":[]}}}))
'''.replace("if WRONG_MODEL else", f"if {wrong_model!r} else"))
    agy.chmod(0o755)
    base = tmp_path / "knowledge"
    with HTTPServer(("127.0.0.1", 0), PaperHandler) as server:
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        try:
            source = f"http://127.0.0.1:{server.server_port}/paper"
            result = subprocess.run([sys.executable, str(SCRIPT), source, "--role", "comparison", "--max-depth", "1",
                                     "--search-id", "search-cli-reading", "--agy", str(agy), "--run-dir", str(tmp_path / "private")],
                                    env={**os.environ, "HOME": str(tmp_path / "safe-home"), "KNOWLEDGE_DIR": str(base)},
                                    capture_output=True, text=True)
        finally:
            server.shutdown()
            thread.join(timeout=5)
    search = yaml.safe_load((base / "citation_map/searches/search-cli-reading.md").read_text().split("---", 2)[1])
    paper = yaml.safe_load(next((base / "citation_map/papers").glob("*.md")).read_text().split("---", 2)[1])
    if wrong_model:
        assert result.returncode == 1 and "agy:model_mismatch" in result.stderr
        assert search["status"] == "interrupted" and paper["text_status"] == "not_requested"
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["agy_calls"] == 1
        assert paper["text_status"] == "available" and paper["summary"] == "CLI fixture summary."
        assert search["generations"][0]["expanded_ids"] == [paper["id"]]
