"""Public registration/validation boundaries using the shared fictional citation map."""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / ".agents/skills/knowledge-control/scripts"
FIXTURE = ROOT / "tests/fixtures/knowledge/citation_map"


def command(base: Path, name: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=ROOT,
                          env={**os.environ, "KNOWLEDGE_DIR": str(base)}, capture_output=True, text=True, check=check)


def metadata(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text().split("---", 2)[1])  # type: ignore[no-any-return]


def seed(base: Path) -> dict[str, str]:
    (base / "nodes").mkdir()
    registry = base / "registry"
    registry.mkdir()
    (registry / "tasks.yaml").write_text("schema_version: 1\ntasks:\n  plcs: {status: active, note: fixture}\n")
    (registry / "deployments.yaml").write_text("schema_version: 1\nconfigs: []\ndeployments: []\n")
    shutil.copyfile(ROOT / "knowledge/registry/relations.yaml", registry / "relations.yaml")
    shutil.copytree(FIXTURE, base / "citation_map")
    return {metadata(p)["title"][-1]: metadata(p)["id"] for p in (base / "citation_map/papers").glob("*.md")}


def test_explicit_search_adoption_records_path_and_joins_by_external_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = seed(tmp_path)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in (tmp_path / "citation_map").rglob("*.md")}
    command(tmp_path, "kg_validate.py")
    assert not (tmp_path / "Papers").exists()  # Discovery/validation alone never adopts.
    command(tmp_path, "kg_papers.py", "--id", "paper-2024-selected-b", "--tasks", "plcs", "--from-search", "search-fixture-comparison-depth2", "--citation-id", ids["B"])
    note = tmp_path / "Papers/paper-2024-selected-b/paper.md"
    data = metadata(note)
    assert data["title"] == "Fixture B"
    assert data["source"] == "https://example.org/fixture-b"
    assert data["summary"].startswith("Fictional B")
    assert data["external_ids"] == {"doi": "10.5555/fixture-b", "arxiv": "2401.00002", "openalex": "W1000002"}
    for token in ("search-fixture-comparison-depth2", "Fixture A", ids["A"], "ref-1", "comparison", "foundation", "Section 3, paragraph 1", "ref-6"):
        assert token in data["reason"]
    assert {p.name for p in note.parent.iterdir()} == {"paper.md"}
    assert {p.name for p in (tmp_path / "Papers").iterdir()} == {"paper-2024-selected-b"}
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path))
    citations = importlib.import_module("kg_citations")
    papers = importlib.import_module("kg_papers")
    assert citations.papers_join(citations.load_citations(), papers.load_papers()) == {ids["B"]: "paper-2024-selected-b"}
    command(tmp_path, "kg_validate.py")
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in (tmp_path / "citation_map").rglob("*.md")}
    duplicate = command(tmp_path, "kg_papers.py", "--id", "paper-2024-alias-b", "--tasks", "plcs", "--title", "Alias B", "--authors", "Author", "--source", "https://example.org/alias-b", "--summary", "Alias", "--reason", "Research question", "--external-id", "arxiv=2401.00002v4", check=False)
    assert duplicate.returncode == 2 and "duplicate literature" in duplicate.stderr
    assert not (tmp_path / "Papers/paper-2024-alias-b").exists()


def test_adoption_rejects_unreached_and_unknown_evidence_before_writing(tmp_path: Path) -> None:
    ids = seed(tmp_path)
    for citation in (ids["C"], "cite-0000000000000000"):
        result = command(tmp_path, "kg_papers.py", "--id", "paper-2024-unreached", "--tasks", "plcs", "--from-search", "search-fixture-comparison-depth2", "--citation-id", citation, check=False)
        assert result.returncode == 2 and "was not reached" in result.stderr
    # An unread leaf has known bibliography but no invented research summary.
    result = command(tmp_path, "kg_papers.py", "--id", "paper-2024-unread", "--tasks", "plcs", "--from-search", "search-fixture-comparison-depth2", "--citation-id", ids["D"], check=False)
    assert result.returncode == 2 and "nonempty summary" in result.stderr
    assert not (tmp_path / "Papers").exists()


def test_knowledge_gate_validates_citation_documents_and_storage_paths(tmp_path: Path) -> None:
    ids = seed(tmp_path)
    path = tmp_path / f"citation_map/papers/{ids['A']}.md"
    raw = path.read_text()
    path.write_text(raw.replace("comparison", "not-a-role", 1))
    result = command(tmp_path, "kg_validate.py", check=False)
    assert result.returncode == 1 and "roles" in result.stdout and "Traceback" not in result.stderr
    path.write_text(raw)
    extra = path.with_name("copied.md")
    shutil.copyfile(path, extra)
    result = command(tmp_path, "kg_validate.py", check=False)
    assert result.returncode == 1 and ("duplicate document id" in result.stdout or "match path" in result.stdout)
    extra.unlink()
    attachment = path.parent / "full-text.pdf"
    attachment.write_bytes(b"forbidden fixture attachment")
    result = command(tmp_path, "kg_validate.py", check=False)
    assert result.returncode == 1 and "only regular Markdown" in result.stdout


@pytest.mark.parametrize("missing", ["source", "summary", "reason"])
def test_manual_registration_requires_primary_url_summary_and_reason(tmp_path: Path, missing: str) -> None:
    seed(tmp_path)
    options = {"title": "Human-selected paper", "source": "https://example.org/manual", "summary": "A research contribution", "reason": "A research question"}
    args = [value for key, value in options.items() if key != missing for value in (f"--{key}", value)]
    result = command(tmp_path, "kg_papers.py", "--id", "paper-2024-manual", "--authors", "Author", "--tasks", "plcs", *args, check=False)
    assert result.returncode == 2 and "primary source URL, nonempty summary and reason required" in result.stderr
    assert not (tmp_path / "Papers").exists()
