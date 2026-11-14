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
    # Cached references of a paper that this search did not expand are outside
    # its evidence, even though the target is known in the global store.
    search = metadata(tmp_path / "citation_map/searches/search-fixture-comparison-depth1.md")
    search.update(id="search-unexpanded-root", max_depth=0, generations=[{"depth": 0, "paper_ids": [ids["A"]], "expanded_ids": []}], issues=[])
    (tmp_path / "citation_map/searches/search-unexpanded-root.md").write_text("---\n" + yaml.safe_dump(search) + "---\n")
    for search_id, citation in (("search-unexpanded-root", ids["C"]), ("search-fixture-comparison-depth2", "cite-0000000000000000")):
        result = command(tmp_path, "kg_papers.py", "--id", "paper-2024-unreached", "--tasks", "plcs", "--from-search", search_id, "--citation-id", citation, check=False)
        assert result.returncode == 2 and "not recorded by" in result.stderr
    # An unread leaf has known bibliography but no invented research summary.
    result = command(tmp_path, "kg_papers.py", "--id", "paper-2024-unread", "--tasks", "plcs", "--from-search", "search-fixture-comparison-depth2", "--citation-id", ids["D"], check=False)
    assert result.returncode == 2 and "nonempty summary" in result.stderr
    assert not (tmp_path / "Papers").exists()


def test_comparison_search_adopts_an_unexpanded_foundation_with_all_decisions(tmp_path: Path) -> None:
    ids = seed(tmp_path)
    command(tmp_path, "kg_papers.py", "--id", "paper-2024-selected-c", "--tasks", "plcs", "--from-search", "search-fixture-comparison-depth2", "--citation-id", ids["C"])
    data = metadata(tmp_path / "Papers/paper-2024-selected-c/paper.md")
    assert data["title"] == "Fixture C" and data["external_ids"] == {"doi": "10.5555/fixture-c"}
    paths = data["reason"].splitlines()[1:]
    assert len(paths) == 2
    assert f"Fixture A ({ids['A']}) / ref-2" in paths[0]
    assert "役割 foundation" in paths[0] and "判定状態 classified" in paths[0]
    assert "Section 3, paragraph 2; Table 1" in paths[0]
    assert f"Fixture A ({ids['A']}) / ref-5" in paths[1]
    assert "判定状態 unclassifiable" in paths[1] and "根拠 記載なし" in paths[1]
    command(tmp_path, "kg_validate.py")


@pytest.mark.parametrize(("search_suffix", "sources"), [("comparison-depth2", ["A", "B"]), ("mixed-depth2", ["A", "B", "C"])])
def test_adoption_records_every_citing_paper_within_the_search(tmp_path: Path, search_suffix: str, sources: list[str]) -> None:
    ids = seed(tmp_path)
    command(tmp_path, "kg_papers.py", "--id", "paper-2024-selected-d", "--tasks", "plcs", "--from-search", f"search-fixture-{search_suffix}", "--citation-id", ids["D"], "--summary", "Human-provided summary of the unread fixture D.")
    paths = metadata(tmp_path / "Papers/paper-2024-selected-d/paper.md")["reason"].splitlines()[1:]
    assert len(paths) == len(sources)
    for source, line in zip(sources, paths, strict=True):
        assert f"Fixture {source} ({ids[source]})" in line
        assert f"/ ref-{7 if source == 'A' else 1};" in line
        assert "判定状態 classified" in line and "Section 3, paragraph" in line
    assert "役割 component, related" in paths[0]  # Outside the selected traversal roles.
    assert "役割 comparison" in paths[1]
    if "C" in sources:
        assert "役割 comparison, foundation" in paths[2]
    command(tmp_path, "kg_validate.py")


def test_conflicting_external_ids_fail_registration_and_the_knowledge_gate(tmp_path: Path) -> None:
    ids = seed(tmp_path)
    args = ("--id", "paper-2024-conflicting-b", "--tasks", "plcs", "--title", "Fixture B", "--authors", "Fixture Author", "--source", "https://example.org/conflicting-b", "--summary", "Conflicting record", "--reason", "Research question", "--external-id", "doi=10.5555/conflicting-b", "--external-id", "arxiv=2401.00002")
    result = command(tmp_path, "kg_papers.py", *args, check=False)
    assert result.returncode == 2 and "conflicting external IDs" in result.stderr
    assert not (tmp_path / "Papers").exists()
    # Preexisting hand-edited Papers metadata must be rejected by the same rule.
    data = metadata(tmp_path / f"citation_map/papers/{ids['B']}.md")
    note = tmp_path / "Papers/paper-2024-conflicting-b/paper.md"
    note.parent.mkdir(parents=True)
    registered = {key: data[key] for key in ("title", "year", "authors", "source", "summary", "external_ids")}
    registered.update(id="paper-2024-conflicting-b", type="paper", tasks=["plcs"], reason="Research question")
    registered["external_ids"]["doi"] = "10.5555/conflicting-b"
    note.write_text("---\n" + yaml.safe_dump(registered) + "---\n")
    result = command(tmp_path, "kg_validate.py", check=False)
    assert result.returncode == 1 and "conflicting external IDs" in result.stdout
    assert "Traceback" not in result.stderr


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
